import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid


from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_LINK_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.pc_utils import referseg_transform_eval, Compose
from PIL import Image
from llava.mm_utils import tokenizer_special_token
import math
import pathlib
import numpy as np
from llava.train.train import DataCollatorForSupervisedDataset
from pointgroup_ops import voxelization_idx
from typing import Dict, Optional, Sequence, List


templates = [
    "<image>\n Please output the segmentation mask according to the following description. \n{description}"
]


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def ponder_collate_fn(batch, max_point=-1):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    # we drop a large data if it exceeds max_point
    # note that directly drop the last one may cause problem
    if max_point > 0:
        accum_num_points = 0
        ret_batches = []
        for batch_id, data in enumerate(batch):
            num_coords = data["coord"].shape[0]
            if accum_num_points + num_coords > max_point:
                continue
            accum_num_points += num_coords
            ret_batches.append(data)
        return ponder_collate_fn(ret_batches)

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [ponder_collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch = {key: ponder_collate_fn([d[key] for d in batch]) for key in batch[0]}
        for key in batch.keys():
            if "offset" in key:
                batch[key] = torch.cumsum(batch[key], dim=0)
        return batch
    else:
        from torch.utils.data.dataloader import default_collate
        return default_collate(batch)


def eval_model(args):
    with open(args.question_file, 'r') as f:
        questions = json.load(f)
        all_list=[]
    for idx, source in enumerate(tqdm(questions)):
        qs=source['conversations'][0]['value'].split("\n")[-1]
        all_list.append(len(qs))
        if len(all_list)>10000:
            all_list.mean()
            sum(all_list) / len(all_list)
        continue

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, _, _ = load_pretrained_model(model_path, args.model_base, model_name, pointcloud_tower_name=args.pointcloud_tower_name)

    with open(args.question_file, 'r') as f:
        questions = json.load(f)
        
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    record = dict()
    for idx, source in enumerate(tqdm(questions)):
        scan_file = source['scene_id']
        if scan_file!="scene0000_00":
            continue
        scan_folder = args.scan_folder
        scan_data_path  = pathlib.Path(scan_folder) / f'{scan_file}.pth'
        superpoint_path = pathlib.Path(scan_folder) / '../super_points' / f'{scan_file}.bin'

        raw_data = torch.load(scan_data_path)
        coord = raw_data['coord']
        color = raw_data['color']
        superpoint_mask = np.fromfile(superpoint_path, dtype=np.int64)

        instance = raw_data['instance_gt']
        object_id = int(source["object_id"])

        gt_mask = instance == object_id

        # data transformation
        transform = Compose(referseg_transform_eval)
        pc_data_dict = dict(
            coord=coord,
            color=color,
            superpoint_mask=superpoint_mask
        )
        pc_data_dict = transform(pc_data_dict)

        grid_coord = pc_data_dict['grid_coord']
        grid_coord = torch.cat([torch.LongTensor(grid_coord.shape[0], 1).fill_(0), grid_coord], 1)
        pc_data_dict['grid_coord'] = grid_coord

        # voxelize
        batch_size = 1
        grid_coords = pc_data_dict["grid_coord"]
        spatial_shape = np.clip((grid_coords.max(0)[0][1:] + 1).numpy(), 128, None)  # long [3]
        voxel_coords, p2v_map, v2p_map = voxelization_idx(grid_coords, batch_size, 4)

        for key in pc_data_dict:
            if key in ["coord", "grid_coord", "feat", "offset"]:
                pc_data_dict[key] = ponder_collate_fn([pc_data_dict[key]])

        # qs = source['description']
        qs=source['conversations'][0]['value'].split("\n")[-1]
        qs = templates[0].format(description=qs)

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        device = model.device
        input_ids = tokenizer_special_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).to(device)
        coord = pc_data_dict["coord"].to(device, dtype=torch.bfloat16)
        voxel_coords = voxel_coords.to(device)
        offset = pc_data_dict["offset"].to(device)
        feat = pc_data_dict["feat"].to(device, dtype=torch.bfloat16)
        p2v_map = p2v_map.to(device)
        v2p_map = v2p_map.to(device)
        superpoint_mask = [torch.tensor(superpoint_mask).to(device)]
        
        # move input tensors to gpu, defaut type is supposed to be bfloat16
        with torch.inference_mode():
            pred_mask = model.generate(
                input_ids,
                # images=image_tensor.unsqueeze(0).half().cuda(),
                # image_sizes=[image.size],
                coord=coord,
                grid_coord=voxel_coords,
                offset=offset,
                feat=feat,
                p2v_map=p2v_map,
                v2p_map=v2p_map,
                spatial_shape=spatial_shape,
                superpoint_mask=superpoint_mask,
                conditions=[pc_data_dict["condition"]],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=64,
                tokenizer=tokenizer,
                click_mask=[[]],
                use_cache=True)
        
        pred_mask = pred_mask.cpu().numpy().astype(bool)[0]
        gt_mask = gt_mask.astype(bool)

def api_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    scan_data_path= os.path.expanduser(args.scan_data_path)
    superpoint_path=os.path.expanduser(args.superpoint_path)
    qs = args.query_text
    
    
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, _, _ = load_pretrained_model(model_path, args.model_base, model_name, pointcloud_tower_name=args.pointcloud_tower_name)
    raw_data = torch.load(scan_data_path)
    coord = raw_data['coord']
    color = raw_data['color']
    superpoint_mask = np.fromfile(superpoint_path, dtype=np.int64)

    instance = raw_data['instance_gt']
    # object_id = int(source["object_id"])

    # gt_mask = instance == object_id

    # data transformation
    transform = Compose(referseg_transform_eval)
    pc_data_dict = dict(
        coord=coord,
        color=color,
        superpoint_mask=superpoint_mask
    )
    pc_data_dict = transform(pc_data_dict)
    grid_coord = pc_data_dict['grid_coord']
    grid_coord = torch.cat([torch.LongTensor(grid_coord.shape[0], 1).fill_(0), grid_coord], 1)
    pc_data_dict['grid_coord'] = grid_coord

    # voxelize
    batch_size = 1
    grid_coords = pc_data_dict["grid_coord"]
    spatial_shape = np.clip((grid_coords.max(0)[0][1:] + 1).numpy(), 128, None)  # long [3]
    voxel_coords, p2v_map, v2p_map = voxelization_idx(grid_coords, batch_size, 4)

    for key in pc_data_dict:
        if key in ["coord", "grid_coord", "feat", "offset"]:
            pc_data_dict[key] = ponder_collate_fn([pc_data_dict[key]])

    qs = templates[0].format(description=qs)

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    device = model.device
    input_ids = tokenizer_special_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).to(device)
    coord = pc_data_dict["coord"].to(device, dtype=torch.bfloat16)
    voxel_coords = voxel_coords.to(device)
    offset = pc_data_dict["offset"].to(device)
    feat = pc_data_dict["feat"].to(device, dtype=torch.bfloat16)
    p2v_map = p2v_map.to(device)
    v2p_map = v2p_map.to(device)
    superpoint_mask = [torch.tensor(superpoint_mask).to(device)]
    
    # move input tensors to gpu, defaut type is supposed to be bfloat16
    with torch.inference_mode():
        pred_mask = model.generate(
            input_ids,
            # images=image_tensor.unsqueeze(0).half().cuda(),
            # image_sizes=[image.size],
            coord=coord,
            grid_coord=voxel_coords,
            offset=offset,
            feat=feat,
            p2v_map=p2v_map,
            v2p_map=v2p_map,
            spatial_shape=spatial_shape,
            superpoint_mask=superpoint_mask,
            conditions=[pc_data_dict["condition"]],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            # no_repeat_ngram_size=3,
            max_new_tokens=64,
            tokenizer=tokenizer,
            click_mask=[[]],
            use_cache=True)
    
    pred_mask = pred_mask.cpu().numpy().astype(bool)[0]
    return pred_mask


        




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-data-path", type=str, default="/inspire/hdd/global_user/xieyuan-24039/Dataset/3D-LLaVA-Data/scannet/train/scene0000_00.pth")
    parser.add_argument("--superpoint-path", type=str, default="/inspire/hdd/global_user/xieyuan-24039/Dataset/3D-LLaVA-Data/scannet/train/../super_points/scene0000_00.bin")
    parser.add_argument("--query-text", type=str, default="a white cabinet in the corner of the room. in the direction from the door and from the inside . it will be on the left, there is a small brown table on the left side of the cabinet and a smaller table on the right side of the cabinet")
    parser.add_argument("--model-path", type=str, default="/inspire/hdd/global_user/xieyuan-24039/sjm/3D-LLaVA/checkpoints/finetune-3d-llava-lora-0611-V1-bs-2-gpu8")
    parser.add_argument("--model-base", type=str, default="/inspire/hdd/global_user/xieyuan-24039/sjm/LLaVA-3D-bf/llava-v1.5-7b")
    parser.add_argument("--pointcloud-tower-name", type=str, default=None)
    parser.add_argument("--scan-folder", type=str, default="/inspire/hdd/global_user/xieyuan-24039/Dataset/3D-LLaVA-Data/scannet/train")
    parser.add_argument("--question-file", type=str, default="/inspire/hdd/global_user/xieyuan-24039/Dataset/3D-LLaVA-Data/train_info/scanrefer_train_3d_llava.json")
    parser.add_argument("--answers-file", type=str, default="./results/predictions/finetune-3d-llava-lora-0611-V1-bs-2-gpu8/referseg_scanrefer_demo/1_0.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--data_version", type=str, default="v0")
    args = parser.parse_args()
    # api_model(args)
    eval_model(args)
