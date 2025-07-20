import os
import math
import argparse
import shortuuid
import json
from typing import Sequence, Mapping

import numpy as np
import torch
import trimesh
from tqdm import tqdm

# -------------------------------------------------------------
# 🟢 project-specific paths — make sure these are reachable
# -------------------------------------------------------------
import sys
current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
submodules_dir = os.path.join(current_script_dir, '3D-LLaVA')
sys.path.append(submodules_dir)

from segmentator import segment_point, compute_vn
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_special_token
from llava.pc_utils import referseg_transform_eval, Compose
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from pointgroup_ops import voxelization_idx

from collections.abc import Sequence, Mapping
from torch.utils.data.dataloader import default_collate

class ModelWrapper:
    def __init__(self, model_path, model_base, pointcloud_tower_name=None, conv_mode="llava_v1", temperature=0.2, top_p=None, num_beams=1):
        disable_torch_init()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv_mode = conv_mode
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams

        model_name = get_model_name_from_path(model_path)
        tokenizer, model, _, _ = load_pretrained_model(model_path, model_base, model_name, pointcloud_tower_name=pointcloud_tower_name)
        self.model = model.eval().to(self.device)
        self.tokenizer = tokenizer

        self.templates = "<image>\n Please output the segmentation mask according to the following description. \n{description}"

        print("[✓] 模型加载完成")

    def ponder_collate_fn(self, batch, max_point=-1):
        if not isinstance(batch, Sequence):
            raise TypeError(f"{type(batch)} is not supported.")

        if max_point > 0:
            accum = 0
            kept = []
            for sample in batch:
                num = sample["coord"].shape[0]
                if accum + num > max_point:
                    continue
                accum += num
                kept.append(sample)
            return self.ponder_collate_fn(kept)

        if isinstance(batch[0], torch.Tensor):
            return torch.cat(list(batch))
        if isinstance(batch[0], np.ndarray):
            return torch.from_numpy(np.concatenate(batch, axis=0))
        if isinstance(batch[0], str):
            return list(batch)

        if isinstance(batch[0], Sequence):
            for b in batch:
                b.append(torch.tensor([b[0].shape[0]]))
            collated = [self.ponder_collate_fn(samples) for samples in zip(*batch)]
            collated[-1] = torch.cumsum(collated[-1], dim=0).int()
            return collated

        if isinstance(batch[0], Mapping):
            collated = {k: self.ponder_collate_fn([d[k] for d in batch]) for k in batch[0]}
            for k in collated:
                if "offset" in k:
                    collated[k] = torch.cumsum(collated[k], dim=0)
            return collated

        return default_collate(batch)

    def preprocess_pointcloud(self, ply_path):
        mesh = trimesh.load_mesh(ply_path)
        coords = mesh.vertices
        colors = mesh.visual.vertex_colors[:, :3]
        vertices = torch.from_numpy(coords.astype(np.float32))
        normals = torch.from_numpy(compute_vn(mesh).astype(np.float32))
        edges = torch.from_numpy(mesh.edges.astype(np.int64))
        superpoint_mask = segment_point(vertices, normals, edges).numpy()

        transform = Compose(referseg_transform_eval)
        pc_data_dict = dict(coord=coords, color=colors, superpoint_mask=superpoint_mask)
        pc_data_dict = transform(pc_data_dict)

        grid_coord = pc_data_dict['grid_coord']
        grid_coord = torch.cat([torch.LongTensor(grid_coord.shape[0], 1).fill_(0), grid_coord], 1)
        pc_data_dict['grid_coord'] = grid_coord

        spatial_shape = np.clip((grid_coord.max(0)[0][1:] + 1).numpy(), 128, None)
        voxel_coords, p2v_map, v2p_map = voxelization_idx(grid_coord, 1, 4)

        for key in ["coord", "grid_coord", "feat", "offset", "condition"]:
            if key in pc_data_dict:
                pc_data_dict[key] = self.ponder_collate_fn([pc_data_dict[key]])

        if "feat" in pc_data_dict and pc_data_dict["feat"].dim() == 3 and pc_data_dict["feat"].shape[0] == 1:
            pc_data_dict["feat"] = pc_data_dict["feat"].squeeze(0)

        return coords, colors, pc_data_dict, voxel_coords, p2v_map, v2p_map, spatial_shape, superpoint_mask

    def inference(self, pc_data_dict, voxel_coords, p2v_map, v2p_map, spatial_shape, query_text, superpoint_mask):
        conv = conv_templates[self.conv_mode].copy()
        query = self.templates.format(description=query_text)
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_special_token(prompt, self.tokenizer, return_tensors='pt').unsqueeze(0).to(self.device)
        coord = pc_data_dict["coord"].to(self.device, dtype=torch.bfloat16)
        offset = pc_data_dict["offset"].to(self.device)
        feat = pc_data_dict["feat"].to(self.device, dtype=torch.bfloat16)
        voxel_coords = voxel_coords.to(self.device)
        p2v_map = p2v_map.to(self.device)
        v2p_map = v2p_map.to(self.device)
        superpoint_tensor = [torch.tensor(superpoint_mask).to(self.device)]

        with torch.inference_mode():
            pred_mask = self.model.generate(
                input_ids,
                coord=coord,
                grid_coord=voxel_coords,
                offset=offset,
                feat=feat,
                p2v_map=p2v_map,
                v2p_map=v2p_map,
                spatial_shape=spatial_shape,
                superpoint_mask=superpoint_tensor,
                conditions=pc_data_dict["condition"],
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=64,
                tokenizer=self.tokenizer,
                click_mask=[[]],
                use_cache=True
            )
        return pred_mask.cpu().numpy().astype(bool)[0]

    def visualize_and_save(self, coords, colors, pred_mask, output_path):
        new_colors = colors.copy()
        highlight = np.array([255, 0, 0], dtype=np.uint8)
        new_colors[pred_mask] = highlight
        pc = trimesh.PointCloud(vertices=coords, colors=new_colors)
        pc.export(output_path)
        print(f"[✓] 可视化结果已保存到 {output_path}")

    def calculate_metrics(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
        """
        计算评估指标
        
        Args:
            pred_mask: 预测掩码 (布尔数组)
            gt_mask: 真实掩码 (布尔数组)
            
        Returns:
            包含 IoU、TP50、TP25 指标的字典
        """
        # 确保掩码是布尔类型
        pred_mask = pred_mask.astype(bool)
        gt_mask = gt_mask.astype(bool)
        
        # 计算交集和并集
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        # 计算 IoU
        iou = intersection / union if union > 0 else 0.0
        
        # 计算 TP50 (IoU > 0.5)
        tp50 = 1 if iou > 0.5 else 0
        
        # 计算 TP25 (IoU > 0.25)
        tp25 = 1 if iou > 0.25 else 0
        
        return {
            "iou": float(iou),
            "tp50": tp50,
            "tp25": tp25
        }

    def save_prediction_jsonl(self, scene_id: str, question_id: int, query_text: str, 
                            pred_mask: np.ndarray, gt_mask: np.ndarray, 
                            model_id: str, output_path: str) -> None:
        """
        将预测结果保存为评估格式的 JSONL
        
        Args:
            scene_id: 场景ID
            question_id: 问题ID
            query_text: 查询文本
            pred_mask: 预测掩码 (布尔数组)
            gt_mask: 真实掩码 (布尔数组)
            model_id: 模型ID
            output_path: 输出文件路径
        """
        # 构建完整的提示词
        full_prompt = self.templates.format(description=query_text)
        
        # 计算评估指标
        metrics = self.calculate_metrics(pred_mask, gt_mask)
        
        # 将掩码转换为整数列表 (0/1)
        pred_mask_list = pred_mask.astype(int).tolist()
        gt_mask_list = gt_mask.astype(int).tolist()
        
        # 构建 JSON 记录
        record = {
            "scene_id": scene_id,
            "question_id": question_id,
            "prompt": full_prompt,
            "model_id": model_id,
            "gt_mask": gt_mask_list,
            "pred_mask": pred_mask_list,
            "iou": metrics["iou"],
            "tp50": metrics["tp50"],
            "tp25": metrics["tp25"]
        }
        
        # 写入 JSONL 文件 - 使用覆盖模式
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"[✓] 评估结果已保存到 {output_path}")
        print(f"    场景ID: {scene_id}, 问题ID: {question_id}")
        print(f"    IoU: {metrics['iou']:.4f}, TP50: {metrics['tp50']}, TP25: {metrics['tp25']}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # optional arguments
    parser.add_argument(
        "--model_path",
        default="./checkpoints/finetune-3d-llava-lora-0611-V1-bs-2-gpu8",
        type=str,
        required=True,
        help='3d llava local model path.',
    )
    parser.add_argument(
        "--model_base",
        default="./checkpoints/llava-v1.5-7b",
        type=str,
        required=True,
        help='3d llava local model base.',
    )
    parser.add_argument(
        "--ply_path",
        default="example_data/demo/sparse/0/points3D_mesh.ply",
        type=str,
        required=True,
        help='input mesh path.',
    )
    parser.add_argument(
        "--output_dir",
        default="example_data/demo/visual_grounding",
        type=str,
        required=True,
        help='output path.',
    )
    parser.add_argument(
        "--scene_id",
        default="scene0000_00",
        type=str,
        help='scene ID for evaluation.',
    )
    parser.add_argument(
        "--gt_mask_path",
        type=str,
        help='path to ground truth mask file (.npy or .txt)',
    )
    args = parser.parse_args()
    
    model = ModelWrapper(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode="llava_v1"
    )

    ply_path = args.ply_path
    coords, colors, pc_data, voxel_coords, p2v_map, v2p_map, spatial_shape, sp_mask = model.preprocess_pointcloud(ply_path)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    
    # 加载真实掩码（如果提供）
    gt_mask = None
    if args.gt_mask_path and os.path.exists(args.gt_mask_path):
        if args.gt_mask_path.endswith('.npy'):
            gt_mask = np.load(args.gt_mask_path)
        elif args.gt_mask_path.endswith('.txt'):
            gt_mask = np.loadtxt(args.gt_mask_path, dtype=int)
        print(f"[✓] 已加载真实掩码: {args.gt_mask_path}")
    else:
        print("[⚠] 未提供真实掩码，将使用随机掩码进行演示")
        # 创建随机掩码用于演示
        gt_mask = np.random.choice([0, 1], size=len(coords), p=[0.9, 0.1]).astype(bool)
    
    question_id = 0
    
    while True:
        try:
            query = input("\n请输入目标描述（输入 q 退出）：\n> ")
            if query.lower() == "q":
                break
            pred = model.inference(pc_data, voxel_coords, p2v_map, v2p_map, spatial_shape, query, sp_mask)
            
            # 创建 JSONL 文件路径
            jsonl_path = os.path.join(output_dir, f"{args.scene_id}_results_{question_id}.jsonl")
            # 保存可视化结果
            output_file = f"{output_dir}/query_{question_id}.ply"
            model.visualize_and_save(coords, colors, pred, output_file)
            
            # 保存评估格式的 JSONL 结果
            model.save_prediction_jsonl(
                scene_id=args.scene_id,
                question_id=question_id,
                query_text=query,
                pred_mask=pred,
                gt_mask=gt_mask,
                model_id=os.path.basename(args.model_path),
                output_path=jsonl_path
            )
            
            question_id += 1
            
        except Exception as e:
            print(f"处理过程中出现错误: {e}")
            continue
