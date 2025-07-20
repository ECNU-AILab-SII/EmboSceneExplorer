# !/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
cd /inspire/hdd/global_user/xieyuan-24039/sjm/3D-LLaVA
source /opt/conda/etc/profile.d/conda.sh
conda activate llava
export PYTHONPATH=$(pwd):$PYTHONPATH
export WANDB_MODE=disabled

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_scan2cap \
#         --scan-folder /inspire/hdd/global_user/xieyuan-24039/Dataset/3D-LLaVA-Data/scannet \
#         --mask3d-inst-folder /inspire/hdd/global_user/xieyuan-24039/Dataset/3D-LLaVA-Data/eval_info/densecap_scanrefer/mask3d_inst_seg \
#         --model-path /inspire/hdd/global_user/xieyuan-24039/sjm/3D-LLaVA/checkpoints/finetune-3d-llava-lora-0527-V5-bs-2 \
#         --model-base /inspire/hdd/global_user/xieyuan-24039/sjm/LLaVA-3D-bf/llava-v1.5-7b \
#         --question-file /inspire/hdd/global_user/xieyuan-24039/Dataset/3D-LLaVA-Data/eval_info/densecap_scanrefer/scan2cap_mask3d_val.json \
#         --answers-file ./results/predictions/${EXP_NAME}/densecap_scanrefer/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --conv-mode vicuna_v1 &
# done

# wait

output_file=./results/predictions/${EXP_NAME}/densecap_scanrefer/merge.jsonl

# Clear out the output file if it exists.
# > "$output_file"

# Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat ./results/predictions/${EXP_NAME}/densecap_scanrefer/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done


python llava/eval/eval_scan2cap.py \
--pred-instance-attribute-file /inspire/hdd/global_user/xieyuan-24039/Dataset/3D-LLaVA-Data/eval_info/densecap_scanrefer/scannet_mask3d_val_attributes.pt \
--gt-instance-attribute-file /inspire/hdd/global_user/xieyuan-24039/Dataset/3D-LLaVA-Data/eval_info/densecap_scanrefer/scannet_val_attributes.pt \
--annotation-file /inspire/hdd/global_user/xieyuan-24039/Dataset/3D-LLaVA-Data/eval_info/densecap_scanrefer/scan2cap_val_corpus.json \
--result-file $output_file \


# Recently, we check the open source code of Video-3DLLM, finding the evaluation metric of Video-3D-LLM is slightly different from ours. Specifically, Video-3DLLM does not exclude "." and "," when compute the scores, resulting in higher result. We also provide the same evaluation to benefit the following works for conducting comparison under the same metric

# python llava/eval/eval_scan2cap_video3dllm_type.py \
# --pred-instance-attribute-file ./playground/data/eval_info/densecap_scanrefer/scannet_mask3d_val_attributes.pt \
# --gt-instance-attribute-file ./playground/data/eval_info/densecap_scanrefer/scannet_val_attributes.pt \
# --annotation-file ./playground/data/eval_info/densecap_scanrefer/scan2cap_val_corpus.json \
# --result-file $output_file \
