# !/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,1,2,3
cd /inspire/hdd/global_user/xieyuan-24039/sjm/3D-llava-0705
source /opt/conda/etc/profile.d/conda.sh
conda activate llava
export PYTHONPATH=$(pwd):$PYTHONPATH
export WANDB_MODE=disabled

export PYTHONPATH=$(pwd)


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

EXP_NAME=finetune-3d-llava-lora-0707-GMM1CL01COT0-margin-0.3-V1-bs-2-gpu8

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_scanrefer \
        --scan-folder /inspire/hdd/global_user/xieyuan-24039/Dataset/3D-LLaVA-Data/scannet/val \
        --model-path /inspire/hdd/global_user/xieyuan-24039/sjm/3D-llava-0705/checkpoints/finetune-3d-llava-lora-0707-GMM1CL01COT0-margin-0.3-V1-bs-2-gpu8 \
        --model-base /inspire/hdd/global_user/xieyuan-24039/sjm/LLaVA-3D-bf/llava-v1.5-7b \
        --question-file /inspire/hdd/global_user/xieyuan-24039/Dataset/3D-LLaVA-Data/eval_info/referseg_scanrefer/ScanRefer_filtered_val.json \
        --answers-file ./results/predictions/$EXP_NAME/referseg_scanrefer/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./results/predictions/$EXP_NAME/referseg_scanrefer/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./results/predictions/$EXP_NAME/referseg_scanrefer/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python llava/eval/eval_refer_seg.py \
    --result-file $output_file
