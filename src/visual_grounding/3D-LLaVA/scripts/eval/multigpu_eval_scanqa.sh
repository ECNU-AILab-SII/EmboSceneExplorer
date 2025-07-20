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

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_scanqa \
        --scan-folder /inspire/hdd/global_user/xieyuan-24039/Dataset/3D-LLaVA-Data/scannet/val \
        --model-path /inspire/hdd/global_user/xieyuan-24039/sjm/3D-LLaVA/checkpoints/finetune-3d-llava-lora-0527-V5-bs-2 \
        --model-base /inspire/hdd/global_user/xieyuan-24039/sjm/LLaVA-3D-bf/llava-v1.5-7b \
        --question-file /inspire/hdd/global_user/xieyuan-24039/Dataset/3D-LLaVA-Data/eval_info/scanqa/scanqa_val_question.jsonl \
        --answers-file ./results/predictions/$EXP_NAME/scanqa/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./results/predictions/$EXP_NAME/scanqa/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./results/predictions/$EXP_NAME/scanqa/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python llava/eval/eval_scanqa.py \
    --annotation-file /inspire/hdd/global_user/xieyuan-24039/Dataset/3D-LLaVA-Data/eval_info/scanqa/scanqa_val_answer.jsonl \
    --result-file $output_file
 