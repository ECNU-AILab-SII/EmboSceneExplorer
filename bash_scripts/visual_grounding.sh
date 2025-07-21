#!/bin/bash


CUDA_VISIBLE_DEVICES=1 python ../src/visual_grounding/interactive_segmentator.py \
        --model_path ../src/visual_grounding/checkpoints/finetune-3d-llava-lora-0611-V2-bs-2-gpu8 \
        --model_base ../src/visual_grounding/checkpoints/llava-v1.5-7b \
        --ply_path ../example_data/demo/sparse/0/points3D_scanet_mesh.ply \
        --output_dir ../example_data/demo/visual_grounding