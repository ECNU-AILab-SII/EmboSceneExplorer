#!/bin/bash

###### 指定场景
python ../src/navigation/nav.py \
    --yaml_path ../submodules/habitat-lab/habitat-lab/habitat/config/benchmark/nav/pointnav/pointnav_scannet.yaml \
    --ply_path ../example_data/demo/visual_grounding/query_0.ply \
    --jsonl_path ../example_data/demo/visual_grounding/scene0000_00_results_0.jsonl \
    --output_path ../example_data/demo/navigation \