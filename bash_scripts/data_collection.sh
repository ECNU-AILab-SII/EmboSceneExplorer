#!/bin/bash

###### 1. 指定场景和动作文件采集
# python ../src/data_collect/viewer_server.py \
#     --scene data/datasets/Scannet/scans/scene0000_00/scene0000_00_vh_clean.glb \
#     --dataset data/datasets/Scannet/scene_dataset_config.json \
#     --action_path data/datasets/Scannet/scans/scene0000_00/action.txt \
#     --output_path output/demo/scene0000_00-7-10 \
#     --sensor_height 0.7 \
#     --feq 20

####### 2. 指定场景自动化采集
data_dir="../example_data/scanet/scans"
scene="scene0000_00"
echo "===================${scene} Begin============================="
glb_path="$data_dir/$scene/${scene}_vh_clean.glb"
config_path="$data_dir/../scene_dataset_config.json"
obs_saved_path="../example_data/demo"
echo "Using scene: $glb_path"
echo "Using config: $config_path"
echo "Output will be saved to: $obs_saved_path"
python ../src/data_collect/viewer_server.py \
    --scene $glb_path \
    --dataset $config_path \
    --output_path $obs_saved_path \
    --sensor_height 0.7 \
    --LOOK 36
echo "===================== ${scene} Done==========================="

######### 3. 遍历指定目录下的Scanet所有场景，自动化采集
# data_dir="data/datasets/Scannet/scans"
# for scene in $(ls $data_dir); do
#     echo "===================${scene} Begin============================="
#     glb_path="$data_dir/$scene/${scene}_vh_clean.glb"
#     config_path="$data_dir/../scene_dataset_config.json"
#     obs_saved_path="output/scanet/new/$scene"
#     echo "Using scene: $glb_path"
#     echo "Using config: $config_path"
#     echo "Output will be saved to: $obs_saved_path"
#     python ../src/data_collect/viewer_server.py \
#         --scene $glb_path \
#         --dataset $config_path \
#         --output_path $obs_saved_path \
#         --sensor_height 0.7
#         --LOOK 2.8
#     echo "===================== ${scene} Done==========================="
#     # zip -q -r output/scanet/new.zip output/scanet/new
# done



