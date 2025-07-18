'''
    支持接口：
        habitatocc.construct_occ()
        habitatocc.vis_local_occ()
        habitatocc.vis_global_occ()
        habitatocc.save_local_occ()
        habitatocc.save_global_occ()
'''
import os
# 必须在导入任何Mayavi相关库前设置
os.environ["ETS_TOOLKIT"] = "qt4"  # 改为使用qt4而不是null
os.environ["QT_API"] = "pyqt5"     # 明确指定Qt版本
os.environ["QT_QPA_PLATFORM"] = "offscreen"  # 强制使用离屏模式
os.environ["PYVISTA_OFF_SCREEN"] = "true"    # PyVista离屏模式
import numpy as np
np.bool = bool

import open3d as o3d
import imageio
import cv2
from datetime import datetime
from tqdm import tqdm
import re
import shutil
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.neighbors import KDTree
import pickle

from utils import read_depth_image, read_depth_image_npy, read_semantic_image, \
                  read_traj_file, read_traj_file_colmap, count_points, read_colmap_intrinsics, \
                  save_pointcloud_ply, merge_pointclouds, depthsemantic_to_pointcloud, split_seq, \
                  draw_occ_png, find_colormap_index
from colormap import colors_map

class HabitatOcc():
    def __init__(self, data_dir, is_generate_pointcloud=True, is_depth_npy=True, \
                is_colmap_extrinsic=True, is_change_depth_resolution=False, is_icp=True, \
                is_rgb_input=False):        # 包含数据的预加载
        self.data_dir = data_dir
        self.saved_dir = "occupancy"
        self.is_generate_pointcloud = is_generate_pointcloud
        self.is_depth_npy = is_depth_npy
        self.is_colmap_extrinsic = is_colmap_extrinsic
        self.is_change_depth_resolution = is_change_depth_resolution
        self.is_icp = is_icp
        self.is_rgb_input = is_rgb_input
        self.voxel_size = 0.08

        ### 数据配置路径
        depth_folder = os.path.join(data_dir, "depth")
        semantic_folder = os.path.join(data_dir, "semantic")
        assert len(os.listdir(depth_folder)) > 0, "The depth folder is empty."
        assert len(os.listdir(semantic_folder)) > 0, "The semantic folder is empty."
        assert len(os.listdir(depth_folder)) == len(os.listdir(semantic_folder)), "The number of depth and semantic files must be the same."
        self.semantic_pointcloud_ply_folder = os.path.join(data_dir, self.saved_dir, "semantic_pointcloud_ply_folder")
        self.merge_all_folder = os.path.join(self.data_dir, self.saved_dir, "semantic_pointcloud_all_ply")
        if not os.path.exists(self.semantic_pointcloud_ply_folder):
            os.makedirs(self.semantic_pointcloud_ply_folder)
        if not os.path.exists(self.merge_all_folder):
            os.makedirs(self.merge_all_folder)

        ### 内外参加载
        intrinsics_file_path = os.path.join(data_dir, "sparse/0/cameras.txt")
        self.camera_intrinsics = read_colmap_intrinsics(intrinsics_file_path)
        pose_file_path = os.path.join(data_dir, "sparse/0/images.txt")
        if is_colmap_extrinsic:
            self.extrinsic_list = read_traj_file_colmap(pose_file_path)
        else:
            self.extrinsic_list = read_traj_file(pose_file)

        # 深度图，语义图加载
        self.depth_files_name = sorted(os.listdir(depth_folder), key=split_seq)
        self.semantic_files_name = sorted(os.listdir(semantic_folder), key=split_seq)
        self.depth_maps = []
        self.semantic_maps = []
        print("data is loading!")
        for depth_file, semantic_file in tqdm(zip(self.depth_files_name, self.semantic_files_name), total=len(self.depth_files_name)):
            depth_image_path = os.path.join(depth_folder, depth_file)
            if is_depth_npy:  ### npy格式
                depth_map = read_depth_image_npy(depth_image_path)
            else:  ### png格式
                depth_map = read_depth_image(depth_image_path)
            if is_change_depth_resolution:  ### 是否修改分辨率
                depth_map = cv2.resize(depth_map, (540, 960), interpolation=cv2.INTER_NEAREST)
            self.depth_maps.append(depth_map)

            semantic_image_path = os.path.join(semantic_folder, semantic_file)
            semantic_map = read_semantic_image(semantic_image_path)
            self.semantic_maps.append(semantic_map)

        # 确保文件列表长度相同
    
    ### 主要负责构建局部/全局的ply文件(语义点云)，局部/全局的pkl文件(语义occ)
    def construct_occ(self):   
        if self.is_generate_pointcloud:
            print("**********构建局部occ开始**********")
            ### 生成并保存逐帧Ply点云
            for i in tqdm(range(len(self.semantic_maps))):
                semantic_pointcloud = depthsemantic_to_pointcloud(self.depth_maps[i], self.semantic_maps[i], self.camera_intrinsics)
                ply_name = os.path.join(self.semantic_pointcloud_ply_folder, self.semantic_files_name[i].rsplit('.', 1)[0] + '.ply')
                save_pointcloud_ply(ply_name, semantic_pointcloud)
            print(f"**********局部occ构建完成 请到{self.semantic_pointcloud_ply_folder}查看结果！**********")
        # 合并点云
        print("**********构建全局occ开始**********")
        merged_pointcloud = merge_pointclouds(self.semantic_pointcloud_ply_folder, self.extrinsic_list, self.is_icp, self.is_colmap_extrinsic)
        current_time = datetime.now()
        time_str = current_time.strftime("%Y%m%d%H%M%S")
        merge_all_path = os.path.join(self.merge_all_folder, f"merge_all_{time_str}"+".ply")
        save_pointcloud_ply(merge_all_path, merged_pointcloud)
        print(f"**********全局occ构建完成 请到{self.merge_all_folder}查看结果！**********")
    
    ### 可视化单帧occ(png格式)
    def vis_local_occ(self):
        print("**********local occ 可视化开始**********")
        #### 保存路径生成
        vis_local_occ_folder = os.path.join(self.data_dir, self.saved_dir, "vis_local_occ")
        shutil.rmtree(vis_local_occ_folder, ignore_errors=True)  # remove output_dir if it exists
        os.makedirs(vis_local_occ_folder, exist_ok=False)

        ply_file_name = sorted(os.listdir(self.semantic_pointcloud_ply_folder), key=split_seq)

        # 调用 draw 函数进行可视化
        for index, file in tqdm(enumerate(ply_file_name), total=len(ply_file_name)):
            point_cloud_path = os.path.join(self.semantic_pointcloud_ply_folder, file)
            point_cloud = o3d.io.read_point_cloud(point_cloud_path)
            colors = np.floor(np.asarray(point_cloud.colors)*255.0).astype(int)    
            color_map = {i: tuple(color) for i, color in enumerate(colors_map)}
            reverse_color_map = {tuple(color[:3]): i for i, color in color_map.items()}
            color_labels = np.array([reverse_color_map.get(tuple(color), -1) for color in colors])
            color_labels = color_labels.reshape(-1, 1)
            points = np.asarray(point_cloud.points)
            to_vis = np.concatenate([points, color_labels], axis=1)
            draw_occ_png(to_vis, voxel_size=self.voxel_size, intrinsic=self.camera_intrinsics, cam_pose=self.extrinsic_list[index], 
                d=0.5, save_path=vis_local_occ_folder, filename=os.path.splitext(file)[0], tag="local")
        print(f"**********local occ 可视化完成 请到{vis_local_occ_folder}查看结果！**********")
    
    ### 可视化全局occ(png格式)
    def vis_global_occ(self): 
        print("**********global occ 可视化开始**********")
        #### 保存路径生成
        vis_local_occ_folder = os.path.join(self.data_dir, self.saved_dir, "vis_global_occ")
        shutil.rmtree(vis_local_occ_folder, ignore_errors=True)  # remove output_dir if it exists
        os.makedirs(vis_local_occ_folder, exist_ok=False)
        
        merge_ply_name = os.listdir(self.merge_all_folder)[0]  ### 只会有一个全局文件
        point_cloud = o3d.io.read_point_cloud(os.path.join(self.merge_all_folder, merge_ply_name))
        colors = np.floor(np.asarray(point_cloud.colors)*255.0).astype(int)    
        color_map = {i: tuple(color) for i, color in enumerate(colors_map)}
        reverse_color_map = {tuple(color[:3]): i for i, color in color_map.items()}
        color_labels = np.array([reverse_color_map.get(tuple(color), -1) for color in colors])
        color_labels = color_labels.reshape(-1, 1)
        points = np.asarray(point_cloud.points)
        to_vis = np.concatenate([points, color_labels], axis=1)
        draw_occ_png(to_vis, voxel_size=self.voxel_size, intrinsic=self.camera_intrinsics, cam_pose=self.extrinsic_list[0], 
            d=0.5, save_path=vis_local_occ_folder, filename=merge_ply_name.split(".")[0], tag="global")
        print(f"**********global occ 可视化结束 请到{vis_local_occ_folder}查看结果！**********")

    ### 构建occ保存为pkl文件
    def save_local_occ(self):
        print(f"**********save local occ开始**********")
        voxel_size = self.voxel_size
        #### 保存路径生成
        save_local_occ_folder = os.path.join(self.data_dir, "save_local_occ")
        shutil.rmtree(save_local_occ_folder, ignore_errors=True)  # remove output_dir if it exists
        os.makedirs(save_local_occ_folder, exist_ok=False)
        if voxel_size == 0.08:
            voxDim = np.asarray([60, 60, 36])  # 0.08 cm  [90 90 54]
            # voxDim = np.asarray([90, 90, 54])  # 0.08 cm  [90 90 54]
        elif voxel_size == 0.04:
            voxDim = np.asarray([120, 120, 72])  # 0.04 cm
        elif svoxel_size == 0.02:
            voxDim = np.asarray([240, 240, 144])  # 0.02 cm
        assert np.all(voxDim * voxel_size == np.asarray([4.8, 4.8, 2.88]))
        voxOriginCam = np.asarray([  ##定义相机初始位置
            [0], [0], [1.44]]
        )
        merge_ply_name = os.listdir(self.merge_all_folder)[0]
        point_cloud_merge = o3d.io.read_point_cloud(os.path.join(self.merge_all_folder, merge_ply_name))
        scene_voxels = np.asarray(point_cloud_merge.points)
        scene_voxels_sem = find_colormap_index(point_cloud_merge)
        for idx in tqdm(range(len(self.extrinsic_list)), total=len(self.extrinsic_list)):
            h, w = self.depth_maps[0].shape[0], self.depth_maps[0].shape[1]
            cam2world = self.extrinsic_list[idx]  ### colmap的Pose是cam2world，其余应该是world2cam
            voxOriginWorld = cam2world[:3, :3] @ voxOriginCam + cam2world[:3, -1:]  ### 相机旋转+平移=相机原点的世界坐标系
            voxOriginWorld2 = deepcopy(voxOriginWorld)
            delta = np.array(
                [[2.4],
                [2.4],
                [1.44]]
            )  # 世界坐标系下的
            # 保留距离原点4.8范围内的场景  
            scene_voxels_delta = np.abs(scene_voxels[:, :3] - voxOriginWorld.reshape(-1))  ### 表示每一行坐标与世界中心坐标的差值
            mask = np.logical_and(scene_voxels_delta[:, 0] <= 4.8,
                                np.logical_and(scene_voxels_delta[:, 1] <= 4.8,
                                                scene_voxels_delta[:, 2] <= 2.88))                               
            ### scene_voxel是全局的，frame_voxel是每一帧，相当于每一帧的点云留下
            frame_voxels = scene_voxels[mask]
            frame_voxels_sem = scene_voxels_sem[mask]
            ### 开始从点云转occ
            # 世界坐标内画出场景的范围   np.arange(开始，结尾，间隔) 取前voxDim个  voxDim就是60x60x36
            xs = np.arange(voxOriginWorld[0, 0], voxOriginWorld[0, 0] + 100 * voxel_size, voxel_size)[:voxDim[0]]
            ys = np.arange(voxOriginWorld[1, 0], voxOriginWorld[1, 0] + 100 * voxel_size, voxel_size)[:voxDim[1]]
            zs = np.arange(voxOriginWorld[2, 0], voxOriginWorld[2, 0] + 100 * voxel_size, voxel_size)[:voxDim[2]]
            ### X 数组表示网格中每个点对应的 x 坐标。可以看到，对于每一行（固定的 y 值），x 坐标是 x 数组的重复。
            ### Y 数组表示网格中每个点对应的 y 坐标。可以看到，对于每一列（固定的 x 值），y 坐标是 y 数组的重复。
            gridPtsWorldX, gridPtsWorldY, gridPtsWorldZ = np.meshgrid(xs, ys, zs)
            ### 转成每一行代表一个三维空间中的点
            gridPtsWorld = np.stack([gridPtsWorldX.flatten(),
                                    gridPtsWorldY.flatten(),
                                    gridPtsWorldZ.flatten()], axis=1)  
            gridPtsLabel = np.zeros((gridPtsWorld.shape[0]))
            gridPtsWorld_color = np.zeros((gridPtsWorld.shape[0], 3))
            ### 对点云用kd树最近邻找到gridPtsWorld对应的voxel
            kdtree = KDTree(frame_voxels[:, :3], leaf_size=10)
            dist, ind = kdtree.query(gridPtsWorld)  # 返回与gridPtsWorld最近的1个邻居。dist表示其距离，ind表示其索引
            dist, ind = dist.reshape(-1), ind.reshape(-1)
            mask = dist <= voxel_size  # 确保最近的邻居，在范围之内
            gridPtsLabel[mask] = frame_voxels_sem[ind[mask]]  # 赋予语义标签
            g = gridPtsLabel.reshape(voxDim[0], voxDim[1], voxDim[2])
            g_not_0 = np.where(g > 0)  # 初始化是0
            if len(g_not_0) == 0:
                continue
            g_not_0_x = g_not_0[0]
            g_not_0_y = g_not_0[1]
            if len(g_not_0_x) == 0:
                continue
            if len(g_not_0_y) == 0:
                continue
            valid_x_min = g_not_0_x.min()
            valid_x_max = g_not_0_x.max()
            valid_y_min = g_not_0_y.min()
            valid_y_max = g_not_0_y.max()
            mask = np.zeros_like(g)
            if valid_x_min != valid_x_max and valid_y_min != valid_y_max:
                mask[valid_x_min:valid_x_max, valid_y_min:valid_y_max, :] = 1
                mask = 1 - mask  #
                mask = mask.astype(np.bool_)
                g[mask] = 255  # 在有效范围以外的区域, 将其label设置为255
            else:
                continue
            frame_voxels = np.zeros((gridPtsWorld.shape[0], 4))
            frame_voxels[:, :3] = gridPtsWorld
            frame_voxels[:, -1] = g.reshape(-1)
            # gridPtsWorld[:, -1] = g.reshape(-1)
            intrinsics = np.array([
                [self.camera_intrinsics['fx'], 0, self.camera_intrinsics['cx']],
                [0, self.camera_intrinsics['fy'], self.camera_intrinsics['cy']],
                [0, 0, 1]
            ])
            # 计算3D点至2D图像的投影点,放到图像坐标系中
            voxels_cam = (np.linalg.inv(cam2world)[:3, :3] @ gridPtsWorld[:, :3].T \
                        + np.linalg.inv(cam2world)[:3, -1:]).T
            voxels_pix = (intrinsics[:3, :3] @ voxels_cam.T).T
            voxels_pix = voxels_pix / voxels_pix[:, -1:]  ### 透视除法，得到最终的像素坐标 (u/w, v/w, 1)
            mask = np.logical_and(voxels_pix[:, 0] >= 0,
                                np.logical_and(voxels_pix[:, 0] < w,
                                                np.logical_and(voxels_pix[:, 1] >= 0,
                                                                np.logical_and(voxels_pix[:, 1] < h,
                                                                            voxels_cam[:, 2] > 0))))  # 视野内的
            inroom = frame_voxels[:, -1] != 255
            mask = np.logical_and(~mask, inroom)  # 如果一个3d point，它没有落在图像上，并且是在房间内，则将其label设置为0（empty）
            frame_voxels[mask, -1] = 0  # empty类别
            # ===================================================================
            # 2025-06-26
            # <<< 可见性过滤 >>>
            # ===================================================================
            # 在这里，`final_labeled_voxels` 包含了当前帧附近的所有体素及其初步标签 (0=empty, 1-12=sem, 255=unknown)
            # 我们现在要基于深度图，把被遮挡的体素找出来，并把它们的标签设为0 (empty)

            # 1. 获取所有体素点的世界坐标和当前标签
            voxel_coords = frame_voxels[:, :3]
            voxel_labels = frame_voxels[:, -1]

            # 2. 将所有体素点转换到相机坐标系
            world2cam = np.linalg.inv(cam2world)
            voxels_cam = (world2cam[:3, :3] @ voxel_coords.T + world2cam[:3, -1:]).T

            # 3. 投影到像素平面
            voxels_pix = (intrinsics[:3, :3] @ voxels_cam.T).T

            depths_in_cam = voxels_cam[:, 2]
            # 防止除以零
            depths_in_cam[depths_in_cam <= 0] = 1e6  # 把相机后方的点深度设为极大值，使其在后续比较中被自然剔除

            us = (voxels_pix[:, 0] / depths_in_cam).astype(int)
            vs = (voxels_pix[:, 1] / depths_in_cam).astype(int)

            # 4. 创建一个遮挡掩码，初始假设所有点都未被遮挡
            # 我们只关心那些有标签的体素（非empty, 非unknown）
            occlusion_mask = np.zeros_like(voxel_labels, dtype=bool)

            # 5. 筛选出需要进行深度测试的体素
            # 条件：a. 在图像范围内 b. 在相机前方 c. 有一个有效的语义标签 (不是empty也不是unknown)
            test_indices = np.where(
                (us >= 0) & (us < w) &
                (vs >= 0) & (vs < h) &
                (depths_in_cam > 0) &
                (voxel_labels > 0) & (voxel_labels != 255)
            )[0]

            if len(test_indices) > 0:
                # 获取这些待测点的像素坐标和计算深度
                test_us = us[test_indices]
                test_vs = vs[test_indices]
                test_depths = depths_in_cam[test_indices]

                # 获取深度图中对应位置的深度值
                gt_depth_values = self.depth_maps[idx][test_vs, test_us]

                # 识别被遮挡的点：其计算深度显著大于GT深度
                is_occluded = test_depths > (gt_depth_values + voxel_size * 5)

                # 将被遮挡的点的索引更新到遮挡掩码中
                occluded_indices = test_indices[is_occluded]
                occlusion_mask[occluded_indices] = True

            # 6. 应用遮挡掩码：将被遮挡的体素标签设为0 (empty)
            final_labels = voxel_labels.copy()
            final_labels[occlusion_mask] = 0
            frame_voxels[:, -1] = final_labels   ### 最后保存的是单帧，被遮挡的体素标签是空，默认看到表面的标签有效
        
            ###### 保存
            target_1_4 = frame_voxels[:, -1].reshape(60, 60, 36)  ### 提取最后一列
            pkl_data = {
                'img': os.path.join(f"{self.data_dir}", "images", f'{idx:05d}.png'),
                'depth_gt': os.path.join(f"{self.data_dir}", "depth", f'{idx:05d}.npy'),
                'cam_pose': self.extrinsic_list[idx],  # camera to world
                'intrinsic': intrinsics,
                'target_1_4': target_1_4,  # 1_4 表示下采样了4倍, 8cm
                'voxel_origin': np.array([frame_voxels[:, 0].min(), frame_voxels[:, 1].min(), frame_voxels[:, 2].min()]),
            }
            with open(os.path.join(f"{save_local_occ_folder}", f'{idx:05d}.pkl'), "wb") as handle:
                pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"**********save local occ结束 请到{save_local_occ_folder}查看结果！**********")

    ### 构建全局occ保存为pkl文件
    def save_global_occ(self):
        print(f"**********save global occ开始**********")
        voxel_size = self.voxel_size
        #### 保存路径生成
        save_global_occ_folder = os.path.join(self.data_dir, "save_global_occ")
        shutil.rmtree(save_global_occ_folder, ignore_errors=True)  # remove output_dir if it exists
        os.makedirs(save_global_occ_folder, exist_ok=False)
        if voxel_size == 0.08:
            voxDim = np.asarray([60, 60, 36])  # 0.08 cm
        elif voxel_size == 0.04:
            voxDim = np.asarray([120, 120, 72])  # 0.04 cm
        elif svoxel_size == 0.02:
            voxDim = np.asarray([240, 240, 144])  # 0.02 cm
        assert np.all(voxDim * voxel_size == np.asarray([4.8, 4.8, 2.88]))
        voxOriginCam = np.asarray([  ##定义相机初始位置
            [0], [0], [1.44]]
        )
        merge_ply_name = os.listdir(self.merge_all_folder)[0]
        point_cloud_merge = o3d.io.read_point_cloud(os.path.join(self.merge_all_folder, merge_ply_name))
        scene_voxels = np.asarray(point_cloud_merge.points)
        scene_voxels_sem = find_colormap_index(point_cloud_merge)

        h, w = self.depth_maps[0].shape[0], self.depth_maps[0].shape[1]
        cam2world = self.extrinsic_list[0]  ### colmap的Pose是cam2world，其余应该是world2cam
        voxOriginWorld = cam2world[:3, :3] @ voxOriginCam + cam2world[:3, -1:]  ### 相机旋转+平移=相机原点的世界坐标系
        voxOriginWorld2 = deepcopy(voxOriginWorld)
        delta = np.array(
            [[2.4],
            [2.4],
            [1.44]]
        )  # 世界坐标系下的

        scene_voxels_delta = np.abs(scene_voxels[:, :3] - voxOriginWorld.reshape(-1))  ### 表示每一行坐标与世界中心坐标的差值
        ### 开始从点云转occ
        # 世界坐标内画出场景的范围   np.arange(开始，结尾，间隔) 取前voxDim个  voxDim就是60x60x36
        xs = np.arange(voxOriginWorld[0, 0], voxOriginWorld[0, 0] + 100 * voxel_size, voxel_size)[:voxDim[0]]
        ys = np.arange(voxOriginWorld[1, 0], voxOriginWorld[1, 0] + 100 * voxel_size, voxel_size)[:voxDim[1]]
        zs = np.arange(voxOriginWorld[2, 0], voxOriginWorld[2, 0] + 100 * voxel_size, voxel_size)[:voxDim[2]]
        ### X 数组表示网格中每个点对应的 x 坐标。可以看到，对于每一行（固定的 y 值），x 坐标是 x 数组的重复。
        ### Y 数组表示网格中每个点对应的 y 坐标。可以看到，对于每一列（固定的 x 值），y 坐标是 y 数组的重复。
        gridPtsWorldX, gridPtsWorldY, gridPtsWorldZ = np.meshgrid(xs, ys, zs)
        ### 转成每一行代表一个三维空间中的点
        gridPtsWorld = np.stack([gridPtsWorldX.flatten(),
                                gridPtsWorldY.flatten(),
                                gridPtsWorldZ.flatten()], axis=1)  
        gridPtsLabel = np.zeros((gridPtsWorld.shape[0]))
        gridPtsWorld_color = np.zeros((gridPtsWorld.shape[0], 3))
        ### 对点云用kd树最近邻找到gridPtsWorld对应的voxel
        kdtree = KDTree(scene_voxels[:, :3], leaf_size=10)
        dist, ind = kdtree.query(gridPtsWorld)  # 返回与gridPtsWorld最近的1个邻居。dist表示其距离，ind表示其索引
        dist, ind = dist.reshape(-1), ind.reshape(-1)
        mask = dist <= voxel_size  # 确保最近的邻居，在范围之内
        gridPtsLabel[mask] = scene_voxels_sem[ind[mask]]  # 赋予语义标签
        
        g = gridPtsLabel.reshape(voxDim[0], voxDim[1], voxDim[2])
        g_not_0 = np.where(g > 0)  # 初始化是0
        g_not_0_x = g_not_0[0]
        g_not_0_y = g_not_0[1]

        valid_x_min = g_not_0_x.min()
        valid_x_max = g_not_0_x.max()
        valid_y_min = g_not_0_y.min()
        valid_y_max = g_not_0_y.max()

        mask = np.zeros_like(g)
        if valid_x_min != valid_x_max and valid_y_min != valid_y_max:
            mask[valid_x_min:valid_x_max, valid_y_min:valid_y_max, :] = 1
            mask = 1 - mask  #
            mask = mask.astype(np.bool_)
            g[mask] = 255  # 在有效范围以外的区域, 将其label设置为255
        else:
            print("valid_x_min != valid_x_max")
            return 
        scene_voxels = np.zeros((gridPtsWorld.shape[0], 4))
        scene_voxels[:, :3] = gridPtsWorld
        scene_voxels[:, -1] = g.reshape(-1)
        # gridPtsWorld[:, -1] = g.reshape(-1)
        intrinsics = np.array([
            [self.camera_intrinsics['fx'], 0, self.camera_intrinsics['cx']],
            [0, self.camera_intrinsics['fy'], self.camera_intrinsics['cy']],
            [0, 0, 1]
        ])
        # 计算3D点至2D图像的投影点,放到图像坐标系中
        voxels_cam = (np.linalg.inv(cam2world)[:3, :3] @ gridPtsWorld[:, :3].T \
                    + np.linalg.inv(cam2world)[:3, -1:]).T
        voxels_pix = (intrinsics[:3, :3] @ voxels_cam.T).T
        voxels_pix = voxels_pix / voxels_pix[:, -1:]  ### 透视除法，得到最终的像素坐标 (u/w, v/w, 1)
        mask = np.logical_and(voxels_pix[:, 0] >= 0,
                            np.logical_and(voxels_pix[:, 0] < w,
                                            np.logical_and(voxels_pix[:, 1] >= 0,
                                                            np.logical_and(voxels_pix[:, 1] < h,
                                                                        voxels_cam[:, 2] > 0))))  # 视野内的
        inroom = scene_voxels[:, -1] != 255
        mask = np.logical_and(~mask, inroom)  # 如果一个3d point，它没有落在图像上，并且是在房间内，则将其label设置为0（empty）
        scene_voxels[mask, -1] = 0  # empty类别
    
        ###### 保存
        target_1_4 = scene_voxels[:, -1].reshape(60, 60, 36)  ### 提取最后一列
        pkl_data = {
            'intrinsic': intrinsics,
            'target_1_4': target_1_4,  # 1_4 表示下采样了4倍, 8cm
            'voxel_origin': np.array([scene_voxels[:, 0].min(), scene_voxels[:, 1].min(), scene_voxels[:, 2].min()]),
        }
        with open(os.path.join(f"{save_global_occ_folder}", merge_ply_name.split(".")[0]+".pkl"), "wb") as handle:
            pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"**********save global occ结束 请到{save_global_occ_folder}查看结果！**********")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # optional arguments
    parser.add_argument(
        "--data_path",
        default="example_data/scanet/scene0000_00",
        type=str,
        required=True,
        help='habitat data saved path for occupancy reconstruction.',
    )
    args = parser.parse_args()
    
    habitatocc = HabitatOcc(args.data_path)
    habitatocc.construct_occ()
    habitatocc.vis_local_occ()
    habitatocc.vis_global_occ()
    # habitatocc.save_local_occ()
    # habitatocc.save_global_occ()
