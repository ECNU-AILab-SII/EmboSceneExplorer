import numpy as np
import open3d as o3d
import imageio
import os
from datetime import datetime
from tqdm import tqdm
import re
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from mayavi import mlab
from colormap import colors_map

### function：自定义排序，observation_semantic_8.png，按照数字从小到大排序
def split_seq(filename):
    match = re.search(r'(\d+)', filename)
    if match:
        # 如果匹配成功，返回提取出的数字（转换为整数）
        return int(match.group(1))
    else:
        # 如果没有匹配到数字，返回一个很大的数，确保该文件排在最后
        return float('inf')

### function：读取深度图(png格式)
def read_depth_image(depth_image_path):
    # 读取深度图，假设深度图是以16位PNG格式存储的
    depth_image = imageio.imread(depth_image_path)
    # print(f"Data type: {depth_image.dtype}, Min: {depth_image.min()}, Max: {depth_image.max()}")  # 检查数据范围
    # 归一化深度图到 0-1 范围
    normalized_depthimage = depth_image / (depth_image.max()-depth_image.min())
    depth_image = normalized_depthimage * 100.0
    depth_image = normalized_depthimage
    return depth_image

### function：读取深度图(npy格式)
def read_depth_image_npy(depth_image_path):
    depth_data = np.load(depth_image_path)
    if len(depth_data.shape) == 3:  ### maybe the batchsize first
        depth_data = depth_data.squeeze(0)
    return depth_data

### function：读取语义图(png格式)——语义值为RGB未归一化颜色
def read_semantic_image(semantic_image_path):
    # 读取语义图，假设语义图是以8位PNG格式存储的
    semantic_image = imageio.imread(semantic_image_path)
    return semantic_image

### function：输入深度图，语义图，内参，返回语义点云
def depthsemantic_to_pointcloud(depth_map, semantic_map, camera_intrinsics):
    if len(depth_map.shape) ==  2:
        height, width = depth_map.shape
    else:   ### 特殊处理下Blender的深度数据   cmap=summber RGBA
        height, width, _ = depth_map.shape
        depth_map = np.dot(depth_map[..., :3], [0.2989, 0.5870, 0.1140])
    fx, fy = camera_intrinsics['fx'], camera_intrinsics['fy']
    cx, cy = camera_intrinsics['cx'], camera_intrinsics['cy']
    # 创建坐标网格
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    x, y = np.meshgrid(x, y)
    # 将像素坐标转换为相机坐标
    X = (x - cx) * depth_map / fx
    Y = (y - cy) * depth_map / fy
    Z = depth_map
    # 转换为世界坐标
    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    if semantic_map.shape[2] == 3:
        colors = semantic_map.reshape(-1, 3).astype(np.float32) / 255.0 
    elif semantic_map.shape[2] == 4:
        colors = semantic_map[:, :, :3].reshape(-1, 3).astype(np.float32) / 255.0   # scannet场景semantic_map: (720, 960, 4)
    # 创建点云
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)
    pointcloud.colors = o3d.utility.Vector3dVector(colors)  # 归一化颜色 
    # print("True or Flase:", pointcloud.has_colors())  # 应该输出 True
    return pointcloud

### function：从非colmap格式中读取外参矩阵
### 格式参考：/data3/gls/code/habitate_demo/data20250522/traj.txt
def read_traj_file(filepath):
    Extrinsic_list = []
    with open(filepath, 'r') as file:
        for line in file:
            # 将每行分割成浮点数
            values = list(map(float, line.split()))
            
            # 组织成4x4矩阵
            matrix = np.array([
                [values[0], values[1], values[2], values[3]],
                [values[4], values[5], values[6], values[7]],
                [values[8], values[9], values[10], values[11]],
                [0, 0, 0, 1]
            ])
            
            # 添加到列表
            Extrinsic_list.append(matrix)
    return Extrinsic_list

### function：工具函数 四元数生成旋转矩阵
def quaternion_to_rotation_matrix(qvec):
    # 将四元数转换为旋转矩阵
    w, x, y, z = qvec
    return np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x**2 + z**2),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])

### function：从colmap格式中读取外参矩阵
### 格式参考：/data3/gls/code/habitate_demo/data20250614/sparse/0/images.txt
def read_traj_file_colmap(filepath):
    ## to do
    Extrinsic_list = []
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            elems = line.split()
            qvec = np.array(tuple(map(float, elems[1:5])))
            tvec = np.array(tuple(map(float, elems[5:8])))
            rotation_matrix = quaternion_to_rotation_matrix(qvec)
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = tvec
            Extrinsic_list.append(transformation_matrix)
    return Extrinsic_list

### function：输入语义点云列表，返回合并语义点云
def merge_pointclouds(semantic_pointcloud_ply_folder, transforms, is_icp, is_colmap_extrinsic):
    ply_file_list = os.listdir(semantic_pointcloud_ply_folder)
    ply_file_list = sorted(ply_file_list, key=split_seq)
    print("merge pointcloud is loading!!")
    pointclouds = []
    for i in tqdm(range(len(ply_file_list))):
        file = ply_file_list[i]
        if file.endswith('.ply'):
            file_path = os.path.join(semantic_pointcloud_ply_folder, file)
            point_cloud = o3d.cuda.pybind.io.read_point_cloud(file_path)
            pointclouds.append(point_cloud)
    if is_icp:
        print("USE ICP!!!")
        world_pointcloud = []
        for pointcloud, transform in zip(pointclouds, transforms):
            if is_colmap_extrinsic:
                pointcloud1 = pointcloud.transform(transform)   ### 如果是colmap格式，那么默认的外参是camera2world
            else:
                pointcloud1 = pointcloud.transform(np.linalg.inv(transform))  ### 如果是普通外参格式，默认外参是world2camera
            pointcloud1 = pointcloud1.voxel_down_sample(voxel_size=0.08)
            world_pointcloud.append(pointcloud1)
        global_pointcloud = world_pointcloud[0]  ### 以第一个作为icp配准初始化
        global_pointcloud.estimate_normals()  # 计算法向量
        
        # 修正ICP部分的关键步骤
        for i in tqdm(range(1, len(world_pointcloud))):
            current_point_cloud = world_pointcloud[i]
            current_point_cloud.estimate_normals()
            # 尝试使用更宽松的阈值和更稳定的估计器
            reg_p2p = o3d.pipelines.registration.registration_icp(
                current_point_cloud, global_pointcloud, max_correspondence_distance=0.05, init=np.eye(4),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )
            print("ICP fitness:", reg_p2p.fitness)
            transformation_icp = reg_p2p.transformation
            current_point_cloud.transform(transformation_icp)
            global_pointcloud += current_point_cloud

    else:
        print("Use normal Transformation!!!")
        global_pointcloud = o3d.cuda.pybind.geometry.PointCloud()
        for pointcloud, transform in tqdm(zip(pointclouds, transforms), total=len(pointclouds)):
            if is_colmap_extrinsic:
                pointcloud1 = pointcloud.transform(transform)
            else:
                pointcloud1 = pointcloud.transform(np.linalg.inv(transform))
            pointcloud1 = transform_point_cloud(pointcloud1)
            pointcloud1 = pointcloud1.voxel_down_sample(voxel_size=0.08)
            global_pointcloud += pointcloud1       

    # 去除重复点
    voxel_size = 0.08
    print("merge before:", count_points(global_pointcloud))
    global_pointcloud = global_pointcloud.voxel_down_sample(voxel_size)
    print("after merge:", count_points(global_pointcloud))
    return global_pointcloud

### function：可视化点云（保存为ply文件）
def save_pointcloud_ply(ply_name, semantic_pointcloud):
    o3d.io.write_point_cloud(ply_name, semantic_pointcloud)
    # print("True or Flase:", semantic_pointcloud.has_colors())  # 应该输出 True

### function：工具函数 返回点云总数
def count_points(point_cloud):
    return len(point_cloud.points)

### function：工具函数 获取COLMAP格式内参
def read_colmap_intrinsics(intrinsics_path):
    with open(intrinsics_path, 'r') as file:
        textline = file.readline()
    # 分割行以获取各个元素
    elements = textline.split()
    # 提取最后四个元素并转换为浮点数
    fx = float(elements[-4])
    fy = float(elements[-3])
    cx = float(elements[-2])
    cy = float(elements[-1])
    # 创建相机内参字典
    camera_intrinsics = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
    return camera_intrinsics

### function：生成OCC voxel并保存为图片，局部和全局视角不同
def draw_occ_png(voxel_label, voxel_size, intrinsic=None, cam_pose=None, d=0.5, save_path=None, filename=None, tag=None):
    mlab.options.offscreen = True
    figure = mlab.figure(size=(1600*0.8, 900*0.8), bgcolor=(1, 1, 1))
    
    if intrinsic is not None and cam_pose is not None:
        assert d > 0, 'camera model d should > 0'
        fx = intrinsic['fx']
        fy = intrinsic['fy']
        cx = intrinsic['cx']
        cy = intrinsic['cy']

        # half of the image plane size
        y = d * 2 * cy / (2 * fy)
        x = d * 2 * cx / (2 * fx)
        
        # camera points (cam frame)
        tri_points = np.array(
            [
                [0, 0, 0],
                [x, y, d],
                [-x, y, d],
                [-x, -y, d],
                [x, -y, d],
            ]
        )
        tri_points = (cam_pose @ np.hstack([tri_points, np.ones((5, 1))]).T).T[:, :3]
        
        # camera points (world frame)
        # tri_points = (cam_pose @ tri_points.T).T
        x = tri_points[:, 0]
        y = tri_points[:, 1]
        z = tri_points[:, 2]
        triangles = [
            (0, 1, 2),
            (0, 1, 4),
            (0, 3, 4),
            (0, 2, 3),
        ]

        # draw cam model
        mlab.triangular_mesh(
            x,
            y,
            z,
            triangles,
            representation="wireframe",
            color=(0, 0, 0),
            line_width=7.5,
        )
    
    # draw occupied voxels
    plt_plot = mlab.points3d(
        voxel_label[:, 0],
        voxel_label[:, 1],
        voxel_label[:, 2],
        voxel_label[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.1 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=0,
        vmax=40,
    )

    plt_plot.glyph.scale_mode = "scale_by_vector"

    plt_plot.module_manager.scalar_lut_manager.lut.table = colors_map
    cam_direction_world = cam_pose
    cam_direction_world = cam_pose[:3, :3] @ np.array([0, 0, 1])
    azimuth = np.arctan2(cam_direction_world[1], cam_direction_world[0]) * 180 / np.pi
    elevation = np.arctan2(cam_direction_world[2], np.linalg.norm(cam_direction_world[:2])) * 180 / np.pi
    if tag == "local":
        mlab.view(azimuth=azimuth+90, elevation=elevation+45+22.5)   ## local是这个
    elif tag == "global":
        mlab.view(azimuth=azimuth-90, elevation=elevation+45)  ## global是这个
    mlab.savefig(os.path.join(save_path, filename+".png"))
    mlab.close()

### function：根据RGB值寻找到colormap中的索引
def find_colormap_index(point_cloud):
    # 2. 将 point_cloud.colors 转换为 [0, 255] 范围的 RGB 值
    colors = np.array(point_cloud.colors) * 255  # 从 [0, 1] 恢复到 [0, 255]
    colors = colors.astype(np.uint8)  # 转换为整数

    # 3. 初始化一个数组存储每个点的 Colmap 索引
    colmap_indices = np.zeros(len(colors), dtype=np.int32)

    # 4. 遍历每个点的颜色，找到最接近的 Colmap 索引
    for i, color in enumerate(colors):
        # 计算当前颜色与所有 Colmap 颜色的欧氏距离
        distances = np.linalg.norm(colors_map[:, :3] - color, axis=1)
        # 找到距离最小的索引
        colmap_indices[i] = np.argmin(distances)
    return colmap_indices