import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from PIL import Image
from colmap_loader import read_extrinsics_text, read_intrinsics_text
from plyfile import PlyData, PlyElement
from typing import NamedTuple

def poisson_reconstruction(points, colors, normals, depth=8, scale=1.1, linear_fit=False):
    """
    使用泊松重建方法从点云生成网格
    
    参数:
        points: 点云坐标 (N, 3)
        colors: 点云颜色 (N, 3)
        normals: 点云法线 (N, 3)
        depth: 泊松重建的深度参数，控制重建的细节程度
        scale: 缩放参数，用于调整重建的尺度
        linear_fit: 是否使用线性拟合来优化重建结果
        
    返回:
        o3d.geometry.TriangleMesh: 重建后的网格
    """
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    
    print("开始泊松重建...")
    # 执行泊松重建
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, 
            depth=depth,
            scale=scale,
            linear_fit=linear_fit
        )
    
    # 可选: 根据密度过滤网格顶点
    # 去除低密度顶点(可能是噪声)
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # 可选: 对网格进行平滑处理
    mesh = mesh.filter_smooth_taubin(number_of_iterations=5)
    
    # 计算顶点法线
    mesh.compute_vertex_normals()
    
    return mesh

def compute_normals(points, k=30):
    """计算点云法线"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 估计法线
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1,  # 搜索半径
            max_nn=k     # 最大近邻数
        )
    )
    
    # 统一法线方向
    try:
        pcd.orient_normals_consistent_tangent_plane(k=k)
    except Exception as e:
        print(f"法线方向统一失败: {str(e)}")
    
    return np.asarray(pcd.normals)

def save_ply_with_normals(points, colors, normals, path):
    """保存带法线的PLY文件"""
    # 确保颜色值在0-255范围内
    colors = (colors * 255.0).astype(np.uint8)
    
    # 创建结构化数组
    vertex = np.zeros(points.shape[0], dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    
    vertex['x'] = points[:, 0]
    vertex['y'] = points[:, 1]
    vertex['z'] = points[:, 2]
    vertex['nx'] = normals[:, 0]
    vertex['ny'] = normals[:, 1]
    vertex['nz'] = normals[:, 2]
    vertex['red'] = colors[:, 0]
    vertex['green'] = colors[:, 1]
    vertex['blue'] = colors[:, 2]
    
    # 创建PlyElement并保存
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(path)
    
def poisson_reconstruction_with_normal(pcd: o3d.geometry.PointCloud, depth: int = 9) -> o3d.geometry.TriangleMesh:
    """泊松表面重建（带法线估计）"""
    print("正在估计点云法线...")
    
    # 估计法线
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1,  # 搜索半径
            max_nn=30    # 最大近邻数
        )
    )
    
    # 统一法线方向（可选）
    try:
        pcd.orient_normals_consistent_tangent_plane(k=30)
    except Exception as e:
        print(f"法线方向统一失败: {str(e)}")
    
    print("正在进行泊松重建...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, 
        depth=depth,
        linear_fit=True,
        n_threads=-1  # 使用所有可用线程
    )
    
    # 去除低密度顶点
    densities = np.asarray(densities)
    if len(densities) > 0:
        density_threshold = np.quantile(densities, 0.01)
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
    
    return mesh

# 坐标系转换
def transform_coordinates(points):
    """
    将点云从 Y-up (X前, Z右) 转换为 Z-up (X前, Y左)
    转换规则：
        new_x = original_x (X保持不变)
        new_y = -original_z (原来的右方向变为左方向)
        new_z = original_y (原来的上方向变为上方向)
    """
    transformed_points = np.zeros_like(points)
    transformed_points[:, 0] = points[:, 0]  # X -> X (forward)
    transformed_points[:, 1] = -points[:, 2]  # Z -> -Y (right to left)
    transformed_points[:, 2] = points[:, 1]   # Y -> Z (up to up)
    return transformed_points

def main(data_dir, trans2scanet=False):
    pose_file = os.path.join(data_dir, "sparse/0/images.txt")
    intrinsics_file = os.path.join(data_dir, "sparse/0/cameras.txt")
    depth_dir = os.path.join(data_dir, "depth")
    color_dir = os.path.join(data_dir, "images")
    output_pcd_path = os.path.join(data_dir, "sparse/0/points3D.ply")
    if trans2scanet:
        output_pcd_path = output_pcd_path.replace("points3D.ply", "points3D_scanet.ply")

    # 加载相机内参
    intrinsics = read_intrinsics_text(intrinsics_file)[1]
    fx, fy, cx, cy = intrinsics.params

    # 加载位姿文件
    poses_dict = read_extrinsics_text(pose_file)  # 形状为 (N, 7)
    num_frames = len(poses_dict)

    # 初始化点云
    colored_pcd = o3d.geometry.PointCloud()
         
    # 多帧加速
    n = 1
    selected_frames = {k:v for i, (k,v) in enumerate(poses_dict.items()) if i % n == 0}
    for frame_id, pose in tqdm(selected_frames.items(), desc=f"Processing every {n}th frame", total=len(selected_frames)):
        idx_chr = os.path.splitext(pose.name)[0].split("_")[-1]
        frame_idx = int(idx_chr)
        try:
            # 加载深度图
            depth = np.load(os.path.join(depth_dir, f"observation_depth_{frame_idx}.npy"))
            
            # 加载并处理RGBA图像（去除Alpha通道）
            color_img = Image.open(os.path.join(color_dir, f"observation_rgb_{frame_idx}.png"))
            rgba = np.array(color_img)
            rgb = rgba[..., :3]  # 提取RGB通道（忽略Alpha）
            
            # 检查数据对齐
            assert depth.shape == rgb.shape[:2], \
                f"Depth和RGB尺寸不匹配 {depth.shape} vs {rgb.shape[:2]}"

            # 反投影点云
            height, width = depth.shape
            u, v = np.meshgrid(np.arange(width), np.arange(height))
            z = depth
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
            colors = rgb.reshape(-1, 3) / 255.0  # 归一化
            
            # [x, y, z] -> [x, -y, -z]
            # https://github.com/facebookresearch/habitat-sim/issues/2494#issuecomment-2466709481
            transform_matrix = np.array([
                [1,  0,  0],
                [0, -1,  0],
                [0,  0, -1]
            ])
            points = points @ transform_matrix.T

            # 过滤无效点（根据场景调整阈值）
            valid_mask = (z.flatten() > 0.1) & (z.flatten() < 100.0)
            points = points[valid_mask]
            colors = colors[valid_mask]
            
            # 不可能为0
            assert len(points) != 0
            
            # 获取当前帧位姿
            position = pose.tvec
            quaternion = pose.qvec
            
            # 构建变换矩阵
            rotation = Rotation.from_quat(quaternion).as_matrix()
            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = rotation
            pose_matrix[:3, 3] = position

            # 转换到世界坐标系
            points_hom = np.hstack([points, np.ones((len(points), 1))])
            points_global = (pose_matrix @ points_hom.T).T[:, :3]
            
            if trans2scanet:
                # 转换到scanet坐标系
                points_global = transform_coordinates(points_global)
                
            # 添加到点云
            colored_pcd.points = o3d.utility.Vector3dVector(
                np.vstack([np.asarray(colored_pcd.points), points_global]))
            colored_pcd.colors = o3d.utility.Vector3dVector(
                np.vstack([np.asarray(colored_pcd.colors), colors]))
        
        except Exception as e:
            print(f"处理帧 {frame_idx} 时出错: {str(e)}")
            continue

    # 点云优化
    colored_pcd = colored_pcd.voxel_down_sample(voxel_size=0.02)  # 体素下采样
    colored_pcd, _ = colored_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)  # 统计滤波去噪

    # 转换为numpy数组
    points = np.asarray(colored_pcd.points)
    colors = np.asarray(colored_pcd.colors)
    
    # # 保存结果
    # o3d.io.write_point_cloud(output_pcd_path, colored_pcd)
    
    # 计算法线
    print("计算点云法线...")
    normals = compute_normals(points)
    
    # 保存带法线的PLY文件
    save_ply_with_normals(points, colors, normals, output_pcd_path)
    print(f"带法线的点云已保存至: {output_pcd_path}")
    
    # 泊松重建
    mesh = poisson_reconstruction(points, colors, normals)
    # 保存mesh
    mesh_path = output_pcd_path.replace(".ply", "_mesh.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"泊松重建mesh已保存至: {mesh_path}")

        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    # optional arguments
    parser.add_argument(
        "--data_dir",
        default="output/scanet/test/scene0050_00",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--trans2scanet",
        action="store_true",
    )
    args = parser.parse_args()
    
    print(f"处理数据：{args.data_dir}")
    main(args.data_dir, args.trans2scanet)