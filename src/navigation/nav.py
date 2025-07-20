import os
from typing import TYPE_CHECKING, Union, cast

import sys
current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
submodules_dir = os.path.join(current_script_dir, 'submodules/habitat-lab/habitat-lab')
sys.path.append(submodules_dir)

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import json
import habitat
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.core.agent import Agent
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
    overlay_frame,
)
from habitat_sim.utils import viz_utils as vut

# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

if TYPE_CHECKING:
    from habitat.core.simulator import Observations
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim


class ShortestPathFollowerAgent(Agent):
    r"""Implementation of the :ref:habitat.core.agent.Agent interface that
    uses :refhabitat.tasks.nav.shortest_path_follower.ShortestPathFollower utility class
    for extracting the action on the shortest path to the goal.
    """

    def __init__(self, env: habitat.Env, goal_radius: float):
        self.env = env
        self.shortest_path_follower = ShortestPathFollower(
            sim=cast("HabitatSim", env.sim),
            goal_radius=goal_radius,
            return_one_hot=False,
        )

    def act(self, observations: "Observations") -> Union[int, np.ndarray]:
        return self.shortest_path_follower.get_next_action(
            cast(NavigationEpisode, self.env.current_episode).goals[0].position
        )

    def reset(self) -> None:
        pass


def example_top_down_map_measure(yaml_path, goal_ply_path, goal_jsonl_path, output_path):
    # Create habitat config
    config = habitat.get_config(
        config_path=yaml_path
    )
    # print(config)
    # Add habitat.tasks.nav.nav.TopDownMap and habitat.tasks.nav.nav.Collisions measures
    with habitat.config.read_write(config):
        config.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=90,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )

    # Create dataset
    dataset = habitat.make_dataset(
        id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
    )

    pcd = o3d.io.read_point_cloud(goal_ply_path)
    points = np.asarray(pcd.points)  # Nx3 的点云坐标
    # print(f"点云总点数: {len(points)}")
    # print(f"原始点云坐标范围:")
    # print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    # print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    # print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    
    # 坐标系平移：将点云坐标系从以0为中心转换为从0开始
    # 计算需要平移的距离
    x_offset = -points[:, 0].min()  # 将X轴最小值移到0
    y_offset = -points[:, 1].min()  # 将Y轴最小值移到0
    z_offset = 0  # Z轴已经是0开始，不需要平移
    
    # 应用平移
    points[:, 0] += x_offset
    points[:, 1] += y_offset
    points[:, 2] += z_offset
    
    # print(f"平移后点云坐标范围:")
    # print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    # print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    # print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    # print(f"平移偏移量: X={x_offset:.3f}, Y={y_offset:.3f}, Z={z_offset:.3f}")
    with open(goal_jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line)
            # if data["scene_id"] == "scene0000_00" and data["question_id"] == 1:
            pre_mask = np.array(data["pred_mask"])
            break
    target_points = points[pre_mask == 1]
    if len(target_points) == 0:
        raise ValueError("未检测到任何目标点。")
    # print("目标点数量:", len(target_points))
    # print("目标点坐标范围:")
    # print("  X: [{:.3f}, {:.3f}]".format(target_points[:, 0].min(),
    #                                      target_points[:, 0].max()))
    # print("  Y: [{:.3f}, {:.3f}]".format(target_points[:, 1].min(),
    #                                      target_points[:, 1].max()))
    # print("  Z: [{:.3f}, {:.3f}]".format(target_points[:, 2].min(),
    #                                      target_points[:, 2].max()))
    target_center = target_points.mean(axis=0)
    
    # 坐标系统转换：点云坐标系 -> Habitat坐标系
    # 点云: [x, y, z] -> Habitat: [x, z, y]
    original_target = target_center.copy()
    target_center[1], target_center[2] = target_center[2], -target_center[1]
    
    # print("坐标转换详情:")
    # print(f"  平移后目标中心: {original_target}")
    # print(f"  转换后坐标: {target_center}")
    # print(f"  转换规则: [x, y, z] -> [x, z, y] (Habitat坐标系)")
    
    # print("目标物体中心坐标为：", target_center)

    # Create simulation environment
    with habitat.Env(config=config, dataset=dataset) as env:
        # Create ShortestPathFollowerAgent agent
        agent = ShortestPathFollowerAgent(
            env=env,
            goal_radius=config.habitat.task.measurements.success.success_distance,
        )

        # Create video of agent navigating in the first episode
        num_episodes = 1
        current_episode = dataset.episodes[0]
        current_episode.start_position = [4.98278317, 0.41561477, -4.59293777]

        new_goal_position = target_center
        new_goal = NavigationGoal(
            position=new_goal_position,
            radius=0.2
        )
       
        lower, upper = env.sim.pathfinder.get_bounds()
        print("NavMesh bounds:")
        print("  lower:", lower)  # Vector3f(x, y, z)
        print("  upper:", upper)
        
        
        # 验证目标坐标是否在边界内，如果不在则调整
        original_target = target_center.copy()
        target_center[0] = np.clip(target_center[0], lower[0], upper[0])
        target_center[1] = np.clip(target_center[1], lower[1], upper[1])
        target_center[2] = np.clip(target_center[2], lower[2], upper[2])
        
        if not np.allclose(original_target, target_center):
            print("⚠️  目标坐标已调整到边界内:")
            print("  原始坐标:", original_target)
            print("  调整后坐标:", target_center)
            # 更新目标位置
            new_goal_position = target_center
            new_goal = NavigationGoal(
                position=new_goal_position,
                radius=0.2
            )

        current_episode.goals = [new_goal]
        print("goals:", current_episode.goals, data["prompt"])
        

        for _ in range(num_episodes):
            # Load the first episode and reset agent
            observations = env.reset()
            agent.reset()

            # Get metrics
            info = env.get_metrics()
            # Concatenate RGB-D observation and topdowm map into one image
            frame = observations_to_image(observations, info)

            info.pop("top_down_map")
            frame = overlay_frame(frame, info)
            # Add fame to vis_frames
            vis_frames = [frame]

            # Repeat the steps above while agent doesn't reach the goal
            while not env.episode_over:
                # Get the next best action
                action = agent.act(observations)
                if action is None:
                    break

                # Step in the environment
                observations = env.step(action)
                info = env.get_metrics()
                frame = observations_to_image(observations, info)

                info.pop("top_down_map")
                frame = overlay_frame(frame, info)
                vis_frames.append(frame)

            current_episode = env.current_episode

            ply_filename = os.path.splitext(os.path.basename(goal_ply_path))[0]
            video_name = f"{os.path.basename(current_episode.scene_id)}_{current_episode.episode_id}_{ply_filename}"
            # Create video from images and save to disk
            os.makedirs(output_path, exist_ok=True)
            images_to_video(
                vis_frames, output_path, video_name, fps=6, quality=9
            )
            vis_frames.clear()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # optional arguments
    parser.add_argument(
        "--yaml_path",
        default="example_data/scanet/pointnav_scannet.yaml",
        type=str,
        help='navigation yaml path',
    )
    parser.add_argument(
        "--ply_path",
        default="./data/navigation_data/black_tv.ply",
        type=str,
        help='navigation point cloud ply path',
    )
    parser.add_argument(
        "--jsonl_path",
        default="./data/navigation_data/scene0000_00_result.jsonl",
        type=str,
        help='navigation json path',
    )
    parser.add_argument(
        "--output_path",
        default="output/xzl/",
        type=str,
        help='navigation output directory path',
    )
    
    args = parser.parse_args()

    output_path = os.path.join(args.output_path, "visualization/")
    example_top_down_map_measure(args.yaml_path, args.ply_path, args.jsonl_path, output_path)
