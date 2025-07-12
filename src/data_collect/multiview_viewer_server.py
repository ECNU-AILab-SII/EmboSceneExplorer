import os
import numpy as np
from PIL import Image
from typing import Any, Dict

import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb

import math
import magnum as mn
import imageio

def make_multiview_camera_cfg(settings):
    sensor_specs = []

    # 使用欧拉角定义方向（单位：弧度）
    camera_orientations = [
        {   # 前视相机（无旋转）
            "name": "front",
            "orientation": [0.0, 0.0, 0.0]  # (pitch, yaw, roll)
        },
        {   # 右视相机（绕Y轴旋转-90度）
            "name": "right",
            "orientation": [0.0, -math.pi/2.0, 0]
        },
        {   # 后视相机（绕Y轴旋转180度）
            "name": "back",
            "orientation": [0.0, math.pi, 0.0]
        },
        {   # 左视相机（绕Y轴旋转90度）
            "name": "left",
            "orientation": [0.0, math.pi/2.0, 0.0]
        }
    ]

    # 公共参数配置
    base_spec = {
        "sensor_type": habitat_sim.SensorType.COLOR,
        "resolution": [settings["height"], settings["width"]],
        "position": [0.0, settings["sensor_height"], 0.0],
        "sensor_subtype": habitat_sim.SensorSubType.PINHOLE,
        "hfov": settings.get("hfov", 90)
    }

    for cam in camera_orientations:
        sensor_spec = habitat_sim.CameraSensorSpec()
        sensor_spec.uuid = f"color_{cam['name']}"
        
        # 应用基础配置
        for key, value in base_spec.items():
            setattr(sensor_spec, key, value)
        
        # 使用欧拉角（Vector3类型）
        sensor_spec.orientation = mn.Vector3(cam["orientation"])

        sensor_specs.append(sensor_spec)

    return sensor_specs


# custom config 
def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.scene_dataset_config_file = settings["scene_dataset"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Here you can specify the amount of displacement in a forward action and the turn angle
    make_action_spec = habitat_sim.agent.ActionSpec
    make_actuation_spec = habitat_sim.agent.ActuationSpec
    MOVE, LOOK = 0.07, 1.5
    
    # all of our possible actions' names
    action_list = [
        "move_left",
        "turn_left",
        "move_right",
        "turn_right",
        "move_backward",
        "look_up",
        "move_forward",
        "look_down",
        "move_down",
        "move_up",
    ]
    
    action_space: Dict[str, habitat_sim.agent.ActionSpec] = {}

    # build our action space map
    for action in action_list:
        actuation_spec_amt = MOVE if "move" in action else LOOK
        action_spec = make_action_spec(
            action, make_actuation_spec(actuation_spec_amt)
        )
        action_space[action] = action_spec
        
    # Create the sensor specifications
    sensor_specs = make_multiview_camera_cfg(settings)

    # Create the agent configuration
    agent_cfg = habitat_sim.agent.AgentConfiguration(
        height=1.5,
        radius=0.1,
        sensor_specifications=sensor_specs,
        action_space=action_space,
        body_type="cylinder",
    )

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def load_action():
    with open("action.txt", "r") as f:
        lines = f.readlines()
        actions = [x.strip() for x in lines]
        return actions
    
def excute_navigation(test_scene, mp3d_scene_dataset, output_dirs):

    sim_settings = {
        "width": 512,  # Spatial resolution of the observations
        "height": 512,
        "scene": test_scene, # Scene path
        "scene_dataset": mp3d_scene_dataset,  # the scene dataset configuration files
        "default_agent": 0,
        "sensor_height": 1.5,  # Height of sensors in meters
        "enable_physics": False,  # kinematics only
    }

    # Create the simulator configuration
    cfg = make_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)

    # Set agent state
    agent = sim.initialize_agent(sim_settings["default_agent"])
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array([-0.6, 0.0, 0.0])  # world space
    agent.set_state(agent_state)

    # Get agent state
    agent_state = agent.get_state()
    print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

    # Get the action queue
    actions = load_action()
    step_count = 0
    combined_images = []
    for action in actions:
        print("action", action)
        observations = sim.step(action)
        
        # 保存所有视角的图像
        for cam_name in ["front", "right", "back", "left"]:
            rgb_obs = observations[f"color_{cam_name}"]
            rgb_img = Image.fromarray(rgb_obs, mode="RGBA")
            rgb_path = os.path.join(output_dirs[cam_name],f"frame_{step_count:05d}.png")
            rgb_img.save(rgb_path)

        # 获取对应视角的RGB数据
        rgb_img_front = Image.fromarray(observations[f"color_front"], mode="RGBA")
        rgb_img_right = Image.fromarray(observations[f"color_right"], mode="RGBA")
        rgb_img_back = Image.fromarray(observations[f"color_back"], mode="RGBA")
        rgb_img_left = Image.fromarray(observations[f"color_left"], mode="RGBA")
        
        # 创建一个新的图像，大小为三倍宽度，三倍高度
        combined_image = Image.new('RGB', (3 * sim_settings["width"], 3 * sim_settings["height"]))
        combined_image.paste(rgb_img_front, (sim_settings["width"], 0))
        combined_image.paste(rgb_img_left, (0, sim_settings["height"]))
        combined_image.paste(rgb_img_right, (2 * sim_settings["width"], sim_settings["height"]))
        combined_image.paste(rgb_img_back, (sim_settings["width"], 2 * sim_settings["height"]))

        # 生成唯一文件名
        combine_rgb_path = os.path.join(output_dirs["combine"], f"frame_{step_count:05d}.png")
        combined_image.save(combine_rgb_path)
        combined_images.append(combined_image)
        
        step_count += 1
        
    # save video
    output_video = "output_video.mp4"
    fps = 15
    imageio.mimsave(output_video, combined_images, fps=fps)
            
        
if __name__ == "__main__":
    
    # 创建保存目录（如果不存在）
    output_dirs = {
        "front": "output/front",
        "right": "output/right",
        "back": "output/back",
        "left": "output/left",
        "combine": "output/combine"
    }

    # 初始化保存目录
    for dir_name, dir_path in output_dirs.items():
        os.makedirs(dir_path, exist_ok=True)
        
    test_scene = "data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"
    mp3d_scene_dataset = "data/scene_datasets/mp3d_example/mp3d.scene_dataset_config.json"
    excute_navigation(test_scene, mp3d_scene_dataset, output_dirs)
    





