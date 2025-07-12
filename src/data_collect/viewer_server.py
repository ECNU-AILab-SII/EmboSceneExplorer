import os
import sys
current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
submodules_dir = os.path.join(current_script_dir, '../../submodules')
sys.path.append(submodules_dir)

import numpy as np
from PIL import Image
from typing import Any, Dict

from color_utils import Scanetv2_40_colors_rgb
import habitat_sim
from habitat_sim.utils.settings import default_sim_settings
from scipy.spatial.transform import Rotation

def save_colmap_cameras(K, camera_file, cam_nums):
    with open(camera_file, 'w') as f:
        for i in range(1, cam_nums+1):  # Starting index at 1
            width = int(K[0, 2] * 2)
            height = int(K[1, 2] * 2)
            f.write(f"{i} PINHOLE {width} {height} {K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]}\n")

def save_colmap_images(camera_poses, images_path, img_list):
    with open(images_path, 'w') as f:
        for i, pose in enumerate(camera_poses, 1): # Starting index at 1
            rotation, position = pose
            q = np.array([rotation.x, rotation.y, rotation.z, rotation.w])
            t = position
            f.write(f"{i} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {i} {img_list[i-1]}\n")
            f.write(f"\n")

def save_observation(action, rgb_obs, semantic_obs, depth_obs, output_path):
    ##  get rgb, depth, semantic images
    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")
    semantic_img, depth_img = None, None
    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(Scanetv2_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
    # if depth_obs.size != 0:
    #     depth = depth_obs / (10 - 0) # normalize depth values to [0, 1]
    #     depth_img = Image.fromarray((depth * 255).astype(np.uint8), mode="L")

    # # Combine depth, RGB, and semantic images into a composite image
    # composite_img = Image.new("RGBA", (rgb_img.width * 3, rgb_img.height))
    # composite_img.paste(rgb_img, (0, 0))
    # composite_img.paste(semantic_img, (rgb_img.width, 0))
    # composite_img.paste(depth_img.convert("RGBA"), (rgb_img.width * 2, 0))

    # Create subdirectories for RGB, depth, and semantic images
    rgb_dir = os.path.join(output_path, "images")
    depth_dir = os.path.join(output_path, "depth")
    semantic_dir = os.path.join(output_path, "semantic")
    # composite_dir = os.path.join(output_path, "composite")

    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(semantic_dir, exist_ok=True)
    # os.makedirs(composite_dir, exist_ok=True)

    # Save images to respective subdirectories
    rgb_path = os.path.join(rgb_dir, f"observation_rgb_{action}.png")
    # depth_path = os.path.join(depth_dir, f"observation_depth_{action}.png")
    depth_file = os.path.join(depth_dir, f"observation_depth_{action}.npy")
    semantic_path = os.path.join(semantic_dir, f"observation_semantic_{action}.png")
    # composite_path = os.path.join(composite_dir, f"observation_{str(action).zfill(5)}.png")

    rgb_img.save(rgb_path)
    # depth_img.save(depth_path)
    np.save(depth_file, depth_obs)
    semantic_img.save(semantic_path)
    # composite_img.save(composite_path)

# custom config 
def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.scene_dataset_config_file = settings["scene_dataset_config_file"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings["height"], settings["width"]]
    color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    color_sensor_spec.clear_color = [1.0, 1.0, 1.0, 1.0]  # 设置背景为白色（RGBA）
    sensor_specs.append(color_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    depth_sensor_spec.clear_color = [1.0, 1.0, 1.0, 1.0]  # 设置背景为白色（RGBA）
    sensor_specs.append(depth_sensor_spec)

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    semantic_sensor_spec.clear_color = [1.0, 1.0, 1.0, 1.0]  # 设置背景为白色（RGBA）
    sensor_specs.append(semantic_sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    make_action_spec = habitat_sim.agent.ActionSpec
    make_actuation_spec = habitat_sim.agent.ActuationSpec
    MOVE, LOOK = settings["MOVE"], settings["LOOK"]
    
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

    agent_cfg = habitat_sim.agent.AgentConfiguration(
        height=1.5,
        radius=0.1,
        sensor_specifications=sensor_specs,
        action_space=action_space,
        body_type="cylinder",
    )

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def get_camera_intrinsics(sim, sensor_name):
    # Get render camera
    render_camera = sim._sensors[sensor_name]._sensor_object.render_camera

    # Get projection matrix
    projection_matrix = render_camera.projection_matrix

    # Get resolution
    viewport_size = render_camera.viewport

    # Intrinsic calculation
    fx = projection_matrix[0, 0] * viewport_size[0] / 2.0
    fy = projection_matrix[1, 1] * viewport_size[1] / 2.0
    cx = (projection_matrix[2, 0] + 1.0) * viewport_size[0] / 2.0
    cy = (projection_matrix[2, 1] + 1.0) * viewport_size[1] / 2.0

    intrinsics = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])
    return intrinsics

def load_action(action_path="action.txt"):
    with open(action_path, "r") as f:
        lines = f.readlines()
        actions = [x.strip() for x in lines]
        return actions
    
def set_agent_to_navmesh_center(sim):
    # 获取场景AABB中心
    scene_graph = sim.get_active_scene_graph()
    aabb = scene_graph.get_root_node().cumulative_bb
    center = aabb.center()

    # 获取NavMesh并找到最近的可导航点
    navmesh = sim.pathfinder
    if not navmesh.is_loaded:
        raise RuntimeError("NavMesh未加载！请确保场景支持路径规划。")

    # 将中心点投影到NavMesh上
    center_on_navmesh = navmesh.snap_point(center)

    # 设置Agent状态
    agent_state = habitat_sim.AgentState()
    agent_state.position = center_on_navmesh
    agent_state.rotation = np.quaternion(1, 0, 0, 0)
    agent = sim.initialize_agent(0, agent_state)
    return agent

def navmesh_config_and_recompute(cfg, agent_id, sim):
    """
    This method is setup to be overridden in for setting config accessibility
    in inherited classes.
    """
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_height = cfg.agents[agent_id].height
    navmesh_settings.agent_radius = cfg.agents[agent_id].radius
    navmesh_settings.include_static_objects = True
    sim.recompute_navmesh(
        sim.pathfinder,
        navmesh_settings,
    )  
          
def semantic_habitat_excute(sim_settings, action_path, output_path, start_position, start_rotation, feq=1):

    # Create the simulator configuration
    cfg = make_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)

    # Set agent state
    agent = sim.initialize_agent(sim_settings["default_agent"])
    if start_position is not None and start_rotation is not None:
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array(start_position)
        agent_state.rotation = np.quaternion(start_rotation[0], start_rotation[1], start_rotation[2], start_rotation[3])
        agent.set_state(agent_state)
    else:
        # using default position with navmesh
        # compute NavMesh if not already loaded by the scene.
        if (not sim.pathfinder.is_loaded and cfg.sim_cfg.scene_id.lower() != "none"):
            navmesh_config_and_recompute(cfg, 0, sim)
        agent = set_agent_to_navmesh_center(sim)
    
    # Get agent state
    agent_state = agent.get_state()
    print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)
    
    # Get intrinsics
    intrinsics = get_camera_intrinsics(sim, "color_sensor")
    print("Camera intrinsics: ")
    print(intrinsics)
    
    # excute actions
    action_idx = 0
    img_list = []
    cam_poses =[]
    
    # Get the action queue
    if action_path is not None:
        actions = load_action(action_path)
        for action in actions:
            action_idx += 1
            print("action", action)
            observations = sim.step(action)
            if action_idx % feq == 0:
                rgb = observations["color_sensor"]
                semantic = observations["semantic_sensor"]
                depth = observations["depth_sensor"]
                save_observation(action_idx, rgb, semantic, depth, output_path)

                # For colmap-style svaing
                img_list.append(f"observation_rgb_{action_idx}.png")
                sensor_state = agent.get_state().sensor_states['color_sensor']
                position = sensor_state.position
                rotation = sensor_state.rotation
                cam_poses.append([rotation, position])
    else:
        for action_idx in range(360 // int(sim_settings["LOOK"])):
            action = "turn_right"
            print("action", action)
            observations = sim.step(action)
            if action_idx % feq == 0:
                rgb = observations["color_sensor"]
                semantic = observations["semantic_sensor"]
                depth = observations["depth_sensor"]
                save_observation(action_idx, rgb, semantic, depth, output_path)

                # For colmap-style svaing
                img_list.append(f"observation_rgb_{action_idx}.png")
                sensor_state = agent.get_state().sensor_states['color_sensor']
                position = sensor_state.position
                rotation = sensor_state.rotation
                cam_poses.append([rotation, position])
            
    # import pdb; pdb.set_trace()
    # create colmap-style directory
    colmap_path = os.path.join(output_path, "sparse/0")
    os.makedirs(colmap_path, exist_ok=True)
    images_file = os.path.join(colmap_path, "images.txt")
    camera_file = os.path.join(colmap_path, "cameras.txt")
    save_colmap_images(cam_poses, images_file, img_list)
    save_colmap_cameras(intrinsics, camera_file, len(img_list))
    

        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # optional arguments
    parser.add_argument(
        "--scene",
        default="sc1_staging_00",
        type=str,
        required=True,
        help='scene/stage file to load (default: "./data/test_assets/scenes/simple_room.glb")',
    )
    parser.add_argument(
        "--dataset",
        default="data/replica_cad_baked_lighting/replicaCAD_baked.scene_dataset_config.json",
        type=str,
        required=True,
        metavar="DATASET",
        help='dataset configuration file to use (default: "default")',
    )
    parser.add_argument(
        "--action_path",
        type=str,
        help="Composite files that the batch renderer will use in-place of simulation assets to improve memory usage and performance. If none is specified, the original scene files will be loaded from disk.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        default="output/oLBMNvg9in8/",
        help="Composite files that the batch renderer will use in-place of simulation assets to improve memory usage and performance. If none is specified, the original scene files will be loaded from disk.",
    )
    parser.add_argument(
        "--disable-physics",
        action="store_true",
        help="disable physics simulation (default: False)",
    )
    parser.add_argument(
        "--use-default-lighting",
        action="store_true",
        help="Override configured lighting to use default lighting for the stage.",
    )
    parser.add_argument(
        "--composite-files",
        type=str,
        nargs="*",
        help="Composite files that the batch renderer will use in-place of simulation assets to improve memory usage and performance. If none is specified, the original scene files will be loaded from disk.",
    )
    parser.add_argument(
        "--num-environments",
        default=1,
        type=int,
        help="Number of concurrent environments to batch render. Note that only the first environment simulates physics and can be controlled.",
    )
    parser.add_argument(
        "--hbao",
        action="store_true",
        help="Enable horizon-based ambient occlusion, which provides soft shadows in corners and crevices.",
    )
    parser.add_argument(
        "--feq",
        default=1,
        type=int,
        help="Frequence of record.",
    )
    parser.add_argument(
        "--width",
        default=1920//2,
        type=int,
        help="Horizontal resolution of the window.",
    )
    parser.add_argument(
        "--height",
        default=1440//2,
        type=int,
        help="Vertical resolution of the window.",
    )
    parser.add_argument(
        "--sensor_height",
        default=1.5,
        type=float,
        help="Vertical resolution of the window.",
    )
    parser.add_argument(
        "--start_position", 
        type=str, 
        help="Start pose as a string"
    )
    parser.add_argument(
        "--start_rotation", 
        type=str, 
        help="Start rotation as a string"
    )
    parser.add_argument(
        "--MOVE",
        default=0.07,
        type=float,
        help="Move stride.",
    )
    parser.add_argument(
        "--LOOK",
        default=1.5,
        type=float,
        help="Look stride.",
    )
    
    args = parser.parse_args()
    
    # custom settings
    print(f"output_path = {args.output_path}")
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    if args.start_position is not None:
        start_position = list(map(float, args.start_position.strip('[]').split(',')))
    else:
        start_position = None

    if args.start_rotation is not None:
        start_rotation = list(map(float, args.start_rotation.strip('[]').split(',')))
    else:
        start_rotation = None

    # Setting up sim_settings
    sim_settings: Dict[str, Any] = default_sim_settings
    sim_settings["width"] = args.width
    sim_settings["height"] = args.height
    sim_settings["sensor_height"] = args.sensor_height
    sim_settings["scene"] = args.scene
    sim_settings["scene_dataset_config_file"] = args.dataset
    sim_settings["enable_physics"] = not args.disable_physics
    sim_settings["use_default_lighting"] = args.use_default_lighting
    sim_settings["num_environments"] = args.num_environments
    sim_settings["enable_hbao"] = args.hbao
    sim_settings["MOVE"] = args.MOVE
    sim_settings["LOOK"] = args.LOOK

    semantic_habitat_excute(sim_settings, args.action_path, args.output_path, start_position, start_rotation, args.feq)

