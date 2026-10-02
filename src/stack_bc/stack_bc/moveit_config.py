"""MoveIt config for the oracle's moveit_py instance (motion.py).

With `collect --scene sensors`, the collision world comes from the octomap that
  ros2 launch ur_gripper_sim_moveit_config ur_moveit.launch.py ur_type:=ur5e gripper:=robotiq_2f85 sensors:=true
builds: that move_group owns the occupancy map monitor, and moveit_py mirrors its scene by
listening to /monitored_planning_scene. moveit_py can't build the octomap itself (MoveItCpp starts
its world-geometry monitor without the octomap monitor). Keep the pipelines here and in
ur_moveit.launch.py in step (both load OMPL + Pilz).
"""

import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder

MOVEIT_PKG = "ur_gripper_sim_moveit_config"
MONITORED_SCENE_TOPIC = "/monitored_planning_scene"   # move_group's published scene


def build_moveit_config(robot_description: str):
    """MoveItConfigs for the live URDF string."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False)
    tmp.write(robot_description)
    tmp.close()
    share = get_package_share_directory(MOVEIT_PKG)
    cfg = os.path.join(share, "config")
    try:
        return (
            MoveItConfigsBuilder(robot_name="ur", package_name=MOVEIT_PKG)
            .robot_description(file_path=tmp.name)
            .robot_description_semantic(file_path=os.path.join(share, "srdf", "ur_gripper_sim.srdf.xacro"),
                                        mappings={"name": "ur5e", "gripper": "robotiq_2f85", "load_gripper": "true"})
            .robot_description_kinematics(file_path=os.path.join(cfg, "kinematics.yaml"))
            .joint_limits(file_path=os.path.join(cfg, "joint_limits.yaml"))
            .trajectory_execution(file_path=os.path.join(cfg, "moveit_controllers_robotiq_2f85.yaml"))
            .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"], default_planning_pipeline="ompl")
            .pilz_cartesian_limits(file_path=os.path.join(cfg, "pilz_cartesian_limits.yaml"))
            .moveit_cpp(file_path=os.path.join(cfg, "moveit_cpp.yaml"))
            .to_moveit_configs()
        )
    finally:
        os.unlink(tmp.name)


def moveit_py_params(robot_description: str, mirror_scene: bool = False):
    """Parameter dict for MoveItPy. mirror_scene=True subscribes its planning scene monitor to
    move_group's published scene (octomap + attached objects) instead of the unused default."""
    config = build_moveit_config(robot_description).to_dict()
    config["use_sim_time"] = True
    config["qos_overrides"] = {"/clock": {"subscription": {
        "reliability": "best_effort", "durability": "volatile", "history": "keep_last", "depth": 1}}}
    if mirror_scene:
        config["planning_scene_monitor_options"]["monitored_planning_scene_topic"] = MONITORED_SCENE_TOPIC
    return config
