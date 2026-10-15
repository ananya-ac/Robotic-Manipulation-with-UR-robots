"""Start the tabletop perception node.

  ros2 launch tabletop_perception perception.launch.py [config:=tabletop_mujoco.yaml]

config: a file in this package's config/ or an absolute path. The node needs Open3D: launch from
a shell with the venv active (see SETUP.md). colcon writes the console script with its own
interpreter (/usr/bin/python3, no Open3D), so it is started through `python3` from PATH -
the venv's when active. Without launch: python3 -m tabletop_perception.perception_node.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context):
    cfg = LaunchConfiguration("config").perform(context)
    if not os.path.isabs(cfg):
        cfg = os.path.join(get_package_share_directory("tabletop_perception"), "config", cfg)
    sim_time = LaunchConfiguration("use_sim_time").perform(context).lower() == "true"
    return [Node(package="tabletop_perception", executable="perception_node", name="perception_node",
                 output="screen", parameters=[cfg, {"use_sim_time": sim_time}], prefix="python3")]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value="tabletop_mujoco.yaml"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        OpaqueFunction(function=_setup),
    ])
