# Starts the ros_gz_bridge parameter_bridge for the D435 wrist camera, driven by
# config/camera_bridge.yaml. Meant to be included (not run standalone) from
# ur_gripper_gz's *_gz_control.launch.py files, gated by a use_camera arg.

from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    bridge_config = PathJoinSubstitution(
        [FindPackageShare("ur_camera_gz"), "config", "camera_bridge.yaml"])

    camera_bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge",
        name="camera_bridge",
        parameters=[{"config_file": bridge_config, "use_sim_time": True}],
        output="screen")

    return LaunchDescription([camera_bridge])
