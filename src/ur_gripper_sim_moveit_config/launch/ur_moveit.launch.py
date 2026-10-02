# MoveIt 2 bringup for the ur_gripper_sim gz-sim robots (UR + Robotiq Hand-E / 2F-85).
#
# Run AFTER a gz bringup is up (reads /robot_description from the running sim and executes on
# its scaled_joint_trajectory_controller + gripper_controller):
#
#   ros2 launch ur_gripper_sim ur_gz_control.launch.py gui:=false          # Hand-E (ur3e)
#   ros2 launch ur_gripper_sim_moveit_config ur_moveit.launch.py ur_type:=ur3e gripper:=hande
#
#   ros2 launch ur_gripper_sim ur_2f85_gz_control.launch.py gui:=false     # 2F-85 (ur5e)
#   ros2 launch ur_gripper_sim_moveit_config ur_moveit.launch.py ur_type:=ur5e gripper:=robotiq_2f85
#
#   ros2 launch ur_bringup_sim robot_mujoco.launch.py rviz:=false               # 2F-85 (ur5e), MuJoCo
#   ros2 launch ur_gripper_sim_moveit_config ur_moveit.launch.py ur_type:=ur5e gripper:=robotiq_2f85 sensors:=true
#
# load_gripper:=false  -> arm-only planning (gripper still collision-exempt, but not a group).
# sensors:=true        -> octomap from the wrist camera's /camera/depth/color/points
#                         (config/wrist_depth_sensors_3d.yaml), shown by RViz's MotionPlanning
#                         display and published on /monitored_planning_scene. stack_bc's
#                         `collect --scene sensors` plans against this scene (its moveit_py
#                         mirrors /monitored_planning_scene), so run this alongside collection.
#
# Pipelines: OMPL (default) + Pilz (LIN/PTP/CIRC) - Pilz is what stack_bc's oracle plans with.
#
# Self-contained: planning pipelines / kinematics / joint_limits are vendored under config/ so
# they can be customized. The gripper is a MoveIt planning group + end_effector; per-gripper
# controller mapping (FollowJointTrajectory for Hand-E, GripperCommand for 2F-85) is selected by
# the `gripper` arg.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder

PKG = "ur_gripper_sim_moveit_config"


def launch_setup(context, *args, **kwargs):
    ur_type = LaunchConfiguration("ur_type").perform(context)
    gripper = LaunchConfiguration("gripper").perform(context)
    load_gripper = LaunchConfiguration("load_gripper").perform(context)
    launch_rviz = LaunchConfiguration("launch_rviz")
    sensors = LaunchConfiguration("sensors").perform(context).lower() == "true"

    share = get_package_share_directory(PKG)
    cfg = os.path.join(share, "config")
    srdf_path = os.path.join(share, "srdf", "ur_gripper_sim.srdf.xacro")
    controllers = os.path.join(cfg, "moveit_controllers_%s.yaml" % gripper)

    # robot_description is NOT set here — move_group reads it from the running bringup's
    # /robot_description topic (arm + gripper). We only supply the semantic + planning config.
    builder = (
        MoveItConfigsBuilder(robot_name="ur", package_name=PKG)
        .robot_description_semantic(
            file_path=srdf_path,
            mappings={"name": ur_type, "gripper": gripper, "load_gripper": load_gripper})
        .robot_description_kinematics(file_path=os.path.join(cfg, "kinematics.yaml"))
        .joint_limits(file_path=os.path.join(cfg, "joint_limits.yaml"))
        .trajectory_execution(file_path=controllers)
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"], default_planning_pipeline="ompl")
        .pilz_cartesian_limits(file_path=os.path.join(cfg, "pilz_cartesian_limits.yaml"))
    )
    if sensors:
        builder = builder.sensors_3d(file_path=os.path.join(cfg, "wrist_depth_sensors_3d.yaml"))
    moveit_config = builder.to_moveit_configs()

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict(), {"use_sim_time": True}],
    )

    rviz_config = PathJoinSubstitution([FindPackageShare(PKG), "config", "moveit.rviz"])
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_moveit",
        output="log",
        condition=IfCondition(launch_rviz),
        arguments=["-d", rviz_config],
        parameters=[
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            {"use_sim_time": True},
        ],
    )

    return [move_group, rviz]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("ur_type", default_value="ur3e",
                              description="UR variant of the running bringup (ur3e Hand-E, ur5e 2F-85)"),
        DeclareLaunchArgument("gripper", default_value="hande",
                              description="Gripper: hande | robotiq_2f85"),
        DeclareLaunchArgument("load_gripper", default_value="true",
                              description="true: gripper is a MoveIt group/end-effector; false: arm-only"),
        DeclareLaunchArgument("launch_rviz", default_value="true",
                              description="Launch RViz MotionPlanning"),
        DeclareLaunchArgument("sensors", default_value="false",
                              description="Build an octomap from /camera/depth/color/points "
                                          "(config/wrist_depth_sensors_3d.yaml)"),
        OpaqueFunction(function=launch_setup),
    ])
