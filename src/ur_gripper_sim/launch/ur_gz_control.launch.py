# Gazebo (gz-sim / Harmonic) bringup for UR + Robotiq Hand-E.
#
# Thin overlay on the upstream ur_simulation_gz vendor package (apt:
# ros-jazzy-ur-simulation-gz). Uses the legacy ur_gripper_gazebo Hand-E URDF
# geometry with gz_ros2_control.
#
#   ros2 launch ur_gripper_sim ur_gz_control.launch.py
#   ros2 launch ur_gripper_sim ur_gz_control.launch.py gui:=false ur_type:=ur3e
#
# Gripper (Hand-E finger_joint) is driven by gripper_controller, a
# JointTrajectoryController in gz. ur_control clients use trajectory mode
# (config/gripper_hande_client.yaml).

import os

import launch
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            RegisterEventHandler, SetEnvironmentVariable, TimerAction)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (Command, FindExecutable, LaunchConfiguration,
                                  PathJoinSubstitution, PythonExpression)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # gz resolves package:// URIs from GZ_SIM_RESOURCE_PATH roots.
    share_roots = [
        os.path.dirname(get_package_share_directory("ur_description")),
        os.path.dirname(get_package_share_directory("ur_gripper_sim")),
    ]
    gz_resource_path = SetEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH",
        os.pathsep.join(share_roots + [os.environ.get("GZ_SIM_RESOURCE_PATH", "")]),
    )

    ur_type = LaunchConfiguration("ur_type")
    gui = LaunchConfiguration("gui")

    pkg = FindPackageShare("ur_gripper_sim")
    controllers_file = PathJoinSubstitution([pkg, "config", "ur_gz_controllers.yaml"])
    xacro_file = PathJoinSubstitution([pkg, "urdf", "ur_gripper_hande_gz.urdf.xacro"])

    robot_description_content = Command([
        FindExecutable(name="xacro"), " ", xacro_file,
        " ur_type:=", ur_type,
        " name:=", ur_type,
        " simulation_controllers:=", controllers_file,
    ])
    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description, {"use_sim_time": True}],
    )

    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
        output="screen",
    )

    # Latch the gripper name so ur_control clients ('--gripper auto') resolve their config.
    active_gripper_pub = Node(
        package="ur_control", executable="active_gripper_publisher",
        parameters=[{"gripper": "robotiq_hande"}], output="screen")

    # Butterworth-filter + zeroable republish of the FT sensor: subscribes to /wrench (from the
    # broadcaster) and publishes /wrench/filtered plus a /wrench/filtered/zero_ftsensor service.
    # ur_control's Arm prefers the filtered topic and needs that service for zero_ft_sensor().
    ft_filter = Node(
        package="ur_control_examples", executable="ft_filter",
        arguments=["-t", "wrench"],
        parameters=[{"use_sim_time": True}], output="screen")

    # bullet-featherstone is REQUIRED for the Hand-E mimic finger (default DART has no
    # mimic-constraint support).
    gz_args = PythonExpression(
        ["'--physics-engine gz-physics-bullet-featherstone-plugin -r -v3 empty.sdf' if '", gui,
         "' == 'true' else '--physics-engine gz-physics-bullet-featherstone-plugin -s -r -v3 empty.sdf'"]
    )
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py"]
        ),
        launch_arguments={"gz_args": gz_args}.items(),
    )

    spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=["-topic", "robot_description", "-name", ur_type, "-allow_renaming", "true"],
    )
    spawn_entity_delayed = TimerAction(period=4.0, actions=[spawn_entity])

    jsb_spawner = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=[
            "joint_state_broadcaster", "-c", "/controller_manager",
            "--controller-manager-timeout", "120",
        ],
    )
    # Publishes the wrist FT sensor on /wrench so ur_control's Arm reads it unchanged
    # (FT_SUBSCRIBER='wrench'). --controller-ros-args remaps the controller node itself
    # (the broadcaster publishes ~/wrench); a Node-level remap would only touch the
    # short-lived spawner process, not the controller running inside controller_manager.
    fts_spawner = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=[
            "force_torque_sensor_broadcaster", "-c", "/controller_manager",
            "--controller-manager-timeout", "120",
            "--controller-ros-args", "-r /force_torque_sensor_broadcaster/wrench:=/wrench",
        ],
    )
    jtc_spawner = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=[
            "scaled_joint_trajectory_controller", "-c", "/controller_manager",
            "--controller-manager-timeout", "120",
        ],
    )
    fvc_spawner = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=[
            "forward_velocity_controller", "-c", "/controller_manager",
            "--inactive", "--controller-manager-timeout", "120",
        ],
    )
    # FZI Cartesian compliance controller: inactive at startup (conflicts with the JTC on
    # the position interfaces), switched on by ur_control's CompliantController. Its FT input
    # (~/ft_sensor_wrench) is remapped to /wrench, where the FT broadcaster publishes.
    compliance_spawner = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=[
            "cartesian_compliance_controller", "-c", "/controller_manager",
            "--inactive", "--controller-manager-timeout", "120",
            "--controller-ros-args",
            "-r /cartesian_compliance_controller/ft_sensor_wrench:=/wrench/filtered",
        ],
    )
    gripper_spawner = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=[
            "gripper_controller", "-c", "/controller_manager",
            "--controller-manager-timeout", "120",
        ],
        condition=launch.conditions.IfCondition(LaunchConfiguration("load_gripper")),
    )

    return LaunchDescription([
        gz_resource_path,
        DeclareLaunchArgument("ur_type", default_value="ur3e",
                              description="UR robot variant (ur3, ur3e, ur5e, ...)"),
        DeclareLaunchArgument("gui", default_value="true",
                              description="Run the Gazebo GUI (false = headless server)"),
        DeclareLaunchArgument("load_gripper", default_value="true",
                              description="Spawn gripper_controller for Hand-E finger_joint"),
        robot_state_publisher,
        clock_bridge,
        active_gripper_pub,
        ft_filter,
        gz_sim,
        spawn_entity_delayed,
        RegisterEventHandler(OnProcessExit(target_action=spawn_entity, on_exit=[jsb_spawner])),
        RegisterEventHandler(OnProcessExit(target_action=jsb_spawner, on_exit=[fts_spawner, jtc_spawner])),
        RegisterEventHandler(
            OnProcessExit(target_action=jtc_spawner,
                          on_exit=[fvc_spawner, compliance_spawner, gripper_spawner])),
    ])
