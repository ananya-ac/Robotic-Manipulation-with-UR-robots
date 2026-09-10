# Gazebo (gz-sim / Harmonic) bringup for the full robot: UR + gripper + D435 wrist
# camera, plus rviz2 showing the color and depth feeds.
#
#   ros2 launch ur_bringup_gz robot_gz.launch.py gripper:=2f85    # default
#   ros2 launch ur_bringup_gz robot_gz.launch.py gripper:=hande
#   ros2 launch ur_bringup_gz robot_gz.launch.py gui:=false rviz:=false
#   ros2 launch ur_bringup_gz robot_gz.launch.py camera_offset:=-0.05   # tune live (default -0.03)
#   ros2 launch ur_bringup_gz robot_gz.launch.py camera_offset_x:=0.02 camera_offset_y:=-0.02
#
# Composes, unmodified:
#   - ur_gripper_gz's config/*.yaml (controller configs, picked by `gripper`)
#   - ur_camera_gz's launch/camera_bridge.launch.py (included directly)
#   - ros_gz_sim's launch/gz_sim.launch.py (included directly)
# The robot_description/spawn/spawner block below can't be reused via
# IncludeLaunchDescription the same way: it must point at ur_bringup_gz's own
# composed xacro (urdf/ur5e_<gripper>_camera_gz.urdf.xacro), which ur_gripper_gz's
# own launch files have no argument to select. Its structure mirrors
# ur_gripper_gz's ur_2f85_gz_control.launch.py / ur_gz_control.launch.py exactly
# (same spawner chain, same event-handler sequencing) since both gripper variants
# already used one identical sequence.
#
# bullet-featherstone is REQUIRED for both grippers' mimic joints (2F-85 four-bar
# linkage, Hand-E mimic finger) — the default DART engine has no mimic support.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            RegisterEventHandler, SetEnvironmentVariable, TimerAction)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (Command, FindExecutable, LaunchConfiguration,
                                  PathJoinSubstitution, PythonExpression)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # gz resolves package:// meshes (rewritten to model:// during URDF->SDF conversion)
    # against these roots. ur_description/robotiq_description/realsense2_description all
    # share /opt/ros/jazzy/share (apt install), so one entry covers all three. ur_gripper_gz
    # (Hand-E meshes) and ur_camera_gz (D435 meshes) each build into their own isolated
    # install/<pkg>/share prefix (no --merge-install), so each needs its own entry -- neither
    # is covered by the apt root above, and neither package's meshes are needed by every
    # variant, but it's harmless to include both unconditionally.
    share_roots = [
        os.path.dirname(get_package_share_directory("ur_description")),
        os.path.dirname(get_package_share_directory("ur_gripper_gz")),
        os.path.dirname(get_package_share_directory("ur_camera_gz")),
    ]
    gz_resource_path = SetEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH",
        os.pathsep.join(share_roots + [os.environ.get("GZ_SIM_RESOURCE_PATH", "")]))

    gripper = LaunchConfiguration("gripper")
    ur_type = LaunchConfiguration("ur_type")
    gui = LaunchConfiguration("gui")
    use_camera = LaunchConfiguration("use_camera")
    camera_offset = LaunchConfiguration("camera_offset")
    camera_offset_x = LaunchConfiguration("camera_offset_x")
    camera_offset_y = LaunchConfiguration("camera_offset_y")

    is_2f85 = ["'", gripper, "' == '2f85'"]
    xacro_filename = PythonExpression(
        ["'ur5e_2f85_camera_gz.urdf.xacro' if "] + is_2f85 + ["else 'ur5e_hande_camera_gz.urdf.xacro'"])
    controllers_filename = PythonExpression(
        ["'ur_gz_2f85_controllers.yaml' if "] + is_2f85 + ["else 'ur_gz_controllers.yaml'"])
    active_gripper_name = PythonExpression(
        ["'robotiq_2f85' if "] + is_2f85 + ["else 'robotiq_hande'"])

    xacro_file = PathJoinSubstitution([FindPackageShare("ur_bringup_gz"), "urdf", xacro_filename])
    controllers_file = PathJoinSubstitution([FindPackageShare("ur_gripper_gz"), "config", controllers_filename])

    robot_description_content = Command([
        FindExecutable(name="xacro"), " ", xacro_file,
        " name:=", ur_type, " ur_type:=", ur_type,
        " simulation_controllers:=", controllers_file,
        " camera_offset:=", camera_offset,
        " camera_offset_x:=", camera_offset_x,
        " camera_offset_y:=", camera_offset_y,
    ])
    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}

    robot_state_publisher = Node(
        package="robot_state_publisher", executable="robot_state_publisher", output="screen",
        parameters=[robot_description, {"use_sim_time": True}])

    clock_bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"], output="screen")

    camera_bridge = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("ur_camera_gz"), "/launch/camera_bridge.launch.py"]),
        condition=IfCondition(use_camera))

    active_gripper_pub = Node(
        package="ur_control", executable="active_gripper_publisher",
        parameters=[{"gripper": active_gripper_name}], output="screen")

    ft_filter = Node(
        package="ur_control_examples", executable="ft_filter",
        arguments=["-t", "wrench"],
        parameters=[{"use_sim_time": True}], output="screen")

    rviz_config = PathJoinSubstitution([FindPackageShare("ur_bringup_gz"), "config", "camera_view.rviz"])
    rviz_node = Node(
        package="rviz2", executable="rviz2", name="rviz2", output="screen",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(LaunchConfiguration("rviz")))

    # worlds/warehouse.sdf (its own package: the world is independent of which
    # robot/gripper/sensor combo is spawned into it) adds the gz-sim Sensors system,
    # which the stock empty.sdf lacks -- without it, rendering-based sensors like the
    # D435 rgbd_camera create their gz topics but never publish any actual frames.
    world_file = PathJoinSubstitution([FindPackageShare("worlds"), "worlds", "warehouse.sdf"])
    gz_args = PythonExpression([
        "'--physics-engine gz-physics-bullet-featherstone-plugin -r -v3 ", world_file, "' if '", gui,
        "' == 'true' else '--physics-engine gz-physics-bullet-featherstone-plugin -s -r -v3 ", world_file, "'"])
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py"]),
        launch_arguments={"gz_args": gz_args}.items())

    spawn_entity = Node(package="ros_gz_sim", executable="create", output="screen",
                        arguments=["-topic", "robot_description", "-name", ur_type, "-allow_renaming", "true"])
    spawn_entity_delayed = TimerAction(period=4.0, actions=[spawn_entity])

    def spawner(name, *extra):
        return Node(package="controller_manager", executable="spawner", output="screen",
                    arguments=[name, "-c", "/controller_manager", "--controller-manager-timeout", "120", *extra])

    jsb = spawner("joint_state_broadcaster")
    jtc = spawner("scaled_joint_trajectory_controller")
    fvc = spawner("forward_velocity_controller", "--inactive")
    compliance = spawner(
        "cartesian_compliance_controller", "--inactive",
        "--controller-ros-args",
        "-r /cartesian_compliance_controller/ft_sensor_wrench:=/wrench/filtered")
    gc = Node(package="controller_manager", executable="spawner", output="screen",
             arguments=["gripper_controller", "-c", "/controller_manager", "--controller-manager-timeout", "120"],
             condition=IfCondition(LaunchConfiguration("load_gripper")))
    fts = spawner("force_torque_sensor_broadcaster",
                  "--controller-ros-args", "-r /force_torque_sensor_broadcaster/wrench:=/wrench")

    return LaunchDescription([
        gz_resource_path,
        DeclareLaunchArgument("gripper", default_value="2f85",
                              description="Which gripper/xacro/controllers to use: '2f85' or 'hande'"),
        DeclareLaunchArgument("ur_type", default_value="ur5e",
                              description="UR robot variant (ur3, ur3e, ur5e, ...)"),
        DeclareLaunchArgument("gui", default_value="true",
                              description="Run the Gazebo GUI (false = headless server)"),
        DeclareLaunchArgument("load_gripper", default_value="true",
                              description="Spawn gripper_controller"),
        DeclareLaunchArgument("use_camera", default_value="true",
                              description="Bring up the D435 gz<->ROS topic bridge"),
        DeclareLaunchArgument("rviz", default_value="true",
                              description="Launch rviz2 showing the color/depth camera feeds"),
        DeclareLaunchArgument("camera_offset", default_value="0",
                              description="Camera distance along tool0's front/back (Z) axis: "
                                          "negative = behind tool0 toward the wrist, positive = "
                                          "forward into the gripper's own envelope. Tune live, "
                                          "e.g. camera_offset:=-0.05"),
        DeclareLaunchArgument("camera_offset_x", default_value="0",
                              description="Camera offset along tool0's left/right (X) axis. "
                                          "Tune live, e.g. camera_offset_x:=0.02"),
        DeclareLaunchArgument("camera_offset_y", default_value="-0.0675",
                              description="Camera offset along tool0's up/down (Y) axis. "
                                          "Tune live, e.g. camera_offset_y:=0.02"),
        robot_state_publisher, clock_bridge, active_gripper_pub, ft_filter, camera_bridge, rviz_node,
        gz_sim, spawn_entity_delayed,
        RegisterEventHandler(OnProcessExit(target_action=spawn_entity, on_exit=[jsb])),
        RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[fts, jtc])),
        RegisterEventHandler(OnProcessExit(target_action=jtc, on_exit=[fvc, compliance, gc])),
    ])
