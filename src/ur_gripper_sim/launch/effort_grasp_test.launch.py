# Phase 0 isolated test (data_plan.md): UR5e + 2F-85 in worlds/tabletop.sdf (table + its own
# cube_0/cube_1/cube_2), gripper_effort_controller only (no gripper_controller, no MoveIt, no
# cameras). Uses dedicated ur_gripper_2f85_gz_effort_test.urdf.xacro /
# ur_gz_2f85_controllers_effort_test.yaml files, not the ones ur_bringup_sim/robot_gz.launch.py
# loads, so this test can't affect the full bringup stack.
#
#   ros2 launch ur_gripper_sim effort_grasp_test.launch.py
#
# Test sequence once running (grasp one of tabletop.sdf's cube_0/cube_1/cube_2):
#   ros2 topic pub -1 /gripper_effort_controller/commands std_msgs/msg/Float64MultiArray "{data: [8.0]}"
#   ros2 action send_goal /scaled_joint_trajectory_controller/follow_joint_trajectory ...
#   ros2 topic pub -1 /gripper_effort_controller/commands std_msgs/msg/Float64MultiArray "{data: [-2.0]}"

import os

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
    share_roots = [
        os.path.dirname(get_package_share_directory("ur_description")),
        os.path.join(get_package_share_directory("cubes_sim"), "models"),
    ]
    gz_resource_path = SetEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH",
        os.pathsep.join(share_roots + [os.environ.get("GZ_SIM_RESOURCE_PATH", "")]))

    ur_type = LaunchConfiguration("ur_type")
    gui = LaunchConfiguration("gui")

    pkg = FindPackageShare("ur_gripper_sim")
    controllers_file = PathJoinSubstitution([pkg, "config", "ur_gz_2f85_controllers_effort_test.yaml"])
    xacro_file = PathJoinSubstitution([pkg, "urdf", "ur_gripper_2f85_gz_effort_test.urdf.xacro"])

    robot_description_content = Command([
        FindExecutable(name="xacro"), " ", xacro_file,
        " name:=", ur_type, " ur_type:=", ur_type,
        " simulation_controllers:=", controllers_file,
    ])
    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}

    robot_state_publisher = Node(
        package="robot_state_publisher", executable="robot_state_publisher", output="screen",
        parameters=[robot_description, {"use_sim_time": True}])
    clock_bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"], output="screen")

    # Resolve a real world file (as ur_bringup_sim/robot_gz.launch.py does) rather than a bare
    # "empty.sdf" name -- bare names resolve via GZ_SIM_RESOURCE_PATH search, which is fragile
    # on a machine with other gz projects' paths on that variable.
    world_file = PathJoinSubstitution([FindPackageShare("worlds"), "worlds", "tabletop.sdf"])

    # bullet-featherstone is REQUIRED for the gripper mimic joints.
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
    gec = spawner("gripper_effort_controller")

    return LaunchDescription([
        gz_resource_path,
        DeclareLaunchArgument("ur_type", default_value="ur5e"),
        DeclareLaunchArgument("gui", default_value="true",
                              description="Run the Gazebo GUI (false = headless server)"),
        robot_state_publisher, clock_bridge, gz_sim, spawn_entity_delayed,
        RegisterEventHandler(OnProcessExit(target_action=spawn_entity, on_exit=[jsb])),
        RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[jtc, gec])),
    ])
