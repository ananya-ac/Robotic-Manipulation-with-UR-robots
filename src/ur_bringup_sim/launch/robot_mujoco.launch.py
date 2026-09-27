# MuJoCo (mujoco_ros2_control) bringup for the full robot: UR + 2F-85 gripper + D435
# wrist camera + tabletop world (table + 3 cubes), plus rviz2 showing the color feed.
#
#   ros2 launch ur_bringup_sim robot_mujoco.launch.py                # default
#   ros2 launch ur_bringup_sim robot_mujoco.launch.py headless:=true
#   ros2 launch ur_bringup_sim robot_mujoco.launch.py rviz:=false
#
# MuJoCo counterpart of robot_gz.launch.py. Structurally different, not just a
# find-and-replace of the Gazebo file, for one real reason: MuJoCo compiles ONE
# simulation from ONE MJCF file tree at startup - there's no live world server to
# spawn a robot into later the way gz-sim has. So this launch file does, at launch
# time, what robot_gz.launch.py's Gazebo backend does for it automatically:
#   1. xacro-generate the robot's URDF (ur5e_2f85_camera_mujoco.urdf.xacro).
#   2. Convert that URDF to MJCF (make_mjcf_from_robot_description.py) - this needs
#      the mujoco/numpy/trimesh/pyyaml/lxml/pycollada/PyKDL-equipped venv this
#      workspace's environment memory describes (~/venvs/ros-jazzy), not the plain
#      system python3, and obj2mjcf on PATH from that same venv.
#   3. Splice the generated robot MJCF into worlds/worlds/tabletop_mujoco.xml (which
#      ships with a ROBOT_MJCF_PATH placeholder for exactly this) via a plain
#      <include>, writing the result alongside the robot MJCF - MJCF <include>
#      resolves relative asset paths (meshdir) against the INCLUDING file's own
#      directory, not the included file's original one, so both must live in the
#      same directory or meshes fail to load.
#   4. Re-run xacro a second time, now pointing mujoco_model_path at that spliced
#      scene file, to get the final robot_description passed to ros2_control_node.
#
# No DetachableJoint/spawn/detach-cubes step (unlike robot_gz.launch.py): the
# tabletop's cubes are compiled into the simulation from the start, and grasping
# depends on the gripper's real contact/friction, not a rigid-attach hack - see
# ur5e_2f85_camera_mujoco.urdf.xacro's own comment on why. No camera ros_gz_bridge
# either: mujoco_ros2_control's CameraPlugin publishes the ROS image/depth/info
# topics directly, no separate bridge process needed.
#
# Episode/cube resets: no custom node needed either - mujoco_ros2_control_node
# auto-creates .../set_free_joint_state and .../reset_world services for exactly
# this (see the migration project memory for the live-verified call pattern).

import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

# This workspace's dedicated venv for mujoco_ros2_control's Python tooling (mujoco,
# numpy, trimesh, pyyaml, lxml, pycollada, PyKDL, obj2mjcf) - see the "ROS/MuJoCo
# venv setup" project memory for why this can't be the plain system python3 or
# whatever venv happens to be active in the launching shell.
MUJOCO_VENV = os.path.expanduser("~/venvs/ros-jazzy")


def launch_setup(context, *args, **kwargs):
    ur_type = LaunchConfiguration("ur_type").perform(context)
    headless = LaunchConfiguration("headless").perform(context)
    camera_offset = LaunchConfiguration("camera_offset").perform(context)
    camera_offset_x = LaunchConfiguration("camera_offset_x").perform(context)
    camera_offset_y = LaunchConfiguration("camera_offset_y").perform(context)
    camera_mass = LaunchConfiguration("camera_mass").perform(context)
    world = LaunchConfiguration("world").perform(context)

    xacro_file = PathJoinSubstitution(
        [FindPackageShare("ur_bringup_sim"), "urdf", "ur5e_2f85_camera_mujoco.urdf.xacro"]
    ).perform(context)
    scene_template = os.path.join(get_package_share_directory("worlds"), "worlds", world)
    convert_script = os.path.join(
        get_package_share_directory("mujoco_ros2_control"), "scripts", "make_mjcf_from_robot_description.py"
    )

    work_dir = tempfile.mkdtemp(prefix="ur_mujoco_")
    print(f"[robot_mujoco.launch.py] Generating MJCF into {work_dir}")

    # Step 1: robot-only URDF, just to feed the converter (no mujoco_model_path yet -
    # we don't know the final scene path until after conversion).
    robot_only_urdf = Command([
        FindExecutable(name="xacro"), " ", xacro_file,
        " name:=", ur_type, " ur_type:=", ur_type,
        " camera_offset:=", camera_offset,
        " camera_offset_x:=", camera_offset_x,
        " camera_offset_y:=", camera_offset_y,
        " camera_mass:=", camera_mass,
    ]).perform(context)
    robot_only_urdf_path = os.path.join(work_dir, "robot_only.urdf")
    with open(robot_only_urdf_path, "w") as f:
        f.write(robot_only_urdf)

    # Step 2: URDF -> MJCF. Needs the venv's python for imports AND its bin/ on PATH
    # for obj2mjcf, which the conversion script shells out to.
    venv_python = os.path.join(MUJOCO_VENV, "bin", "python3")
    env = dict(os.environ)
    env["PATH"] = os.path.join(MUJOCO_VENV, "bin") + os.pathsep + env.get("PATH", "")
    result = subprocess.run(
        [venv_python, convert_script,
         "--urdf", robot_only_urdf_path,
         "--output", work_dir,
         "--save_only", "--convert_stl_to_obj"],
        env=env, capture_output=True, text=True,
    )
    if result.returncode != 0:
        # make_mjcf_from_robot_description.py doesn't always surface the real cause in
        # its own exit code/traceback (see the migration memory's obj2mjcf note) -
        # dump both streams so a failure here is debuggable rather than a bare
        # non-zero exit.
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"MJCF conversion failed (exit {result.returncode}); see output above")
    robot_mjcf_filename = "mujoco_description_formatted.xml"

    # Step 2.5: MuJoCo pads a keyframe's qpos with ZEROS (not each body's own default
    # pose/qpos0) when it's shorter than the compiled model's nq - confirmed empirically
    # after the tabletop's cubes were found sitting at world (0,0,0) instead of their table
    # positions on this scene's "home" keyframe. The robot MJCF's own "home" key only ever
    # lists its own 12 DOF (all it knows about when make_mjcf_from_robot_description.py
    # writes it) - once <include>d into a scene that adds more free-joint bodies (the 3
    # cubes here, 7 qpos values each), that keyframe needs extending or every included body
    # collapses to the origin on reset. Fixed here, not at the xacro level, because the robot
    # description is deliberately scene-agnostic - it has no way to know what (if anything)
    # will be composed around it; the launch file is what actually knows the chosen world's
    # extra bodies, so it's the right place to complete the keyframe for them. Parses the
    # scene template for any <body> with a <freejoint/> (order matches compiled qpos order:
    # bodies declared after the <include> in tabletop_mujoco.xml compile after the robot's
    # own DOF, confirmed via the qpos indices actually observed - 12, 19, 26 for the 3 cubes).
    scene_tree = ET.parse(scene_template)
    extra_qpos_values = []
    for body in scene_tree.getroot().iter("body"):
        if body.find("freejoint") is not None:
            extra_qpos_values.append(body.get("pos", "0 0 0"))
            extra_qpos_values.append(body.get("quat", "1 0 0 0"))
    if extra_qpos_values:
        robot_mjcf_path = os.path.join(work_dir, robot_mjcf_filename)
        with open(robot_mjcf_path) as f:
            robot_mjcf = f.read()
        match = re.search(r'qpos="([^"]*)"', robot_mjcf)
        if match is None:
            raise RuntimeError("Expected a <key ... qpos=\"...\"> in the generated robot MJCF")
        new_qpos = match.group(1).rstrip() + "  " + "  ".join(extra_qpos_values)
        robot_mjcf = robot_mjcf[:match.start(1)] + new_qpos + robot_mjcf[match.end(1):]
        with open(robot_mjcf_path, "w") as f:
            f.write(robot_mjcf)

    # Step 3: splice the robot MJCF into the scene template, in the SAME directory
    # (see module docstring on why meshdir resolution requires this).
    with open(scene_template) as f:
        scene_xml = f.read()
    scene_xml = scene_xml.replace("ROBOT_MJCF_PATH", robot_mjcf_filename)
    scene_path = os.path.join(work_dir, "scene.xml")
    with open(scene_path, "w") as f:
        f.write(scene_xml)

    # Step 4: final robot_description, now pointing mujoco_model at the spliced scene.
    robot_description_content = Command([
        FindExecutable(name="xacro"), " ", xacro_file,
        " name:=", ur_type, " ur_type:=", ur_type,
        " camera_offset:=", camera_offset,
        " camera_offset_x:=", camera_offset_x,
        " camera_offset_y:=", camera_offset_y,
        " camera_mass:=", camera_mass,
        " headless:=", headless,
        " mujoco_model_path:=", scene_path,
    ]).perform(context)
    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}

    controllers_file = PathJoinSubstitution(
        [FindPackageShare("ur_gripper_sim"), "config", "ur_mujoco_2f85_controllers.yaml"]
    )
    mujoco_plugins_file = PathJoinSubstitution(
        [FindPackageShare("ur_bringup_sim"), "config", "mujoco_plugins.yaml"]
    )

    robot_state_publisher = Node(
        package="robot_state_publisher", executable="robot_state_publisher", output="screen",
        parameters=[robot_description, {"use_sim_time": True}])

    ros2_control_node = Node(
        package="mujoco_ros2_control", executable="ros2_control_node", output="screen",
        parameters=[robot_description, controllers_file, mujoco_plugins_file, {"use_sim_time": True}])

    active_gripper_pub = Node(
        package="ur_control", executable="active_gripper_publisher",
        parameters=[{"gripper": "robotiq_2f85"}], output="screen")

    ft_filter = Node(
        package="ur_control_examples", executable="ft_filter",
        arguments=["-t", "wrench"],
        parameters=[{"use_sim_time": True}], output="screen")

    rviz_config = PathJoinSubstitution([FindPackageShare("ur_bringup_sim"), "config", "camera_view.rviz"])
    rviz_node = Node(
        package="rviz2", executable="rviz2", name="rviz2", output="screen",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(LaunchConfiguration("rviz")))

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

    return [
        robot_state_publisher, ros2_control_node,
        active_gripper_pub, ft_filter, rviz_node,
        jsb, jtc, fvc, compliance, gc, fts,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="tabletop_mujoco.xml",
                              description="MJCF scene (in worlds/worlds/) to splice the robot into"),
        DeclareLaunchArgument("ur_type", default_value="ur5e",
                              description="UR robot variant (ur3, ur3e, ur5e, ...)"),
        DeclareLaunchArgument("headless", default_value="false",
                              description="Run MuJoCo without its own viewer window"),
        DeclareLaunchArgument("load_gripper", default_value="true",
                              description="Spawn gripper_controller"),
        DeclareLaunchArgument("rviz", default_value="true",
                              description="Launch rviz2 with the camera view config"),
        DeclareLaunchArgument("camera_offset", default_value="0",
                              description="Wrist camera Z offset from tool0 (see the xacro's own comment)"),
        DeclareLaunchArgument("camera_offset_x", default_value="0",
                              description="Wrist camera X offset from tool0"),
        DeclareLaunchArgument("camera_offset_y", default_value="-0.0675",
                              description="Wrist camera Y offset from tool0"),
        DeclareLaunchArgument("camera_mass", default_value="0.072",
                              description="Diagnostic only: D435 mass in kg (near-zero to "
                                           "test whether camera weight on wrist_3 explains "
                                           "observed wrist behavior)"),
        OpaqueFunction(function=launch_setup),
    ])
