"""Offline build of the compiled robot+tabletop MJCF scene, without launching ROS.

Mirrors robot_mujoco.launch.py's launch_setup() steps 1-3 (xacro -> URDF, URDF -> MJCF via
make_mjcf_from_robot_description.py, keyframe qpos padding for the scene's free-joint bodies,
splice into the tabletop template) so offline tools - the IK feasibility map, oracle dry runs,
and later the qpos-replay renderer for the RGB-D phase - load exactly the model the live sim
compiles. Keep the two in sync if either changes.

  build_scene --out ~/.cache/stack_bc/scene      # writes scene.xml + robot MJCF + assets/
"""

import argparse
import os
import re
import subprocess
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory

MUJOCO_VENV = os.path.expanduser("~/venvs/ros-jazzy")
DEFAULT_OUT = os.path.expanduser("~/.cache/stack_bc/scene")
ROBOT_MJCF_FILENAME = "mujoco_description_formatted.xml"

# Launch-file defaults for robot_mujoco.launch.py's xacro args.
XACRO_ARGS = {
    "name": "ur5e", "ur_type": "ur5e",
    "camera_offset": "0", "camera_offset_x": "0", "camera_offset_y": "-0.0675",
    "camera_mass": "0.072",
}


def build_scene(out_dir=DEFAULT_OUT, world="tabletop_mujoco.xml"):
    """Generate the spliced scene into out_dir and return the scene.xml path."""
    os.makedirs(out_dir, exist_ok=True)
    xacro_file = os.path.join(get_package_share_directory("ur_bringup_sim"), "urdf",
                              "ur5e_2f85_camera_mujoco.urdf.xacro")
    scene_template = os.path.join(get_package_share_directory("worlds"), "worlds", world)
    convert_script = os.path.join(get_package_share_directory("mujoco_ros2_control"), "scripts",
                                  "make_mjcf_from_robot_description.py")

    urdf = subprocess.run(["xacro", xacro_file] + [f"{k}:={v}" for k, v in XACRO_ARGS.items()],
                          check=True, capture_output=True, text=True).stdout
    urdf_path = os.path.join(out_dir, "robot_only.urdf")
    with open(urdf_path, "w") as f:
        f.write(urdf)

    env = dict(os.environ)
    env["PATH"] = os.path.join(MUJOCO_VENV, "bin") + os.pathsep + env.get("PATH", "")
    result = subprocess.run(
        [os.path.join(MUJOCO_VENV, "bin", "python3"), convert_script, "--urdf", urdf_path,
         "--output", out_dir, "--save_only", "--convert_stl_to_obj"],
        env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"MJCF conversion failed (exit {result.returncode})")

    # Pad the robot's "home" keyframe with the scene's free-joint bodies (see the launch
    # file's Step 2.5 for why MuJoCo's zero-padding would otherwise drop them at the origin).
    extra = []
    for body in ET.parse(scene_template).getroot().iter("body"):
        if body.find("freejoint") is not None:
            extra += [body.get("pos", "0 0 0"), body.get("quat", "1 0 0 0")]
    robot_mjcf_path = os.path.join(out_dir, ROBOT_MJCF_FILENAME)
    with open(robot_mjcf_path) as f:
        robot_mjcf = f.read()
    if extra:
        match = re.search(r'qpos="([^"]*)"', robot_mjcf)
        if match is None:
            raise RuntimeError('Expected a <key ... qpos="..."> in the generated robot MJCF')
        new_qpos = match.group(1).rstrip() + "  " + "  ".join(extra)
        robot_mjcf = robot_mjcf[:match.start(1)] + new_qpos + robot_mjcf[match.end(1):]
        with open(robot_mjcf_path, "w") as f:
            f.write(robot_mjcf)

    with open(scene_template) as f:
        scene_xml = f.read().replace("ROBOT_MJCF_PATH", ROBOT_MJCF_FILENAME)
    scene_path = os.path.join(out_dir, "scene.xml")
    with open(scene_path, "w") as f:
        f.write(scene_xml)
    return scene_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--world", default="tabletop_mujoco.xml")
    args = parser.parse_args()
    print(build_scene(args.out, args.world))


if __name__ == "__main__":
    main()
