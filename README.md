# Setup

Recreation steps for this workspace.

## Third-party dependencies

Pinned in `dependencies.repos` (same `vcstool` mechanism `ur_python_utilities` itself
uses for its own dependencies; ours pins both repos directly, so its nested
`ur_python_utilities/dependencies.repos` is not needed for this reproduction):

```bash
mkdir -p ur_ws_jazzy/src
cd ur_ws_jazzy
vcs import src < dependencies.repos
```

`ur_simulation_gz` is not cloned as source; it's pulled in as the apt binary package
`ros-jazzy-ur-simulation-gz` via `rosdep install` below.

## First-party packages

Already part of this workspace under `src/`, not cloned from anywhere:

- `ur_camera_gz` — D435 wrist camera description and gz-sim sensor bridge
- `ur_bringup_gz` — composed robot (arm + gripper + camera) and its gz-sim launch
- `worlds` — gz-sim world files (`warehouse.sdf`)

## D435 camera meshes

`ur_camera_gz`'s visual meshes (`meshes/d435i/`) are vendored from Google DeepMind's
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(`realsense_d435i` package, Apache-2.0). Not committed to git (~45MB, largely
un-decimated Blender exports); fetch them once after cloning, before building:

```bash
cd ur_ws_jazzy/src/ur_camera_gz/meshes
./fetch_meshes.sh
```

Provenance, license, and per-part color mapping are documented in
`ur_camera_gz/meshes/README.md`.

## Build

```bash
cd ur_ws_jazzy
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install \
  --packages-skip cartesian_controller_simulation cartesian_controller_tests
source install/setup.bash
```

## Run

```bash
ros2 launch ur_bringup_gz robot_gz.launch.py gripper:=2f85
ros2 launch ur_gripper_gz_moveit_config ur_moveit.launch.py ur_type:=ur5e gripper:=robotiq_2f85
```
