# Setup

## Third-party dependencies

```bash
mkdir -p ur_ws_jazzy/src
cd ur_ws_jazzy
vcs import src < dependencies.repos
```

## D435 camera meshes

```bash
cd ur_ws_jazzy/src/ur_camera_gz/meshes
./fetch_meshes.sh
```

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

## Acknowledgments

- [cambel/ur3](https://github.com/cambel/ur3) (`ros2-jazzy` branch) — UR control library, gripper gz-sim bringup, MoveIt config
- [omron-sinicx/cartesian_controllers](https://github.com/omron-sinicx/cartesian_controllers) (`ros2-jazzy` branch) — Cartesian compliance controller
- [UniversalRobots/Universal_Robots_ROS2_GZ_Simulation](https://github.com/UniversalRobots/Universal_Robots_ROS2_GZ_Simulation) — source of the `ur_simulation_gz` apt package
- [Google DeepMind MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (`realsense_d435i`) — D435 wrist camera visual meshes, fetched by `fetch_meshes.sh`; license preserved at `src/ur_camera_gz/meshes/d435i/LICENSE` (Apache-2.0)
- [OpenRobotics industrial-warehouse](https://fuel.gazebosim.org/1.0/OpenRobotics/worlds/industrial-warehouse) on Gazebo Fuel — warehouse world scene
