# Machine setup: MuJoCo sim + stack_bc data collection + Diffusion Policy training

Reproduces this workspace on a fresh machine. Tested on Ubuntu 24.04 with ROS 2 Jazzy, an NVIDIA
RTX 4070 (driver 595) for training. Steps marked **(sudo)** change the system; everything else is
user- or workspace-local.

Three environments are involved, deliberately kept apart:

| Environment | Python | Used for |
|---|---|---|
| ROS 2 Jazzy (`/opt/ros/jazzy`) | system 3.12 | sim, controllers, MoveIt, building the workspace |
| `~/venvs/ros-jazzy` venv, layered on a sourced ROS | 3.12 | MuJoCo tooling, `stack_bc` (collect, evaluate, replay), `tabletop_perception` node |
| `robodiff` conda env | 3.10 | Diffusion Policy training + `policy_server.py` |

Workspace packages on this branch (`mujoco-migration`):

| Package | Role |
|---|---|
| `ur_bringup_sim`, `ur_gripper_sim`, `ur_camera_sim`, `worlds` | MuJoCo sim of the UR5e + Robotiq 2F-85 + wrist D435 on a tabletop (`robot_mujoco.launch.py`) |
| `ur_gripper_sim_moveit_config` | MoveIt config (OMPL + Pilz; optional wrist-depth octomap, `sensors:=true`) |
| `mujoco_dr_plugin` | C++ sim plugin: recolours materials at runtime (`/color_randomizer/set_colors`, colour domain randomization) |
| `tabletop_perception`, `tabletop_perception_msgs` | Point-cloud perception for `oracle_obj_unknown` (segmentation, multi-view fusion, antipodal grasps) |
| `stack_bc` | Cube-stacking BC pipeline: oracles, data collection, evaluation, DAgger tooling; `dp/` holds the diffusion_policy adapter and configs |
| `ur_python_utilities`, `cartesian_controllers` | Upstream dependencies (`dependencies.repos`) |

## 1. System packages (sudo)

ROS 2 Jazzy itself (desktop) per the official install guide, then:

```bash
sudo apt install \
  python3-vcstool python3-colcon-common-extensions python3-rosdep python3-pykdl \
  ros-jazzy-mujoco-ros2-control ros-jazzy-mujoco-ros2-control-plugins \
  ros-jazzy-mujoco-ros2-control-msgs ros-jazzy-mujoco-ros2-control-demos \
  ros-jazzy-moveit ros-jazzy-moveit-py ros-jazzy-pilz-industrial-motion-planner \
  ros-jazzy-moveit-ros-perception
```

Versions this was validated with: `mujoco-ros2-control` (+ `-plugins`, `-msgs`) 0.1.1,
`mujoco-vendor` 0.1.0, `moveit`/`moveit-py`/`pilz`/`moveit-ros-perception` 2.12.4.
`moveit-ros-perception` provides the octomap updater used by `ur_moveit.launch.py sensors:=true`
(`collect --scene sensors` and `--oracle obj_unknown`).
The remaining ROS dependencies (UR/Robotiq/RealSense descriptions, controllers, gz packages) are
declared in package manifests and installed by `rosdep` in step 2. The packages above are not
declared anywhere rosdep sees, so they must be installed explicitly.

## 2. Workspace

```bash
git clone -b mujoco-migration https://github.com/ananya-ac/Robotic-Manipulation-with-UR-robots.git ur_ws_jazzy
cd ur_ws_jazzy
vcs import src < dependencies.repos          # upstream cambel/ur3 + cartesian_controllers
(cd src/ur_camera_sim/meshes && ./fetch_meshes.sh)
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y     # (sudo, via rosdep)
colcon build --symlink-install \
  --packages-skip cartesian_controller_simulation cartesian_controller_tests
```

**Build in a terminal where the venv (section 3) is NOT active.** `tabletop_perception_msgs` and
`mujoco_dr_plugin` generate ROS interfaces (rosidl), which needs the system Python's `empy`; with
the venv active the build fails. If your shell auto-activates a venv, build with a clean
environment:

```bash
env -i HOME=$HOME PATH=/usr/bin:/bin bash -c 'source /opt/ros/jazzy/setup.bash && \
  /usr/bin/colcon build --symlink-install --packages-skip cartesian_controller_simulation cartesian_controller_tests'
```

**Re-source after building a new package.** `install/setup.bash` fixes the package list when it is
sourced, so a terminal sourced before a package existed can't find it (e.g. the sim fails with
"class mujoco_dr_plugin/ColorRandomizerPlugin ... does not exist"). Re-run
`source ~/ur_ws_jazzy/install/setup.bash`, or open a new terminal.

No patches to the vendored `cambel/ur3` checkout are needed for `stack_bc` (see
`notes/eaik_urdf_comment_bug.md` for the one known upstream bug and why it no longer matters here).

## 3. ROS Python venv (`~/venvs/ros-jazzy`)

`mujoco_ros2_control`'s URDF->MJCF converter, `stack_bc` and the `tabletop_perception` node need
packages that the system Python doesn't have. A plain venv (no system site-packages) is layered on
top of a sourced ROS:

```bash
python3 -m venv ~/venvs/ros-jazzy
~/venvs/ros-jazzy/bin/python3 -m pip install \
  mujoco==3.13.0 numpy trimesh pyyaml lxml pycollada obj2mjcf \
  EAIK==1.2.2 numpy-quaternion "zarr<3" numcodecs pyzmq scipy lark imageio imageio-ffmpeg \
  open3d==0.20.0 imagecodecs
# PyKDL is apt-only (the PyPI package doesn't build): copy the apt module into the venv.
cp /usr/lib/python3/dist-packages/PyKDL.cpython-312-x86_64-linux-gnu.so \
   ~/venvs/ros-jazzy/lib/python3.12/site-packages/
```

Validated versions: mujoco 3.13.0, EAIK 1.2.2, zarr 2.18.7, numcodecs 0.15.1, pyzmq 27.2.0,
trimesh 5.1.0, lxml 6.1.3, pycollada 0.9.3, numpy-quaternion 2024.0.13, obj2mjcf 0.0.25,
open3d 0.20.0, imagecodecs 2026.10.10.

- `zarr<3` matters: datasets are written in the zarr v2 format that diffusion_policy's
  `ReplayBuffer` reads.
- `open3d`: the `tabletop_perception` node (`perception.launch.py` starts it with `python3` from
  PATH, i.e. the venv's, because colcon writes a `/usr/bin/python3` shebang that has no Open3D).
- `imagecodecs`: JPEG 2000 compression of recorded wrist images (`collect --wrist-images`). Data is
  written through `stack_bc/image_codecs.py`, whose codec config matches diffusion_policy's vendored
  codec, so any imagecodecs version works here.
- Optional, for Jupyter notebooks on this interpreter: `pip install ipykernel matplotlib`.

Every ROS terminal: source ROS first, then the workspace, then activate the venv:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ur_ws_jazzy/install/setup.bash
source ~/venvs/ros-jazzy/bin/activate
```

(`robot_mujoco.launch.py` calls the venv's Python for MJCF conversion itself, by the fixed path
`~/venvs/ros-jazzy`, so keep that location.)

## 4. Diffusion Policy (`robodiff` conda env)

The upstream repo pins Python 3.9 / torch 1.12 / CUDA 11.6, which has no native kernels for
Ada-generation GPUs (RTX 40xx). This env uses modern torch and pins only what the repo's code is
sensitive to:

```bash
git clone https://github.com/real-stanford/diffusion_policy.git ~/diffusion_policy
git -C ~/diffusion_policy checkout 5ba07ac6661db573af695b419a7947ecb704690f   # validated commit

conda create -y -n robodiff python=3.10
conda activate robodiff
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install "numpy==1.26.4" "zarr==2.16.1" "numcodecs==0.12.1" "hydra-core==1.2.0" "einops==0.4.1" \
  tqdm "dill==0.3.8" "diffusers==0.11.1" "huggingface_hub==0.25.2" "wandb==0.16.6" \
  threadpoolctl termcolor psutil click matplotlib scipy numba "pandas<2.3" pyzmq "setuptools<70" \
  "imagecodecs==2022.9.26"
```

- `imagecodecs==2022.9.26` (the repo's own pin) is needed to read wrist-image datasets: call the
  repo's `diffusion_policy.codecs.imagecodecs_numcodecs.register_codecs()` before opening them
  (its image datasets do this already).
- LSTM-GMM baseline (robomimic BC-RNN):
  `pip install --no-deps robomimic==0.2.0 && pip install h5py tensorboardX imageio`.
  `--no-deps` skips `egl_probe`, a native build that often fails and is only needed for
  robomimic's rendering environments, not its policies.
- `diffusers==0.11.1` is the version the repo's schedulers/UNet code was written against, and it
  needs `huggingface_hub<0.26` (newer hubs removed an API it imports).
- `pytorch3d` is intentionally not installed: only the repo's `RotationTransformer` imports it, and
  `stack_bc` data is already in pytorch3d's rot6d convention.
- No `pip install -e` of the repo: its package has no `__init__.py`, so an editable install maps
  nothing. `train.py` is run from the clone's root, and `policy_server.py` adds the clone to
  `sys.path` itself (`--dp-root`).

Training/eval commands: `src/stack_bc/dp/README.md`.

## 5. Sim configuration

`ros2 launch ur_bringup_sim robot_mujoco.launch.py [world:=...] [rviz:=false] [pointcloud:=false] [headless:=true]`

| World (`world:=`) | Contents | Offline scene (replay, states) |
|---|---|---|
| `tabletop_mujoco.xml` (default) | table + 3 cubes | `~/.cache/stack_bc/scene` |
| `tabletop_marker_mujoco.xml` | + a visual-only stack-location marker; `stack_bc` builds the tower on it | `~/.cache/stack_bc/scene_marker` (pass `--scene` to `stack_bc.replay`) |

Sim plugins (`ur_bringup_sim/config/mujoco_plugins.yaml`, loaded in every launch):

- `free_joint_state_publisher`: cube (and marker) poses at 100 Hz.
- `mujoco_camera_plugin`: wrist RGB + depth at **20 Hz**. Without an explicit entry the plugin
  auto-registers at 5 Hz, too slow for the 10 Hz recorder.
- `color_randomizer` (`mujoco_dr_plugin`): idle until `/color_randomizer/set_colors` is called
  (`collect --randomize-colors`).

Camera topics: `/camera/color/image_raw` (640x480 rgb8, ~0.9 MB per frame),
`/camera/color/camera_info`, `/camera/depth/image_rect_raw`. **Subscribe with RELIABLE QoS.** With
Fast DDS (the default RMW) a frame travels as many UDP fragments, and a BEST_EFFORT reader drops
the whole frame when one fragment is lost. Measured at 20 Hz: best-effort received 8 Hz (64% of
frames lost), reliable 20 Hz (0 lost). `stack_bc.camera.WristCamera` does this; best-effort viewers
(e.g. `ros2 topic hz`) show a choppy stream that is not what gets recorded.

GPU: the sim renders on whatever GPU OpenGL picks. On a hybrid laptop that is the integrated GPU
(Mesa), which is enough: the camera holds 20 Hz on Intel graphics with rviz running. To render on
the NVIDIA GPU instead (optional), export `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia`
before launching.

Other launches:

- `ros2 launch ur_gripper_sim_moveit_config ur_moveit.launch.py ur_type:=ur5e gripper:=robotiq_2f85 sensors:=true`:
  move_group with the wrist-depth octomap (`collect --scene sensors`, `--oracle obj_unknown`).
- `ros2 launch tabletop_perception perception.launch.py` (venv active): perception node for
  `--oracle obj_unknown`.

Main `stack_bc.collect` modes (all flags in `python3 -m stack_bc.collect -h`):

| Data | Sim world | Flags |
|---|---|---|
| Original state-based demos | default | (none) |
| OMPL transits (DAgger plan) | either | `--transit ompl` |
| Marker task + colour DR | marker | `--transit ompl --randomize-colors` |
| + wrist images (episodes start at the overview pose `task.SCAN_Q`) | marker | `--transit ompl --randomize-colors --wrist-images` |
| Perception oracle | default + `ur_moveit.launch.py sensors:=true` + `perception.launch.py` | `--oracle obj_unknown` |

Details: `notes/bc_data_pipeline_and_training.md`, `notes/domain_randomization.md`,
`notes/dagger_implementation.md`, `notes/perception_stack_report.md` (local notes, untracked).

## 6. Verify

```bash
# ROS side (terminal with ROS + workspace + venv active)
python3 -m stack_bc.scene                      # offline MJCF build -> ~/.cache/stack_bc/scene/scene.xml
ros2 launch ur_bringup_sim robot_mujoco.launch.py headless:=true rviz:=false
python3 -m stack_bc.collect --out /tmp/check.zarr --episodes 1 --seed 1000   # in another terminal

# Marker world, colour DR and wrist images (sim relaunched with world:=tabletop_marker_mujoco.xml)
ros2 service call /color_randomizer/set_colors mujoco_dr_plugin/srv/SetColors \
  "{materials: [table_wood], rgba: [0.1, 0.6, 0.9, 1.0]}"          # table turns blue
python3 -m stack_bc.collect --out /tmp/check_wrist.zarr --episodes 1 --seed 6000 \
  --transit ompl --randomize-colors --wrist-images                  # expect "[wrist frames: 0 stale ...]"

# Training side
conda activate robodiff && cd ~/diffusion_policy
PYTHONPATH=~/ur_ws_jazzy/src/stack_bc/dp python train.py \
  --config-dir=$HOME/ur_ws_jazzy/src/stack_bc/dp/config --config-name=train_stack_lowdim.yaml \
  task.dataset.zarr_path=/tmp/check.zarr task.dataset.val_ratio=0.0 training.num_epochs=2 \
  hydra.run.dir=/tmp/dp_check
```

## Known issues

- `ft_filter` (from `ur_control_examples`) dies at launch with `No module named 'quaternion'`: its
  entry point is hard-shebanged to `/usr/bin/python3`, which doesn't see the venv. Nothing in the
  MuJoCo/`stack_bc` pipeline uses it.
- Only one sim may run at a time: two instances publish competing `/clock` and `/joint_states`
  (rviz reports "jump back in time").
- `stack_bc.collect` exits via `os._exit`: moveit_py segfaults during normal interpreter teardown.
  All data is committed before that point.
- One sim plugin failing to load stops the ones after it: when `color_randomizer` was missing, the
  sim also skipped `free_joint_state_publisher`, and `collect` failed with "no cube poses". Plugin
  errors are in the sim's log, `~/.ros/log/ros2_control_node_<pid>_*.log`.
- The ROS daemon can keep listing a service from an earlier launch, so `ros2 service call` waits
  forever. `ros2 daemon stop` clears it.
- `moveit_cpp.yaml`'s `plan_request_params` (planning_time, planning_attempts) are not applied:
  `PlanRequestParameters(robot, "plan_request_params")` looks them up one namespace deeper, so
  MoveIt's defaults are used (in collection and in offline planning alike).
- `tabletop_marker_mujoco.xml` is lit more dimly than `tabletop_mujoco.xml` on purpose (peak gain
  ~1.0 instead of ~1.6), so randomized colours don't clip to white. Marker-world data recorded
  before 2026-10-17 used the old, brighter lights; don't mix it with new data.
