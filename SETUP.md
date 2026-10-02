# Machine setup: MuJoCo sim + stack_bc data collection + Diffusion Policy training

Reproduces this workspace on a fresh machine. Tested on Ubuntu 24.04 with ROS 2 Jazzy, an NVIDIA
RTX 4070 (driver 595) for training. Steps marked **(sudo)** change the system; everything else is
user- or workspace-local.

Three environments are involved, deliberately kept apart:

| Environment | Python | Used for |
|---|---|---|
| ROS 2 Jazzy (`/opt/ros/jazzy`) | system 3.12 | sim, controllers, MoveIt |
| `~/venvs/ros-jazzy` venv, layered on a sourced ROS | 3.12 | MuJoCo tooling, `stack_bc` (collect, evaluate) |
| `robodiff` conda env | 3.10 | Diffusion Policy training + `policy_server.py` |

## 1. System packages (sudo)

ROS 2 Jazzy itself (desktop) per the official install guide, then:

```bash
sudo apt install \
  python3-vcstool python3-colcon-common-extensions python3-rosdep python3-pykdl \
  ros-jazzy-mujoco-ros2-control ros-jazzy-mujoco-ros2-control-plugins \
  ros-jazzy-mujoco-ros2-control-msgs ros-jazzy-mujoco-ros2-control-demos \
  ros-jazzy-moveit ros-jazzy-moveit-py ros-jazzy-pilz-industrial-motion-planner
```

Versions this was validated with: `mujoco-ros2-control` 0.1.1, `moveit`/`moveit-py`/`pilz` 2.12.4.
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

No patches to the vendored `cambel/ur3` checkout are needed for `stack_bc` (see
`notes/eaik_urdf_comment_bug.md` for the one known upstream bug and why it no longer matters here).

## 3. ROS Python venv (`~/venvs/ros-jazzy`)

`mujoco_ros2_control`'s URDF->MJCF converter and `stack_bc` need packages that the system Python
doesn't have. A plain venv (no system site-packages) is layered on top of a sourced ROS:

```bash
python3 -m venv ~/venvs/ros-jazzy
~/venvs/ros-jazzy/bin/python3 -m pip install \
  mujoco==3.13.0 numpy trimesh pyyaml lxml pycollada obj2mjcf \
  EAIK==1.2.2 numpy-quaternion "zarr<3" numcodecs pyzmq scipy lark imageio imageio-ffmpeg
# PyKDL is apt-only (the PyPI package doesn't build): copy the apt module into the venv.
cp /usr/lib/python3/dist-packages/PyKDL.cpython-312-x86_64-linux-gnu.so \
   ~/venvs/ros-jazzy/lib/python3.12/site-packages/
```

Validated versions: mujoco 3.13.0, EAIK 1.2.2, zarr 2.18.7, numcodecs 0.15.1, pyzmq 27.2.0,
trimesh 5.1.0, lxml 6.1.3, pycollada 0.9.3, numpy-quaternion 2024.0.13, obj2mjcf 0.0.25.
`zarr<3` matters: datasets are written in the zarr v2 format that diffusion_policy's
`ReplayBuffer` reads.

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
  threadpoolctl termcolor psutil click matplotlib scipy numba "pandas<2.3" pyzmq "setuptools<70"
```

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

## 5. Verify

```bash
# ROS side (terminal with ROS + workspace + venv active)
python3 -m stack_bc.scene                      # offline MJCF build -> ~/.cache/stack_bc/scene/scene.xml
ros2 launch ur_bringup_sim robot_mujoco.launch.py headless:=true rviz:=false
python3 -m stack_bc.collect --out /tmp/check.zarr --episodes 1 --seed 1000   # in another terminal

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
