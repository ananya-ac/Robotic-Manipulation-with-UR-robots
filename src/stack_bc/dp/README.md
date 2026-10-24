# Diffusion Policy training for stack_bc data

Extensions that let an **unmodified** clone of
[real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) train on the
zarr datasets written by `stack_bc.collect`:

| Path | What |
|---|---|
| `stack_dp/stack_lowdim_dataset.py` | `StackLowdimDataset`: windows via the repo's `SequenceSampler`; range normalizer on positions/width, identity on rot6d dims (paper App. A.1) |
| `stack_dp/null_lowdim_runner.py` | No-op env runner (rollouts happen in the ROS/MuJoCo sim, not during training) |
| `config/train_stack_lowdim.yaml` | CNN lowdim workspace, paper defaults (To=2, Ta=8, Tp=16, ...); checkpoints ranked by `train_action_mse_error` |
| `config/task/stack_lowdim.yaml` | Dataset path (`~/data/stack_bc/overfit.zarr` by default) and dims (obs 37, action 10) |
| `stack_dp/stack_image_dataset.py` | `StackImageDataset`: wrist-camera data (`collect --wrist-images`); obs keys `wrist`, `proprio`, optional `wrist_context`; images kept JPEG 2000-compressed in RAM |
| `stack_dp/null_image_runner.py` | No-op env runner for image policies |
| `config/train_stack_wrist_image.yaml` | CNN image workspace, the paper's real-world setup (ResNet-18 + GroupNorm, crop 216x288, To=2, Ta=8, Tp=16) |
| `config/task/stack_wrist_image.yaml` | Overview-start: 2 wrist frames + proprio (dataset `~/data/stack_bc/wrist.zarr` by default) |
| `config/task/stack_wrist_context_image.yaml` | Overview + context: the same + the episode's frame 0 (the overview) as a constant extra image |
| `stack_dp/depth.py` | Depth (uint16 mm) -> network input [0, 1]; shared by the dataset and `policy_server.py` |
| `stack_dp/vision.py` | `get_resnet_in_chans` (ResNet for 1/4 input channels) and `RGBDCrossAttnEncoder` (variant C) |
| `config/train_stack_wrist_depth.yaml` | The depth ablation's train config: as `train_stack_wrist_image.yaml`, encoder from the task |
| `config/task/stack_wrist_rgbd.yaml`, `stack_wrist_depth_late.yaml`, `stack_wrist_depth_xattn.yaml` | Depth variants A (early fusion), B (late fusion), C (cross-attention) |
| `config/train_stack_station_image.yaml` | DP-paper station (front + wrist cameras): as `train_stack_wrist_image.yaml` with the paper's real Push-T Ta=6 |
| `config/task/stack_station_image.yaml` | Station: front + wrist frames + proprio (dataset `~/data/stack_bc/station.zarr`; obs key `front` = `data/front_rgb`) |

Environment setup (the `robodiff` conda env) is in the workspace's `SETUP.md`.

## Train

Run from the diffusion_policy clone (its `train.py`); this directory goes on `PYTHONPATH` so the
configs' `stack_dp.*` targets import:

```bash
conda activate robodiff
cd ~/diffusion_policy
PYTHONPATH=~/ur_ws_jazzy/src/stack_bc/dp python train.py \
  --config-dir=$HOME/ur_ws_jazzy/src/stack_bc/dp/config --config-name=train_stack_lowdim.yaml \
  task.dataset.zarr_path=$HOME/data/stack_bc/overfit.zarr task.dataset.val_ratio=0.0 \
  training.num_epochs=1000 hydra.run.dir=data/outputs/overfit_stack
```

Outputs (checkpoints, `logs.json.txt`, offline wandb) go to `hydra.run.dir` under the clone.
Re-running with the same `hydra.run.dir` resumes from `latest.ckpt`.

### LSTM-GMM baseline

Same task, dataset and normalizer; only the policy differs (`config/train_lstm_gmm_stack_lowdim.yaml`:
robomimic BC-RNN, 2-layer LSTM 400, 5-mode GMM, 10-step training sequences). It is a recurrent
one-step policy (To = 1, Ta = 1), and checkpoints are ranked by its GMM negative log-likelihood
(`train_loss`), which is not comparable to the diffusion loss - compare policies with
`scripts/offline_eval.py` instead.

```bash
PYTHONPATH=~/ur_ws_jazzy/src/stack_bc/dp python train.py \
  --config-dir=$HOME/ur_ws_jazzy/src/stack_bc/dp/config --config-name=train_lstm_gmm_stack_lowdim.yaml \
  task.dataset.zarr_path=$HOME/data/stack_bc/overfit.zarr task.dataset.val_ratio=0.0 \
  training.num_epochs=2000 hydra.run.dir=data/outputs/overfit_stack_lstm_gmm
```

### Wrist-camera policies

Data from `python3 -m stack_bc.collect ... --transit ompl --randomize-colors --wrist-images` (marker
world): every episode starts at the overview pose `task.SCAN_Q`, and each tick stores the wrist frame
(`data/wrist_rgb`, 240x320) next to the usual arrays. One dataset trains both observation variants:

| Variant | Task config | Policy input per step (To = 2) |
|---|---|---|
| Overview-start | `stack_wrist_image` | wrist frames t-1, t + proprio `state[:, :10]` (EE pos, EE rot6d, gripper width) |
| Overview + context | `stack_wrist_context_image` | the same + the episode's frame 0 (the overview), constant for the episode |

Cube and marker poses are not inputs. The robodiff env needs `imagecodecs==2022.9.26` (SETUP.md).

```bash
PYTHONPATH=~/ur_ws_jazzy/src/stack_bc/dp python train.py \
  --config-dir=$HOME/ur_ws_jazzy/src/stack_bc/dp/config --config-name=train_stack_wrist_image.yaml \
  task=stack_wrist_image task.dataset.zarr_path=$HOME/data/stack_bc/wrist.zarr \
  hydra.run.dir=data/outputs/wrist_overview
# overview + context: task=stack_wrist_context_image, hydra.run.dir=data/outputs/wrist_context
```

DP-paper station (front + wrist cameras, recorded in `world:=dp_station_mujoco.xml` with
`collect --wrist-images --front-images`):

```bash
PYTHONPATH=~/ur_ws_jazzy/src/stack_bc/dp python train.py \
  --config-dir=$HOME/ur_ws_jazzy/src/stack_bc/dp/config --config-name=train_stack_station_image.yaml \
  task.dataset.zarr_path=$HOME/data/stack_bc/station.zarr hydra.run.dir=data/outputs/station
```

Two ResNet-18 trunks, like the context variant: on the 8 GB laptop only as a smoke test with a smaller
U-Net (`"policy.down_dims=[256,512,1024]"`, batch 8); train on the bigger GPU.

Memory: with the paper-sized U-Net (down_dims 512/1024/2048) the context variant does not fit an 8 GB
laptop GPU even at batch 8 (two ResNet-18 encoders + AdamW state); the overview-start variant does
at small batches. Train on a bigger GPU at the default batch 64, or reduce
`dataloader.batch_size` / `policy.down_dims` for quick tests.

### Depth ablation (wrist RGB + depth)

Depth comes from the same wrist render as the RGB (`data/wrist_depth`, uint16 mm, pixel-aligned):
record it live with `collect ... --wrist-images --wrist-depth`, or add it to an existing wrist dataset
from its logged qpos (~2 ms per frame):

```bash
python3 -m stack_bc.backfill_depth --dataset ~/data/stack_bc/wrist.zarr --out ~/data/stack_bc/wrist_depth.zarr
python3 -m stack_bc.backfill_depth --dataset <live --wrist-depth dataset> --check   # rendered vs live depth
```

Three ways of conditioning on depth, trained with one config (`train_stack_wrist_depth.yaml`) so
they differ only in the observation encoder (all with `imagenet_norm: False`):

| Variant | Task config | Encoder | Params |
|---|---|---|---|
| A: early fusion | `stack_wrist_rgbd` | one ResNet-18, 4-channel input (RGB + depth) | 11.2M |
| B: late fusion | `stack_wrist_depth_late` | separate RGB and depth ResNet-18s, pooled features concatenated | 22.3M |
| C: cross-attention | `stack_wrist_depth_xattn` | `stack_dp.vision.RGBDCrossAttnEncoder`: shared crop, 2 x 63 spatial tokens, 16 latents x 2 cross/self-attention blocks | 26.8M |

Depth enters the network through `stack_dp/depth.py` (0.03-1.0 m -> [0, 1], invalid -> 1), the same
function at training and serving time.

```bash
for task in stack_wrist_rgbd stack_wrist_depth_late stack_wrist_depth_xattn; do
  PYTHONPATH=~/ur_ws_jazzy/src/stack_bc/dp python train.py \
    --config-dir=$HOME/ur_ws_jazzy/src/stack_bc/dp/config --config-name=train_stack_wrist_depth.yaml \
    task=$task task.dataset.zarr_path=$HOME/data/stack_bc/wrist_depth.zarr hydra.run.dir=data/outputs/depth_$task
done
# reference without depth, same encoder normalization:
#   --config-name=train_stack_wrist_image.yaml policy.obs_encoder.imagenet_norm=False ...
```

Every task config uses `val_ratio: 0.1`; compare runs by `val_loss` and held-out rollouts rather than
training error (the RGB-only run overfit). B and C have two trunks: train them on the bigger GPU (on
the 8 GB laptop only with a smaller U-Net, e.g. `"policy.down_dims=[256,512,1024]"`, for smoke tests).

## Compare policies offline

`scripts/offline_eval.py` replays every demo episode through each policy exactly as a rollout would
(reset per episode, last To demo observations every Ta ticks) and reports accuracy vs the demo
actions and smoothness, with an optional per-episode plot:

```bash
python ~/ur_ws_jazzy/src/stack_bc/scripts/offline_eval.py --dataset ~/data/stack_bc/overfit.zarr \
  --ckpt DP=<dp run>/checkpoints/latest.ckpt --ckpt LSTM-GMM=<gmm run>/checkpoints/latest.ckpt \
  --plot ~/ur_ws_jazzy/videos/offline_compare.png
```

## Evaluate in the sim

Three terminals: the sim, the policy server (`robodiff` env), and the ROS-side client.

```bash
ros2 launch ur_bringup_sim robot_mujoco.launch.py rviz:=false

conda activate robodiff
python ~/ur_ws_jazzy/src/stack_bc/scripts/policy_server.py \
  --ckpt ~/diffusion_policy/data/outputs/overfit_stack/checkpoints/latest.ckpt   # --inference-steps 16 (DDIM) default

cd ~/ur_ws_jazzy && source install/setup.bash
python3 -m stack_bc.evaluate --dataset ~/data/stack_bc/overfit.zarr --episode 0 1 2 [--record /tmp/rollout.zarr]
```

The server only needs the diffusion_policy clone (`--dp-root`, default `~/diffusion_policy`);
checkpoints don't reference `stack_dp` at load time. It serves LSTM-GMM checkpoints the same way;
the client then queries every tick and resets the policy's recurrent state at each episode start.
Caveat for comparisons: a one-step policy is executed as one single-point trajectory per tick, which
eases toward every setpoint (visibly less smooth than DP's 8-step chunks), so part of any
closed-loop smoothness gap is the execution scheme, not the policy - check `offline_eval.py` too.

Wrist-camera checkpoints work with the same commands; the sim must run the marker world
(`world:=tabletop_marker_mujoco.xml`; rviz can stay on) and `--dataset` should be a wrist dataset,
whose episodes give the layouts, marker and colours. The server recognizes image checkpoints by their
`shape_meta` and reports the expected obs keys in `info`. The client then starts each rollout at
`task.SCAN_Q`, sends the last 2 live wrist frames + proprio (+ the first tick's frame as
`wrist_context` for the context variant), prints the number of stale wrist frames per episode, and
`--record` also stores the frames. Measured with a smoke checkpoint: ~90 ms per query (16 DDIM steps).

Depth checkpoints (A/B/C) also work with the same commands: the server reports the raw inputs it
needs (`info["inputs"]`), the client then subscribes to the depth stream too and sends live RGB +
depth from the same render, and the server builds the checkpoint's keys (e.g. `wrist_rgbd`) exactly as
the dataset does. The server puts `src/stack_bc/dp` on `sys.path` itself (`--stack-dp-root`), which the
cross-attention checkpoints need. Measured with smoke checkpoints: ~75-95 ms per query.
