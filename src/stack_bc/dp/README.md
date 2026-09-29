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
checkpoints don't reference `stack_dp` at load time.
