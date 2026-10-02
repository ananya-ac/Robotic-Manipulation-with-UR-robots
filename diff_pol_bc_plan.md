# Behavior-Cloning Phase — Agent Task Spec

**Objective:** collect expert cube-stacking demonstrations in the ROS 2 + MuJoCo sim and train a **Diffusion Policy (behavior cloning)** on them, reproducing the paper's methodology on this task.

This file is a seed for a command-line coding agent. Decisions marked **LOCKED** are settled — implement them, don't re-litigate. Items under **CONFIRM** must be resolved with the user or by inspecting the workspace before coding. Companion docs (context only): `stacking_data_collection_plan.md`, `ros2_mujoco_migration.md`.

---

## Current state (done)

- Stack migrated to **ROS 2 Jazzy + MuJoCo** (`mujoco_ros2_control` SystemInterface + MoveIt); toolchain validated.
- Robot: **UR5e** (6-DOF) + **Robotiq 2F-85** gripper, from MuJoCo Menagerie MJCF.
- Arm = **position**-controlled; gripper = **effort/force**-controlled. Grasping is stable.
- Cube scene **randomization implemented** (per-episode randomized poses; reset via free-joint `qpos`).

## Scope

- **In scope:** recorder → scripted oracle → data collection → Diffusion Policy reproduction + training + sim eval, **state-based** first.
- **Out of scope (this phase):** offline RL, reward shaping, DAgger, real-robot transfer. Do **not** add a reward channel. (A per-episode success flag is required, but it is a label for filtering/eval, not a reward.)
- **Forward-compat:** the BC schema must be a strict subset of what offline RL would need, so RL can later be enabled by adding a `reward` array — nothing else should need re-collection.

---

## LOCKED design decisions

- **Action `Da = 10`, task-space, absolute:** EE position (3) + EE orientation as 6D rotation rep (6, Zhou et al. 2019, left **unnormalized**) + gripper width (1). Arm-independent (do NOT use joint-space).
- **Observation (state-based phase):** proprioception = EE pose (9) + gripper width (1), plus ground-truth cube poses (privileged, sim-only). Observation is conditioning, not diffused.
- **Rates:** log at a fixed **10 Hz in sim time** (`use_sim_time`, tick off `/clock` or MuJoCo sim time — never wall time).
- **Do NOT pre-chunk into sequences.** Store dense per-timestep `(obs_t, action_t)` + episode boundaries. Horizons are training-time hyperparameters.
- **Horizons (paper defaults):** `To=2, Tp=16, Ta=8`.
- **Model:** **CNN-based** Diffusion Policy (not Transformer) as baseline — works with little tuning per the paper.
- **Always log RGB + depth from the first episode** even though BC training starts state-based, so the vision/3D phase needs no re-collection.
- **Gripper action = target width** (measured finger width at next tick under force control), not raw torque.

---

## Data schema (zarr ReplayBuffer — target format for `diffusion_policy`)

```
<dataset>.zarr
├── data/                         # arrays equal length = total steps across all episodes
│   ├── action    (T, 10)          # EE pos(3) + 6D rot(6) + gripper width(1)
│   ├── state     (T, Ds)          # EE pose(9) + gripper width(1) + cube poses  (state-based obs)
│   ├── rgb       (T, K, H, W, 3)   # K camera views; logged now, used in vision phase
│   └── depth     (T, K, H, W)      # depth maps; logged now
└── meta/
    ├── episode_ends (E,)          # cumulative step index at end of each episode
    ├── episode_success (E,)       # bool per episode (filter/eval label)
    └── cam_params                 # per-view intrinsics + extrinsics
```

Alternative: emit robomimic-style HDF5 and reuse robomimic loaders — pick whichever matches the training config chosen in Task 4.

---

## Tasks (ordered)

### Task 1 — Lean recorder → zarr
- **Goal:** record dense `(obs, action)` at 10 Hz sim time into the zarr schema above.
- **Steps:** sample latest cached values on a fixed sim-time timer (not `ApproximateTimeSynchronizer`); obs EE pose via FK/TF; cube poses read directly from MuJoCo `mjData` (cleaner than bridging); action = next-tick desired EE pose + target width; append per episode with `episode_ends`. Also capture RGB + depth + `cam_params`.
- **Recommended:** record raw rollouts to **rosbag2 (mcap)** and convert offline to zarr, so schema changes don't require re-running episodes.
- **Done when:** one episode round-trips to a valid zarr readable by a standalone script; array lengths consistent; `episode_ends` correct.

### Task 2 — Scripted pick-stack oracle
- **Goal:** deterministic expert that stacks N cubes.
- **Steps:** Cartesian waypoints — pre-grasp above cube → descend → force-close → **verify grasp** → lift → transit above stack → descend → release → retreat; per cube. Use UR5e joint order `[shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]`; IK per waypoint. Randomize stack order across episodes (multimodality).
- **Grasp-success gate (required):** after force-close, verify finger width settled on cube / contact on both pads before lifting; on failure, retry or mark episode failed — never log a failed grasp as expert data.
- **Done when:** oracle stacks reliably from randomized starts; `episode_success` label computed (cube stable within tolerance for k ticks).
- **CONFIRM (pending user decision):** scripted-waypoint (recommended: fast/deterministic, fine for uncluttered stacking) vs MoveIt Task Constructor (collision-aware, heavier). Do not implement Task 2 until the user chooses.

### Task 3 — Collect debug dataset
- **Goal:** ~200 successful episodes; validate the loop.
- **Steps:** run oracle + recorder headless; filter to successes; sanity-check zarr (value ranges sane, action-vs-next-obs pose consistency, no NaNs, episode lengths reasonable); confirm cubes spawn non-overlapping and inside UR5e reach.
- **Done when:** ~200-episode zarr passes checks.

### Task 4 — Reproduce the Push-T benchmark (validate training code BEFORE trusting own data)
- **Goal:** de-risk the training/eval pipeline against a known number. This is a sanity check on the DP code, unrelated to the stacking task.
- **Steps:** clone `github.com/real-stanford/diffusion_policy`; install per README; train the **CNN** variant on the provided **Push-T** dataset; reach near the repo's reported success. (Optional second check: robomimic **Lift** low-dim, for a manipulation-flavored confirmation.)
- **Done when:** reproduced Push-T success rate is within a reasonable margin of the published number, **using the repo's own eval protocol** (see Guardrails).

### Task 5 — Dataset adapter for own data
- **Goal:** load the Task 3 zarr into DP's sequence sampler.
- **Steps:** thin dataset subclass reading `data/*` + `meta/episode_ends` into `(obs, action)` sequences with `To/Tp/Ta`; wire a config from the reproduced CNN low-dim workspace.
- **Done when:** a training run starts and overfits a tiny subset (sanity).

### Task 6 — Train DP on own data (state-based, CNN)
- **Steps:** train with paper defaults (`To=2, Tp=16, Ta=8`), normalizer left to DP's `LinearNormalizer` (rotation dims untouched).
- **Done when:** training converges; checkpoints saved.

### Task 7 — Sim rollout eval
- **Steps:** roll out the policy in the MuJoCo stacking env; report success rate over a fixed set of seeds/inits vs the oracle; log failure modes.
- **Done when:** a success-rate number exists with a documented eval protocol.

### Task 8 — Scale
- **Steps:** collect toward ~1000 successful episodes; retrain; re-eval.

---

## Guardrails / known gotchas

- **Match the eval protocol, not just training.** DP's reported numbers depend on number of rollout envs/seeds and checkpoint-selection rule. Mismatched eval → wrong "failed to reproduce" conclusions. Fix the protocol before comparing.
- **CNN, not Transformer.** Transformer needs careful HP tuning; CNN is the low-effort baseline.
- **State-based first.** Cheapest to train; validates data before adding the vision encoder (ResNet-18 + spatial softmax + GroupNorm) later.
- **Sim time, not wall time**, for all logging.
- **No reward channel** this phase (BC only); keep `episode_success` as a label.
- **Gripper coupling is native in MJCF** (tendon/equality) — no mimic-joint workaround needed.
- **Camera placement (D435, for the logged frames):** scene/overhead cam ~0.4–0.8 m from workspace; a wrist cam breaches the D435 min-Z (~16.8 cm @ 848×480) during final approach. Depth is idealized in sim — real transfer will need a structured stereo-noise model (later phase).

## CONFIRM before coding
- **Oracle: scripted waypoints vs MoveIt Task Constructor — user is deciding; do not start Task 2 until resolved.** (Scripted = deterministic/fast, fine for uncluttered stacking; MTC = collision-aware, heavier.)

Resolved: ROS 2 **Jazzy**; gripper **Robotiq 2F-85**; Task 4 benchmark **Push-T**.

## References
- Diffusion Policy: `github.com/real-stanford/diffusion_policy`
- MuJoCo Menagerie (UR5e, Robotiq 2F-85): `github.com/google-deepmind/mujoco_menagerie`
- `mujoco_ros2_control` (MoveIt-org and DFKI variants)
- Reference stack (UR5e + 2F-85 + D435 + ROS 2 Jazzy + MuJoCo): `github.com/MuRain37/so101-ros2-mujoco`