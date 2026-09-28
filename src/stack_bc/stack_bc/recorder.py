"""Per-tick episode buffer + zarr writer in diffusion_policy's ReplayBuffer layout.

<dataset>.zarr
  data/state          (T, 37)  float32   spaces.state_vector  (policy observation, state-based)
  data/action         (T, 10)  float32   spaces.action_vector (desired EE pose + desired width)
  data/qpos           (T, nq)  float64   full MuJoCo qpos (arm, gripper, cube free joints) -> exact
                                          offline re-rendering of any camera for the RGB-D phase
  data/q_arm          (T, 6)   float32   measured arm joints
  data/width          (T,)     float32   measured fingertip gap (also inside state)
  data/desired_width  (T,)     float32   (also inside action; kept separately for the paper's
                                          "actual and desired" observation variant, Sec. 7.1)
  data/timestamp      (T,)     float64   sim time of each tick
  meta/episode_ends   (E,)     int64     cumulative end index (ReplayBuffer convention)
  meta/episode_success (E,)    bool      label for filtering/eval, NOT a reward
  meta/episode_seed   (E,)     int64
  meta/episode_order  (E, 3)   int64     stack order (base first)

Dense per-tick (obs_t, action_t) - no pre-chunking; horizons are training-time choices. There is
deliberately no reward array: offline RL later only needs one added alongside these.
"""

import numpy as np
import zarr
from numcodecs import Blosc

from stack_bc import spaces, task
from stack_bc.gripper import GRIPPER_JOINTS
from stack_bc.kinematics import ARM_JOINTS
from stack_bc.rotations import matrix_to_quat

COMPRESSOR = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
CHUNK_STEPS = 1024


class EpisodeBuffer:
    def __init__(self, model):
        """model: the live MjModel (for qpos addresses)."""
        self.arm_adr = [model.jnt_qposadr[model.joint(n).id] for n in ARM_JOINTS]
        self.grip_adr = [model.jnt_qposadr[model.joint(n).id] for n in GRIPPER_JOINTS]
        self.cube_adr = [model.jnt_qposadr[model.body(n).jntadr[0]] for n in task.CUBE_NAMES]
        self.nq = model.nq
        self.rows = {k: [] for k in ("state", "action", "qpos", "q_arm", "width", "desired_width", "timestamp")}

    def __len__(self):
        return len(self.rows["state"])

    def record(self, snap, T_cmd, width_cmd):
        qpos = np.zeros(self.nq)
        qpos[self.arm_adr] = snap.q_arm
        qpos[self.grip_adr] = snap.q_gripper
        for adr, p, R in zip(self.cube_adr, snap.cube_pos, snap.cube_rot):
            x, y, z, w = matrix_to_quat(R)
            qpos[adr:adr + 7] = [*p, w, x, y, z]  # MuJoCo free joint: pos + quat wxyz
        r = self.rows
        r["state"].append(spaces.state_vector(snap))
        r["action"].append(spaces.action_vector(T_cmd, width_cmd))
        r["qpos"].append(qpos)
        r["q_arm"].append(snap.q_arm.astype(np.float32))
        r["width"].append(np.float32(snap.width))
        r["desired_width"].append(np.float32(width_cmd))
        r["timestamp"].append(snap.stamp)

    def arrays(self):
        dtypes = {"qpos": np.float64, "timestamp": np.float64}
        return {k: np.asarray(v, dtype=dtypes.get(k, np.float32)) for k, v in self.rows.items()}


class ReplayWriter:
    def __init__(self, path):
        self.root = zarr.open_group(path, mode="a")
        self.data = self.root.require_group("data")
        self.meta = self.root.require_group("meta")

    @property
    def n_episodes(self):
        return self.meta["episode_ends"].shape[0] if "episode_ends" in self.meta else 0

    @property
    def n_steps(self):
        return int(self.meta["episode_ends"][-1]) if self.n_episodes else 0

    def append_episode(self, arrays, success, seed, order):
        n = len(next(iter(arrays.values())))
        assert all(len(v) == n for v in arrays.values()), "ragged episode arrays"
        for key, value in arrays.items():
            if key not in self.data:
                self.data.create_dataset(key, shape=(0,) + value.shape[1:], dtype=value.dtype,
                                         chunks=(CHUNK_STEPS,) + value.shape[1:], compressor=COMPRESSOR)
            self.data[key].append(value)
        end = self.n_steps + n
        self._append_meta("episode_success", np.array([success], dtype=bool))
        self._append_meta("episode_seed", np.array([seed], dtype=np.int64))
        self._append_meta("episode_order", np.asarray([order], dtype=np.int64))
        # episode_ends LAST: it is the commit marker. A crash before it leaves only trailing rows
        # past episode_ends[-1], which truncate_partial() drops on the next open.
        self._append_meta("episode_ends", np.array([end], dtype=np.int64))

    def _append_meta(self, key, value):
        if key not in self.meta:
            self.meta.create_dataset(key, shape=(0,) + value.shape[1:], dtype=value.dtype,
                                     chunks=(1024,) + value.shape[1:], compressor=None)
        self.meta[key].append(value)

    def truncate_partial(self):
        """Drop data rows (and meta rows) past the last complete episode, e.g. after a crash."""
        n_ep = self.meta["episode_ends"].shape[0] if "episode_ends" in self.meta else 0
        for key in ("episode_success", "episode_seed", "episode_order"):
            if key in self.meta and self.meta[key].shape[0] > n_ep:
                self.meta[key].resize((n_ep,) + self.meta[key].shape[1:])
        end = self.n_steps
        for key in self.data:
            if self.data[key].shape[0] > end:
                self.data[key].resize((end,) + self.data[key].shape[1:])
