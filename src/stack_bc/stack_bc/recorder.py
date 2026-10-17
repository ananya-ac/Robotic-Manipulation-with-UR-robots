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
  data/obj_pose_perceived (T, 27) float32 oracle_obj_unknown only: the oracle's perceived object
                                          poses (pos + rot6d per cube, NaN until seen), for analysis
  attrs["oracle"]                         "obj_known" | "obj_unknown" (untagged = obj_known)
  attrs["transit"]                        obj_known only: "lin" | "ompl" (untagged = lin)
Marker world (tabletop_marker_mujoco.xml; the tower is built on a stack-location marker):
  data/target_pose    (T, 3)   float32   marker center (spaces.target_vector; constant per episode;
                                          also in qpos as the stack_marker free body)
  meta/episode_target (E, 2)   float64   marker xy
  attrs["marker"]                         "on" (untagged = "off")
Colour domain randomization (randomize.py):
  meta/episode_rgba   (E, 5, 4) float32  RGBA per randomize.COLOR_SLOTS (cube_0..2, table, marker)
  attrs["dr"]                             "colors" (untagged = "none")
Wrist images (collect --wrist-images; camera.py):
  data/wrist_rgb      (T, 240, 320, 3) uint8  wrist camera RGB, newest frame at each tick, JPEG2k
                                          (level 50, one frame per chunk; stack_bc.image_codecs.Jpeg2k,
                                          config-compatible with diffusion_policy's vendored codec:
                                          read there after its register_codecs())
  data/wrist_stamp    (T,)     float64   sim time of that frame (<= the tick's timestamp)
  attrs["images"]                         "wrist_rgb" (untagged = "none")
  attrs["start"]                          "scan": episodes start at task.SCAN_Q, so frame 0 is the
                                          overview of the whole spawn region (untagged = "home")

Dense per-tick (obs_t, action_t) - no pre-chunking; horizons are training-time choices. There is
deliberately no reward array: offline RL later only needs one added alongside these.
"""

import numpy as np
import zarr
from numcodecs import Blosc

from stack_bc import spaces, task
from stack_bc.image_codecs import Jpeg2k, register_image_codecs
from stack_bc.env import has_marker
from stack_bc.gripper import GRIPPER_JOINTS
from stack_bc.kinematics import ARM_JOINTS
from stack_bc.rotations import matrix_to_quat

COMPRESSOR = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
CHUNK_STEPS = 1024
IMAGE_KEYS = ("wrist_rgb",)       # uint8 image arrays: lossy JPEG2k, one frame per chunk
JPEG2K_LEVEL = 50                 # as diffusion_policy's real-world data conversion


def _compressor_and_chunks(key, value):
    if key in IMAGE_KEYS:
        return Jpeg2k(level=JPEG2K_LEVEL), (1,) + value.shape[1:]
    return COMPRESSOR, (CHUNK_STEPS,) + value.shape[1:]


class EpisodeBuffer:
    def __init__(self, model):
        """model: the live MjModel (for qpos addresses)."""
        self.arm_adr = [model.jnt_qposadr[model.joint(n).id] for n in ARM_JOINTS]
        self.grip_adr = [model.jnt_qposadr[model.joint(n).id] for n in GRIPPER_JOINTS]
        self.cube_adr = [model.jnt_qposadr[model.body(n).jntadr[0]] for n in task.CUBE_NAMES]
        self.marker_adr = model.jnt_qposadr[model.body(task.MARKER_NAME).jntadr[0]] if has_marker(model) else None
        self.nq = model.nq
        self.rows = {k: [] for k in ("state", "action", "qpos", "q_arm", "width", "desired_width", "timestamp")}

    def __len__(self):
        return len(self.rows["state"])

    def record(self, snap, T_cmd, width_cmd, perceived=None, image=None):
        """perceived: optional (N_CUBES * 9,) the oracle's perceived object poses (oracle_obj_unknown)
        -> data/obj_pose_perceived. Absent for oracle_obj_known, whose schema is unchanged.
        A snapshot with a marker also logs data/target_pose.
        image: optional (stamp, (240, 320, 3) uint8) from camera.WristCamera.latest() ->
        data/wrist_rgb + data/wrist_stamp."""
        qpos = np.zeros(self.nq)
        qpos[self.arm_adr] = snap.q_arm
        qpos[self.grip_adr] = snap.q_gripper
        for adr, p, R in zip(self.cube_adr, snap.cube_pos, snap.cube_rot):
            x, y, z, w = matrix_to_quat(R)
            qpos[adr:adr + 7] = [*p, w, x, y, z]  # MuJoCo free joint: pos + quat wxyz
        if self.marker_adr is not None:
            # A disc lying flat: reset_world places it with identity orientation, and nothing turns it.
            qpos[self.marker_adr:self.marker_adr + 7] = [*snap.marker_pos, 1.0, 0.0, 0.0, 0.0]
        r = self.rows
        r["state"].append(spaces.state_vector(snap))
        r["action"].append(spaces.action_vector(T_cmd, width_cmd))
        r["qpos"].append(qpos)
        r["q_arm"].append(snap.q_arm.astype(np.float32))
        r["width"].append(np.float32(snap.width))
        r["desired_width"].append(np.float32(width_cmd))
        r["timestamp"].append(snap.stamp)
        if perceived is not None:
            r.setdefault("obj_pose_perceived", []).append(np.asarray(perceived, np.float32))
        if snap.marker_pos is not None:
            r.setdefault("target_pose", []).append(spaces.target_vector(snap))
        if image is not None:
            r.setdefault("wrist_stamp", []).append(image[0])
            r.setdefault("wrist_rgb", []).append(image[1])

    def arrays(self):
        dtypes = {"qpos": np.float64, "timestamp": np.float64, "wrist_stamp": np.float64, "wrist_rgb": np.uint8}
        return {k: np.asarray(v, dtype=dtypes.get(k, np.float32)) for k, v in self.rows.items()}


class ReplayWriter:
    # Dataset tags: attrs key -> the value an untagged dataset that already has data implies.
    TAGS = {"oracle": "obj_known", "transit": "lin", "marker": "off", "dr": "none", "images": "none",
            "start": "home"}

    def __init__(self, path, oracle=None, transit=None, marker=None, dr=None, images=None, start=None):
        """oracle: 'obj_known' | 'obj_unknown' -> root.attrs['oracle']. transit: 'lin' | 'ompl'
        (oracle_obj_known's transit planner). marker: 'on' | 'off'. dr: 'colors' | 'none'.
        images: 'wrist_rgb' | 'none'. start: 'scan' | 'home' (recorded episodes' first pose).
        Appending episodes with a different tag than the dataset's is refused (an untagged
        dataset implies TAGS)."""
        register_image_codecs()           # so image arrays of an existing dataset can be opened
        self.root = zarr.open_group(path, mode="a")
        for key, value in (("oracle", oracle), ("transit", transit), ("marker", marker), ("dr", dr),
                           ("images", images), ("start", start)):
            if value is None:
                continue
            existing = self.root.attrs.get(key, self.TAGS[key] if "data" in self.root else None)
            if existing is not None and existing != value:
                raise ValueError(f"{path} holds {key}={existing} episodes; refusing to add {key}={value} ones")
            self.root.attrs[key] = value
        self.data = self.root.require_group("data")
        self.meta = self.root.require_group("meta")

    @property
    def n_episodes(self):
        return self.meta["episode_ends"].shape[0] if "episode_ends" in self.meta else 0

    @property
    def n_steps(self):
        return int(self.meta["episode_ends"][-1]) if self.n_episodes else 0

    def append_episode(self, arrays, success, seed, order, extra_meta=None):
        """extra_meta: optional {meta key: per-episode value} (e.g. episode_target, episode_rgba),
        appended with the other meta rows, before the episode_ends commit marker."""
        n = len(next(iter(arrays.values())))
        assert all(len(v) == n for v in arrays.values()), "ragged episode arrays"
        for key, value in arrays.items():
            if key not in self.data:
                compressor, chunks = _compressor_and_chunks(key, value)
                self.data.create_dataset(key, shape=(0,) + value.shape[1:], dtype=value.dtype,
                                         chunks=chunks, compressor=compressor)
            self.data[key].append(value)
        end = self.n_steps + n
        self._append_meta("episode_success", np.array([success], dtype=bool))
        self._append_meta("episode_seed", np.array([seed], dtype=np.int64))
        self._append_meta("episode_order", np.asarray([order], dtype=np.int64))
        for key, value in (extra_meta or {}).items():
            self._append_meta(key, np.asarray(value)[None])
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
        for key in self.meta:
            if key != "episode_ends" and self.meta[key].shape[0] > n_ep:
                self.meta[key].resize((n_ep,) + self.meta[key].shape[1:])
        end = self.n_steps
        for key in self.data:
            if self.data[key].shape[0] > end:
                self.data[key].resize((end,) + self.data[key].shape[1:])
