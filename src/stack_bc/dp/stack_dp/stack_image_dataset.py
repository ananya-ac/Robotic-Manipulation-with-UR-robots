"""Wrist-image dataset for the UR5e cube-stacking task (stack_bc collect --wrist-images).

Zarr layout (ReplayBuffer format, written by stack_bc.recorder):
  data/wrist_rgb   (T, 240, 320, 3) uint8 : wrist camera RGB at each 10 Hz tick (JPEG 2000)
  data/front_rgb   (T, 240, 320, 3) uint8 : DP-paper station only (collect --front-images): the fixed
                                            front camera, the paper's second policy view (JPEG 2000)
  data/wrist_depth (T, 240, 320) uint16   : wrist depth in mm from the same render (0 = invalid);
                                            live (collect --wrist-depth) or backfill_depth.py
  data/state       (T, 37)                : proprio = the first 10 dims: EE pos(3) + EE rot6d(6)
                                            + measured gripper width(1); the cube poses after it
                                            are NOT given to image policies
  data/action      (T, 10)                : desired EE pos(3) + desired EE rot6d(6) + desired width(1)
  meta/episode_ends, meta/episode_success

Observation keys (chosen by the task config's shape_meta; one dataset serves every variant):
  wrist         rgb (3, H, W) the last n_obs_steps wrist frames
  proprio       low_dim (10)  state[:, :10] for the same steps
  wrist_context rgb (3, H, W) optional: the episode's frame 0 (the overview at task.SCAN_Q),
                              repeated for every obs step                  (overview + context)
  wrist_depth   rgb (1, H, W) depth, stack_dp.depth.depth_unit -> [0, 1]   (depth variants B, C)
  wrist_rgbd    rgb (4, H, W) RGB + depth stacked as one image              (depth variant A)
  front         rgb (3, H, W) the last n_obs_steps front-camera frames       (DP-paper station)
"rgb" is diffusion_policy's type for any image key (it routes the key to an image encoder).

Images are copied into memory still compressed (zarr MemoryStore, no recompression) and decoded per
sample, as diffusion_policy's real-world datasets do. Normalization: image keys (colour and depth
alike, already in [0, 1]) by the image range normalizer; proprio and action range-scaled to [-1, 1]
except rot6d dims (identity), as stack_lowdim_dataset.
"""

import copy
from typing import Dict

import numpy as np
import torch
import zarr

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer

from stack_dp.depth import depth_unit
from stack_dp.stack_lowdim_dataset import ACTION_ROT6D_DIMS, _range_except

register_codecs(verbose=False)

PROPRIO_DIM = 10
PROPRIO_ROT6D_DIMS = list(range(3, 9))
RGB_ARRAY = "wrist_rgb"           # zarr arrays
DEPTH_ARRAY = "wrist_depth"
# Other camera views: obs key -> zarr array, each read and decoded like wrist_rgb.
VIEW_ARRAYS = {"front": "front_rgb"}
OBS_IMAGE_KEYS = ("wrist", "wrist_context", "wrist_depth", "wrist_rgbd") + tuple(VIEW_ARRAYS)
OBS_LOWDIM_KEYS = ("proprio",)
NEEDS_RGB = {"wrist", "wrist_context", "wrist_rgbd"}
NEEDS_DEPTH = {"wrist_depth", "wrist_rgbd"}


def rgb_chw(frames):
    """(..., H, W, 3) uint8 -> (..., 3, H, W) float32 in [0, 1]."""
    return np.moveaxis(frames, -1, -3).astype(np.float32) / 255.0


class StackImageDataset(BaseImageDataset):
    def __init__(self,
                 shape_meta: dict,
                 zarr_path,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 n_obs_steps=None,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None):
        super().__init__()
        obs_meta = shape_meta["obs"]
        unknown = set(obs_meta) - set(OBS_IMAGE_KEYS) - set(OBS_LOWDIM_KEYS)
        if unknown:
            raise ValueError(f"unsupported obs keys {sorted(unknown)}; expected {OBS_IMAGE_KEYS + OBS_LOWDIM_KEYS}")
        if "proprio" not in obs_meta or not ({"wrist", "wrist_rgbd"} | set(VIEW_ARRAYS)) & set(obs_meta):
            raise ValueError(f"shape_meta.obs needs 'proprio' and one of wrist, wrist_rgbd, {', '.join(VIEW_ARRAYS)}")
        self.obs_keys = [k for k in OBS_IMAGE_KEYS if k in obs_meta]
        arrays = ["state", "action"]
        if NEEDS_RGB & set(obs_meta):
            arrays.append(RGB_ARRAY)
        if NEEDS_DEPTH & set(obs_meta):
            arrays.append(DEPTH_ARRAY)
            if DEPTH_ARRAY not in zarr.open(zarr_path, "r")["data"]:
                raise ValueError(f"{zarr_path} has no data/{DEPTH_ARRAY}: record with collect --wrist-depth "
                                 "or add it with python3 -m stack_bc.backfill_depth")
        root = zarr.open(zarr_path, "r")
        for key, array in VIEW_ARRAYS.items():
            if key in obs_meta:
                if array not in root["data"]:
                    raise ValueError(f"{zarr_path} has no data/{array}: record in the station world with "
                                     "collect --wrist-images --front-images")
                arrays.append(array)
        self.arrays = arrays

        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, store=zarr.MemoryStore(), keys=arrays)
        n_episodes = self.replay_buffer.n_episodes
        success = np.ones(n_episodes, dtype=bool)
        if "episode_success" in self.replay_buffer.meta:
            success = np.asarray(self.replay_buffer.meta["episode_success"][:], dtype=bool)
        val_mask = get_val_mask(n_episodes=n_episodes, val_ratio=val_ratio, seed=seed) & success
        train_mask = downsample_mask(mask=~val_mask & success, max_n=max_train_episodes, seed=seed)

        # Only the first n_obs_steps of images / proprio in a window are used: don't decode the rest.
        key_first_k = {k: n_obs_steps for k in arrays if k != "action"} if n_obs_steps is not None else {}
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, sequence_length=horizon, pad_before=pad_before,
            pad_after=pad_after, keys=arrays, key_first_k=key_first_k, episode_mask=train_mask)

        self.episode_ends = np.asarray(self.replay_buffer.episode_ends[:])
        if "wrist_context" in obs_meta:
            starts = np.concatenate([[0], self.episode_ends[:-1]])
            self.context_frames = np.stack([self.replay_buffer[RGB_ARRAY][int(s)] for s in starts])  # (E, H, W, 3)
        self.key_first_k = key_first_k
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, sequence_length=self.horizon, pad_before=self.pad_before,
            pad_after=self.pad_after, keys=self.arrays, key_first_k=self.key_first_k,
            episode_mask=self.val_mask)
        val_set.train_mask = self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer["action"] = _range_except(self.replay_buffer["action"][:], ACTION_ROT6D_DIMS)
        normalizer["proprio"] = _range_except(self.replay_buffer["state"][:, :PROPRIO_DIM], PROPRIO_ROT6D_DIMS)
        for key in self.obs_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"][:])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        T_slice = slice(self.n_obs_steps)          # all steps when n_obs_steps is None
        obs = {"proprio": sample["state"][T_slice, :PROPRIO_DIM].astype(np.float32)}       # (To, 10)
        rgb = rgb_chw(sample[RGB_ARRAY][T_slice]) if RGB_ARRAY in sample else None          # (To, 3, H, W)
        depth = depth_unit(sample[DEPTH_ARRAY][T_slice])[:, None] if DEPTH_ARRAY in sample else None  # (To, 1, H, W)
        if "wrist" in self.obs_keys:
            obs["wrist"] = rgb
        if "wrist_depth" in self.obs_keys:
            obs["wrist_depth"] = depth
        if "wrist_rgbd" in self.obs_keys:
            obs["wrist_rgbd"] = np.concatenate([rgb, depth], axis=1)                         # (To, 4, H, W)
        for key, array in VIEW_ARRAYS.items():
            if key in self.obs_keys:
                obs[key] = rgb_chw(sample[array][T_slice])                                    # (To, 3, H, W)
        if "wrist_context" in self.obs_keys:
            # The window's episode: its first buffer index lies inside it (padding repeats inside too).
            buffer_start = self.sampler.indices[idx][0]
            episode = int(np.searchsorted(self.episode_ends, buffer_start, side="right"))
            frame = rgb_chw(self.context_frames[episode])
            obs["wrist_context"] = np.repeat(frame[None], len(obs["proprio"]), axis=0)     # (To, 3, H, W)
        data = {"obs": obs, "action": sample["action"].astype(np.float32)}
        return dict_apply(data, torch.from_numpy)
