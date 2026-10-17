"""Wrist-image dataset for the UR5e cube-stacking task (stack_bc collect --wrist-images).

Zarr layout (ReplayBuffer format, written by stack_bc.recorder):
  data/wrist_rgb (T, 240, 320, 3) uint8 : wrist camera RGB at each 10 Hz tick (JPEG 2000)
  data/state     (T, 37)                : proprio = the first 10 dims: EE pos(3) + EE rot6d(6)
                                          + measured gripper width(1); the cube poses after it
                                          are NOT given to image policies
  data/action    (T, 10)                : desired EE pos(3) + desired EE rot6d(6) + desired width(1)
  meta/episode_ends, meta/episode_success

Observation keys (chosen by the task config's shape_meta; one dataset serves both variants):
  wrist         rgb     the last n_obs_steps wrist frames                       (overview-start)
  proprio       low_dim state[:, :10] for the same steps
  wrist_context rgb     optional: the episode's frame 0, i.e. the overview the
                        recorded episode starts with (collect starts at task.SCAN_Q),
                        repeated for every obs step                            (overview + context)

Images are copied into memory still JPEG 2000-compressed (zarr MemoryStore, no recompression) and
decoded per sample, as diffusion_policy's real-world datasets do: ~19 kB per frame instead of 230 kB.
Normalization: rgb to [0, 1] by the image range normalizer (as the repo's image datasets);
proprio and action range-scaled to [-1, 1] except rot6d dims (identity), as stack_lowdim_dataset.
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

from stack_dp.stack_lowdim_dataset import ACTION_ROT6D_DIMS, _range_except

register_codecs(verbose=False)

PROPRIO_DIM = 10
PROPRIO_ROT6D_DIMS = list(range(3, 9))
IMAGE_KEY = "wrist_rgb"           # zarr array
OBS_RGB_KEYS = ("wrist", "wrist_context")
OBS_LOWDIM_KEYS = ("proprio",)


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
        unknown = set(obs_meta) - set(OBS_RGB_KEYS) - set(OBS_LOWDIM_KEYS)
        if unknown:
            raise ValueError(f"unsupported obs keys {sorted(unknown)}; expected {OBS_RGB_KEYS + OBS_LOWDIM_KEYS}")
        if "wrist" not in obs_meta or "proprio" not in obs_meta:
            raise ValueError("shape_meta.obs needs 'wrist' and 'proprio'")
        self.use_context = "wrist_context" in obs_meta

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, store=zarr.MemoryStore(), keys=[IMAGE_KEY, "state", "action"])
        n_episodes = self.replay_buffer.n_episodes
        success = np.ones(n_episodes, dtype=bool)
        if "episode_success" in self.replay_buffer.meta:
            success = np.asarray(self.replay_buffer.meta["episode_success"][:], dtype=bool)
        val_mask = get_val_mask(n_episodes=n_episodes, val_ratio=val_ratio, seed=seed) & success
        train_mask = downsample_mask(mask=~val_mask & success, max_n=max_train_episodes, seed=seed)

        # Only the first n_obs_steps of images / proprio in a window are used: don't decode the rest.
        key_first_k = {IMAGE_KEY: n_obs_steps, "state": n_obs_steps} if n_obs_steps is not None else {}
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, sequence_length=horizon, pad_before=pad_before,
            pad_after=pad_after, keys=[IMAGE_KEY, "state", "action"], key_first_k=key_first_k,
            episode_mask=train_mask)

        self.episode_ends = np.asarray(self.replay_buffer.episode_ends[:])
        if self.use_context:
            starts = np.concatenate([[0], self.episode_ends[:-1]])
            self.context_frames = np.stack([self.replay_buffer[IMAGE_KEY][int(s)] for s in starts])  # (E, H, W, 3)
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
            pad_after=self.pad_after, keys=[IMAGE_KEY, "state", "action"], key_first_k=self.key_first_k,
            episode_mask=self.val_mask)
        val_set.train_mask = self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer["action"] = _range_except(self.replay_buffer["action"][:], ACTION_ROT6D_DIMS)
        normalizer["proprio"] = _range_except(self.replay_buffer["state"][:, :PROPRIO_DIM], PROPRIO_ROT6D_DIMS)
        normalizer["wrist"] = get_image_range_normalizer()
        if self.use_context:
            normalizer["wrist_context"] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"][:])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        T_slice = slice(self.n_obs_steps)          # all steps when n_obs_steps is None
        wrist = sample[IMAGE_KEY][T_slice]
        obs = {
            "wrist": np.moveaxis(wrist, -1, 1).astype(np.float32) / 255.0,             # (To, 3, H, W)
            "proprio": sample["state"][T_slice, :PROPRIO_DIM].astype(np.float32),       # (To, 10)
        }
        if self.use_context:
            # The window's episode: its first buffer index lies inside it (padding repeats inside too).
            buffer_start = self.sampler.indices[idx][0]
            episode = int(np.searchsorted(self.episode_ends, buffer_start, side="right"))
            frame = np.moveaxis(self.context_frames[episode], -1, 0).astype(np.float32) / 255.0
            obs["wrist_context"] = np.repeat(frame[None], len(wrist), axis=0)           # (To, 3, H, W)
        data = {"obs": obs, "action": sample["action"].astype(np.float32)}
        return dict_apply(data, torch.from_numpy)
