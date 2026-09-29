"""Low-dim dataset for the UR5e cube-stacking task collected by ur_ws_jazzy's stack_bc package.

Zarr layout (ReplayBuffer format, written by stack_bc.recorder):
  data/state  (T, 37): EE pos(3) + EE rot6d(6) + measured gripper width(1)
                       + 3 cubes x [pos(3) + canonical rot6d(6)]          -> policy 'obs'
  data/action (T, 10): desired EE pos(3) + desired EE rot6d(6) + desired gripper width(1)
  meta/episode_ends, meta/episode_success (+ seed/order, unused here)

rot6d = first two ROWS of the rotation matrix (pytorch3d convention). Normalization follows the
paper (App. A.1): position/width dims are range-normalized to [-1, 1]; rotation dims are left
unchanged - for the action as the paper specifies, and for the observation too, because the cube
rot6d entries are near-constant (upright cubes) and range-scaling them would blow sim noise up to
the full [-1, 1] range.
"""

import copy
from typing import Dict

import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

N_CUBES = 3
STATE_ROT6D_DIMS = [d for s in [range(3, 9)] + [range(10 + 9 * i + 3, 10 + 9 * i + 9) for i in range(N_CUBES)]
                    for d in s]
ACTION_ROT6D_DIMS = list(range(3, 9))


def _range_except(data, identity_dims, output_min=-1.0, output_max=1.0, range_eps=1e-4):
    """Per-dim [-1, 1] range normalizer, identity on identity_dims. Dims with a range below
    range_eps are only re-centered (paper App. A.1: don't scale near-constant dims)."""
    data = data.reshape(-1, data.shape[-1]).astype(np.float32)
    dmin, dmax = data.min(axis=0), data.max(axis=0)
    rng = dmax - dmin
    scale = (output_max - output_min) / np.where(rng < range_eps, output_max - output_min, rng)
    offset = output_min - scale * dmin
    flat = rng < range_eps
    offset[flat] = (output_max + output_min) / 2 - dmin[flat]
    scale[identity_dims], offset[identity_dims] = 1.0, 0.0
    stats = {"min": dmin, "max": dmax, "mean": data.mean(axis=0), "std": data.std(axis=0)}
    return SingleFieldLinearNormalizer.create_manual(scale=scale, offset=offset, input_stats_dict=stats)


class StackLowdimDataset(BaseLowdimDataset):
    def __init__(self,
                 zarr_path,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 obs_key='state',
                 action_key='action',
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=[obs_key, action_key])
        n_episodes = self.replay_buffer.n_episodes

        # Only successful demonstrations (stack_bc writes only successes, but stay safe).
        success = np.ones(n_episodes, dtype=bool)
        if 'episode_success' in self.replay_buffer.meta:
            success = np.asarray(self.replay_buffer.meta['episode_success'][:], dtype=bool)

        val_mask = get_val_mask(n_episodes=n_episodes, val_ratio=val_ratio, seed=seed) & success
        train_mask = downsample_mask(mask=~val_mask & success, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)
        self.obs_key = obs_key
        self.action_key = action_key
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask)
        val_set.train_mask = self.val_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        normalizer = LinearNormalizer()
        normalizer['obs'] = _range_except(self.replay_buffer[self.obs_key], STATE_ROT6D_DIMS)
        normalizer['action'] = _range_except(self.replay_buffer[self.action_key], ACTION_ROT6D_DIMS)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = {
            'obs': sample[self.obs_key].astype(np.float32),        # (horizon, 37)
            'action': sample[self.action_key].astype(np.float32),  # (horizon, 10)
        }
        return dict_apply(data, torch.from_numpy)
