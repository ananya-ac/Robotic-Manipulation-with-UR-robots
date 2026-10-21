"""Wrist depth -> network input. One definition for training (stack_image_dataset) and serving
(scripts/policy_server.py), as image_obs_tensors is for RGB. numpy only.

Stored / sent depth: uint16 millimetres, 0 = invalid (stack_bc.imageproc). Network input: float in
[0, 1], linear in metres between D_NEAR and D_FAR, invalid -> 1 (as far as possible). Seen ranges:
fingers / grasp ~0.03-0.1 m, hover ~0.28 m, overview at SCAN_Q ~0.44 m to the table, the floor
beyond the table edge ~0.87 m; the cubes stand out from the table by their 45 mm height.
The policy's image range normalizer then maps [0, 1] to [-1, 1] like the RGB channels.
"""

import numpy as np

D_NEAR = 0.03      # m
D_FAR = 1.0        # m


def depth_unit(depth_mm):
    """(..., H, W) uint16 mm -> (..., H, W) float32 in [0, 1]."""
    d = np.asarray(depth_mm).astype(np.float32) / 1000.0
    u = (np.clip(d, D_NEAR, D_FAR) - D_NEAR) / (D_FAR - D_NEAR)
    u[np.asarray(depth_mm) == 0] = 1.0
    return u.astype(np.float32)
