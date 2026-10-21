"""Wrist-camera frame processing shared by live capture (camera.py) and offline tools
(backfill_depth.py, wrist_coverage.py). ROS-free: numpy only.

The sim renders the wrist camera at SOURCE_SHAPE; stored frames are half that in each direction:
  RGB   (240, 320, 3) uint8   2x2 block mean
  depth (240, 320)    uint16  millimetres, 0 = invalid, 2x2 minimum over valid pixels
Both use the same 2x2 grid, so stored depth stays pixel-aligned with the stored RGB.
"""

import numpy as np

SOURCE_SHAPE = (480, 640)        # what the sim publishes; downsampling is an exact factor of 2
IMAGE_SHAPE = (240, 320, 3)      # stored H, W, C (uint8 RGB)
DEPTH_SHAPE = (240, 320)         # stored H, W (uint16 millimetres, 0 = invalid)
DEPTH_MAX_MM = 10000             # beyond this (or NaN / inf / <= 0) a pixel counts as invalid


def downsample2(rgb):
    """(2H, 2W, 3) uint8 -> (H, W, 3) uint8 by averaging 2x2 blocks."""
    h, w = rgb.shape[0] // 2, rgb.shape[1] // 2
    a = rgb[:2 * h, :2 * w]
    acc = a[0::2, 0::2].astype(np.uint16)   # strided adds: ~10x faster than reshape + sum
    acc += a[1::2, 0::2]
    acc += a[0::2, 1::2]
    acc += a[1::2, 1::2]
    acc += 2                                 # round to nearest
    acc >>= 2
    return acc.astype(np.uint8)


def depth_m_to_mm(depth_m):
    """float metres -> uint16 millimetres; NaN / inf / <= 0 / beyond DEPTH_MAX_MM -> 0 (invalid)."""
    d = np.asarray(depth_m, dtype=np.float32)
    mm = np.rint(d * 1000.0)
    bad = ~np.isfinite(mm) | (mm <= 0) | (mm > DEPTH_MAX_MM)
    mm[bad] = 0
    return mm.astype(np.uint16)


def downsample2_min(depth_mm):
    """(2H, 2W) uint16 mm -> (H, W) by the minimum over each 2x2 block's VALID pixels (the nearest
    surface; 0 if none is valid). Not the mean: averaging across a cube edge would invent a surface
    between cube and table. Not nearest-pixel: that aliases thin edges."""
    h, w = depth_mm.shape[0] // 2, depth_mm.shape[1] // 2
    a = depth_mm[:2 * h, :2 * w].astype(np.int32)
    a[a == 0] = np.iinfo(np.int32).max                     # invalid never wins the minimum
    m = np.minimum(np.minimum(a[0::2, 0::2], a[1::2, 0::2]), np.minimum(a[0::2, 1::2], a[1::2, 1::2]))
    m[m == np.iinfo(np.int32).max] = 0
    return m.astype(np.uint16)


def depth_frame(depth_m):
    """Full-resolution float metres -> the stored (240, 320) uint16 mm depth frame."""
    return downsample2_min(depth_m_to_mm(depth_m))
