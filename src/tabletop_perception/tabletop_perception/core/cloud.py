"""RGB-D -> point cloud back-projection (pinhole, optical frame: +z forward, +x right, +y down)."""

import numpy as np


def pixel_rays(width, height, fx, fy, cx, cy, stride=1):
    """(h', w', 2) normalized (x/z, y/z) for every stride-th pixel."""
    u = np.arange(0, width, stride)
    v = np.arange(0, height, stride)
    uu, vv = np.meshgrid(u, v)
    return np.stack([(uu - cx) / fx, (vv - cy) / fy], axis=-1).astype(np.float32)


def backproject(depth, rays, stride=1, min_range=0.05, max_range=1.2, rgb=None):
    """depth (H, W) metres -> (N, 3) points in the camera optical frame [, (N, 3) colors in 0..1].

    rays: from pixel_rays(..., stride). rgb: optional (H, W, 3) uint8 image aligned with depth.
    """
    z = depth[::stride, ::stride]
    valid = np.isfinite(z) & (z > min_range) & (z < max_range)
    zv = z[valid]
    r = rays[valid]
    pts = np.empty((zv.size, 3), dtype=np.float32)
    pts[:, 0] = r[:, 0] * zv
    pts[:, 1] = r[:, 1] * zv
    pts[:, 2] = zv
    if rgb is None:
        return pts
    cols = rgb[::stride, ::stride][valid].astype(np.float32) / 255.0
    return pts, cols
