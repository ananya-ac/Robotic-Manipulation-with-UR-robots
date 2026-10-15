"""Small geometry/color helpers shared by segmentation and grasping (numpy/scipy only)."""

import numpy as np
from scipy.spatial import ConvexHull, QhullError


def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rgb_to_hsv(rgb):
    """(N, 3) rgb in [0, 1] -> (N, 3) hsv, hue in degrees [0, 360)."""
    rgb = np.asarray(rgb, dtype=float)
    mx, mn = rgb.max(axis=1), rgb.min(axis=1)
    delta = mx - mn
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    h = np.zeros(len(rgb))
    nz = delta > 1e-9
    rm, gm, bm = nz & (mx == r), nz & (mx == g) & (mx != r), nz & (mx == b) & (mx != r) & (mx != g)
    h[rm] = (60.0 * ((g[rm] - b[rm]) / delta[rm])) % 360.0
    h[gm] = 60.0 * ((b[gm] - r[gm]) / delta[gm]) + 120.0
    h[bm] = 60.0 * ((r[bm] - g[bm]) / delta[bm]) + 240.0
    s = np.where(mx > 1e-9, delta / np.maximum(mx, 1e-9), 0.0)
    return np.stack([h, s, mx], axis=1)


def hue_distance(h1, h2):
    """Circular distance between hues in degrees (broadcasts)."""
    d = np.abs(np.asarray(h1) - np.asarray(h2)) % 360.0
    return np.minimum(d, 360.0 - d)


def min_area_rect(xy):
    """Minimum-area bounding rectangle of 2D points (rotating calipers over the convex hull).

    Returns (center (2,), size (2,) [along the rect's x, y], yaw) where yaw is the angle of the
    rect's x axis. size[0] >= size[1] is NOT enforced; see canonical_box_yaw.
    """
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 3:
        c = xy.mean(axis=0) if len(xy) else np.zeros(2)
        return c, np.zeros(2), 0.0
    try:
        hull = xy[ConvexHull(xy).vertices]
    except QhullError:  # collinear / degenerate
        hull = xy
    edges = np.roll(hull, -1, axis=0) - hull
    angles = np.unique(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), np.pi / 2))
    best = None
    for a in angles:
        R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])   # world -> rect frame
        p = hull @ R.T
        lo, hi = p.min(axis=0), p.max(axis=0)
        area = np.prod(hi - lo)
        if best is None or area < best[0]:
            center_r = (lo + hi) / 2
            best = (area, R.T @ center_r, hi - lo, a)
    _, center, size, yaw = best
    return center, size, float(yaw)


def canonical_box_yaw(size, yaw, square_tol=0.15):
    """Canonicalize a footprint rectangle's yaw by its symmetry.

    Square-ish (sides within square_tol): 4-fold, yaw in [-pi/4, pi/4).
    Otherwise: 2-fold about the LONG axis, yaw in [-pi/2, pi/2), size returned as (long, short).
    """
    a, b = float(size[0]), float(size[1])
    if max(a, b) > 0 and abs(a - b) / max(a, b) <= square_tol:
        y = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
        return np.array([a, b]), y, 4
    if b > a:
        a, b, yaw = b, a, yaw + np.pi / 2
    y = (yaw + np.pi / 2) % np.pi - np.pi / 2
    return np.array([a, b]), y, 2
