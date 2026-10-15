"""Tabletop segmentation: colored world-frame point cloud -> objects on the support surface.

Pipeline (detect):
  1. crop to the workspace box
  2. RANSAC support plane (must be near-horizontal), removed; its height is the default support
  3. self-filter: drop points inside caller-given boxes (robot links)
  4. DBSCAN clusters (tiny clusters dropped)
  5. split touching objects (e.g. a stack) by per-point palette class. Without a palette,
     touching objects are NOT split - a known limitation.
  6. per segment: class (mean hue vs palette); yaw + footprint from the minimum-area rectangle of
     the top face; bottom = lowest observed point when side faces are seen (snapped to the plane),
     else (top face only) the highest OTHER segment under the footprint, else the plane; height =
     top - bottom. The object is modeled as its footprint extruded down to its support (prism);
     no cube prior.

Everything is in the world frame with +z up. Pure numpy/scipy/Open3D: no ROS.
"""

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from tabletop_perception.core.geometry import (canonical_box_yaw, hue_distance, min_area_rect,
                                               rgb_to_hsv, rot_z)


@dataclass
class SegmentationConfig:
    workspace_min: np.ndarray
    workspace_max: np.ndarray
    plane_distance: float = 0.004       # RANSAC inlier distance (m)
    plane_max_tilt_deg: float = 10.0    # reject planes whose normal is further than this from +z
    plane_iterations: int = 500
    dbscan_eps: float = 0.012
    dbscan_min_points: int = 10
    min_object_points: int = 40
    top_band: float = 0.006             # top face = points within this of the segment's top
    fill_pixel: float = 0.004           # raster cell for the top-face fill test (> point spacing)
    min_top_fill: float = 0.5           # top band must cover this much of its rectangle
    min_side_extent: float = 0.010      # segment spans this much in z -> side faces seen, bottom observed
    on_support_tol: float = 0.012       # observed bottom this close to the plane -> rests on the plane
    support_margin: float = 0.002       # footprint shrink when searching for what an object rests on
    palette: dict = field(default_factory=dict)   # name -> rgb (0..1)
    min_saturation: float = 0.35        # less saturated points are not color-classified
    max_hue_dist_deg: float = 30.0
    square_tol: float = 0.15            # footprint sides within this fraction -> 4-fold symmetric

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        ws = d.pop("workspace")
        palette = {k: np.asarray(v, float) for k, v in d.pop("palette", {}).items()}
        return cls(workspace_min=np.asarray(ws["min"], float), workspace_max=np.asarray(ws["max"], float),
                   palette=palette, **d)


@dataclass
class ObjectEstimate:
    name: str                   # palette class, or "unknown_<i>"
    center: np.ndarray          # (3,) world
    yaw: float                  # canonicalized by footprint symmetry
    extent: np.ndarray          # (3,) footprint long/short (or equal) + height
    symmetry: int               # 4 (square footprint) or 2
    top_z: float
    bottom_z: float             # support height under it (plane, or the object it sits on)
    top_visible: bool
    points: np.ndarray          # (N, 3) segment points
    n_points: int

    @property
    def R(self):
        return rot_z(self.yaw)


@dataclass
class SceneEstimate:
    objects: list               # [ObjectEstimate]
    plane: np.ndarray           # (4,) a, b, c, d with c > 0 (normal up)
    scene_points: np.ndarray    # all non-plane, non-robot points in the workspace (for grasp checks)
    message: str = ""

    def get(self, name):
        return next((o for o in self.objects if o.name == name), None)


def plane_height(plane, xy):
    a, b, c, d = plane
    xy = np.atleast_2d(xy)
    return -(a * xy[:, 0] + b * xy[:, 1] + d) / c


def crop(points, colors, lo, hi):
    keep = np.all((points >= lo) & (points <= hi), axis=1)
    return points[keep], colors[keep]


def fit_support_plane(points, cfg):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    model, inliers = pcd.segment_plane(cfg.plane_distance, 3, cfg.plane_iterations)
    model = np.asarray(model, float)
    if model[2] < 0:
        model = -model
    tilt = np.degrees(np.arccos(np.clip(model[2] / np.linalg.norm(model[:3]), -1, 1)))
    if tilt > cfg.plane_max_tilt_deg:
        raise ValueError(f"largest plane is tilted {tilt:.0f} deg - no horizontal support surface in view")
    model /= np.linalg.norm(model[:3])
    mask = np.zeros(len(points), bool)
    mask[inliers] = True
    # Also drop anything at/below the plane (noise under it, the table's front edge, ...).
    mask |= points[:, 2] <= plane_height(model, points[:, :2]) + cfg.plane_distance
    return model, mask


def fill_ratio(xy, pixel):
    """Fraction of the points' minimum-area rectangle covered by occupied pixel cells."""
    center, size, yaw = min_area_rect(xy)
    area = size[0] * size[1]
    if area <= pixel * pixel:
        return 0.0
    c, s = np.cos(yaw), np.sin(yaw)
    local = (xy - center) @ np.array([[c, -s], [s, c]])
    cells = np.unique(np.floor(local / pixel).astype(int), axis=0)
    return min(len(cells) * pixel * pixel / area, 1.0)


def classify_points(colors, cfg):
    """Per-point palette index (-1 = unclassified)."""
    if not cfg.palette:
        return np.full(len(colors), -1)
    hsv = rgb_to_hsv(colors)
    pal_h = rgb_to_hsv(np.stack(list(cfg.palette.values())))[:, 0]
    d = hue_distance(hsv[:, 0:1], pal_h[None, :])
    lab = np.argmin(d, axis=1)
    lab[(d[np.arange(len(lab)), lab] > cfg.max_hue_dist_deg) | (hsv[:, 1] < cfg.min_saturation)] = -1
    return lab


def split_by_class(points, labels):
    """Split one cluster into per-class index arrays; unclassified points join the nearest
    classified point's class. Returns [(label, idx)]; label -1 if nothing was classified."""
    known = labels >= 0
    if not known.any():
        return [(-1, np.arange(len(points)))]
    lab = labels.copy()
    if (~known).any():
        _, nn = cKDTree(points[known]).query(points[~known])
        lab[~known] = labels[known][nn]
    return [(c, np.flatnonzero(lab == c)) for c in np.unique(lab)]


def footprint(pts, cfg):
    """Top-face analysis of one segment -> (top_z, top_visible, center_xy, size, yaw)."""
    top_z = float(np.percentile(pts[:, 2], 98))
    top = pts[pts[:, 2] >= top_z - cfg.top_band]
    # Top face visible if the top band FILLS its bounding rectangle - a real horizontal face -
    # rather than tracing its edges, as the upper rims of side faces do when the top is covered
    # (a cube under another one).
    top_visible = len(top) >= 10 and fill_ratio(top[:, :2], cfg.fill_pixel) >= cfg.min_top_fill
    center_xy, size, yaw = min_area_rect((top if top_visible else pts)[:, :2])
    return top_z, bool(top_visible), center_xy, size, yaw


def support_height(center_xy, size, yaw, below_z, others, plane, cfg):
    """What the object rests on: the highest point of OTHER segments under its footprint (shrunk
    by support_margin) and below its own lowest point, else the support plane. Works whether or
    not its side faces are visible - a straight-down view sees only top faces."""
    support = float(plane_height(plane, np.asarray(center_xy)[None])[0])
    if len(others):
        c, s = np.cos(yaw), np.sin(yaw)
        local = (others[:, :2] - center_xy) @ np.array([[c, -s], [s, c]])
        half = np.maximum(np.asarray(size) / 2 - cfg.support_margin, 0.002)
        under = np.all(np.abs(local) <= half, axis=1) & (others[:, 2] < below_z)
        if under.sum() >= 10:
            support = max(support, float(np.percentile(others[under, 2], 98)))
    return support


def estimate_object(name, pts, others, plane, cfg):
    """One segment -> ObjectEstimate. others: points of all OTHER segments (support search)."""
    top_z, top_visible, center_xy, size, yaw = footprint(pts, cfg)
    own_low = float(np.percentile(pts[:, 2], 2))
    if top_z - own_low >= cfg.min_side_extent:
        # Side faces seen: the bottom is observed (snapped to the plane when it reaches it).
        plane_z = float(plane_height(plane, np.asarray(center_xy)[None])[0])
        bottom = plane_z if own_low - plane_z < cfg.on_support_tol else own_low
    else:
        # Only the top face seen (straight-down view): infer what it rests on.
        bottom = support_height(center_xy, size, yaw, own_low - cfg.top_band, others, plane, cfg)
    size, yaw, sym = canonical_box_yaw(size, yaw, cfg.square_tol)
    height = top_z - bottom
    return ObjectEstimate(name=name, center=np.array([center_xy[0], center_xy[1], bottom + height / 2]),
                          yaw=float(yaw), extent=np.array([size[0], size[1], height]), symmetry=sym,
                          top_z=top_z, bottom_z=bottom, top_visible=top_visible, points=pts,
                          n_points=len(pts))


def detect(points, colors, cfg, exclude_boxes=()):
    """points (N, 3) world, colors (N, 3) in 0..1 -> SceneEstimate.

    exclude_boxes: iterable of (T_world_box (4, 4), half_extents (3,)) - points inside are removed
    (robot self-filter; the caller supplies link poses).
    """
    points, colors = crop(np.asarray(points, float), np.asarray(colors, float), cfg.workspace_min, cfg.workspace_max)
    if len(points) < 100:
        raise ValueError(f"only {len(points)} points in the workspace - is the camera looking at it?")
    plane, on_plane = fit_support_plane(points, cfg)
    points, colors = points[~on_plane], colors[~on_plane]
    for T, half in exclude_boxes:
        local = (points - T[:3, 3]) @ T[:3, :3]
        inside = np.all(np.abs(local) <= half, axis=1)
        points, colors = points[~inside], colors[~inside]
    if len(points) == 0:
        return SceneEstimate([], plane, points, "nothing above the support plane")

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cluster = np.asarray(pcd.cluster_dbscan(cfg.dbscan_eps, cfg.dbscan_min_points))
    labels = classify_points(colors, cfg)
    names = list(cfg.palette.keys())

    segments, n_unknown = [], 0          # (name, indices into points)
    for c in range(cluster.max() + 1):
        idx = np.flatnonzero(cluster == c)
        if len(idx) < cfg.min_object_points:
            continue
        for lab, sub in split_by_class(points[idx], labels[idx]):
            if len(sub) < cfg.min_object_points:
                continue
            if lab >= 0:
                name = names[lab]
            else:
                name, n_unknown = f"unknown_{n_unknown}", n_unknown + 1
            segments.append((name, idx[sub]))
    owner = np.full(len(points), -1)
    for k, (_, gi) in enumerate(segments):
        owner[gi] = k
    objects = [estimate_object(name, points[gi], points[(owner >= 0) & (owner != k)], plane, cfg)
               for k, (name, gi) in enumerate(segments)]

    # A palette class seen as several segments (e.g. split by an occluder): keep the largest.
    best = {}
    for o in objects:
        if o.name not in best or o.n_points > best[o.name].n_points:
            best[o.name] = o
    objects = sorted(best.values(), key=lambda o: o.name)
    return SceneEstimate(objects, plane, points, f"{len(objects)} objects")
