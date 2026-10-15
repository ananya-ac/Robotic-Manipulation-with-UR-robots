"""Antipodal grasp sampling for a parallel-jaw gripper, approaching from above.

Two samplers (GraspConfig.method):

  antipodal_3d (default; for fused multi-view clouds) - classic 3D antipodal sampling: a contact
    p1 on a near-vertical surface, its partner p2 along the inward normal ray, both connecting
    lines inside their friction cones, using real surface normals. Approaches: the direction
    closest to straight down that is perpendicular to the closing axis, plus tilts about the
    closing axis up to max_tilt_deg. Needs both opposite faces observed - hence multi-view.

  outline (single top-down view) - Dex-Net-style (Mahler et al. 2017, antipodal sampler on
    depth-image edges): project the object along the approach, take the footprint outline and
    its 2D normals as stand-ins for the unseen side faces. Straight-down approaches only;
    assumes prism-like objects.

Every candidate is then rejected if the fingers' lowest point (closed) would come within
`table_clearance` of the support plane, if any scene point falls inside a gripper body (palm,
open fingers, linkage), or if any OTHER object's point lies in the closing region; survivors are
scored (antipodal quality, centering, width margin, [3D: contact height, tilt]) and de-duplicated.
"""

from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.spatial import cKDTree

from tabletop_perception.core.segmentation import plane_height


@dataclass
class GraspConfig:
    method: str = "antipodal_3d"     # or "outline"
    friction_coef: float = 0.5
    width_margin: float = 0.010      # keep width <= stroke - margin
    min_width: float = 0.005
    table_clearance: float = 0.005
    pixel: float = 0.002             # footprint raster resolution (m)
    n_samples: int = 400             # outline points tried as first contact
    max_candidates: int = 20
    dedupe_dist: float = 0.004
    dedupe_angle_deg: float = 8.0
    w_antipodal: float = 1.0
    w_center: float = 1.0
    w_width: float = 0.3
    seed: int = 0
    collision_radius: float = 0.20   # m; scene points further than this (xy) from the object are ignored
    # antipodal_3d only
    normal_radius: float = 0.006     # normal estimation neighbourhood (m)
    max_side_normal_z: float = 0.6   # first contacts on surfaces this close to vertical
    min_flatness: float = 0.97       # contacts only where neighbouring normals agree (not edges)
    ray_radius: float = 0.003        # partner search: max distance from the inward normal ray
    max_tilt_deg: float = 25.0       # approach within this of straight down
    tilt_step_deg: float = 12.5
    w_height: float = 0.5            # prefer contacts at the object's mid-height
    w_tilt: float = 0.3              # prefer vertical approaches

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


@dataclass
class Grasp:
    T: np.ndarray         # (4, 4) tip pose in world: x = closing axis, z = approach
    width: float          # contact separation (m)
    score: float
    contacts: np.ndarray  # (2, 3) world

    @property
    def approach(self):
        return self.T[:3, 2]


def footprint_outline(xy, pixel):
    """Rasterize projected points, fill, and return outline points + outward unit normals (2D)."""
    lo = xy.min(axis=0) - 4 * pixel
    ij = np.floor((xy - lo) / pixel).astype(int)
    shape = ij.max(axis=0) + 5
    mask = np.zeros((shape[1], shape[0]), bool)            # rows = y, cols = x
    mask[ij[:, 1], ij[:, 0]] = True
    mask = ndimage.binary_closing(mask, iterations=2)
    mask = ndimage.binary_fill_holes(mask)
    edge = mask & ~ndimage.binary_erosion(mask)
    smooth = ndimage.gaussian_filter(mask.astype(float), 1.5)
    gy, gx = np.gradient(smooth)
    rows, cols = np.nonzero(edge)
    n = -np.stack([gx[rows, cols], gy[rows, cols]], axis=1)  # gradient points inward
    norm = np.linalg.norm(n, axis=1)
    ok = norm > 1e-6
    pts = lo + (np.stack([cols, rows], axis=1)[ok] + 0.5) * pixel
    return pts, n[ok] / norm[ok, None]


def _pair_candidates(outline, normals, cfg, max_w, rng):
    """For sampled first contacts, the best antipodal partner on the outline: [(p1, p2, quality)]."""
    cos_cone = np.cos(np.arctan(cfg.friction_coef))
    out = []
    for i in rng.choice(len(outline), size=min(cfg.n_samples, len(outline)), replace=False):
        p1, n1 = outline[i], normals[i]
        v = outline - p1
        w = np.linalg.norm(v, axis=1)
        ok = (w >= cfg.min_width) & (w <= max_w)
        if not ok.any():
            continue
        d = v[ok] / w[ok, None]
        q1 = d @ -n1                                  # line inside the cone at p1 (pushing inward)
        q2 = np.einsum("ij,ij->i", d, normals[ok])    # and at p2 (its outward normal along d)
        q = np.where((q1 >= cos_cone) & (q2 >= cos_cone), np.minimum(q1, q2), -np.inf)
        k = int(np.argmax(q))
        if np.isfinite(q[k]):
            out.append((p1, outline[np.flatnonzero(ok)[k]], float(q[k])))
    return out


def _dedupe(grasps, cfg):
    kept = []
    for g in grasps:
        yaw = np.arctan2(g.T[1, 0], g.T[0, 0])
        dup = False
        for k in kept:
            dyaw = abs((yaw - np.arctan2(k.T[1, 0], k.T[0, 0]) + np.pi / 2) % np.pi - np.pi / 2)  # 180-deg symmetric
            if np.linalg.norm(g.T[:2, 3] - k.T[:2, 3]) < cfg.dedupe_dist and np.degrees(dyaw) < cfg.dedupe_angle_deg:
                dup = True
                break
        if not dup:
            kept.append(g)
        if len(kept) >= cfg.max_candidates:
            break
    return kept


class _Checker:
    """Collision / clearance tests shared by both samplers."""

    def __init__(self, obj, scene, gripper, cfg):
        self.scene, self.gripper, self.cfg = scene, gripper, cfg
        # Only points that could touch the gripper: horizontally within collision_radius of the
        # object (the gripper's boxes span < 8 cm from the tip; tilts shift the palm < 8 cm more).
        pts = scene.scene_points
        if len(pts):
            pts = pts[np.linalg.norm(pts[:, :2] - obj.center[:2], axis=1) <= cfg.collision_radius]
        self.pts = pts
        self.others = self.pts
        if len(self.pts):
            d, _ = cKDTree(obj.points).query(self.pts, distance_upper_bound=1e-6)
            self.others = self.pts[~(d <= 1e-6)]

    def ok(self, T):
        lowest = T[:3, 3] + self.gripper.finger_reach * T[:3, 2]          # fingertips when closed
        if lowest[2] < plane_height(self.scene.plane, lowest[None, :2])[0] + self.cfg.table_clearance - 1e-9:
            return False
        if len(self.pts) and self.gripper.collides((self.pts - T[:3, 3]) @ T[:3, :3]):
            return False
        if len(self.others) and self.gripper.closing_region.contains((self.others - T[:3, 3]) @ T[:3, :3]).any():
            return False
        return True


def _frame(x, approach, contact_mid, contact_depth):
    """Tip pose: x = closing axis, z = approach, contacts `contact_depth` behind the tip."""
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, np.cross(approach, x), approach])
    T[:3, 3] = contact_mid + contact_depth * approach
    return T


def _score(cfg, q, mid_xy, obj, width, max_w, extra=0.0):
    half_diag = 0.5 * np.linalg.norm(obj.extent[:2]) + 1e-6
    centering = 1.0 - min(np.linalg.norm(mid_xy - obj.center[:2]) / half_diag, 1.0)
    return float(cfg.w_antipodal * q + cfg.w_center * centering + cfg.w_width * (max_w - width) / max_w + extra)


def outline_grasps(obj, scene, gripper, cfg):
    """Top-down grasps from the footprint outline (single-view friendly; see module docstring)."""
    rng = np.random.default_rng(cfg.seed)
    outline, normals = footprint_outline(obj.points[:, :2], cfg.pixel)
    if len(outline) < 8:
        return []
    max_w = gripper.stroke - cfg.width_margin
    # Contacts at mid-height, but high enough that the fingers clear whatever the object sits on.
    z_min = obj.bottom_z + cfg.table_clearance + gripper.finger_reach + gripper.contact_depth
    z_mid = max(0.5 * (obj.bottom_z + obj.top_z), z_min)
    if z_mid > obj.top_z - 0.003:
        return []   # too flat to grasp by its sides with this gripper
    check = _Checker(obj, scene, gripper, cfg)
    down = np.array([0.0, 0.0, -1.0])
    grasps = []
    for p1, p2, q in _pair_candidates(outline, normals, cfg, max_w, rng):
        x2 = (p2 - p1) / np.linalg.norm(p2 - p1)
        mid = np.array([*(0.5 * (p1 + p2)), z_mid])
        T = _frame(np.array([x2[0], x2[1], 0.0]), down, mid, gripper.contact_depth)
        if not check.ok(T):
            continue
        width = float(np.linalg.norm(p2 - p1))
        grasps.append(Grasp(T=T, width=width, score=_score(cfg, q, mid[:2], obj, width, max_w),
                            contacts=np.array([[*p1, z_mid], [*p2, z_mid]])))
    grasps.sort(key=lambda g: -g.score)
    return _dedupe(grasps, cfg)


def object_normals(points, radius):
    """Surface normals from a (fused) object cloud, oriented outward from the object's centroid
    (fine for convex-ish objects; MuJoCo cubes/boxes here)."""
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, float)))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    n = np.asarray(pcd.normals)
    flip = np.einsum("ij,ij->i", n, points - points.mean(axis=0)) < 0
    n[flip] *= -1
    return n


def normal_consistency(points, normals, radius):
    """Mean agreement (dot product) of each point's normal with its neighbours' within `radius`.
    ~1 on flat faces; drops at edges/corners, whose normals are averages of two faces and would
    otherwise let corner-to-corner (diagonal) pairs pass the friction-cone test."""
    tree = cKDTree(points)
    out = np.empty(len(points))
    for i, nb in enumerate(tree.query_ball_point(points, radius)):
        out[i] = np.mean(normals[nb] @ normals[i])
    return out


def antipodal_3d_grasps(obj, scene, gripper, cfg):
    """Classic 3D antipodal sampling on a (fused, multi-view) object cloud.

    For a random contact p1 on a near-vertical surface, the partner p2 is the object point along the
    inward normal ray whose outward normal faces back along the line (both contacts' lines inside
    their friction cones). The approach is the direction closest to straight down that is
    perpendicular to the closing axis, plus +-tilt_step rotations about the closing axis, kept
    within max_tilt_deg of vertical. Contacts sit `contact_depth` behind the tip.
    """
    rng = np.random.default_rng(cfg.seed)
    pts = obj.points
    if len(pts) < 30:
        return []
    n = object_normals(pts, cfg.normal_radius)
    flat = normal_consistency(pts, n, cfg.normal_radius) >= cfg.min_flatness
    cos_cone = np.cos(np.arctan(cfg.friction_coef))
    max_w = gripper.stroke - cfg.width_margin
    side = np.flatnonzero((np.abs(n[:, 2]) < cfg.max_side_normal_z) & flat)
    if len(side) == 0:
        return []
    check = _Checker(obj, scene, gripper, cfg)
    down = np.array([0.0, 0.0, -1.0])
    tilts = [0.0] + [s * t for t in np.arange(cfg.tilt_step_deg, cfg.max_tilt_deg + 1e-9, cfg.tilt_step_deg)
                     for s in (1, -1)]                       # most vertical first
    center_h = 0.5 * (obj.bottom_z + obj.top_z)
    half_h = max(0.5 * (obj.top_z - obj.bottom_z), 1e-3)

    grasps = []
    for i in rng.choice(side, size=min(cfg.n_samples, len(side)), replace=False):
        p1, n1 = pts[i], n[i]
        v = pts - p1
        t = v @ -n1                                          # distance along the inward ray
        lateral = np.linalg.norm(v - np.outer(t, -n1), axis=1)
        cand = np.flatnonzero((t >= cfg.min_width) & (t <= max_w) & (lateral <= cfg.ray_radius) & flat)
        if len(cand) == 0:
            continue
        d = v[cand] / np.linalg.norm(v[cand], axis=1, keepdims=True)
        q1, q2 = d @ -n1, np.einsum("ij,ij->i", d, n[cand])
        q = np.where((q1 >= cos_cone) & (q2 >= cos_cone), np.minimum(q1, q2), -np.inf)
        k = int(np.argmax(q))
        if not np.isfinite(q[k]):
            continue
        p2 = pts[cand[k]]
        x = (p2 - p1) / np.linalg.norm(p2 - p1)
        a0 = down - (down @ x) * x                           # closest-to-vertical approach
        if np.linalg.norm(a0) < 1e-6:
            continue
        a0 /= np.linalg.norm(a0)
        mid = 0.5 * (p1 + p2)
        width = float(np.linalg.norm(p2 - p1))
        for tilt in tilts:
            r = np.radians(tilt)
            a = a0 * np.cos(r) + np.cross(x, a0) * np.sin(r)   # rotate about the closing axis
            if np.degrees(np.arccos(np.clip(a @ down, -1, 1))) > cfg.max_tilt_deg:
                continue
            T = _frame(x, a, mid, gripper.contact_depth)
            if not check.ok(T):
                continue
            height_term = -cfg.w_height * abs(mid[2] - center_h) / half_h
            tilt_term = -cfg.w_tilt * abs(np.degrees(np.arccos(np.clip(a @ down, -1, 1)))) / max(cfg.max_tilt_deg, 1)
            grasps.append(Grasp(T=T, width=width, contacts=np.array([p1, p2]),
                                score=_score(cfg, float(q[k]), mid[:2], obj, width, max_w, height_term + tilt_term)))
            break   # tilts are tried most-vertical first: keep the first collision-free one
    grasps.sort(key=lambda g: -g.score)
    return _dedupe(grasps, cfg)


def antipodal_grasps(obj, scene, gripper, cfg):
    """obj: ObjectEstimate, scene: SceneEstimate -> [Grasp], best first. cfg.method selects the
    sampler: 'antipodal_3d' (fused multi-view clouds) or 'outline' (single top-down view)."""
    if scene.plane[2] / np.linalg.norm(scene.plane[:3]) < 0.95:
        raise ValueError("support plane is not horizontal; grasps are sampled relative to vertical")
    if cfg.method == "outline":
        return outline_grasps(obj, scene, gripper, cfg)
    if cfg.method == "antipodal_3d":
        return antipodal_3d_grasps(obj, scene, gripper, cfg)
    raise ValueError(f"unknown grasp method {cfg.method!r}")
