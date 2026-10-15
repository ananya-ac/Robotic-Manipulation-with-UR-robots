"""Core tests on synthetic single-view scenes: table plane + colored boxes (top face and the two
sides facing the camera only, as one wrist view would see them). No ROS, no simulator."""

import os

import numpy as np
import pytest
import yaml

from tabletop_perception.core import (GraspConfig, GripperModel, SegmentationConfig, antipodal_grasps,
                                      detect)
from tabletop_perception.core.fusion import VoxelMap
from tabletop_perception.core.views import orbit_views

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")
TABLE_Z = 0.30
RED, GREEN, BLUE = (0.8, 0.1, 0.1), (0.1, 0.7, 0.1), (0.1, 0.1, 0.8)


def seg_cfg():
    return SegmentationConfig.from_dict({
        "workspace": {"min": [-0.92, -0.32, 0.27], "max": [-0.28, 0.32, 0.60]},
        "palette": {"cube_0": RED, "cube_1": GREEN, "cube_2": BLUE},
    })


def gripper():
    with open(os.path.join(CONFIG_DIR, "robotiq_2f85.yaml")) as f:
        return GripperModel.from_dict(yaml.safe_load(f))


def box_points(center, size, yaw, rgb, step=0.002, faces=("top", "+x", "+y")):
    """Surface samples of a box (in its frame, then rotated by yaw about z)."""
    hx, hy, hz = np.asarray(size) / 2
    u = np.arange(-1, 1 + 1e-9, step / max(hx, hy, hz))
    pts = []
    for f in faces:
        a, b = np.meshgrid(u, u)
        a, b = a.ravel(), b.ravel()
        if f == "top":
            p = np.stack([a * hx, b * hy, np.full_like(a, hz)], 1)
        elif f == "+x":
            p = np.stack([np.full_like(a, hx), a * hy, b * hz], 1)
        elif f == "+y":
            p = np.stack([a * hx, np.full_like(a, hy), b * hz], 1)
        elif f == "-x":
            p = np.stack([np.full_like(a, -hx), a * hy, b * hz], 1)
        elif f == "-y":
            p = np.stack([a * hx, np.full_like(a, -hy), b * hz], 1)
        pts.append(p)
    p = np.unique(np.round(np.vstack(pts), 5), axis=0)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    p = p @ R.T + center
    return p, np.tile(rgb, (len(p), 1))


def scene(objs, seed=0, noise=0.0008):
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.arange(-0.9, -0.3, 0.004), np.arange(-0.3, 0.3, 0.004))
    table = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, TABLE_Z)], 1)
    pts, cols = [table], [np.tile((0.55, 0.4, 0.28), (len(table), 1))]
    for o in objs:
        p, c = box_points(*o)
        pts.append(p)
        cols.append(c)
    P = np.vstack(pts)
    return P + rng.normal(0, noise, P.shape), np.vstack(cols)


CUBE = (0.045, 0.045, 0.045)


def cube_on_table(x, y, yaw, rgb):
    return (np.array([x, y, TABLE_Z + 0.0225]), CUBE, yaw, rgb)


def yaw_err_deg(a, b, period=np.pi / 2):
    return np.degrees(abs((a - b + period / 2) % period - period / 2))


def test_three_cubes_poses_and_classes():
    truth = {"cube_0": (-0.50, -0.10, 0.3), "cube_1": (-0.45, 0.10, -0.6), "cube_2": (-0.60, 0.02, 0.0)}
    objs = [cube_on_table(x, y, yaw, rgb) for (x, y, yaw), rgb in zip(truth.values(), (RED, GREEN, BLUE))]
    est = detect(*scene(objs), seg_cfg())
    assert sorted(o.name for o in est.objects) == sorted(truth)
    assert abs(-est.plane[3] / est.plane[2] - TABLE_Z) < 0.002
    for o in est.objects:
        x, y, yaw = truth[o.name]
        assert np.linalg.norm(o.center[:2] - [x, y]) < 0.004, o.name
        assert abs(o.center[2] - (TABLE_Z + 0.0225)) < 0.004, o.name
        assert yaw_err_deg(o.yaw, yaw) < 4.0, o.name
        assert np.allclose(o.extent, CUBE, atol=0.005), (o.name, o.extent)
        assert o.symmetry == 4 and o.top_visible


def test_stack_is_split_by_color():
    base = cube_on_table(-0.5, 0.0, 0.2, RED)
    top = (np.array([-0.5, 0.0, TABLE_Z + 0.045 + 0.0225]), CUBE, 0.2, GREEN)
    base = (base[0], base[1], base[2], base[3], 0.002, ("+x", "+y"))    # its top face is covered
    est = detect(*scene([base, top]), seg_cfg())
    o0, o1 = est.get("cube_0"), est.get("cube_1")
    assert o0 is not None and o1 is not None
    assert not o0.top_visible and o1.top_visible
    assert abs(o1.bottom_z - (TABLE_Z + 0.045)) < 0.005      # sits on the red cube, not the table
    assert abs(o1.center[2] - (TABLE_Z + 0.0675)) < 0.004
    assert abs(o0.top_z - (TABLE_Z + 0.045)) < 0.004


def test_top_face_only_view_gets_full_height():
    # Straight-down view: no side faces at all, so the bottom can't be observed. Cubes on the table
    # must still get their full height (support = plane), and a cube resting on another object
    # whose top is visible must find that support under its footprint.
    red = (np.array([-0.5, -0.05, TABLE_Z + 0.0225]), CUBE, 0.1, RED, 0.002, ("top",))
    shelf = (np.array([-0.5, 0.10, TABLE_Z + 0.01]), (0.10, 0.10, 0.02), 0.0, GREEN, 0.002, ("top",))
    blue = (np.array([-0.5, 0.10, TABLE_Z + 0.02 + 0.0225]), CUBE, 0.0, BLUE, 0.002, ("top",))
    est = detect(*scene([red, shelf, blue]), seg_cfg())
    r, b = est.get("cube_0"), est.get("cube_2")
    assert abs(r.extent[2] - 0.045) < 0.004 and abs(r.center[2] - (TABLE_Z + 0.0225)) < 0.003
    assert abs(b.bottom_z - (TABLE_Z + 0.02)) < 0.004       # on the green slab, not the table
    assert abs(b.extent[2] - 0.045) < 0.005


def test_non_cube_box_extent_and_2fold_yaw():
    box = (np.array([-0.55, 0.0, TABLE_Z + 0.015]), (0.08, 0.03, 0.03), 0.5, BLUE)
    o = detect(*scene([box]), seg_cfg()).get("cube_2")
    assert o.symmetry == 2
    assert np.allclose(o.extent, (0.08, 0.03, 0.03), atol=0.005)
    assert yaw_err_deg(o.yaw, 0.5, period=np.pi) < 3.0


def test_grasps_on_cube_are_antipodal_and_face_aligned():
    yaw = 0.35
    est = detect(*scene([cube_on_table(-0.5, 0.0, yaw, RED)]), seg_cfg())
    o = est.get("cube_0")
    gs = antipodal_grasps(o, est, gripper(), GraspConfig(method="outline"))
    assert gs, "no grasps"
    g = gs[0]
    assert abs(g.width - 0.045) < 0.006
    closing_yaw = np.arctan2(g.T[1, 0], g.T[0, 0])
    assert yaw_err_deg(closing_yaw, yaw) < 6.0            # fingers on opposite faces
    assert np.allclose(g.T[:3, 2], [0, 0, -1])            # straight down
    assert abs(g.T[2, 3] - (TABLE_Z + 0.0225)) < 0.004    # tip at mid-height (cube center)
    assert np.linalg.norm(g.T[:2, 3] - o.center[:2]) < 0.006


def test_neighbor_blocks_grasp_axis():
    # A tall green box right next to the red cube along world x: grasps closing along x would put
    # a finger into it, so every returned grasp must close along y.
    red = cube_on_table(-0.5, 0.0, 0.0, RED)
    wall = (np.array([-0.5 + 0.0225 + 0.012 + 0.01, 0.0, TABLE_Z + 0.05]), (0.02, 0.12, 0.10), 0.0, GREEN)
    est = detect(*scene([red, wall]), seg_cfg())
    gs = antipodal_grasps(est.get("cube_0"), est, gripper(), GraspConfig(method="outline"))
    assert gs
    for g in gs:
        assert abs(g.T[0, 0]) < 0.2, "a grasp closes along x into the neighbor"


def test_too_wide_object_has_no_grasp():
    big = (np.array([-0.55, 0.0, TABLE_Z + 0.03]), (0.12, 0.12, 0.06), 0.0, BLUE)
    est = detect(*scene([big]), seg_cfg())
    assert antipodal_grasps(est.get("cube_2"), est, gripper(), GraspConfig(method="outline")) == []


ALL_SIDES = ("top", "+x", "-x", "+y", "-y")


def full_cube(x, y, yaw, rgb, z=TABLE_Z + 0.0225):
    return (np.array([x, y, z]), CUBE, yaw, rgb, 0.002, ALL_SIDES)


def test_3d_grasps_on_fully_observed_cube():
    yaw = -0.4
    est = detect(*scene([full_cube(-0.5, 0.0, yaw, RED)]), seg_cfg())
    o = est.get("cube_0")
    gs = antipodal_grasps(o, est, gripper(), GraspConfig(method="antipodal_3d"))
    assert gs, "no 3D grasps"
    g = gs[0]
    assert abs(g.width - 0.045) < 0.004
    assert yaw_err_deg(np.arctan2(g.T[1, 0], g.T[0, 0]), yaw) < 6.0
    assert np.degrees(np.arccos(-g.T[2, 2])) < 5.0              # best one ~straight down (noise tilts it)
    assert abs(g.T[2, 3] - (TABLE_Z + 0.0225)) < 0.006          # contacts near mid-height
    for g in gs:
        assert np.degrees(np.arccos(np.clip(-g.T[2, 2], -1, 1))) <= 25.0 + 1e-6


def test_3d_grasps_need_both_faces():
    # Single view (top, +x, +y only): no opposite faces observed -> the 3D sampler finds nothing,
    # which is exactly why multi-view fusion is used with it.
    est = detect(*scene([cube_on_table(-0.5, 0.0, 0.0, RED)]), seg_cfg())
    assert antipodal_grasps(est.get("cube_0"), est, gripper(), GraspConfig(method="antipodal_3d")) == []


def test_3d_neighbor_blocks_grasp_axis():
    # A tall wall 12 mm from the cube's +x face: grasps closing (nearly) along x would put a finger
    # into it. Control: without the wall such grasps exist.
    def along_x(gs):
        return [g for g in gs if abs(g.T[0, 0]) > 0.9]
    red = full_cube(-0.5, 0.0, 0.0, RED)
    wall = (np.array([-0.5 + 0.0225 + 0.012 + 0.01, 0.0, TABLE_Z + 0.05]), (0.02, 0.12, 0.10), 0.0, GREEN,
            0.002, ALL_SIDES)
    cfg = GraspConfig(method="antipodal_3d", max_candidates=50)
    alone = detect(*scene([red]), seg_cfg())
    assert along_x(antipodal_grasps(alone.get("cube_0"), alone, gripper(), cfg))
    est = detect(*scene([red, wall]), seg_cfg())
    gs = antipodal_grasps(est.get("cube_0"), est, gripper(), cfg)
    assert gs and not along_x(gs)


def test_fusion_merges_views_without_duplicates():
    a, ca = box_points(np.array([-0.5, 0.0, TABLE_Z + 0.0225]), CUBE, 0.0, RED, faces=("top", "+x", "+y"))
    b, cb = box_points(np.array([-0.5, 0.0, TABLE_Z + 0.0225]), CUBE, 0.0, RED, faces=("top", "-x", "-y"))
    m = VoxelMap(0.002)
    n1 = m.add(a, ca)
    n2 = m.add(b, cb)
    n3 = m.add(b, cb)                                   # re-adding the same view adds nothing
    assert n1 < n2 and n3 == n2 and m.n_views == 3
    once = VoxelMap(0.002).add(np.vstack([a, b]), np.vstack([ca, cb]))   # both views in one go
    assert abs(n2 - once) <= 0.02 * once
    assert np.allclose(m.colors.mean(axis=0), RED, atol=1e-6)            # colors averaged, not mixed


def test_orbit_views_look_at_target():
    target = np.array([-0.47, 0.0, 0.32])
    for el, az, T in orbit_views(target, 0.3, [50, 65], range(0, 360, 90)):
        assert abs(np.linalg.norm(T[:3, 3] - target) - 0.3) < 1e-9
        ray = (target - T[:3, 3]) / 0.3
        assert np.allclose(T[:3, 2], ray)               # optical axis on the target
        assert abs(T[2, 0]) < 1e-9                      # image x horizontal (no roll)
        assert T[2, 1] < 0                              # image y points downward-ish


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
