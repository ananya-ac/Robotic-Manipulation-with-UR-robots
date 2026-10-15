"""Build core configs from the node's flat parameter dict (as in config/tabletop_mujoco.yaml).

ROS-free, so offline tools can load the same YAML the node uses:
    params = load_node_params(path)  ->  configs_from_params(params)
"""

import os

import numpy as np
import yaml

from tabletop_perception.core.grasping import GraspConfig
from tabletop_perception.core.gripper import GripperModel
from tabletop_perception.core.segmentation import SegmentationConfig

SEG_KEYS = ("plane_distance", "plane_max_tilt_deg", "plane_iterations", "dbscan_eps", "dbscan_min_points",
            "min_object_points", "top_band", "fill_pixel", "min_top_fill", "min_side_extent", "on_support_tol", "support_margin",
            "min_saturation", "max_hue_dist_deg", "square_tol")
GRASP_KEYS = ("friction_coef", "width_margin", "min_width", "table_clearance", "pixel", "n_samples",
              "max_candidates", "dedupe_dist", "dedupe_angle_deg", "w_antipodal", "w_center", "w_width", "seed", "collision_radius",
              "method", "normal_radius", "max_side_normal_z", "min_flatness", "ray_radius", "max_tilt_deg",
              "tilt_step_deg", "w_height", "w_tilt")


def config_dir():
    """This package's config/ (installed share dir if available, else the source tree)."""
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory("tabletop_perception"), "config")
    except Exception:
        return os.path.join(os.path.dirname(__file__), "..", "config")


def load_node_params(path):
    with open(path) as f:
        d = yaml.safe_load(f)
    return next(iter(d.values()))["ros__parameters"]


def configs_from_params(p):
    """-> (SegmentationConfig, GraspConfig, GripperModel, self_filter [(frame, center(3), half(3))])"""
    names = list(p.get("palette_names", []))
    rgb = np.asarray(p.get("palette_rgb", []), float).reshape(-1, 3)
    if len(names) != len(rgb):
        raise ValueError("palette_names and palette_rgb (3 per name) differ in length")
    seg = SegmentationConfig.from_dict({
        "workspace": {"min": p["workspace_min"], "max": p["workspace_max"]},
        "palette": dict(zip(names, rgb)),
        **{k: p[k] for k in SEG_KEYS if k in p}})
    grasp = GraspConfig.from_dict({k: p[k] for k in GRASP_KEYS if k in p})
    gpath = p.get("gripper_config", "robotiq_2f85.yaml")
    if not os.path.isabs(gpath):
        gpath = os.path.join(config_dir(), gpath)
    with open(gpath) as f:
        gripper = GripperModel.from_dict(yaml.safe_load(f))
    frames = list(p.get("self_filter_frames", []))
    centers = np.asarray(p.get("self_filter_centers", []), float).reshape(-1, 3)
    halves = np.asarray(p.get("self_filter_half_extents", []), float).reshape(-1, 3)
    self_filter = list(zip(frames, centers, halves))
    return seg, grasp, gripper, self_filter
