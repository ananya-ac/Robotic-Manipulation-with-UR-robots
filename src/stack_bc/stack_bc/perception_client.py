"""Client for tabletop_perception's node (oracle_obj_unknown). Plain service wrappers that turn
the messages into numpy poses; all perception happens in the node.

  ros2 launch tabletop_perception perception.launch.py   (venv active - it needs Open3D)
"""

from dataclasses import dataclass

import numpy as np
from std_srvs.srv import Trigger
from tabletop_perception_msgs.srv import DetectObjects, PlanGrasps

from stack_bc import rotations as rot
from stack_bc.scene_sensing import _call

NODE = "/perception_node"
HINT = "is `ros2 launch tabletop_perception perception.launch.py` running (venv active)?"


@dataclass
class Perceived:
    center: np.ndarray     # (3,) world
    R: np.ndarray          # (3, 3) yaw-only
    extent: np.ndarray     # (3,) footprint (2) + height
    top_visible: bool

    @property
    def T(self):
        return rot.pose_matrix(self.center, self.R)

    @property
    def top_z(self):
        return float(self.center[2] + self.extent[2] / 2)


@dataclass
class GraspPose:
    T: np.ndarray          # (4, 4) gripper_tip_link target: x = closing axis, z = approach
    width: float
    score: float


def _pose_to_matrix(p):
    return rot.pose_matrix([p.position.x, p.position.y, p.position.z],
                           rot.quat_to_matrix([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]))


class PerceptionClient:
    def __init__(self, node, timeout=10.0):
        self.timeout = timeout
        self._reset = node.create_client(Trigger, f"{NODE}/reset_map")
        self._add = node.create_client(Trigger, f"{NODE}/add_view")
        self._detect = node.create_client(DetectObjects, f"{NODE}/detect_objects")
        self._grasps = node.create_client(PlanGrasps, f"{NODE}/plan_grasps")

    def reset_map(self):
        res = _call(self._reset, Trigger.Request(), "reset_map", self.timeout, HINT)
        if not res.success:
            raise RuntimeError(f"reset_map: {res.message}")

    def add_view(self):
        """Capture the camera's current view (a frame newer than this call) into the fused map."""
        res = _call(self._add, Trigger.Request(), "add_view", self.timeout, HINT)
        if not res.success:
            raise RuntimeError(f"add_view: {res.message}")
        return res.message

    def detect(self, names=(), use_map=True, fresh=True):
        """-> ({name: Perceived}, message). Missing names are simply absent (check the dict)."""
        req = DetectObjects.Request(names=list(names), use_map=use_map, fresh=fresh)
        res = _call(self._detect, req, "detect_objects", self.timeout, HINT)
        out = {}
        for det in res.detections.detections:
            if not det.results:
                continue
            name = det.results[0].hypothesis.class_id
            T = _pose_to_matrix(det.bbox.center)
            out[name] = Perceived(center=T[:3, 3].copy(), R=T[:3, :3].copy(),
                                  extent=np.array([det.bbox.size.x, det.bbox.size.y, det.bbox.size.z]),
                                  top_visible=det.results[0].hypothesis.score >= 0.99)
        return out, res.message

    def grasps(self, name, max_candidates=0, timeout=30.0):
        """Grasps for `name` from the node's latest detection, best first."""
        res = _call(self._grasps, PlanGrasps.Request(name=name, max_candidates=max_candidates),
                    "plan_grasps", timeout, HINT)
        return [GraspPose(T=_pose_to_matrix(g.tip_pose.pose), width=g.width, score=g.score) for g in res.grasps], \
            res.message
