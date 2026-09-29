"""Arm kinematics for the stacking pipeline: KDL FK + EAIK analytical IK, base_link -> gripper_tip_link.

Shared by the offline feasibility map and the live executor so both agree on what is reachable.
Wraps ur_control's EAIKKinematics (for its URDF parsing and EAIK->KDL tip-frame alignment) but
queries every EAIK branch itself, because the wrapper (a) returns EAIK's least-squares
"solutions" for unreachable poses as if they were exact, and (b) picks the branch closest to the
seed without accounting for the arm joints' +-2pi range, so an equivalent configuration one turn
away can lose to a genuinely different branch.

The URDF is sanitized here (sanitize_urdf) before it reaches ur_control: upstream
EAIKKinematics strips extension blocks with regexes, which break on our xacro output's comments
that mention "<ros2_control>" (see notes/eaik_urdf_comment_bug.md). Doing it with a real XML
parser means stack_bc runs against an unmodified upstream ur_control checkout.

Poses are (pos[3], quat_xyzw[4]) in base_link, matching ur_control/PyKDL conventions; the
*_world variants take/return world-frame poses (the frame cube poses arrive in). base_link is NOT
aligned with world in this robot description (+90 deg about z), so never mix the two.
"""

import xml.etree.ElementTree as ET

import numpy as np
import PyKDL

from ur_control import transformations
from ur_control.eaik_kinematics import EAIKKinematics
from ur_pykdl import ur_kinematics

from stack_bc.rotations import matrix_to_quat

BASE_LINK = "base_link"
TIP_LINK = "gripper_tip_link"
ARM_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
# URDF limits for the UR5e arm (elbow is +-pi, everything else +-2pi) - see the compiled MJCF.
JOINT_LOWER = np.array([-2, -2, -1, -2, -2, -2]) * np.pi
JOINT_UPPER = -JOINT_LOWER

FK_POS_TOL = 1e-4   # m - rejects EAIK least-squares approximations of unreachable poses
FK_ROT_TOL = 1e-3   # rad

NON_URDF_TAGS = ("ros2_control", "mujoco_inputs", "gazebo")


def sanitize_urdf(urdf: str) -> str:
    """Plain URDF: comments dropped (ElementTree's default parser discards them) and top-level
    ros2_control / mujoco_inputs / gazebo extension blocks removed as elements, not by regex."""
    root = ET.fromstring(urdf)
    for child in list(root):
        if child.tag in NON_URDF_TAGS:
            root.remove(child)
    return ET.tostring(root, encoding="unicode")


class ArmKinematics:
    def __init__(self, urdf: str, base_link: str = BASE_LINK, tip_link: str = TIP_LINK):
        urdf = sanitize_urdf(urdf)
        self.kdl = ur_kinematics(base_link=base_link, ee_link=tip_link, robot_description=urdf)
        self._eaik = EAIKKinematics(self.kdl, robot_description=urdf)
        # Fixed world -> base_link transform: a zero-joint KDL chain (ur_kinematics assumes 6 joints).
        chain = self.kdl._kdl_tree.getChain("world", base_link)
        frame = PyKDL.Frame()
        PyKDL.ChainFkSolverPos_recursive(chain).JntToCart(PyKDL.JntArray(chain.getNrOfJoints()), frame)
        self.T_world_base = np.eye(4)
        self.T_world_base[:3, 3] = [frame.p[i] for i in range(3)]
        self.T_world_base[:3, :3] = [[frame.M[i, j] for j in range(3)] for i in range(3)]
        self.T_base_world = np.linalg.inv(self.T_world_base)

    def fk(self, q):
        pose = np.asarray(self.kdl.forward(q), dtype=float)
        return pose[:3], pose[3:]

    def fk_matrix(self, q):
        return transformations.pose_to_transform(np.asarray(self.kdl.forward(q), dtype=float))

    def fk_world(self, q):
        return self.T_world_base @ self.fk_matrix(q)

    def ik_world(self, T_world, seed):
        pos, quat = matrix_to_pose(self.T_base_world @ T_world)
        return self.ik(pos, quat, seed)

    def ik_all(self, pos, quat_xyzw):
        """All exact IK branches (each unwrapped to lie within joint limits), possibly empty."""
        T = transformations.pose_to_transform(np.concatenate([pos, quat_xyzw]))
        ik = self._eaik._bot.IK(T @ np.linalg.inv(self._eaik._T_eaik_to_ee))
        sols = []
        for q in (ik.Q if ik.Q is not None else []):
            q = np.asarray(q, dtype=float)
            if not np.all(np.isfinite(q)):
                continue
            q = (q + np.pi) % (2 * np.pi) - np.pi  # canonical representative in [-pi, pi)
            if not self._matches(q, T):
                continue
            sols.append(q)
        return sols

    def ik(self, pos, quat_xyzw, seed):
        """Exact IK branch closest to seed (choosing each joint's 2pi representative closest to
        the seed within limits), or None if the pose is unreachable."""
        seed = np.asarray(seed, dtype=float)
        best, best_d = None, np.inf
        for q in self.ik_all(pos, quat_xyzw):
            q = self._nearest_representative(q, seed)
            if q is None:
                continue
            d = np.linalg.norm(q - seed)
            if d < best_d:
                best, best_d = q, d
        return best

    def _matches(self, q, T_target):
        T = self.fk_matrix(q)
        if np.linalg.norm(T[:3, 3] - T_target[:3, 3]) > FK_POS_TOL:
            return False
        R_err = T[:3, :3].T @ T_target[:3, :3]
        angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0))
        return angle < FK_ROT_TOL

    @staticmethod
    def _nearest_representative(q, seed):
        q = q + 2 * np.pi * np.round((seed - q) / (2 * np.pi))
        # Nudge back inside limits if the nearest turn lies outside (only elbow's +-pi can bind).
        q = np.where(q > JOINT_UPPER, q - 2 * np.pi, q)
        q = np.where(q < JOINT_LOWER, q + 2 * np.pi, q)
        if np.any(q > JOINT_UPPER) or np.any(q < JOINT_LOWER):
            return None
        return q


def matrix_to_pose(T):
    """4x4 -> (pos, quat_xyzw)."""
    return T[:3, 3].copy(), matrix_to_quat(T[:3, :3])
