"""Rotation conversions (numpy only). Quaternions are xyzw throughout, matching ROS/PyKDL.

The 6D representation (Zhou et al. 2019) matches pytorch3d's matrix_to_rotation_6d /
rotation_6d_to_matrix exactly - the functions diffusion_policy's RotationTransformer calls - so
data written here and actions decoded at eval time use the same convention: rot6d is the first
two ROWS of the rotation matrix, and decoding Gram-Schmidts them back into rows.
"""

import numpy as np


def quat_to_matrix(q_xyzw):
    x, y, z, w = np.asarray(q_xyzw, dtype=float) / np.linalg.norm(q_xyzw)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R):
    """3x3 rotation -> xyzw quaternion with w >= 0."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    return q if q[3] >= 0 else -q


def matrix_to_rot6d(R):
    """(..., 3, 3) -> (..., 6): first two rows, as pytorch3d.matrix_to_rotation_6d."""
    R = np.asarray(R, dtype=float)
    return R[..., :2, :].reshape(*R.shape[:-2], 6)


def rot6d_to_matrix(d6):
    """(..., 6) -> (..., 3, 3), as pytorch3d.rotation_6d_to_matrix (Gram-Schmidt on rows)."""
    d6 = np.asarray(d6, dtype=float)
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def _cube_symmetries():
    """The 24 proper rotations mapping a cube onto itself (signed permutation matrices, det +1)."""
    import itertools
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            M[range(3), perm] = signs
            if np.isclose(np.linalg.det(M), 1.0):
                mats.append(M)
    return np.array(mats)


CUBE_SYMMETRIES = _cube_symmetries()


def canonical_cube_rotation(R):
    """Among R's 24 physically identical cube orientations (R @ S), the one closest to identity,
    so identical cube states always give the same observation. For an upright cube this is the
    yaw representative in (-pi/4, pi/4]."""
    cands = R @ CUBE_SYMMETRIES
    return cands[np.argmax(np.trace(cands, axis1=1, axis2=2))]


def slerp(R0, R1, a):
    """Spherical interpolation between two rotation matrices, a in [0, 1]."""
    q0, q1 = matrix_to_quat(R0), matrix_to_quat(R1)
    d = np.dot(q0, q1)
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
    else:
        th = np.arccos(d)
        q = np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1
    return quat_to_matrix(q / np.linalg.norm(q))


def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def top_down_grasp(yaw):
    """World-frame gripper_tip_link orientation pointing straight down (tip z -> world -z), with
    the tip x axis at `yaw` in the world xy plane. Same family as mujoco_grasp_test.py's validated
    vertical approach, quat (1, 0, 0, 0) in base_link (= yaw +pi/2 here, since base_link is +90
    deg about z from world). A parallel-jaw grasp is symmetric under yaw + pi."""
    return rot_z(yaw) @ np.diag([1.0, -1.0, -1.0])


def yaw_of(R):
    """Yaw of a rotation's x axis projected onto the world xy plane."""
    return np.arctan2(R[1, 0], R[0, 0])


def wrap_to_quarter_turn(angle):
    """Map an angle onto the (-pi/4, pi/4] representative mod pi/2 (a cube's 4-fold yaw symmetry)."""
    return (angle + np.pi / 4) % (np.pi / 2) - np.pi / 4


def angle_between(R1, R2):
    return float(np.arccos(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1.0, 1.0)))


def pose_matrix(pos, R):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T
