"""Observation / action vector layouts - the single definition shared by recorder and eval.

state  (37) = EE pos (3) + EE rot6d (6) + measured gripper width (1)
              + for each cube in task.CUBE_NAMES order: pos (3) + canonical rot6d (6)
action (10) = desired EE pos (3) + desired EE rot6d (6) + desired gripper width (1)

"Desired" follows the Diffusion Policy paper (Sec. 7.1, App. D.0.1): actions are setpoints, not
measured next states. rot6d = first two rows of the rotation matrix (pytorch3d convention, see
rotations.py); leave those dims un-normalized at training time (paper App. A.1).
"""

import numpy as np

from stack_bc import rotations as rot
from stack_bc import task

STATE_DIM = 10 + 9 * task.N_CUBES
ACTION_DIM = 10
STATE_ROT6D_SLICES = [slice(3, 9)] + [slice(10 + 9 * i + 3, 10 + 9 * i + 9) for i in range(task.N_CUBES)]
ACTION_ROT6D_SLICES = [slice(3, 9)]


def state_vector(snap):
    parts = [snap.ee[:3, 3], rot.matrix_to_rot6d(snap.ee[:3, :3]), [snap.width]]
    for p, R in zip(snap.cube_pos, snap.cube_rot):
        parts += [p, rot.matrix_to_rot6d(rot.canonical_cube_rotation(R))]
    return np.concatenate(parts).astype(np.float32)


def action_vector(T_world, width):
    return np.concatenate([T_world[:3, 3], rot.matrix_to_rot6d(T_world[:3, :3]), [width]]).astype(np.float32)


def action_to_target(a):
    """Inverse of action_vector: (4x4 world EE target, width). Re-orthonormalizes rot6d."""
    a = np.asarray(a, dtype=float)
    return rot.pose_matrix(a[:3], rot.rot6d_to_matrix(a[3:9])), float(a[9])
