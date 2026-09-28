"""Robotiq 2F-85 width <-> joint conversions.

"Width" is the gap between the two fingertip collision pads (MuJoCo mj_geomDistance), which reads
85.0 mm fully open, matching the real 2F-85's stroke.

- measured width: FK on ALL six gripper joints as reported on /joint_states. The knuckle angle
  alone is a poor proxy here: the mimic linkage is equality-constrained and soft under load (the
  passive joints deviate 15-29% from their mimic ratio while squeezing), e.g. the knuckle stalls
  near 0.66 rad on a 45 mm cube, which the ideal linkage would call ~16 mm.
- commanded width -> knuckle target: the ideal-linkage table below (commands are targets for a
  force-limited position controller; closing on an object simply stalls short of the target).
"""

import mujoco
import numpy as np

GRIPPER_JOINTS = ["robotiq_85_left_knuckle_joint", "robotiq_85_left_finger_tip_joint",
                  "robotiq_85_right_knuckle_joint", "robotiq_85_right_finger_tip_joint",
                  "robotiq_85_left_inner_knuckle_joint", "robotiq_85_right_inner_knuckle_joint"]
KNUCKLE_JOINT = GRIPPER_JOINTS[0]
KNUCKLE_MAX = 0.8

# Ideal-linkage calibration (passive joints set exactly by their mimic multipliers), measured
# offline with mj_geomDistance between the fingertip collision pads.
_KNUCKLE_TABLE = np.array([0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])
_WIDTH_TABLE = np.array([85.0, 76.0, 66.3, 56.0, 45.3, 34.3, 23.0, 11.6, 0.2]) / 1000.0
WIDTH_OPEN = float(_WIDTH_TABLE[0])
WIDTH_CLOSED = 0.0


def width_to_knuckle(width):
    w = float(np.clip(width, _WIDTH_TABLE[-1], _WIDTH_TABLE[0]))
    return float(np.interp(w, _WIDTH_TABLE[::-1], _KNUCKLE_TABLE[::-1]))


class WidthGauge:
    """Measured fingertip gap from the six gripper joint positions (FK in a private MjData)."""

    def __init__(self, model: mujoco.MjModel):
        self.m = model
        self.d = mujoco.MjData(model)
        self.qadr = [model.jnt_qposadr[model.joint(n).id] for n in GRIPPER_JOINTS]
        left = model.body("robotiq_85_left_finger_tip_link").id
        right = model.body("robotiq_85_right_finger_tip_link").id
        is_col = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        self.left = [g for g in range(model.ngeom) if model.geom_bodyid[g] == left and is_col[g]]
        self.right = [g for g in range(model.ngeom) if model.geom_bodyid[g] == right and is_col[g]]
        assert self.left and self.right, "fingertip collision geoms not found"

    def __call__(self, q_gripper):
        """q_gripper: positions in GRIPPER_JOINTS order."""
        self.d.qpos[self.qadr] = q_gripper
        mujoco.mj_kinematics(self.m, self.d)
        return max(0.0, min(mujoco.mj_geomDistance(self.m, self.d, a, b, 0.2, None)
                            for a in self.left for b in self.right))
