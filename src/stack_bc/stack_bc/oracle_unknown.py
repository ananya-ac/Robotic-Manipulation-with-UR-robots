"""oracle_obj_unknown: pick-and-stack from perception only (no ground-truth object poses).

The object model comes from the pre-episode multi-view scan (active_scan.py): fused-cloud object
poses/extents and antipodal grasps from tabletop_perception's node. Ground truth is used only by
collect.py to label success and to record the policy state - never here.

Per stacked object (stack order = random permutation of class names, as oracle_obj_known):
  pick:  grasps for src (best first) -> for each: choose the 180-deg flip closest to the current
         wrist, IK-check grasp + pre-grasp (10 cm back along the approach) -> OMPL to the pre-grasp
         (octomap-checked; next grasp if it fails) -> inside scene.contact(): Pilz LIN descend,
         force-limited close, GATE, LIN lift back to the pre-grasp
  gate:  no ground truth: the measured gap must match the grasp's contact width (+-12 mm) and not
         change during the lift (no slip). Failure: open, retry with the next grasp (2 attempts).
  carry: the held object is attached in the PLANNING SCENE only (scene.attach_object - no physics;
         the sim holds it by contact/friction) at its perceived pose in the hand
  place: target = src's perceived bottom on dst's top face + PLACE_CLEARANCE, yaw aligned to dst's
         footprint symmetry nearest src's current yaw -> OMPL to pre-place -> inside contact():
         LIN descend, open, detach, LIN retreat. src's model pose is updated to the placed pose.

Inherits _move / _set_gripper / _contact from ScriptedStackOracle (unchanged).
"""

import numpy as np

from stack_bc import rotations as rot
from stack_bc import task
from stack_bc.gripper import WIDTH_CLOSED, WIDTH_OPEN
from stack_bc.motion import PlanningFailure
from stack_bc.oracle import OMPL_VEL, V_APPROACH, V_LIFT, OracleFailure, ScriptedStackOracle
from stack_bc.perception_client import Perceived

PREGRASP = 0.10            # m back along the approach
GATE_WIDTH_TOL = 0.012     # m, measured gap vs the grasp's contact width
GATE_SLIP_TOL = 0.003      # m, gap change during the lift
MAX_GRASPS_TRIED = 6       # candidates tried for planning per pick attempt
MAX_PICK_ATTEMPTS = 2


def _flip(T):
    """The same parallel-jaw grasp rotated 180 deg about the approach."""
    return T @ rot.pose_matrix([0, 0, 0], rot.rot_z(np.pi))


def _back(T, d):
    out = T.copy()
    out[:3, 3] -= d * T[:3, 2]
    return out


class PerceptionStackOracle(ScriptedStackOracle):
    def __init__(self, sim, env, motion, gripper, scene, perception, objects):
        super().__init__(sim, env, motion, gripper, scene)
        self.perception = perception
        self.objects = dict(objects)     # name -> Perceived (current belief; updated on place)
        self.T_tip_obj = None

    def run(self, order):
        names = [task.CUBE_NAMES[i] for i in order]
        missing = [n for n in names if n not in self.objects]
        if missing:
            raise OracleFailure(f"not perceived: {missing}")
        self.cmd = self.env.snapshot().ee.copy()
        for k in range(1, len(names)):
            src, dst = names[k], names[k - 1]
            for attempt in range(MAX_PICK_ATTEMPTS):
                if self._pick(src, skip=attempt * MAX_GRASPS_TRIED):
                    break
                if attempt == MAX_PICK_ATTEMPTS - 1:
                    raise OracleFailure(f"grasp of {src} failed {MAX_PICK_ATTEMPTS}x")
            self._place(src, dst)
        self.phase = "done"

    # ---- helpers -------------------------------------------------------------------------------

    def _reachable(self, T, seed):
        q = self.env.kin.ik_world(T, seed)
        return q if q is not None and self.env.kin.ik_world(_back(T, PREGRASP), q) is not None else None

    def _orient(self, T):
        """Pick the 180-deg flip of T whose IK is closest to the current arm (smallest wrist turn)."""
        q_now = self.env.snapshot().q_arm
        best = None
        for Tc in (T, _flip(T)):
            q = self._reachable(Tc, q_now)
            if q is not None and (best is None or np.linalg.norm(q - q_now) < best[0]):
                best = (np.linalg.norm(q - q_now), Tc)
        return None if best is None else best[1]

    # ---- pick / place ------------------------------------------------------------------------

    def _pick(self, src, skip=0):
        self.phase = f"pick_{src}"
        try:
            grasps, msg = self.perception.grasps(src)
        except RuntimeError as exc:      # node unavailable / timed out: fail this episode, not the run
            raise OracleFailure(f"plan_grasps({src}): {exc}") from exc
        print(f"[oracle] {msg}", flush=True)
        if not grasps:
            raise OracleFailure(f"no grasps for {src}: {msg}")
        for g in grasps[skip:skip + MAX_GRASPS_TRIED]:
            T = self._orient(g.T)
            if T is None:
                continue
            try:
                self.motion.ompl(_back(T, PREGRASP), vel_scale=OMPL_VEL, label=f"{self.phase}/transit")
            except PlanningFailure:
                continue
            self.cmd = _back(T, PREGRASP)
            return self._grasp(src, T, g.width)
        raise OracleFailure(f"{src}: none of {len(grasps[skip:skip + MAX_GRASPS_TRIED])} grasps reachable/plannable")

    def _grasp(self, src, T, width):
        with self._contact():
            self._move(T, V_APPROACH, "descend")
            held = self._set_gripper(WIDTH_CLOSED).width
            self._move(_back(T, PREGRASP), V_LIFT, "lift")
            self.sim.sleep(0.3)
            after = self.env.snapshot().width
            ok = abs(held - width) <= GATE_WIDTH_TOL and abs(after - held) <= GATE_SLIP_TOL \
                and held > WIDTH_CLOSED + 0.005
            self.grasps.append(dict(object=src, width=held, expected=width, after_lift=after, ok=ok))
            if not ok:
                self._set_gripper(WIDTH_OPEN)
                return False
        # In-hand pose from what was perceived and commanded - not from the simulator.
        self.T_tip_obj = np.linalg.inv(T) @ self.objects[src].T
        if self.scene is not None:
            self.scene.attach_object(self.objects[src].extent, self.T_tip_obj)
        return True

    def _place(self, src, dst):
        self.phase = f"place_{src}_on_{dst}"
        s, d = self.objects[src], self.objects[dst]
        # Yaw: dst's footprint symmetry (4-fold if square) nearest src's current yaw.
        period = np.pi / 2 if abs(d.extent[0] - d.extent[1]) <= 0.15 * max(d.extent[:2]) else np.pi
        src_yaw, dst_yaw = rot.yaw_of(s.R), rot.yaw_of(d.R)
        yaw = dst_yaw + period * np.round((src_yaw - dst_yaw) / period)
        center = np.array([d.center[0], d.center[1], d.top_z + s.extent[2] / 2])
        T_obj = rot.pose_matrix(center + [0, 0, task.PLACE_CLEARANCE], rot.rot_z(yaw))
        T_place = T_obj @ np.linalg.inv(self.T_tip_obj)
        T_pre = _back(T_place, PREGRASP)
        self.motion.ompl(T_pre, vel_scale=OMPL_VEL, label=f"{self.phase}/transit")
        self.cmd = T_pre
        with self._contact():
            self._move(T_place, V_APPROACH, "descend")
            self._set_gripper(WIDTH_OPEN)
            if self.scene is not None:
                self.scene.detach_object()
            self._move(T_pre, V_LIFT, "retreat")
        self.objects[src] = Perceived(center=center, R=rot.rot_z(yaw), extent=s.extent, top_visible=True)
        self.T_tip_obj = None

    def perceived_state(self):
        """(N_CUBES * 9,) perceived pos + canonical rot6d per cube (NaN if unknown), for logging."""
        out = []
        for name in task.CUBE_NAMES:
            o = self.objects.get(name)
            if o is None:
                out.append(np.full(9, np.nan))
            else:
                out.append(np.concatenate([o.center, rot.matrix_to_rot6d(rot.canonical_cube_rotation(o.R))]))
        return np.concatenate(out).astype(np.float32)
