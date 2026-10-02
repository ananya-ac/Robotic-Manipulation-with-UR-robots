"""Scripted pick-and-stack expert, executed through MoveIt (Pilz LIN for every move).

Runs sequentially (plan -> execute -> check -> next move) while the recorder samples a 10 Hz
stream in its own thread. Every move is a straight top-down line (tip z -> world -z) between
waypoints inside the IK-feasible set the feasibility map checked. Uses privileged ground-truth
cube poses (sim only), re-read at each decision point so a nudged cube is re-targeted.

Per cube placed on the stack (stack order = a random permutation; order[0] is the base):
  rise/transit at a safe height -> hover over the cube -> slow descend -> close (width 0, force-
  limited) -> GRASP GATE -> lift -> transit at max(pick hover, place hover) -> hover over the stack
  -> slow descend to contact + clearance -> open -> retreat up.

Grasp gate (never log a failed grasp as expert data): after the gripper stops closing, the measured
fingertip gap must be consistent with a cube between the pads, and a 1.5 cm test lift must bring
the cube up with the tip. On failure: open, back off, retry once from the cube's current pose;
a second failure ends the episode as failed.
"""

from contextlib import nullcontext

import numpy as np

from stack_bc import rotations as rot
from stack_bc import task
from stack_bc.gripper import WIDTH_CLOSED, WIDTH_OPEN

V_FREE = 0.20          # m/s, free-space moves (paper's UR5 station caps at 0.43)
V_APPROACH = 0.05      # m/s, final descend onto a cube / the stack
V_LIFT = 0.10          # m/s
MIN_MOVE = 0.002       # m; skip moves shorter than this (no rotation change either)
TEST_LIFT = 0.015      # m
GRASP_WIDTH_RANGE = (task.CUBE_SIZE - 0.012, task.CUBE_SIZE + 0.004)  # measured gap holding a cube
GRIPPER_SETTLE = 0.3   # s of sim time with |d width| < 0.5 mm per 0.1 s -> gripper stopped
GRIPPER_TIMEOUT = 2.0  # s
MAX_GRASP_ATTEMPTS = 2


class OracleFailure(RuntimeError):
    pass


def canonical_grasp_yaw(cube_R, reference_yaw):
    """Grasp yaw aligned with the cube's faces (4-fold symmetric) closest to reference_yaw."""
    return reference_yaw + rot.wrap_to_quarter_turn(rot.yaw_of(cube_R) - reference_yaw)


class ScriptedStackOracle:
    def __init__(self, sim, env, motion, gripper, scene=None):
        """gripper: an EEExecutor (only its set_width is used - force-limited GripperCommand).
        scene: a scene_sensing.SceneManager when planning against the sensor-built octomap."""
        self.sim, self.env, self.motion, self.gripper = sim, env, motion, gripper
        self.scene = scene
        self.phase = "start"
        self.cmd = None       # last commanded tip pose (4x4 world)
        self.grasps = []

    def run(self, order):
        """Stack cubes in `order` (base first). Raises OracleFailure; sets phase 'done' on success."""
        self.cmd = self.env.snapshot().ee.copy()
        for k in range(1, len(order)):
            src, dst = order[k], order[k - 1]
            for attempt in range(MAX_GRASP_ATTEMPTS):
                if self._pick(src):
                    break
                if attempt == MAX_GRASP_ATTEMPTS - 1:
                    raise OracleFailure(f"grasp of cube {src} failed {MAX_GRASP_ATTEMPTS}x")
            self._place(src, dst, level=k)
        self.phase = "done"

    # ---- primitives --------------------------------------------------------------------------

    def _move(self, T_goal, speed, label):
        dist = np.linalg.norm(T_goal[:3, 3] - self.cmd[:3, 3])
        if dist < MIN_MOVE and rot.angle_between(T_goal[:3, :3], self.cmd[:3, :3]) < 1e-3:
            return
        self.motion.lin(T_goal, speed, label=f"{self.phase}/{label}")
        self.cmd = T_goal.copy()

    def _transit(self, T_goal, via_z, contact_rise=False):
        """Rise to via_z, move horizontally (rotating) to above the goal, descend to the goal.
        contact_rise: the vertical rise starts in contact (carried cube still overlapping the
        voxels of the spot it was lifted from), so plan it with octomap contact allowed."""
        up = self.cmd.copy()
        up[2, 3] = max(via_z, self.cmd[2, 3])
        with self._contact() if contact_rise else nullcontext():
            self._move(up, V_LIFT, "rise")
        over = T_goal.copy()
        over[2, 3] = up[2, 3]
        self._move(over, V_FREE, "over")
        self._move(T_goal, V_FREE, "down")

    def _contact(self):
        """Grasp/place segments: the octomap holds the target cube / stack itself, so these short
        vertical moves may touch it. Transits outside this block stay fully collision-checked."""
        return self.scene.contact() if self.scene is not None else nullcontext()

    def _set_gripper(self, width):
        """Command a width and wait (sim time) until the measured gap stops moving."""
        self.gripper.set_width(width)
        history, t_end = [], self.sim.now() + GRIPPER_TIMEOUT
        n_settle = int(round(GRIPPER_SETTLE / 0.1))
        while self.sim.now() < t_end:
            self.sim.sleep(0.1)
            history.append(self.env.snapshot().width)
            if len(history) > n_settle and np.all(np.abs(np.diff(history[-n_settle - 1:])) < 0.0005):
                break
        return self.env.snapshot()

    # ---- pick / place ------------------------------------------------------------------------

    def _pick(self, src):
        self.phase = f"pick{src}"
        snap = self.env.snapshot()
        yaw = canonical_grasp_yaw(snap.cube_rot[src], rot.yaw_of(self.cmd[:3, :3]))
        R_grasp = rot.top_down_grasp(yaw)
        hover = rot.pose_matrix(snap.cube_pos[src] + [0, 0, task.HOVER_HEIGHT], R_grasp)
        self._transit(hover, hover[2, 3])

        with self._contact():
            cube_p = self.env.snapshot().cube_pos[src]  # re-read: anything nudged during the approach
            self._move(rot.pose_matrix(cube_p, R_grasp), V_APPROACH, "descend")
            held_width = self._set_gripper(WIDTH_CLOSED).width

            before = self.env.snapshot()
            test = self.cmd.copy()
            test[2, 3] += TEST_LIFT
            self._move(test, V_LIFT, "test_lift")
            self.sim.sleep(0.3)
            after = self.env.snapshot()
            # Compare the cube's rise to the TIP's actual rise, not the commanded TEST_LIFT: closing
            # the fingers pushes the tip up a few mm (seen: 4 mm), so the tip itself only rises ~12 mm
            # of the commanded 15 and a held cube follows it almost exactly.
            tip_rise = after.ee[2, 3] - before.ee[2, 3]
            lifted = after.cube_pos[src][2] - before.cube_pos[src][2]
            ok = (GRASP_WIDTH_RANGE[0] <= held_width <= GRASP_WIDTH_RANGE[1]) and \
                lifted > max(0.005, 0.7 * tip_rise)
            self.grasps.append(dict(cube=src, width=held_width, lifted=lifted, tip_rise=tip_rise, ok=ok))
            if not ok:
                self._set_gripper(WIDTH_OPEN)
                back = self.cmd.copy()
                back[2, 3] = self.env.snapshot().cube_pos[src][2] + task.HOVER_HEIGHT
                self._move(back, V_LIFT, "back_off")
        if ok and self.scene is not None:
            # Nominal grasp (tip at the cube center, faces aligned with the grasp yaw), not the
            # simulator's cube pose: this is what the robot itself knows about what it holds.
            self.scene.attach_cube(np.eye(4))
        return ok

    def _place(self, src, dst, level):
        self.phase = f"place{src}->{dst}"
        snap = self.env.snapshot()
        # Carried cube's pose relative to the tip, measured now (it may have settled in the grip).
        T_tip_cube = np.linalg.inv(snap.ee) @ rot.pose_matrix(snap.cube_pos[src], snap.cube_rot[src])
        base_p, base_R = snap.cube_pos[dst], snap.cube_rot[dst]
        # Desired carried-cube pose: centered over the stack, faces aligned with the cube below
        # (the aligned yaw nearest the carried cube's current yaw -> smallest wrist turn).
        carried_R = snap.cube_rot[src]
        cube_yaw_goal = canonical_grasp_yaw(base_R, rot.yaw_of(carried_R))
        R_cube_goal = rot.rot_z(cube_yaw_goal - rot.yaw_of(carried_R)) @ carried_R
        target_center_z = task.CUBE_REST_Z + level * task.CUBE_SIZE
        T_cube_goal = rot.pose_matrix([base_p[0], base_p[1], target_center_z + task.PLACE_CLEARANCE], R_cube_goal)
        T_place = T_cube_goal @ np.linalg.inv(T_tip_cube)
        T_place[:3, :3] = rot.top_down_grasp(rot.yaw_of(T_place[:3, :3]))  # keep the tip exactly vertical
        hover = T_place.copy()
        hover[2, 3] += task.HOVER_HEIGHT
        # Carry at max(pick hover, place hover): cmd is at the test-lift height (grasp + TEST_LIFT).
        via_z = max(hover[2, 3], self.cmd[2, 3] + (task.HOVER_HEIGHT - TEST_LIFT))
        self._transit(hover, via_z, contact_rise=True)

        with self._contact():
            self._move(T_place, V_APPROACH, "descend")
            self._set_gripper(WIDTH_OPEN)
            if self.scene is not None:
                self.scene.detach_cube()
            up = self.cmd.copy()
            up[2, 3] += task.HOVER_HEIGHT
            self._move(up, V_LIFT, "retreat")
