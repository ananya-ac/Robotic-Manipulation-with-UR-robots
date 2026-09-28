"""EEExecutor: absolute end-effector targets -> arm + gripper commands.

The ONE path by which both the scripted oracle and the learned policy move the robot, so the
logged action space is exactly what gets executed at eval time. Mirrors the Diffusion Policy
paper's UR5 station (App. D.0.1): desired EE poses at 10 Hz, linearly interpolated by the
low-level controller, with a speed cap.

Arm: IK (EAIK, seeded along the plan from the previous commanded configuration so the branch
stays continuous) -> JointTrajectory on scaled_joint_trajectory_controller's topic interface.
Each tick sends this tick's target plus a lookahead of future targets (the oracle's remaining
segment, or the policy's predicted action chunk) with joint velocities, so JTC tracks one smooth
spline; see send_plan.

Gripper: desired width -> knuckle target on gripper_controller's GripperCommand action,
force-limited (max_effort) so "closed" (width 0) stalls on the object with a real squeeze.
"""

import numpy as np
from builtin_interfaces.msg import Duration
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from stack_bc.gripper import width_to_knuckle
from stack_bc.kinematics import ARM_JOINTS

TRAJ_TOPIC = "/scaled_joint_trajectory_controller/joint_trajectory"
GRIPPER_ACTION = "/gripper_controller/gripper_cmd"

MAX_JOINT_STEP = 0.35         # rad per 0.1 s tick (~200 deg/s) - larger means an IK branch flip
LOOKAHEAD = 10                # plan points sent per tick (1 s at 10 Hz)
MAX_EE_SPEED = 0.43           # m/s, as the paper's UR5 interpolation controller (App. D.0.1)
GRIPPER_MAX_EFFORT = 50.0
GRIPPER_RESEND_TOL = 0.002    # m; only send a new gripper goal when the target moves this much


class IKFailure(RuntimeError):
    pass


def _duration(t):
    return Duration(sec=int(t), nanosec=int(round((t - int(t)) * 1e9)))


class EEExecutor:
    def __init__(self, node, kin, dt=0.1):
        self.node = node
        self.kin = kin
        self.dt = dt
        self._traj_pub = node.create_publisher(JointTrajectory, TRAJ_TOPIC, 10)
        self._gripper = ActionClient(node, GripperCommand, GRIPPER_ACTION)
        self.q_cmd = None             # last commanded arm configuration (IK seed)
        self.ee_cmd = None            # last commanded EE pose (4x4 world)
        self.width_cmd = None

    def sync(self, q_arm, ee):
        """Re-seed from the measured state (call after reset or any externally driven motion)."""
        self.q_cmd = np.asarray(q_arm, dtype=float).copy()
        self.ee_cmd = ee.copy()

    def solve(self, T_world, seed):
        q = self.kin.ik_world(T_world, seed)
        if q is None:
            raise IKFailure(f"no IK for target at {np.round(T_world[:3, 3], 3)}")
        if np.max(np.abs(q - seed)) > MAX_JOINT_STEP:
            raise IKFailure(f"IK branch jump {np.round(q - seed, 2)} at {np.round(T_world[:3, 3], 3)}")
        return q

    def send_plan(self, targets, width, stops=True, horizon=LOOKAHEAD):
        """Command this tick's target (targets[0], reached dt from now) plus up to horizon-1 future
        targets as lookahead, all dt apart, as ONE trajectory with positions AND joint velocities.

        JTC then follows a continuous spline through the plan instead of easing to a stop at every
        10 Hz point (which is what a positions-only, one-point-per-tick stream does - visibly
        jerky). The plan is re-sent every tick, so only targets[0] is committed (it is this tick's
        logged action); the rest is replaced before it is reached. stops=True means the motion
        genuinely ends at targets[-1] (zero velocity there); False for a truncated/receding plan.
        Returns the (speed-clamped) target actually commanded for this tick."""
        plan = list(targets[:horizon])
        ends_here = stops and len(targets) <= horizon
        Ts, qs = [], [self.q_cmd]
        prev_T = self.ee_cmd
        for T in plan:
            T = self._speed_clamp(T, prev_T)
            qs.append(self.solve(T, qs[-1]))
            Ts.append(T)
            prev_T = T
        points = []
        for k in range(1, len(qs)):
            if k < len(qs) - 1:
                v = (qs[k + 1] - qs[k - 1]) / (2 * self.dt)
            elif ends_here:
                v = np.zeros(len(ARM_JOINTS))
            else:
                v = (qs[k] - qs[k - 1]) / self.dt
            points.append(JointTrajectoryPoint(positions=qs[k].tolist(), velocities=v.tolist(),
                                               time_from_start=_duration(k * self.dt)))
        self._traj_pub.publish(JointTrajectory(joint_names=ARM_JOINTS, points=points))  # zero stamp = now
        self.q_cmd, self.ee_cmd = qs[1], Ts[0]
        self.set_width(width)
        return Ts[0]

    def send(self, T_world, width):
        """Single target that the motion stops at (e.g. holding still)."""
        return self.send_plan([T_world], width, stops=True)

    def move_joints(self, q_target, duration):
        """Plain joint-space move (resets/homing only - not a logged action)."""
        msg = JointTrajectory(joint_names=ARM_JOINTS, points=[JointTrajectoryPoint(
            positions=list(map(float, q_target)), time_from_start=_duration(duration))])
        self._traj_pub.publish(msg)
        self.q_cmd = np.asarray(q_target, dtype=float).copy()
        self.ee_cmd = self.kin.fk_world(self.q_cmd)

    def set_width(self, width, force=False):
        width = float(width)
        if not force and self.width_cmd is not None and abs(width - self.width_cmd) < GRIPPER_RESEND_TOL:
            return
        goal = GripperCommand.Goal()
        goal.command.position = width_to_knuckle(width)
        goal.command.max_effort = GRIPPER_MAX_EFFORT
        self._gripper.send_goal_async(goal)  # fire-and-forget: progress is observed via joint_states
        self.width_cmd = width

    def wait_for_servers(self, timeout=10.0):
        if not self._gripper.wait_for_server(timeout_sec=timeout):
            raise RuntimeError(f"{GRIPPER_ACTION} unavailable")

    def _speed_clamp(self, T, prev_T):
        """Limit the step from prev_T to MAX_EE_SPEED * dt (a safety net for policy outputs; the
        oracle's plans never hit it)."""
        if prev_T is None:
            return T
        step = T[:3, 3] - prev_T[:3, 3]
        max_step = MAX_EE_SPEED * self.dt
        n = np.linalg.norm(step)
        if n <= max_step:
            return T
        T = T.copy()
        T[:3, 3] = prev_T[:3, 3] + step * (max_step / n)
        return T
