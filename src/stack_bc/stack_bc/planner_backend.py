"""Offline MoveIt planning from arbitrary joint states - no sim, no robot (the DAgger expert).

The same planners and request helpers as the live oracle (motion.py): Pilz LIN for straight
moves, OMPL (default planner, RRTConnect) for free-space transits, against the ground-truth world
of scene_gt.GTWorld. Plans are returned, never executed.

Start states: taken from the planning scene's current state with the arm joints overwritten. That
state carries the scene's attached bodies (the held cube), which a fresh RobotState would not, so
carry plans are checked with the cube in the hand. (Setting arm joints through the
`joint_positions` dict setter; RobotState.set_joint_group_positions segfaulted in this moveit_py,
see motion.py.)

Verified offline (Step 0 of the DAgger plan): MoveItPy starts without a sim given
moveit_config.moveit_py_params(offline=True), plans start exactly at the requested state, OMPL
median ~30 ms, LIN ~1 ms.
"""

from dataclasses import dataclass

import numpy as np
import rclpy
from moveit.planning import MoveItPy

from stack_bc.kinematics import ARM_JOINTS
from stack_bc.moveit_config import moveit_py_params
from stack_bc.motion import ARM_GROUP, MAX_TRANS_VEL, plan_params, set_pose_goal, trajectory_arrays
from stack_bc.scene_gt import GTWorld


@dataclass
class Plan:
    times: np.ndarray       # (N,) s from the plan's start
    q: np.ndarray           # (N, 6) ARM_JOINTS order

    @property
    def duration(self):
        return float(self.times[-1])

    def q_at(self, t):
        """Joint configuration at plan time t (linear between points, clamped to the ends) -
        the same interpolation as motion.MoveItMotion.desired_q, which labeled the demos."""
        tau = np.clip(t, self.times[0], self.times[-1])
        return np.array([np.interp(tau, self.times, self.q[:, j]) for j in range(self.q.shape[1])])


class OfflinePlanner:
    def __init__(self, urdf, node_name="stack_bc_offline_planner"):
        if not rclpy.ok():
            rclpy.init()
        self.robot = MoveItPy(node_name=node_name, config_dict=moveit_py_params(urdf, offline=True))
        self.arm = self.robot.get_planning_component(ARM_GROUP)
        self.psm = self.robot.get_planning_scene_monitor()
        self.world = GTWorld(self.psm, urdf)

    def _start_state(self, q0):
        with self.psm.read_only() as scene:
            rs = scene.current_state
        jp = rs.joint_positions
        jp.update(zip(ARM_JOINTS, map(float, q0)))
        rs.joint_positions = jp
        rs.update()
        return rs

    def colliding(self, q0):
        """Whether the arm at q0 (with the held cube, if any) collides with the current world."""
        rs = self._start_state(q0)
        with self.psm.read_only() as scene:
            return bool(scene.is_state_colliding(robot_state=rs, joint_model_group_name=ARM_GROUP, verbose=False))

    def _plan(self, q0, T_goal, params):
        self.arm.set_start_state(robot_state=self._start_state(q0))
        set_pose_goal(self.arm, T_goal)
        result = self.arm.plan(single_plan_parameters=params)
        if not result:
            return None
        return Plan(*trajectory_arrays(result))

    def lin(self, q0, T_goal, speed, acc_scale=0.3):
        """Pilz LIN from q0 to T_goal at `speed` m/s (motion.MoveItMotion.lin's request), or None."""
        return self._plan(q0, T_goal, plan_params(self.robot, "LIN", speed / MAX_TRANS_VEL, acc_scale))

    def ompl(self, q0, T_goal, vel_scale, acc_scale=0.3):
        """OMPL from q0 to T_goal (motion.MoveItMotion.ompl's request), or None. Random: repeated
        calls give different paths."""
        return self._plan(q0, T_goal, plan_params(self.robot, "", vel_scale, acc_scale, pipeline="ompl"))
