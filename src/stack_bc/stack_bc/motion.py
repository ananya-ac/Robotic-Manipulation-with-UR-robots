"""MoveIt (moveit_py) motion for the scripted oracle: Pilz LIN for every task move.

Each move is planned as one time-parameterized trajectory and executed by JTC through MoveIt's
FollowJointTrajectory path - smooth, like the earlier mujoco_grasp_test.py setup - instead of
being streamed point-by-point. Pilz LIN gives straight Cartesian lines with trapezoidal velocity
profiles and is deterministic (unlike OMPL), so repeated demos of the same situation look the same.

The planned trajectory of the move in progress (with its sim-time start) is kept so the recorder
can log the DESIRED pose at any tick (paper Sec. 7.1: actions are setpoints, not measured states).

moveit_py setup (config builder, use_sim_time, /clock QoS overrides, gripper-server settle) is
taken from ur_control_examples/mujoco_grasp_test.py, where each piece is explained.

Homing between episodes is NOT done here (it is never recorded): EEExecutor.move_joints sends a
plain joint trajectory. A Pilz PTP to a RobotState goal segfaulted inside moveit_py's
RobotState.set_joint_group_positions binding, even with the robot model kept alive.
"""

import os
import tempfile
import threading

import numpy as np
from ament_index_python.packages import get_package_share_directory
from moveit.core.kinematic_constraints import construct_link_constraint
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit_configs_utils import MoveItConfigsBuilder

from stack_bc import rotations as rot
from stack_bc.kinematics import ARM_JOINTS

ARM_GROUP = "ur_manipulator"
TIP_LINK = "gripper_tip_link"
MAX_TRANS_VEL = 1.0      # pilz_cartesian_limits.yaml max_trans_vel; LIN speed = this * scaling


class PlanningFailure(RuntimeError):
    pass


class MoveItMotion:
    def __init__(self, robot_description: str, sim_now):
        """robot_description: the live URDF string. sim_now: callable -> current sim time (s)."""
        self._sim_now = sim_now
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False)
        tmp.write(robot_description)
        tmp.close()
        share = get_package_share_directory("ur_gripper_sim_moveit_config")
        cfg = os.path.join(share, "config")
        moveit_config = (
            MoveItConfigsBuilder(robot_name="ur", package_name="ur_gripper_sim_moveit_config")
            .robot_description(file_path=tmp.name)
            .robot_description_semantic(file_path=os.path.join(share, "srdf", "ur_gripper_sim.srdf.xacro"),
                                        mappings={"name": "ur5e", "gripper": "robotiq_2f85", "load_gripper": "true"})
            .robot_description_kinematics(file_path=os.path.join(cfg, "kinematics.yaml"))
            .joint_limits(file_path=os.path.join(cfg, "joint_limits.yaml"))
            .trajectory_execution(file_path=os.path.join(cfg, "moveit_controllers_robotiq_2f85.yaml"))
            .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"], default_planning_pipeline="ompl")
            .pilz_cartesian_limits(file_path=os.path.join(cfg, "pilz_cartesian_limits.yaml"))
            .moveit_cpp(file_path=os.path.join(cfg, "moveit_cpp.yaml"))
            .to_moveit_configs()
        )
        config = moveit_config.to_dict()
        config["use_sim_time"] = True
        config["qos_overrides"] = {"/clock": {"subscription": {
            "reliability": "best_effort", "durability": "volatile", "history": "keep_last", "depth": 1}}}
        self.robot = MoveItPy(node_name="stack_bc_moveit", config_dict=config)
        os.unlink(tmp.name)
        self.arm = self.robot.get_planning_component(ARM_GROUP)

        self._lock = threading.Lock()
        self._active = None   # (t0_sim, times (N,), positions (N, 6)) of the executing/last move

    # ---- planning + execution ----------------------------------------------------------------

    def _params(self, planner_id, vel_scale, acc_scale):
        p = PlanRequestParameters(self.robot, "plan_request_params")
        p.planning_pipeline = "pilz_industrial_motion_planner"
        p.planner_id = planner_id
        p.max_velocity_scaling_factor = float(vel_scale)
        p.max_acceleration_scaling_factor = float(acc_scale)
        return p

    def _plan_and_execute(self, params, label):
        self.arm.set_start_state_to_current_state()
        result = self.arm.plan(single_plan_parameters=params)
        if not result:
            raise PlanningFailure(f"{label}: Pilz {params.planner_id} planning failed")
        jt = result.trajectory.get_robot_trajectory_msg().joint_trajectory
        idx = [jt.joint_names.index(n) for n in ARM_JOINTS]
        times = np.array([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in jt.points])
        positions = np.array([[p.positions[i] for i in idx] for p in jt.points])
        with self._lock:
            self._active = (self._sim_now(), times, positions)
        status = self.robot.execute(result.trajectory, controllers=[])  # blocks until JTC reports done
        status = str(getattr(status, "status", status))  # ExecutionStatus -> "SUCCEEDED", "ABORTED", ...
        self.last_status = (label, status)
        if status != "SUCCEEDED":
            print(f"[motion] {label}: execution {status}", flush=True)

    def lin(self, T_world, speed, acc_scale=0.3, label="lin"):
        """Straight-line move of gripper_tip_link to T_world (4x4, world frame) at `speed` m/s."""
        pos, quat = T_world[:3, 3], rot.matrix_to_quat(T_world[:3, :3])
        self.arm.set_goal_state(motion_plan_constraints=[construct_link_constraint(
            TIP_LINK, "world", cartesian_position=list(map(float, pos)), cartesian_position_tolerance=0.001,
            orientation=list(map(float, quat)), orientation_tolerance=0.01)])
        self._plan_and_execute(self._params("LIN", speed / MAX_TRANS_VEL, acc_scale), label)

    # ---- desired state for the recorder ------------------------------------------------------

    def desired_q(self, t_sim):
        """Planned arm configuration at sim time t_sim (linear between trajectory points; holds the
        last point after the move ends). None before the first move."""
        with self._lock:
            active = self._active
        if active is None:
            return None
        t0, times, positions = active
        tau = np.clip(t_sim - t0, times[0], times[-1])
        return np.array([np.interp(tau, times, positions[:, j]) for j in range(positions.shape[1])])

    def forget(self):
        """Drop the stored trajectory (after a reset teleports the robot)."""
        with self._lock:
            self._active = None

    def shutdown(self):
        self.robot.shutdown()
