"""Live interface to the MuJoCo stacking sim (robot_mujoco.launch.py must be running).

StackEnv caches the latest /joint_states and cube poses, exposes a snapshot of everything the
recorder and oracle need (world-frame EE pose, measured gripper width, cube poses, full robot
joint state), resets episodes to randomized cube placements, and checks stack success.

Everything is in sim time, taken from the /joint_states header stamps: StackEnv feeds each stamp
into the runtime.StampClock it is given, and callers tick off that clock - never wall time.
"""

import re
import threading
from dataclasses import dataclass

import mujoco
import numpy as np
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from mujoco_ros2_control_msgs.msg import FreeJointState
from mujoco_ros2_control_msgs.msg import FreeJointStateArray
from mujoco_ros2_control_msgs.srv import ResetWorld

from stack_bc import rotations as rot
from stack_bc import task
from stack_bc.gripper import GRIPPER_JOINTS, WidthGauge
from stack_bc.kinematics import ARM_JOINTS, ArmKinematics

CUBE_TOPIC = "/free_joint_state_publisher/free_joint_states"
RESET_SERVICE = "/mujoco_ros2_control_node/reset_world"


@dataclass
class Snapshot:
    stamp: float               # sim time (s) of the newer of the two source messages
    q_arm: np.ndarray          # (6,) ARM_JOINTS order
    q_gripper: np.ndarray      # (6,) GRIPPER_JOINTS order
    ee: np.ndarray             # (4, 4) world-frame gripper_tip_link pose
    width: float               # measured fingertip gap (m)
    cube_pos: np.ndarray       # (N_CUBES, 3) world, task.CUBE_NAMES order
    cube_rot: np.ndarray       # (N_CUBES, 3, 3)


def wait_for_topic(node, topic, msg_type, qos, timeout=15.0):
    event, holder = threading.Event(), {}

    def cb(msg):
        holder["msg"] = msg
        event.set()
    sub = node.create_subscription(msg_type, topic, cb, qos)
    event.wait(timeout)
    node.destroy_subscription(sub)
    return holder.get("msg")


class StackEnv:
    def __init__(self, node, clock):
        """node must be spun by an executor in another thread; clock is a runtime.StampClock."""
        self.node = node
        self.clock = clock
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST)
        rd = wait_for_topic(node, "/robot_description", String, latched)
        if rd is None:
            raise RuntimeError("no /robot_description - is robot_mujoco.launch.py running?")
        self.urdf = rd.data
        self.kin = ArmKinematics(rd.data)
        # Load the exact MJCF the live sim compiled (the launch file bakes its path into the URDF).
        match = re.search(r'<param name="mujoco_model">([^<]+)</param>', rd.data)
        if match is None:
            raise RuntimeError("mujoco_model param not found in /robot_description")
        self.model = mujoco.MjModel.from_xml_path(match.group(1).strip())
        self.width_gauge = WidthGauge(self.model)

        self._lock = threading.Lock()
        self._joints, self._joints_stamp = None, None
        self._cubes, self._cubes_stamp = None, None
        node.create_subscription(JointState, "/joint_states", self._on_joints, 10)
        node.create_subscription(FreeJointStateArray, CUBE_TOPIC, self._on_cubes, 10)
        self._reset_client = node.create_client(ResetWorld, RESET_SERVICE)

    # ---- state -------------------------------------------------------------------------------

    def _on_joints(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self._lock:
            self._joints = dict(zip(msg.name, msg.position))
            self._joints_stamp = stamp
        self.clock.update(stamp)

    def _on_cubes(self, msg):
        poses = {}
        for fj in msg.free_joints:
            p, o = fj.pose.pose.position, fj.pose.pose.orientation
            poses[fj.name] = (np.array([p.x, p.y, p.z]), rot.quat_to_matrix([o.x, o.y, o.z, o.w]))
        with self._lock:
            self._cubes = poses
            self._cubes_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def ready(self):
        with self._lock:
            return self._joints is not None and self._cubes is not None

    def snapshot(self) -> Snapshot:
        with self._lock:
            joints, cubes = self._joints, self._cubes
            stamp = max(self._joints_stamp, self._cubes_stamp)
        q_arm = np.array([joints[n] for n in ARM_JOINTS])
        q_grip = np.array([joints[n] for n in GRIPPER_JOINTS])
        return Snapshot(
            stamp=stamp, q_arm=q_arm, q_gripper=q_grip, ee=self.kin.fk_world(q_arm),
            width=self.width_gauge(q_grip),
            cube_pos=np.stack([cubes[n][0] for n in task.CUBE_NAMES]),
            cube_rot=np.stack([cubes[n][1] for n in task.CUBE_NAMES]))

    # ---- episodes ----------------------------------------------------------------------------

    @staticmethod
    def sample_cubes(rng):
        """Non-overlapping cube xy + yaw inside the IK-feasible spawn region."""
        for _ in range(1000):
            xy = np.column_stack([rng.uniform(*task.SPAWN_X, task.N_CUBES),
                                  rng.uniform(*task.SPAWN_Y, task.N_CUBES)])
            dists = np.linalg.norm(xy[:, None] - xy[None], axis=-1)[np.triu_indices(task.N_CUBES, 1)]
            if np.all(dists >= task.MIN_CUBE_SPACING):
                return xy, rng.uniform(-np.pi, np.pi, task.N_CUBES)
        raise RuntimeError("could not place cubes; spawn region too small for MIN_CUBE_SPACING")

    def reset_world(self, cube_xy, cube_yaw, timeout=10.0):
        """Atomic reset: robot to the "home" keyframe (arm + gripper open), cubes to the given
        poses at rest on the table, all velocities zero. The caller must already have commanded
        the arm controller to home, or JTC will drag the arm back to its last setpoint."""
        req = ResetWorld.Request()
        req.keyframe = "home"
        for name, (x, y), yaw in zip(task.CUBE_NAMES, cube_xy, cube_yaw):
            fj = FreeJointState()
            fj.name = name
            fj.pose.pose.position.x, fj.pose.pose.position.y = float(x), float(y)
            fj.pose.pose.position.z = float(task.CUBE_REST_Z)
            qx, qy, qz, qw = rot.matrix_to_quat(rot.rot_z(yaw))
            o = fj.pose.pose.orientation
            o.x, o.y, o.z, o.w = float(qx), float(qy), float(qz), float(qw)
            req.state_overrides.free_joints.append(fj)
        if not self._reset_client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"{RESET_SERVICE} unavailable")
        done = threading.Event()
        future = self._reset_client.call_async(req)
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout) or not future.result().success:
            raise RuntimeError(f"reset_world failed: {future.result().message if future.done() else 'timeout'}")


def stack_success(snap: Snapshot, order, xy_tol=0.01, z_tol=0.005, tilt_tol=np.deg2rad(10)):
    """True when cube order[k] sits on order[k-1] (k >= 1) within tolerance, base on the table.

    Tolerances: xy offset from the cube below, center height vs the ideal stacked height, and
    tilt of each cube's z axis from vertical. Stability over time is the caller's job (require
    this to hold for several consecutive ticks)."""
    base = order[0]
    if abs(snap.cube_pos[base, 2] - task.CUBE_REST_Z) > z_tol:
        return False
    for k in range(len(order)):
        # A cube resting on any face counts as upright: tilt of whichever body axis is closest
        # to world z (row 2 of R holds each body axis's world-z component).
        tilt = np.arccos(np.clip(np.max(np.abs(snap.cube_rot[order[k]][2, :])), -1, 1))
        if tilt > tilt_tol:
            return False
        if k == 0:
            continue
        below, cube = snap.cube_pos[order[k - 1]], snap.cube_pos[order[k]]
        if np.linalg.norm(cube[:2] - below[:2]) > xy_tol:
            return False
        if abs(cube[2] - (task.CUBE_REST_Z + k * task.CUBE_SIZE)) > z_tol:
            return False
    return True
