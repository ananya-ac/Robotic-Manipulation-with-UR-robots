"""Ground-truth planning-scene world: table + one box per cube at its simulator pose.

Used where OMPL plans free-space transits for oracle_obj_known (collect --transit ompl) and by the
DAgger expert offline (planner_backend.OfflinePlanner). Both build the SAME world from the same
cube poses, so a relabeled transit is planned against what the demo's transit was planned against
(the octomap of --scene sensors is never logged, so it couldn't be rebuilt offline). The policy
observes GT cube poses too, so a GT world doesn't give the expert anything the policy lacks.

Everything is a moveit_py-local edit of the given PlanningSceneMonitor (only moveit_py plans;
no move_group is involved):
  * GTWorld.set(): table, the cubes not held, and the held cube (if any) as an attached body on
    gripper_tip_link at its measured in-hand pose, so carry plans check it against the stack.
  * GTWorld.allow_contact(): the short vertical grasp/place segments (the oracle's contact()
    blocks) touch the target cube / stack / table with the gripper and the held cube; those pairs
    are allowed there. Arm links stay checked against everything.
GTScene adapts GTWorld to the interface the oracle uses for scene_sensing.SceneManager.
"""

import re
from contextlib import contextmanager

import numpy as np
from geometry_msgs.msg import Pose
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
from shape_msgs.msg import SolidPrimitive

from stack_bc import rotations as rot
from stack_bc import task
from stack_bc.scene_sensing import GRIPPER_LINK_PATTERN, HELD_ID, TIP_LINK

TABLE_ID = "table"
TABLE_THICKNESS = 0.02
HELD_MAX_DIST = 0.03    # m, tip to cube center for a cube to count as the one in the hand


def _pose(T):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = map(float, T[:3, 3])
    x, y, z, w = rot.matrix_to_quat(T[:3, :3])
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, (x, y, z, w))
    return pose


def _box(obj_id, frame, T, dims):
    co = CollisionObject(id=obj_id, operation=CollisionObject.ADD)
    co.header.frame_id = frame
    co.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(d) for d in dims])]
    co.primitive_poses = [_pose(T)]
    return co


def held_cube(snap):
    """Index of the cube centered between the fingers (tip within HELD_MAX_DIST), else None.
    Geometry only - whether it is actually grasped is the caller's call."""
    d = np.linalg.norm(snap.cube_pos - snap.ee[:3, 3], axis=1)
    i = int(np.argmin(d))
    return i if d[i] < HELD_MAX_DIST else None


class GTWorld:
    def __init__(self, psm, urdf):
        """psm: a moveit_py PlanningSceneMonitor. urdf: the robot description (gripper links)."""
        self.psm = psm
        self.gripper_links = [n for n in re.findall(r'<link\s+name="([^"]+)"', urdf)
                              if re.match(GRIPPER_LINK_PATTERN, n)]
        if not self.gripper_links:
            raise RuntimeError("no gripper links found in robot_description")
        self._attached = False
        self._contact = False

    def set(self, cube_pos, cube_rot, held=None, T_tip_held=None):
        """Replace the world. held: index of the cube in the hand (attached at T_tip_held, the
        tip -> cube transform) or None."""
        if self._attached:
            aco = AttachedCollisionObject(link_name=TIP_LINK)
            aco.object.id, aco.object.operation = HELD_ID, CollisionObject.REMOVE
            self.psm.process_attached_collision_object(aco)
            self._attached = False
        table = rot.pose_matrix([*task.TABLE_CENTER_XY, task.TABLE_TOP_Z - TABLE_THICKNESS / 2], np.eye(3))
        with self.psm.read_write() as scene:
            scene.remove_all_collision_objects()   # also drops the detached copy MoveIt re-adds
            scene.apply_collision_object(_box(TABLE_ID, "world", table,
                                              [2 * task.TABLE_HALF_EXTENT] * 2 + [TABLE_THICKNESS]))
            for i, name in enumerate(task.CUBE_NAMES):
                if i != held:
                    scene.apply_collision_object(_box(name, "world", rot.pose_matrix(cube_pos[i], cube_rot[i]),
                                                      [task.CUBE_SIZE] * 3))
        if held is not None:
            aco = AttachedCollisionObject(link_name=TIP_LINK, touch_links=self.gripper_links)
            aco.object = _box(HELD_ID, TIP_LINK, T_tip_held, [task.CUBE_SIZE] * 3)
            self.psm.process_attached_collision_object(aco)
            self._attached = True
        self.allow_contact(self._contact)  # removing objects may drop their ACM entries

    def allow_contact(self, allowed):
        """Allow (or forbid again) gripper/held-cube contact with the table and all cubes."""
        self._contact = allowed
        with self.psm.read_write() as scene:
            acm = scene.allowed_collision_matrix
            for name in self.gripper_links + [HELD_ID]:
                for obj in (TABLE_ID, *task.CUBE_NAMES):
                    acm.set_entry(obj, name, allowed)


class GTScene:
    """The oracle's scene interface (as scene_sensing.SceneManager) over a live GTWorld."""

    def __init__(self, env, psm, urdf):
        self.env = env
        self.world = GTWorld(psm, urdf)
        self.held, self.T_tip_held = None, None

    def refresh(self):
        """Rebuild the world from the current GT cube poses (call before every plan)."""
        snap = self.env.snapshot()
        self.world.set(snap.cube_pos, snap.cube_rot, self.held, self.T_tip_held)

    def clear(self):
        self.held, self.T_tip_held = None, None
        self.refresh()

    @contextmanager
    def contact(self):
        self.world.allow_contact(True)
        try:
            yield
        finally:
            self.world.allow_contact(False)

    def attach_cube(self, T_tip_cube=None):
        """The cube between the fingers rides on the tip at its MEASURED in-hand pose (GT; the
        argument is the sensor scene's nominal pose and is ignored here)."""
        snap = self.env.snapshot()
        self.held = held_cube(snap)
        if self.held is None:
            raise RuntimeError("attach_cube: no cube between the fingers")
        self.T_tip_held = np.linalg.inv(snap.ee) @ rot.pose_matrix(snap.cube_pos[self.held], snap.cube_rot[self.held])
        self.refresh()

    def detach_cube(self):
        self.held, self.T_tip_held = None, None
        self.refresh()
