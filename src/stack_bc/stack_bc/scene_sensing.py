"""Sensor-built planning scene for the oracle (collect.py --scene sensors).

The collision world MoveIt plans against is an octomap built only from the wrist depth camera,
owned by the move_group of `ur_moveit.launch.py ... sensors:=true` and mirrored into the oracle's
moveit_py (moveit_config.py). No ground-truth geometry (table, cubes) is ever inserted.

SceneManager drives that scene through an episode:
  * clear() after a reset (cubes teleported; old voxels are stale), then wait_for_map() at the
    scan pose once fresh depth has been integrated.
  * contact(): the octomap contains the very cube being grasped and the stack being placed on, so
    the short vertical grasp/place segments are planned with octomap-vs-gripper (and held cube)
    collisions allowed. Transit moves stay fully checked. The relaxation is a moveit_py-local ACM
    edit: only moveit_py plans, and move_group's diffs never carry an ACM unless its own changes.
  * attach_cube()/detach_cube(): the held cube becomes an attached body on gripper_tip_link. It
    goes to move_group too, whose self-filter then removes the cube from the depth cloud (so it
    doesn't leave voxels that collide with itself), and to moveit_py so carry moves check it.
    The pose is the tip->cube transform the oracle measured at grasp time.
"""

import re
import threading
from contextlib import contextmanager

from geometry_msgs.msg import Pose
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive
from std_srvs.srv import Empty

from stack_bc import rotations as rot
from stack_bc import task
from stack_bc.moveit_config import MONITORED_SCENE_TOPIC

OCTOMAP_ID = "<octomap>"            # MoveIt's collision-world name for the octomap
HELD_ID = "held_cube"
TIP_LINK = "gripper_tip_link"
GRIPPER_LINK_PATTERN = r"^(robotiq_85_\w+|ur_to_robotiq_link|gripper_mount_link|gripper_tip_link)$"
MIN_OCTOMAP_BYTES = 1000            # a scan of the table is far larger; an empty tree is ~0


def _call(client, req, name, timeout=5.0):
    if not client.wait_for_service(timeout_sec=timeout):
        raise RuntimeError(f"{name} unavailable - is ur_moveit.launch.py ... sensors:=true running?")
    done = threading.Event()
    future = client.call_async(req)
    future.add_done_callback(lambda _: done.set())
    if not done.wait(timeout):
        raise RuntimeError(f"{name} timed out")
    return future.result()


class SceneManager:
    def __init__(self, node, urdf, scene_monitor):
        """node: spun elsewhere (SimNode). scene_monitor: MoveItMotion.scene_monitor (moveit_py)."""
        self.psm = scene_monitor
        link_names = re.findall(r'<link\s+name="([^"]+)"', urdf)
        self.gripper_links = [n for n in link_names if re.match(GRIPPER_LINK_PATTERN, n)]
        if not self.gripper_links:
            raise RuntimeError("no gripper links found in robot_description")
        self._clear = node.create_client(Empty, "/clear_octomap")
        self._apply = node.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self._cv = threading.Condition()
        self._map_updates = 0
        node.create_subscription(PlanningScene, MONITORED_SCENE_TOPIC, self._on_scene, 10)
        self.holding = False

    def _on_scene(self, msg):
        if len(msg.world.octomap.octomap.data) >= MIN_OCTOMAP_BYTES:
            with self._cv:
                self._map_updates += 1
                self._cv.notify_all()

    # ---- octomap -----------------------------------------------------------------------------

    def clear(self):
        _call(self._clear, Empty.Request(), "/clear_octomap")
        self.psm.clear_octomap()   # don't wait for move_group's empty-map diff to arrive

    def wait_for_map(self, n_updates=2, timeout=5.0):
        """Block until n non-empty octomap updates arrive from now on (call it once the camera
        is still at the scan pose, so they integrate that view)."""
        with self._cv:
            target = self._map_updates + n_updates
            if not self._cv.wait_for(lambda: self._map_updates >= target, timeout):
                raise RuntimeError(f"no octomap updates on {MONITORED_SCENE_TOPIC} within {timeout}s - "
                                   "check /camera/depth/color/points and move_group's sensor plugin")

    # ---- contact phases ----------------------------------------------------------------------

    def _set_octomap_allowed(self, allowed):
        # HELD_ID is always included (an entry for an absent body is harmless), so attaching or
        # detaching inside a contact block can't leave its entry stale.
        with self.psm.read_write() as scene:
            acm = scene.allowed_collision_matrix
            for name in self.gripper_links + [HELD_ID]:
                acm.set_entry(OCTOMAP_ID, name, allowed)

    @contextmanager
    def contact(self):
        """Moves inside this block may touch the octomap with the gripper / held cube."""
        self._set_octomap_allowed(True)
        try:
            yield
        finally:
            self._set_octomap_allowed(False)

    # ---- held cube ---------------------------------------------------------------------------

    def attach_cube(self, T_tip_cube):
        box = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[task.CUBE_SIZE] * 3)
        pose = Pose()
        p = pose.position
        p.x, p.y, p.z = map(float, T_tip_cube[:3, 3])
        o = pose.orientation
        o.x, o.y, o.z, o.w = map(float, rot.matrix_to_quat(T_tip_cube[:3, :3]))
        aco = AttachedCollisionObject(link_name=TIP_LINK, touch_links=self.gripper_links)
        aco.object.id = HELD_ID
        aco.object.header.frame_id = TIP_LINK
        aco.object.primitives = [box]
        aco.object.primitive_poses = [pose]
        aco.object.operation = CollisionObject.ADD
        self._apply_attached(aco, remove_world=False)
        self.holding = True

    def detach_cube(self):
        if not self.holding:
            return
        aco = AttachedCollisionObject(link_name=TIP_LINK)
        aco.object.id = HELD_ID
        aco.object.operation = CollisionObject.REMOVE
        self._apply_attached(aco, remove_world=True)
        self.holding = False

    def _apply_attached(self, aco, remove_world):
        # MoveIt re-adds a detached body to the world; remove that copy in the same diff.
        world_rm = CollisionObject(id=HELD_ID, operation=CollisionObject.REMOVE)
        diff = PlanningScene(is_diff=True)
        diff.robot_state.is_diff = True
        diff.robot_state.attached_collision_objects = [aco]
        if remove_world:
            diff.world.collision_objects = [world_rm]
        if not _call(self._apply, ApplyPlanningScene.Request(scene=diff), "/apply_planning_scene").success:
            raise RuntimeError(f"/apply_planning_scene rejected {HELD_ID} update")
        self.psm.process_attached_collision_object(aco)
        if remove_world:
            self.psm.process_collision_object(world_rm)
