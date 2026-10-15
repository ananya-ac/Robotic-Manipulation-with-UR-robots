"""Active multi-view scan for oracle_obj_unknown - runs after each reset, BEFORE recording.

  1. clear the octomap and the perception node's fused map; move to task.SCAN_Q (top view of the
     spawn region), wait for the octomap, add the view, detect -> rough object poses
  2. for each object: 4 views around it at 45 deg off its faces (tabletop_perception's
     object_views), so every side face is seen twice; each camera pose becomes a tip pose through
     the camera's fixed mount, IK-checked, reached with OMPL (octomap-checked), and added
  3. detect on the fused map -> the oracle's object model for the whole episode
  4. back to HOME_Q (recorded episodes start there, as for oracle_obj_known)

Offline, this view set saw all 5 faces of every cube and gave < 1.1 mm / 1.3 deg pose error
(stack_bc.perception_check --views object).
"""

import numpy as np
import PyKDL

from stack_bc import rotations as rot
from stack_bc import task
from stack_bc.motion import PlanningFailure
from tabletop_perception.core.views import object_views

CAMERA_FRAME = "camera_color_toein_frame"   # the optical frame the wrist camera renders from
VIEW_RADIUS = 0.28                          # m, camera to object center
VIEW_ELEVATION = 55.0                       # deg above the object's horizontal plane
MOVE_TIME = 2.0                             # s, unrecorded joint moves to/from the scan pose


def camera_mount(kin):
    """Fixed gripper_tip_link -> camera optical frame transform, from the URDF (KDL)."""
    chain = kin.kdl._kdl_tree.getChain("world", CAMERA_FRAME)
    q = PyKDL.JntArray(chain.getNrOfJoints())
    for i, v in enumerate(task.HOME_Q):
        q[i] = v
    f = PyKDL.Frame()
    PyKDL.ChainFkSolverPos_recursive(chain).JntToCart(q, f)
    T_cam = rot.pose_matrix([f.p[i] for i in range(3)], np.array([[f.M[i, j] for j in range(3)] for i in range(3)]))
    return np.linalg.inv(kin.fk_world(task.HOME_Q)) @ T_cam


class ActiveScanner:
    def __init__(self, sim, env, motion, ex, scene, perception, log=print):
        self.sim, self.env, self.motion, self.ex = sim, env, motion, ex
        self.scene, self.perception, self.log = scene, perception, log
        self.T_tip_cam = camera_mount(env.kin)
        self.T_cam_tip = np.linalg.inv(self.T_tip_cam)

    def _view_pose(self, T_cam, seed):
        """Camera pose -> reachable tip pose (the image may be rolled 180 deg; either is fine)."""
        for roll in (0.0, np.pi):
            T_tip = T_cam @ rot.pose_matrix([0, 0, 0], rot.rot_z(roll)) @ self.T_cam_tip
            if self.env.kin.ik_world(T_tip, seed) is not None:
                return T_tip
        return None

    def scan(self):
        """Returns {name: Perceived} from the fused map. Leaves the arm at HOME_Q."""
        self.scene.detach_cube()          # a failed episode may have left something attached
        self.scene.clear()
        self.perception.reset_map()
        self.ex.move_joints(task.SCAN_Q, MOVE_TIME)
        self.sim.sleep(MOVE_TIME + 0.3)
        self.scene.wait_for_map()
        self.perception.add_view()
        rough, msg = self.perception.detect(use_map=True)
        if not rough:
            raise RuntimeError(f"scan: nothing detected at the scan pose ({msg})")

        n_ok, n_skip = 1, 0
        for name, o in sorted(rough.items()):
            yaw = rot.yaw_of(o.R)
            for _, _, T_cam in object_views(o.center, yaw, VIEW_RADIUS, [VIEW_ELEVATION]):
                T_tip = self._view_pose(T_cam, self.env.snapshot().q_arm)
                if T_tip is None:
                    n_skip += 1
                    continue
                try:
                    self.motion.ompl(T_tip, vel_scale=0.3, label=f"scan/{name}")
                except PlanningFailure:
                    n_skip += 1
                    continue
                self.sim.sleep(0.2)
                self.perception.add_view()
                n_ok += 1
        objects, msg = self.perception.detect(use_map=True)
        self.log(f"[scan] {n_ok} views ({n_skip} unreachable/unplannable), {msg}")

        self.ex.move_joints(task.HOME_Q, MOVE_TIME)
        self.sim.sleep(MOVE_TIME + 0.3)
        self.motion.forget()
        return objects
