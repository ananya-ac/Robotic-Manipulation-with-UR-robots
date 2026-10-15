"""Offline accuracy of tabletop_perception on the sim's own renders (no ROS, no running sim).

  python3 -m stack_bc.perception_check [--seeds 100] [--views orbit|scan] [--grasp-method antipodal_3d|outline]

For each seed: place the cubes like a collection reset (StackEnv.sample_cubes), render RGB + depth
from the wrist camera with mujoco.Renderer at task.SCAN_Q and (--views orbit) at every IK-reachable
orbit viewpoint around the spawn center, back-project each through camera_color_toein_frame (the
optical site the camera renders from), fuse them (VoxelMap, as the node's add_view does), then run
the perception core exactly as the node does (same YAML) and compare with ground truth:
  - class (all three cubes found with the right names)
  - position / height / yaw (mod 90 deg) error
  - grasps: >= 1 candidate per cube, width ~ cube size, fingers on opposite faces, and the best
    grasp + its pre-grasp (10 cm above) reachable by IK.
"""

import argparse
import os

import mujoco
import numpy as np

from stack_bc import rotations as rot
from stack_bc import task
from stack_bc.env import StackEnv
from stack_bc.kinematics import ARM_JOINTS, ArmKinematics
from stack_bc.scene import DEFAULT_OUT, build_scene
from tabletop_perception.core import antipodal_grasps, backproject, detect, pixel_rays
from tabletop_perception.core.fusion import VoxelMap
from tabletop_perception.core.views import object_views, orbit_views
from tabletop_perception.params import config_dir, configs_from_params, load_node_params

PREGRASP_HEIGHT = 0.10


class SimCamera:
    def __init__(self, model, camera="camera", site="camera_color_toein_frame"):
        self.m = model
        self.d = mujoco.MjData(model)
        self.cam, self.site = camera, model.site(site).id
        w, h = model.cam_resolution[model.camera(camera).id]
        fy = 0.5 * h / np.tan(np.radians(model.cam_fovy[model.camera(camera).id]) / 2)
        self.rays = pixel_rays(w, h, fy, fy, w / 2, h / 2, stride=2)
        self.rgb_r = mujoco.Renderer(model, height=h, width=w)
        self.depth_r = mujoco.Renderer(model, height=h, width=w)
        self.depth_r.enable_depth_rendering()
        self.opt = mujoco.MjvOption()
        self.opt.sitegroup[:] = 0          # no site markers (one floats between the fingers)

    def set_state(self, q_arm, cube_xy, cube_yaw):
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        for name, q in zip(ARM_JOINTS, q_arm):
            self.d.qpos[self.m.joint(name).qposadr[0]] = q
        for name, (x, y), yaw in zip(task.CUBE_NAMES, cube_xy, cube_yaw):
            adr = self.m.jnt_qposadr[self.m.body(name).jntadr[0]]
            qx, qy, qz, qw = rot.matrix_to_quat(rot.rot_z(yaw))
            self.d.qpos[adr:adr + 7] = [x, y, task.CUBE_REST_Z, qw, qx, qy, qz]
        mujoco.mj_forward(self.m, self.d)

    def cloud_world(self):
        self.rgb_r.update_scene(self.d, camera=self.cam, scene_option=self.opt)
        rgb = self.rgb_r.render()
        self.depth_r.update_scene(self.d, camera=self.cam, scene_option=self.opt)
        depth = self.depth_r.render()
        pts, cols = backproject(depth, self.rays, stride=2, rgb=rgb)
        R = self.d.site_xmat[self.site].reshape(3, 3)
        return pts @ R.T + self.d.site_xpos[self.site], cols, rgb

    def site_pose(self, name):
        i = self.m.site(name).id
        T = np.eye(4)
        T[:3, :3] = self.d.site_xmat[i].reshape(3, 3)
        T[:3, 3] = self.d.site_xpos[i]
        return T


def orbit_joint_targets(cam, kin, radius, elevations, azimuths):
    """IK-reachable arm configurations for orbit views around the spawn center (camera poses ->
    tip poses through the camera's fixed mount, measured from the model at SCAN_Q)."""
    cam.set_state(task.SCAN_Q, np.zeros((task.N_CUBES, 2)) + [-0.45, 0.0], np.zeros(task.N_CUBES))
    T_tip_cam = np.linalg.inv(cam.site_pose("gripper_tip_link")) @ cam.site_pose("camera_color_toein_frame")
    target = [np.mean(task.SPAWN_X), np.mean(task.SPAWN_Y), task.CUBE_REST_Z]
    out = []
    for el, az, T_cam in orbit_views(target, radius, elevations, azimuths):
        q = None
        for roll in (0.0, np.pi):          # the image may be upside down; either is a valid view
            Tc = T_cam.copy()
            Tc[:3, :3] = T_cam[:3, :3] @ rot.rot_z(roll)
            q = kin.ik_world(Tc @ np.linalg.inv(T_tip_cam), task.SCAN_Q)
            if q is not None:
                break
        if q is not None:
            out.append(((el, az), q))
    return out


def views_to_joints(cam, kin, cam_views, seed):
    """Camera poses -> reachable arm configurations ((label, q)), trying each listed elevation
    for an azimuth until one is reachable (higher views reach further)."""
    cam.set_state(task.SCAN_Q, np.zeros((task.N_CUBES, 2)) + [-0.45, 0.0], np.zeros(task.N_CUBES))
    T_tip_cam = np.linalg.inv(cam.site_pose("gripper_tip_link")) @ cam.site_pose("camera_color_toein_frame")
    out = []
    for label, T_cam in cam_views:
        for roll in (0.0, np.pi):
            Tc = T_cam.copy()
            Tc[:3, :3] = T_cam[:3, :3] @ rot.rot_z(roll)
            q = kin.ik_world(Tc @ np.linalg.inv(T_tip_cam), seed)
            if q is not None:
                out.append((label, q))
                break
    return out


def exclude_boxes(cam, self_filter):
    out = []
    for frame, center, half in self_filter:
        T = cam.site_pose(frame)
        T[:3, 3] = T[:3, :3] @ center + T[:3, 3]
        out.append((T, half))
    return out


def face_coverage(points, xy, yaw, h=task.CUBE_SIZE / 2):
    """Fraction of the cube's 4 side faces + top that received points (>= 30% of each face's 4 mm
    cells), in the cube's own frame."""
    local = (points - [xy[0], xy[1], task.CUBE_REST_Z]) @ rot.rot_z(yaw)
    faces = {"top": (2, 1), "+x": (0, 1), "-x": (0, -1), "+y": (1, 1), "-y": (1, -1)}
    seen = 0
    for axis, sign in faces.values():
        on = np.abs(local[:, axis] - sign * h) < 0.003
        other = [a for a in range(3) if a != axis]
        cells = np.unique(np.floor(local[on][:, other] / 0.004).astype(int), axis=0)
        seen += len(cells) >= 0.3 * (2 * h / 0.004) ** 2
    return seen / len(faces)


def reachable(kin, T, seed):
    for flip in (0.0, np.pi):    # a parallel-jaw grasp is the same grasp rotated 180 deg about z
        Tf = T.copy()
        Tf[:3, :3] = T[:3, :3] @ rot.rot_z(flip)
        pre = Tf.copy()
        pre[:3, 3] -= PREGRASP_HEIGHT * Tf[:3, 2]
        q = kin.ik_world(Tf, seed)
        if q is not None and kin.ik_world(pre, q) is not None:
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--config", default=os.path.join(config_dir(), "tabletop_mujoco.yaml"))
    ap.add_argument("--save", default=None, help="directory: save RGB of failing seeds")
    ap.add_argument("--views", choices=("object", "orbit", "scan"), default="object",
                    help="object: scan view, then 4 views around each detected object; orbit: one ring "
                         "around the spawn center; scan: top view only")
    ap.add_argument("--radius", type=float, default=0.30)
    ap.add_argument("--elevations", type=float, nargs="+", default=[50.0, 65.0])
    ap.add_argument("--azimuths", type=float, nargs="+", default=[0, 45, 90, 135, 180, 225, 270, 315])
    ap.add_argument("--grasp-method", default=None, help="override the config's grasp method")
    args = ap.parse_args()

    scene_path = build_scene(DEFAULT_OUT)
    model = mujoco.MjModel.from_xml_path(scene_path)
    with open(os.path.join(DEFAULT_OUT, "robot_only.urdf")) as f:
        kin = ArmKinematics(f.read())
    seg, gcfg, gripper, self_filter = configs_from_params(load_node_params(args.config))
    if args.grasp_method:
        gcfg.method = args.grasp_method
    cam = SimCamera(model)
    views = [("scan", task.SCAN_Q)]
    if args.views == "orbit":
        views += orbit_joint_targets(cam, kin, args.radius, args.elevations, args.azimuths)
    print(f"views: {args.views} ({len(views)} fixed), grasp method {gcfg.method}")
    coverage, n_views = [], []

    pos_err, z_err, yaw_err, width_err, face_err = [], [], [], [], []
    fails = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        xy, yaw = StackEnv.sample_cubes(rng)
        fused = VoxelMap(0.002)
        seed_views = list(views)
        if args.views == "object":                    # phase 1: top view -> rough object poses
            cam.set_state(task.SCAN_Q, xy, yaw)
            p0, c0, _ = cam.cloud_world()
            try:
                rough = detect(p0, c0, seg, exclude_boxes(cam, self_filter)).objects
            except ValueError:
                rough = []
            cams = []
            for o in rough:                           # phase 2: views around each object
                for el in args.elevations:
                    cams += [((o.name, el, round(az)), T) for _, az, T in
                             object_views(o.center, o.yaw, args.radius, [el])]
            seed_views += views_to_joints(cam, kin, cams, task.SCAN_Q)
            n_views.append(len(seed_views))
        for _, q in seed_views:
            cam.set_state(q, xy, yaw)
            pts, cols, rgb = cam.cloud_world()
            for T, half in exclude_boxes(cam, self_filter):          # self-filter per view
                inside = np.all(np.abs((pts - T[:3, 3]) @ T[:3, :3]) <= half, axis=1)
                pts, cols = pts[~inside], cols[~inside]
            fused.add(pts, cols)
        try:
            est = detect(fused.points, fused.colors, seg)
        except ValueError as exc:
            fails.append((seed, f"detect: {exc}"))
            continue
        for i, name in enumerate(task.CUBE_NAMES):
            o = est.get(name)
            if o is None:
                fails.append((seed, f"{name} not found ({[x.name for x in est.objects]})"))
                continue
            coverage.append(face_coverage(o.points, xy[i], yaw[i]))
            pos_err.append(np.linalg.norm(o.center[:2] - xy[i]))
            z_err.append(abs(o.center[2] - task.CUBE_REST_Z))
            yaw_err.append(np.degrees(abs((o.yaw - yaw[i] + np.pi / 4) % (np.pi / 2) - np.pi / 4)))
            gs = antipodal_grasps(o, est, gripper, gcfg)
            if not gs:
                fails.append((seed, f"{name}: no grasp"))
                continue
            g = gs[0]
            width_err.append(abs(g.width - task.CUBE_SIZE))
            cyaw = np.arctan2(g.T[1, 0], g.T[0, 0])
            face_err.append(np.degrees(abs((cyaw - yaw[i] + np.pi / 4) % (np.pi / 2) - np.pi / 4)))
            if not any(reachable(kin, gg.T, task.HOME_Q) for gg in gs[:5]):
                fails.append((seed, f"{name}: top-5 grasps unreachable"))
        if args.save and fails and fails[-1][0] == seed:
            import imageio
            os.makedirs(args.save, exist_ok=True)
            imageio.imwrite(os.path.join(args.save, f"seed{seed:03d}.png"), rgb)

    def stat(name, v, unit, scale=1.0):
        v = np.asarray(v) * scale
        if len(v):
            print(f"  {name:22s} median {np.median(v):6.2f} {unit}   p95 {np.percentile(v, 95):6.2f}   max {v.max():6.2f}")

    n = args.seeds * task.N_CUBES
    print(f"{args.seeds} scenes / {n} cubes, {len(fails)} failures")
    stat("views per scene", n_views, "")
    stat("faces observed", coverage, "of 5", 5)
    stat("xy position error", pos_err, "mm", 1000)
    stat("center height error", z_err, "mm", 1000)
    stat("yaw error (mod 90)", yaw_err, "deg")
    stat("grasp width error", width_err, "mm", 1000)
    stat("grasp axis vs faces", face_err, "deg")
    for seed, why in fails[:20]:
        print(f"  seed {seed}: {why}")


if __name__ == "__main__":
    main()
