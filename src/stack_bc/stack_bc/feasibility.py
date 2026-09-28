"""Offline IK feasibility map of the tabletop, used to set the cube randomization region.

For each (x, y) grid cell and each sampled cube yaw, checks that a straight vertical top-down
approach - hover (target + HOVER_HEIGHT) down to the target - has a continuous exact IK solution
(EAIK, seeded from HOME_Q, no branch flips between 1 cm steps), trying the cube's four
symmetric grasp yaws and keeping the one closest to home. Checked at three target heights:
level 0 (picking a cube off the table) and levels 1-2 (placing onto a 1- or 2-high stack). The
lowest configuration is also checked for robot contact with the table/floor in the compiled
MuJoCo scene (cubes moved out of the way).

A cell is usable for spawning when every level passes for every sampled yaw - any spawned cube
may end up picked or be chosen as the stack base. The largest all-feasible axis-aligned rectangle
is reported; copy it into task.py's SPAWN_* bounds.

  feasibility [--scene ~/.cache/stack_bc/scene] [--step 0.02] [--yaws 8]
"""

import argparse
import os

import mujoco
import numpy as np

from stack_bc import rotations as rot
from stack_bc import task
from stack_bc.kinematics import ArmKinematics
from stack_bc.scene import DEFAULT_OUT

MAX_JOINT_STEP = 0.25  # rad between consecutive 1 cm samples; larger means an IK branch flip
LEVELS = (0, 1, 2)


def vertical_path(kin, xy, yaw, z_target, seed):
    """Joint configs from hover down to z_target, or None. Returns the list top-to-bottom."""
    zs = np.arange(z_target + task.HOVER_HEIGHT, z_target - 1e-9, -0.01)
    if not np.isclose(zs[-1], z_target):
        zs = np.append(zs, z_target)
    R = rot.top_down_grasp(yaw)
    qs, q_prev = [], seed
    for z in zs:
        q = kin.ik_world(rot.pose_matrix([xy[0], xy[1], z], R), q_prev)
        if q is None or (qs and np.max(np.abs(q - q_prev)) > MAX_JOINT_STEP):
            return None
        qs.append(q)
        q_prev = q
    return qs


def best_grasp_path(kin, xy, cube_yaw, z_target):
    """Path for the symmetric grasp yaw (cube_yaw + k*pi/2) whose hover config is closest to home."""
    best, best_d = None, np.inf
    for k in range(4):
        path = vertical_path(kin, xy, cube_yaw + k * np.pi / 2, z_target, task.HOME_Q)
        if path is not None:
            d = np.linalg.norm(path[0] - task.HOME_Q)
            if d < best_d:
                best, best_d = path, d
    return best


class ContactChecker:
    """Robot-vs-table/floor contact at a given arm config, gripper open, cubes parked far away."""

    def __init__(self, scene_path):
        self.m = mujoco.MjModel.from_xml_path(scene_path)
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        for i, name in enumerate(task.CUBE_NAMES):
            adr = self.m.jnt_qposadr[self.m.body(name).jntadr[0]]
            self.d.qpos[adr:adr + 7] = [5.0 + i, 5.0, 0.1, 1, 0, 0, 0]
        self.robot_bodies = {self.m.body(n).id for n in
                             ["shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link",
                              "wrist_2_link", "wrist_3_link"]} | {
            i for i in range(self.m.nbody) if self.m.body(i).name.startswith("robotiq_85")}
        self.env_geoms = {g for g in range(self.m.ngeom)
                          if self.m.geom_bodyid[g] in (0, self.m.body("table").id)}

    def collides(self, q_arm):
        self.d.qpos[:6] = q_arm
        self.d.qpos[6:12] = 0.0  # gripper open
        mujoco.mj_forward(self.m, self.d)
        for c in self.d.contact[:self.d.ncon]:
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            if (b1 in self.robot_bodies and c.geom2 in self.env_geoms) or \
               (b2 in self.robot_bodies and c.geom1 in self.env_geoms):
                if c.dist < 0:
                    return True
        return False


def largest_rectangle(mask):
    """Largest all-True axis-aligned rectangle in a 2D bool grid -> (i0, i1, j0, j1) inclusive."""
    best, best_area = None, 0
    heights = np.zeros(mask.shape[1], dtype=int)
    for i in range(mask.shape[0]):
        heights = np.where(mask[i], heights + 1, 0)
        stack = []
        for j in range(mask.shape[1] + 1):
            h = heights[j] if j < mask.shape[1] else 0
            start = j
            while stack and stack[-1][1] >= h:
                start, sh = stack.pop()
                area = sh * (j - start)
                if area > best_area:
                    best_area, best = area, (i - sh + 1, i, start, j - 1)
            stack.append((start, h))
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default=DEFAULT_OUT)
    parser.add_argument("--step", type=float, default=0.02)
    parser.add_argument("--yaws", type=int, default=8, help="cube yaw samples over [0, pi/2)")
    parser.add_argument("--margin", type=float, default=0.04, help="keep cube centers this far inside the table edge")
    args = parser.parse_args()

    kin = ArmKinematics(open(os.path.join(args.scene, "robot_only.urdf")).read())
    checker = ContactChecker(os.path.join(args.scene, "scene.xml"))

    lo = task.TABLE_CENTER_XY - task.TABLE_HALF_EXTENT + args.margin
    hi = task.TABLE_CENTER_XY + task.TABLE_HALF_EXTENT - args.margin
    xs = np.arange(lo[0], hi[0] + 1e-9, args.step)
    ys = np.arange(lo[1], hi[1] + 1e-9, args.step)
    yaws = np.arange(args.yaws) * (np.pi / 2) / args.yaws

    ok = np.zeros((len(LEVELS), len(xs), len(ys)), dtype=float)
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            for li, level in enumerate(LEVELS):
                z = task.CUBE_REST_Z + level * task.CUBE_SIZE
                n_ok = 0
                for yaw in yaws:
                    path = best_grasp_path(kin, (x, y), yaw, z)
                    if path is not None and not checker.collides(path[-1]):
                        n_ok += 1
                ok[li, ix, iy] = n_ok / len(yaws)

    full = np.all(ok == 1.0, axis=0)
    print("Feasibility (rows = x from %.2f to %.2f, cols = y from %.2f to %.2f); "
          "'#' all levels+yaws ok, digit = worst level's fraction*10, '.' none" % (xs[0], xs[-1], ys[0], ys[-1]))
    worst = ok.min(axis=0)
    for ix in range(len(xs)):
        row = "".join("#" if full[ix, iy] else ("." if worst[ix, iy] == 0 else str(min(9, int(worst[ix, iy] * 10))))
                      for iy in range(len(ys)))
        print(f"x={xs[ix]:+.2f} {row}")
    for li, level in enumerate(LEVELS):
        print(f"level {level}: {np.mean(ok[li] == 1.0) * 100:.0f}% of cells fully feasible")

    rect = largest_rectangle(full)
    if rect is None:
        print("No fully feasible cell.")
    else:
        i0, i1, j0, j1 = rect
        print(f"Largest all-feasible rectangle: x in [{xs[i0]:.2f}, {xs[i1]:.2f}], y in [{ys[j0]:.2f}, {ys[j1]:.2f}]")
    out = os.path.join(args.scene, "feasibility.npz")
    np.savez(out, xs=xs, ys=ys, yaws=yaws, levels=np.array(LEVELS), ok=ok)
    print("saved", out)


if __name__ == "__main__":
    main()
