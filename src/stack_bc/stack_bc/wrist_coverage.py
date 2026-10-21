"""How often does the wrist camera see the cube the task needs next? (offline, from logged qpos)

  python3 -m stack_bc.wrist_coverage --dataset ~/data/stack_bc/<set>.zarr [--episodes 0 1 ...]
      [--scene ~/.cache/stack_bc/scene_marker] [--min-pixels 50] [--sheet out.png]

Renders the wrist camera (MJCF camera "camera", 240x320 as recorded) with segmentation for every
tick, counts each cube's visible pixels (occlusion included), and works out per tick which cube
the robot needs to see:
  * reaching: the next cube to pick (order[k], not yet stacked)
  * carrying: the cube it goes onto (order[k-1]); marker world, base carry: the marker
A cube counts as visible with >= --min-pixels pixels. Reports the visible fraction per phase and
the longest blind stretch per episode - long blind stretches are where a 2-frame wrist-only
policy has nothing to go on, i.e. where the overview context image should matter.
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import zarr  # noqa: E402

from stack_bc import task  # noqa: E402
from stack_bc.imageproc import IMAGE_SHAPE  # noqa: E402
from stack_bc.scene import DEFAULT_OUT  # noqa: E402

HELD_MAX_DIST = 0.03      # tip-to-cube-centre distance for "in the hand" (as scene_gt)
LIFTED = 0.005            # m above rest height


def body_geoms(model, name):
    bid = model.body(name).id
    return {g for g in range(model.ngeom) if model.geom_bodyid[g] == bid}


def run(dataset, scene_dir, episodes, min_pixels, sheet):
    model = mujoco.MjModel.from_xml_path(os.path.join(scene_dir, "scene.xml"))
    data = mujoco.MjData(model)
    root = zarr.open(os.path.expanduser(dataset), "r")
    if root["data/qpos"].shape[1] != model.nq:
        raise ValueError("dataset qpos does not match the scene (marker datasets need --scene ~/.cache/stack_bc/scene_marker)")
    ends = root["meta/episode_ends"][:]
    starts = np.r_[0, ends[:-1]]
    orders = root["meta/episode_order"][:]
    targets = root["meta/episode_target"][:] if "episode_target" in root["meta"] else None
    cube_geoms = [body_geoms(model, n) for n in task.CUBE_NAMES]
    marker_geoms = body_geoms(model, task.MARKER_NAME) if targets is not None else set()
    cube_adr = [model.jnt_qposadr[model.body(n).jntadr[0]] for n in task.CUBE_NAMES]
    tip_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip_link")
    cam = mujoco.MjvCamera()
    cam.type, cam.fixedcamid = mujoco.mjtCamera.mjCAMERA_FIXED, model.camera("camera").id
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0

    stats = {"reach": [0, 0], "carry": [0, 0]}
    sheet_frames = []
    episodes = episodes if episodes else range(len(ends))
    h, w = IMAGE_SHAPE[:2]
    with mujoco.Renderer(model, h, w) as seg, mujoco.Renderer(model, h, w) as rgb:
        seg.enable_segmentation_rendering()
        for e in episodes:
            order = [int(i) for i in orders[e]]
            qpos = root["data/qpos"][starts[e]:ends[e]]
            target_xy = targets[e] if targets is not None else None
            k, blind, worst_blind, ep_vis = 0, 0, 0, []
            for t, q in enumerate(qpos):
                data.qpos[:] = q
                mujoco.mj_forward(model, data)
                ee = data.site_xpos[tip_site] if tip_site >= 0 else data.xpos[model.body("gripper_tip_link").id]
                pos = np.array([q[a:a + 3] for a in cube_adr])
                # stage: k = index in order of the cube being picked / carried next
                while k < len(order) and _stacked(pos, order, k, target_xy):
                    k += 1
                if k >= len(order):
                    break                                   # task done; the rest is the idle tail
                src = order[k]
                held = np.linalg.norm(pos[src] - ee) < HELD_MAX_DIST and pos[src][2] > task.CUBE_REST_Z + LIFTED
                phase = "carry" if held else "reach"
                if phase == "reach":
                    need = cube_geoms[src]
                elif k == 0:
                    need = marker_geoms
                else:
                    need = cube_geoms[order[k - 1]]
                seg.update_scene(data, camera=cam, scene_option=opt)
                ids = seg.render()
                objs = ids[..., 0][ids[..., 1] == mujoco.mjtObj.mjOBJ_GEOM]
                visible = np.isin(objs, list(need)).sum() >= min_pixels
                stats[phase][0] += visible
                stats[phase][1] += 1
                ep_vis.append(visible)
                blind = 0 if visible else blind + 1
                worst_blind = max(worst_blind, blind)
                if sheet and e == episodes[0] and t % 25 == 0:
                    rgb.update_scene(data, camera=cam, scene_option=opt)
                    frame = rgb.render().copy()
                    frame[:6] = (0, 200, 0) if visible else (220, 0, 0)    # bar: needed cube visible?
                    sheet_frames.append(frame)
            print(f"episode {e}: needed object visible in {np.mean(ep_vis) * 100:.0f}% of {len(ep_vis)} task ticks, "
                  f"longest blind stretch {worst_blind * 0.1:.1f} s", flush=True)
    for phase, (v, n) in stats.items():
        if n:
            print(f"{phase}: needed object visible in {v / n * 100:.0f}% of {n} ticks")
    if sheet and sheet_frames:
        import imageio.v2 as imageio
        cols = 6
        while len(sheet_frames) % cols:
            sheet_frames.append(np.zeros_like(sheet_frames[0]))
        rows = [np.hstack(sheet_frames[i:i + cols]) for i in range(0, len(sheet_frames), cols)]
        imageio.imwrite(os.path.expanduser(sheet), np.vstack(rows))
        print(f"wrote {sheet} (every 2.5 s of episode {episodes[0]}; green bar = needed object visible)")


def _stacked(pos, order, k, target_xy):
    """Whether order[k] is already in place: on the cube below, or (k = 0, marker world) resting
    on the marker. Without a marker the base is in place from the start."""
    c = pos[order[k]]
    if k == 0:
        return target_xy is None or (np.linalg.norm(c[:2] - target_xy) < task.MARKER_XY_TOL
                                     and abs(c[2] - task.CUBE_REST_Z) < 0.005)
    below = pos[order[k - 1]]
    return np.linalg.norm(c[:2] - below[:2]) < 0.01 and abs(c[2] - below[2] - task.CUBE_SIZE) < 0.005


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--episodes", type=int, nargs="*", default=None)
    p.add_argument("--scene", default=DEFAULT_OUT, help="offline scene dir (marker datasets: ~/.cache/stack_bc/scene_marker)")
    p.add_argument("--min-pixels", type=int, default=50)
    p.add_argument("--sheet", default=None, help="optional contact sheet PNG of sample frames")
    a = p.parse_args()
    run(a.dataset, os.path.expanduser(a.scene), a.episodes, a.min_pixels, a.sheet)


if __name__ == "__main__":
    main()
