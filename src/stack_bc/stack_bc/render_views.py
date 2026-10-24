"""Render any camera of a recorded stack_bc episode from its logged qpos (offline, no sim).

  # videos of the offline-only DP-station cameras (full D415 720p frame), episodes 0 and 1:
  python3 -m stack_bc.render_views --dataset ~/data/stack_bc/station.zarr --views topdown,side,front_b \\
      --size 1280x720 --episodes 0 1 --out ~/data/stack_bc/station_views
  # live vs offline: re-render a recorded camera and compare it with the dataset's frames:
  python3 -m stack_bc.render_views --dataset ~/data/stack_bc/station.zarr --check front
  # add an offline camera to the dataset as a policy-size array (data/<view>_rgb, 240x320, JPEG2k):
  python3 -m stack_bc.render_views --dataset ~/data/stack_bc/station.zarr --add front_b

Every tick's full MuJoCo qpos is logged (arm, gripper, cube and marker free joints), so a frame is
mj_forward on that qpos + a render of the chosen camera with the same compiled model the sim used
(states.dataset_model picks it from the dataset's tags). World cameras are fixed; the wrist camera's
pose comes from the forward kinematics. Colour domain randomization is reproduced from
meta/episode_rgba (randomize.COLOR_SLOTS). Time resolution is the recorder's 10 Hz tick.

Views: MJCF camera names (front_a, front_b, topdown, side in the station world) or "wrist" / "front"
(= the MJCF wrist camera "camera" / front_a). --size 640x480 (default) renders the policy framing,
stored/compared after the same 2x2 mean as the live path (camera.py); 1280x720 renders the full
16:9 frame of a fovy-42.5 (D415-like) camera, i.e. what the paper's cameras recorded before the
4:3 crop.
"""

import argparse
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as iio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import zarr  # noqa: E402

from stack_bc.image_codecs import register_image_codecs  # noqa: E402
from stack_bc.imageproc import IMAGE_SHAPE, downsample2  # noqa: E402
from stack_bc.randomize import COLOR_SLOTS  # noqa: E402
from stack_bc.recorder import compressor_and_chunks  # noqa: E402
from stack_bc.states import dataset_model, episode_slices  # noqa: E402

ALIASES = {"wrist": "camera", "front": "front_a"}
RECORDED = {"camera": "wrist_rgb", "front_a": "front_rgb"}   # MJCF camera -> live-recorded array
FPS = 10                                                       # recorder tick rate


class ViewRenderer:
    def __init__(self, model, width=640, height=480):
        self.m, self.d = model, mujoco.MjData(model)
        self.opt = mujoco.MjvOption()
        self.opt.sitegroup[:] = 0                       # sites are not in the live camera images either
        self.r = mujoco.Renderer(model, height=height, width=width)
        self.mat_ids = [model.material(mat).id for _, mat in COLOR_SLOTS]
        self.default_rgba = model.mat_rgba[self.mat_ids].copy()

    def set_colors(self, rgba):
        """rgba: (len(COLOR_SLOTS), 4) of one episode, or None for the world's own colours."""
        self.m.mat_rgba[self.mat_ids] = self.default_rgba if rgba is None else rgba

    def __call__(self, qpos, camera):
        self.d.qpos[:] = qpos
        mujoco.mj_forward(self.m, self.d)
        self.r.update_scene(self.d, camera=camera, scene_option=self.opt)
        return self.r.render()


def camera_name(view, model):
    name = ALIASES.get(view, view)
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) < 0:
        names = [model.camera(i).name for i in range(model.ncam)]
        raise ValueError(f"no camera {view!r} in this dataset's world (cameras: {names})")
    return name


def episode_colors(root, ep):
    return root["meta/episode_rgba"][ep] if "episode_rgba" in root["meta"] else None


def videos(root, views, size, episodes, out):
    model, _ = dataset_model(root)
    render = ViewRenderer(model, *size)
    cams = [camera_name(v, model) for v in views]
    slices = episode_slices(root)
    os.makedirs(out, exist_ok=True)
    for ep in episodes:
        render.set_colors(episode_colors(root, ep))
        qpos = root["data/qpos"][slices[ep]]
        for view, cam in zip(views, cams):
            path = os.path.join(out, f"ep{ep:04d}_{view}.mp4")
            with iio.get_writer(path, fps=FPS, quality=8, macro_block_size=8) as w:
                for q in qpos:
                    w.append_data(render(q, cam))
            print(f"wrote {path} ({len(qpos)} frames {size[0]}x{size[1]})", flush=True)


def check(root, view, max_frames=300, seed=0):
    """Offline re-render vs the live-recorded frames of a camera (policy size, after the 2x2 mean)."""
    model, _ = dataset_model(root)
    cam = camera_name(view, model)
    key = RECORDED.get(cam)
    if key is None or key not in root["data"]:
        raise ValueError(f"{view}: no live-recorded array to compare with (have {list(root['data'])})")
    render = ViewRenderer(model)
    ends = root["meta/episode_ends"][:]
    n = int(ends[-1])
    idx = np.sort(np.random.default_rng(seed).choice(n, size=min(n, max_frames), replace=False))
    diffs = []
    for i in idx:
        render.set_colors(episode_colors(root, int(np.searchsorted(ends, i, side="right"))))
        rend = downsample2(render(root["data/qpos"][i], cam)).astype(np.int16)
        live = root["data"][key][i].astype(np.int16)
        diffs.append(np.abs(rend - live))
    d = np.stack(diffs)
    per_frame = d.mean(axis=(1, 2, 3))
    print(f"{view} ({cam} vs data/{key}), {len(idx)} frames: mean |live - rendered| {d.mean():.2f} / 255 "
          f"(per frame median {np.median(per_frame):.2f}, worst {per_frame.max():.2f}); "
          f"pixels off by > 20: {100 * (d.max(-1) > 20).mean():.2f}%")


def add(root, view):
    """Write data/<view>_rgb (T, 240, 320, 3) for an offline camera, like a live-recorded one."""
    model, _ = dataset_model(root)
    cam = camera_name(view, model)
    key = f"{view}_rgb"
    render = ViewRenderer(model)
    n = root["data/qpos"].shape[0]
    compressor, chunks = compressor_and_chunks("wrist_rgb", np.zeros((1,) + IMAGE_SHAPE, np.uint8))
    arr = root["data"].create_dataset(key, shape=(n,) + IMAGE_SHAPE, dtype=np.uint8, chunks=chunks,
                                      compressor=compressor, overwrite=True)
    t0 = time.time()
    for ep, sl in enumerate(episode_slices(root)):
        render.set_colors(episode_colors(root, ep))
        arr[sl] = np.stack([downsample2(render(q, cam)) for q in root["data/qpos"][sl]])
    root.attrs[f"{key}_rendered"] = "offline"
    print(f"wrote data/{key} {arr.shape} in {time.time() - t0:.0f} s", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--views", help="comma-separated cameras to render as mp4 per episode (needs --out)")
    g.add_argument("--check", metavar="VIEW", help="compare a live-recorded camera (wrist / front) with its re-render")
    g.add_argument("--add", metavar="VIEW", help="add data/<VIEW>_rgb (policy size) rendered offline, in place")
    p.add_argument("--size", default="640x480", help="WxH for --views (1280x720 = full D415 frame)")
    p.add_argument("--episodes", type=int, nargs="+", default=[0])
    p.add_argument("--out", help="output directory for --views")
    a = p.parse_args()
    register_image_codecs()
    path = os.path.expanduser(a.dataset)
    if a.views:
        if not a.out:
            p.error("--views needs --out")
        size = tuple(int(v) for v in a.size.lower().split("x"))
        videos(zarr.open(path, "r"), a.views.split(","), size, a.episodes, os.path.expanduser(a.out))
    elif a.check:
        check(zarr.open(path, "r"), a.check)
    else:
        add(zarr.open_group(path, mode="a"), a.add)


if __name__ == "__main__":
    main()
