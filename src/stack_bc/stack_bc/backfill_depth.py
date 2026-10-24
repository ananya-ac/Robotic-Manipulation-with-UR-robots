"""Add wrist depth to an existing wrist-image dataset by rendering it from the logged qpos (offline).

  python3 -m stack_bc.backfill_depth --dataset ~/data/stack_bc/wrist.zarr --out ~/data/stack_bc/wrist_depth.zarr
  python3 -m stack_bc.backfill_depth --dataset <live-depth dataset> --check     # rendered vs recorded

Every tick's full MuJoCo qpos is in the dataset, so the wrist camera's depth can be rendered with the
same model the sim compiled (offline scene from stack_bc.scene; marker datasets use the marker
world). Each frame goes through imageproc.depth_frame, exactly as live frames from
camera.WristCamera do (uint16 mm, 2x2 minimum), and is written as data/wrist_depth with the
recorder's compression, so a backfilled dataset reads like a live --wrist-depth one. Colours (domain
randomization) don't affect depth, so they are not needed.

--out copies the dataset first (the source is never modified); --in-place adds the array to the
dataset itself. --check renders and compares against an existing (live) data/wrist_depth without
writing: it validates that the sim plugin's depth convention matches the offline renderer.
"""

import argparse
import os
import shutil
import time

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import zarr  # noqa: E402

from stack_bc.imageproc import DEPTH_SHAPE, SOURCE_SHAPE, depth_frame  # noqa: E402
from stack_bc.image_codecs import register_image_codecs  # noqa: E402
from stack_bc.recorder import compressor_and_chunks  # noqa: E402
from stack_bc.states import dataset_model  # noqa: E402

CAMERA = "camera"     # the wrist camera in the compiled MJCF (CameraPlugin renders the same one)
BACKFILL_TAG = "wrist_depth_backfilled"


class DepthRenderer:
    def __init__(self, model):
        self.m = model
        self.d = mujoco.MjData(model)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = model.camera(CAMERA).id
        self.opt = mujoco.MjvOption()
        self.opt.sitegroup[:] = 0          # sites are not in the live camera images either
        self.r = mujoco.Renderer(model, height=SOURCE_SHAPE[0], width=SOURCE_SHAPE[1])
        self.r.enable_depth_rendering()

    def __call__(self, qpos):
        """(240, 320) uint16 mm depth frame for one qpos row."""
        self.d.qpos[:] = qpos
        mujoco.mj_forward(self.m, self.d)
        self.r.update_scene(self.d, camera=self.cam, scene_option=self.opt)
        return depth_frame(self.r.render())


def model_for(root):
    return dataset_model(root)[0]


def backfill(path, overwrite=False, log_every=2000):
    register_image_codecs()
    root = zarr.open_group(path, mode="a")
    if "wrist_depth" in root["data"] and not overwrite:
        raise ValueError(f"{path} already has data/wrist_depth (attrs depth={root.attrs.get('depth')}); "
                         "pass --overwrite to replace it")
    render = DepthRenderer(model_for(root))
    qpos = root["data/qpos"]
    n = qpos.shape[0]
    dummy = np.zeros((1,) + DEPTH_SHAPE, np.uint16)
    compressor, chunks = compressor_and_chunks("wrist_depth", dummy)
    arr = root["data"].create_dataset("wrist_depth", shape=(n,) + DEPTH_SHAPE, dtype=np.uint16, chunks=chunks,
                                      compressor=compressor, overwrite=True)
    t0 = time.time()
    for start in range(0, n, 1000):                   # read qpos in blocks, write frame by frame
        block = qpos[start:start + 1000]
        arr[start:start + len(block)] = np.stack([render(q) for q in block])
        if (start // 1000) % max(1, log_every // 1000) == 0:
            print(f"  {start + len(block)}/{n} frames ({(time.time() - t0) / (start + len(block)) * 1000:.1f} ms/frame)",
                  flush=True)
    root.attrs["depth"] = BACKFILL_TAG
    print(f"wrote data/wrist_depth {arr.shape} to {path} in {time.time() - t0:.0f} s", flush=True)


def check(path, max_frames=500, seed=0):
    """Rendered vs recorded depth on a sample of ticks (only meaningful for live-recorded depth)."""
    register_image_codecs()
    root = zarr.open(path, "r")
    if "wrist_depth" not in root["data"]:
        raise ValueError(f"{path} has no data/wrist_depth to check against")
    render = DepthRenderer(model_for(root))
    n = root["data/qpos"].shape[0]
    idx = np.sort(np.random.default_rng(seed).choice(n, size=min(n, max_frames), replace=False))
    diffs, mask_agree, n_valid = [], [], 0
    for i in idx:
        live = root["data/wrist_depth"][i].astype(np.int32)
        rend = render(root["data/qpos"][i]).astype(np.int32)
        both = (live > 0) & (rend > 0)
        mask_agree.append(np.mean((live > 0) == (rend > 0)))
        diffs.append(np.abs(live[both] - rend[both]))
        n_valid += both.sum()
    d = np.concatenate(diffs)
    print(f"{len(idx)} frames, {n_valid} valid pixel pairs: |live - rendered| median {np.median(d):.1f} mm, "
          f"p95 {np.percentile(d, 95):.1f} mm, max {d.max()} mm; valid-mask agreement {np.mean(mask_agree) * 100:.2f}% "
          f"(dataset attrs depth={root.attrs.get('depth')})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--out", help="write a copy with data/wrist_depth here (the source is not modified)")
    g.add_argument("--in-place", action="store_true", help="add data/wrist_depth to --dataset itself")
    g.add_argument("--check", action="store_true", help="compare rendered depth with the recorded data/wrist_depth")
    p.add_argument("--overwrite", action="store_true", help="replace an existing data/wrist_depth")
    a = p.parse_args()
    src = os.path.expanduser(a.dataset)
    if a.check:
        check(src)
        return
    if a.out:
        dst = os.path.expanduser(a.out)
        if os.path.exists(dst):
            raise SystemExit(f"{dst} exists; choose a new --out")
        print(f"copying {src} -> {dst}", flush=True)
        shutil.copytree(src, dst)
        src = dst
    backfill(src, overwrite=a.overwrite)


if __name__ == "__main__":
    main()
