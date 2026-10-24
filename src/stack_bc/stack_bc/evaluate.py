"""Closed-loop policy rollout in the MuJoCo sim (sim + scripts/policy_server.py must be running).

  python3 -m stack_bc.evaluate --dataset ~/data/stack_bc/overfit.zarr --episode 0 [--record out.zarr]

Resets the scene to the initial cube layout of a recorded episode (xy + yaw of each cube, read
from that episode's first qpos row), then runs the policy with receding-horizon control as in the
paper (Sec. 2.3): every n_action_steps ticks, send the last n_obs_steps observations to the policy
server, execute the returned action chunk (one positions-only trajectory; the paper linearly
interpolates its 10 Hz setpoints the same way, App. D.0.1), and repeat. Observations are built
with stack_bc.spaces.state_vector - the same function the recorder used for training data.

Gripper: the demos only ever command fully open (85 mm) or closed (0 = force-limited squeeze), so
the predicted width is snapped to the nearer of the two; sending an intermediate width to the
position-controlled gripper would give a weak or no grasp.

Success: the same stack check as data collection, for ANY stack order (the policy isn't told the
demo's order), held for SUCCESS_HOLD_TICKS consecutive ticks.

Image policies (server info has obs_keys; dp/stack_dp/stack_image_dataset.py): the rollout starts
at task.SCAN_Q like `collect --wrist-images` episodes, and each query sends the last n_obs_steps
live wrist frames (camera.WristCamera, 240x320 uint8) + proprio (state_vector[:10]); the
overview + context variant also gets the frame of the first tick (the overview) as
"wrist_context", constant for the episode - exactly what the dataset feeds at training time.
--record then also stores the wrist frames. Policies with a "front" input (DP-paper station world,
dp_station_mujoco.xml) also get the station's fixed front camera (camera.FRONT_IMAGE_TOPIC).
"""

import argparse
import itertools
import os
import time
from collections import deque

import numpy as np
import zarr
import zmq

from stack_bc import rotations as rot
from stack_bc import spaces, task
from stack_bc.camera import CAMERA_SETTLE, DEPTH_TOPIC, FRONT_IMAGE_TOPIC, IMAGE_TOPIC, MAX_IMAGE_AGE, WristCamera
from stack_bc.env import StackEnv, has_marker, stack_success
from stack_bc.randomize import ColorRandomizer
from stack_bc.executor import EEExecutor, IKFailure
from stack_bc.gripper import WIDTH_CLOSED, WIDTH_OPEN
from stack_bc.recorder import EpisodeBuffer, ReplayWriter
from stack_bc.runtime import SimNode

DT = 0.1
SUCCESS_HOLD_TICKS = 10
GRIPPER_THRESHOLD = (WIDTH_OPEN + WIDTH_CLOSED) / 2


def initial_layout(dataset_path, episode, model):
    """(xy (N,2), yaw (N,), demo stack order, marker xy or None, episode rgba or None) of a
    recorded episode's first tick. The marker is only reproduced when the live sim is a marker
    world too."""
    root = zarr.open(os.path.expanduser(dataset_path), "r")
    ends = root["meta/episode_ends"][:]
    if not 0 <= episode < len(ends):
        raise ValueError(f"episode {episode} out of range (dataset has {len(ends)})")
    start = 0 if episode == 0 else int(ends[episode - 1])
    qpos = root["data/qpos"][start]
    xy, yaw = [], []
    for name in task.CUBE_NAMES:
        adr = model.jnt_qposadr[model.body(name).jntadr[0]]
        w, x, y, z = qpos[adr + 3:adr + 7]                    # MuJoCo free joint: pos + quat wxyz
        xy.append(qpos[adr:adr + 2])
        yaw.append(rot.yaw_of(rot.quat_to_matrix([x, y, z, w])))
    meta = root["meta"]
    order = [int(i) for i in meta["episode_order"][episode]] if "episode_order" in meta else None
    marker_xy = np.asarray(meta["episode_target"][episode]) if "episode_target" in meta else None
    if has_marker(model) != (marker_xy is not None):
        raise ValueError(f"{dataset_path} {'has' if marker_xy is not None else 'has no'} stack marker but the "
                         "running sim's world doesn't match (robot_mujoco.launch.py world:=...)")
    rgba = np.asarray(meta["episode_rgba"][episode]) if "episode_rgba" in meta else None
    return np.array(xy), np.array(yaw), order, marker_xy, rgba


class PolicyClient:
    def __init__(self, port, timeout_s=10.0):
        self.sock = zmq.Context().socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(f"tcp://127.0.0.1:{port}")

    def _call(self, req):
        self.sock.send_pyobj(req)
        try:
            rep = self.sock.recv_pyobj()
        except zmq.Again:
            raise RuntimeError("no reply from policy server - is scripts/policy_server.py running?")
        if "error" in rep:
            raise RuntimeError(f"policy server: {rep['error']}")
        return rep

    def info(self):
        return self._call({"cmd": "info"})

    def reset(self):
        """Start-of-episode: clears the policy's recurrent state (LSTM-GMM); no-op for DP."""
        self._call({"cmd": "reset"})

    def act(self, obs):
        # Image policies get a dict (uint8 frames + float proprio, see ObsHistory); low-dim a float array.
        payload = {k: np.asarray(v) for k, v in obs.items()} if isinstance(obs, dict) else np.asarray(obs, np.float32)
        rep = self._call({"cmd": "act", "obs": payload})
        return rep["action"], rep["latency_s"]


def any_order_success(snap, target_xy=None):
    return any(stack_success(snap, list(p), target_xy=target_xy) for p in itertools.permutations(range(task.N_CUBES)))


def is_image_policy(info):
    return bool(info.get("obs_keys"))


def policy_inputs(info):
    """Raw inputs an image policy wants (server info "inputs"; older servers: its obs keys)."""
    return set(info.get("inputs") or info.get("obs_keys") or ())


def needs_depth(info):
    return "wrist_depth" in policy_inputs(info)


def needs_front(info):
    return "front" in policy_inputs(info)


class ObsHistory:
    """The last n_obs_steps observations in the policy server's input format.

    Low-dim policies: (To, obs_dim) state_vector rows. Image policies: the server's raw "inputs":
    {"wrist": (To, H, W, 3) uint8, "proprio": (To, 10), ["wrist_depth": (To, H, W) uint16 mm],
    ["wrist_context": (To, H, W, 3)], ["front": (To, H, W, 3) uint8]}; the server builds the checkpoint's keys from them (e.g.
    wrist_rgbd). Like the training windows, the history starts filled with the first observation
    (pad_before repeats it); the context frame is the first tick's wrist frame, i.e. the overview
    at task.SCAN_Q. Colour and depth come from the same render (camera.latest_rgbd); a pair
    without a shared stamp counts as stale."""

    def __init__(self, info, camera=None, front_camera=None):
        self.To = info["n_obs_steps"]
        self.inputs = policy_inputs(info)
        if self.inputs and camera is None:
            raise ValueError("image policy needs a camera")
        self.depth = "wrist_depth" in self.inputs
        if self.depth and not camera.depth:
            raise ValueError("depth policy needs WristCamera(depth=True)")
        self.front_camera = front_camera if "front" in self.inputs else None
        if "front" in self.inputs and front_camera is None:
            raise ValueError("front-camera policy needs the station's front camera")
        self.camera = camera
        self.stale, self._last_stamp, self._last_front_stamp = 0, None, None
        self.front_frame = None   # newest (stamp, rgb) front frame, for --record

    def _frame(self, snap):
        if self.depth:
            stamp, rgb, depth, matched = self.camera.latest_rgbd()
        else:
            (stamp, rgb), depth, matched = self.camera.latest(), None, True
        if rgb is None:
            raise RuntimeError("no wrist image")
        if snap.stamp - stamp > MAX_IMAGE_AGE or stamp == self._last_stamp or not matched:
            self.stale += 1
        self._last_stamp = stamp
        return (stamp, rgb) if depth is None else (stamp, rgb, depth)

    def _front(self, snap):
        stamp, rgb = self.front_camera.latest()
        if rgb is None:
            raise RuntimeError("no front image")
        if snap.stamp - stamp > MAX_IMAGE_AGE or stamp == self._last_front_stamp:
            self.stale += 1
        self._last_front_stamp = stamp
        self.front_frame = (stamp, rgb)
        return rgb

    def start(self, snap):
        """First tick: fill the history with this observation. Returns the frame (or None)."""
        if not self.inputs:
            self.state = deque([spaces.state_vector(snap)] * self.To, maxlen=self.To)
            return None
        frame = self._frame(snap)
        self.context = frame[1]
        self.wrist = deque([frame[1]] * self.To, maxlen=self.To)
        if self.depth:
            self.wrist_depth = deque([frame[2]] * self.To, maxlen=self.To)
        if self.front_camera is not None:
            self.front = deque([self._front(snap)] * self.To, maxlen=self.To)
        self.proprio = deque([spaces.state_vector(snap)[:10]] * self.To, maxlen=self.To)
        return frame

    def push(self, snap):
        """Later ticks. Returns the frame (or None)."""
        if not self.inputs:
            self.state.append(spaces.state_vector(snap))
            return None
        frame = self._frame(snap)
        self.wrist.append(frame[1])
        if self.depth:
            self.wrist_depth.append(frame[2])
        if self.front_camera is not None:
            self.front.append(self._front(snap))
        self.proprio.append(spaces.state_vector(snap)[:10])
        return frame

    def obs(self):
        if not self.inputs:
            return np.stack(self.state).astype(np.float32)
        out = {"proprio": np.stack(self.proprio).astype(np.float32)}
        if "wrist" in self.inputs:
            out["wrist"] = np.stack(self.wrist)
        if self.depth:
            out["wrist_depth"] = np.stack(self.wrist_depth)
        if "wrist_context" in self.inputs:
            out["wrist_context"] = np.repeat(self.context[None], self.To, axis=0)
        if self.front_camera is not None:
            out["front"] = np.stack(self.front)
        return out


def rollout(sim, env, ex, client, info, xy, yaw, max_time, buf=None, log=print, marker_xy=None, colors=None,
            camera=None, front_camera=None):
    """marker_xy: stack-location marker (marker world; success requires the tower on it).
    colors: (randomize.ColorRandomizer, rgba) applied after the reset.
    camera: camera.WristCamera, required for image policies (which start at task.SCAN_Q).
    front_camera: camera.WristCamera on FRONT_IMAGE_TOPIC, required for policies with a "front" input."""
    To, Ta = info["n_obs_steps"], info["n_action_steps"]
    ex.move_joints(task.HOME_Q, 2.5)
    ex.set_width(WIDTH_OPEN, force=True)
    sim.sleep(2.8)
    env.reset_world(xy, yaw, marker_xy)
    if colors is not None:
        colors[0].apply(colors[1])
    sim.sleep(0.5)
    if is_image_policy(info):
        ex.move_joints(task.SCAN_Q, task.SCAN_MOVE_TIME)   # as collect --wrist-images episodes start
        sim.sleep(task.SCAN_MOVE_TIME + CAMERA_SETTLE)
    snap = env.snapshot()
    ex.sync(snap.q_arm, snap.ee)

    client.reset()
    history = ObsHistory(info, camera, front_camera)
    chunk, targets, latencies = None, None, []
    hold, reason, success = 0, "timeout", False
    t_start = t_next = sim.now()
    tick = 0
    while sim.now() - t_start < max_time:
        sim.wait_until(t_next)
        t_next += DT
        snap = env.snapshot()
        frame = history.start(snap) if tick == 0 else history.push(snap)

        if tick % Ta == 0:
            chunk, lat = client.act(history.obs())
            latencies.append(lat)
            try:
                targets = ex.send_chunk([spaces.action_to_target(a)[0] for a in chunk])
            except IKFailure as exc:
                reason = f"IK: {exc}"
                break
        i = tick % Ta
        width = WIDTH_OPEN if chunk[i][9] > GRIPPER_THRESHOLD else WIDTH_CLOSED
        ex.set_width(width)
        if buf is not None:
            buf.record(snap, targets[i], width, image=frame, front=history.front_frame)

        hold = hold + 1 if any_order_success(snap, marker_xy) else 0
        if hold >= SUCCESS_HOLD_TICKS:
            success, reason = True, "ok"
            break
        tick += 1
    return dict(success=success, reason=reason, sim_time=sim.now() - t_start, ticks=tick,
                latency_mean=float(np.mean(latencies)) if latencies else float("nan"),
                latency_max=float(np.max(latencies)) if latencies else float("nan"),
                final_cube_pos=env.snapshot().cube_pos, stale_images=history.stale)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="zarr whose episode's initial layout to reproduce")
    parser.add_argument("--episode", type=int, nargs="+", default=[0], help="episode index(es) in --dataset")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--max-time", type=float, default=60.0, help="s of sim time per rollout")
    parser.add_argument("--record", default=None, help="optional zarr to save rollouts to (demo format)")
    args = parser.parse_args()

    sim = SimNode("stack_bc_evaluate")
    env = StackEnv(sim.node, sim.clock)
    ex = EEExecutor(sim.node, env.kin, dt=DT)
    ex.wait_for_servers()
    if not sim.wait_for(env.ready, 10.0):
        raise RuntimeError("no /joint_states or cube poses")
    client = PolicyClient(args.port)
    info = client.info()
    print(f"policy: {info}", flush=True)
    camera = None
    if is_image_policy(info):
        camera = WristCamera(sim.node, depth=needs_depth(info))
        if not sim.wait_for(camera.ready, 10.0):
            raise RuntimeError(f"image policy but no wrist images on {IMAGE_TOPIC}"
                               + (f" / {DEPTH_TOPIC}" if needs_depth(info) else ""))
    front_camera = None
    if needs_front(info):
        if not env.has_station:
            raise RuntimeError("the policy takes the front camera: launch the station world "
                               "(world:=dp_station_mujoco.xml)")
        front_camera = WristCamera(sim.node, FRONT_IMAGE_TOPIC, name="front")
        if not sim.wait_for(front_camera.ready, 10.0):
            raise RuntimeError(f"no front images on {FRONT_IMAGE_TOPIC}")
    writer = None
    if args.record:
        writer = ReplayWriter(os.path.expanduser(args.record),
                              images=("wrist_rgb,front_rgb" if front_camera else "wrist_rgb") if camera else "none",
                              start="scan" if camera else "home",
                              depth="wrist_depth" if camera is not None and camera.depth else "none",
                              station="dp" if env.has_station else "off")

    results = []
    randomizer = None
    for ep in args.episode:
        xy, yaw, demo_order, marker_xy, rgba = initial_layout(args.dataset, ep, env.model)
        if rgba is not None and randomizer is None:
            randomizer = ColorRandomizer(sim.node)
        buf = EpisodeBuffer(env.model) if writer else None
        t0 = time.time()
        res = rollout(sim, env, ex, client, info, xy, yaw, args.max_time, buf, marker_xy=marker_xy,
                      colors=(randomizer, rgba) if rgba is not None else None, camera=camera,
                      front_camera=front_camera)
        results.append(res)
        if writer and len(buf):
            extra_meta = {}
            if marker_xy is not None:
                extra_meta["episode_target"] = marker_xy
            if rgba is not None:
                extra_meta["episode_rgba"] = rgba
            writer.append_episode(buf.arrays(), res["success"], ep, demo_order or [-1] * task.N_CUBES, extra_meta)
        print(f"episode {ep} (demo order {demo_order}): {'SUCCESS' if res['success'] else 'FAIL'} "
              f"({res['reason']}) | {res['sim_time']:.1f}s sim, {time.time() - t0:.0f}s wall | "
              f"policy latency mean {res['latency_mean'] * 1000:.0f} ms, max {res['latency_max'] * 1000:.0f} ms"
              + (f" | stale wrist frames {res['stale_images']}" if camera else ""), flush=True)
        print("  final cube positions:", np.round(res["final_cube_pos"], 3).tolist(), flush=True)
    n_ok = sum(r["success"] for r in results)
    print(f"success {n_ok}/{len(results)}", flush=True)
    sim.shutdown()


if __name__ == "__main__":
    main()
