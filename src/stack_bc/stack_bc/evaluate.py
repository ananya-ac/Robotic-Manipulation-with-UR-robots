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
from stack_bc.env import StackEnv, stack_success
from stack_bc.executor import EEExecutor, IKFailure
from stack_bc.gripper import WIDTH_CLOSED, WIDTH_OPEN
from stack_bc.recorder import EpisodeBuffer, ReplayWriter
from stack_bc.runtime import SimNode

DT = 0.1
SUCCESS_HOLD_TICKS = 10
GRIPPER_THRESHOLD = (WIDTH_OPEN + WIDTH_CLOSED) / 2


def initial_layout(dataset_path, episode, model):
    """(xy (N,2), yaw (N,), demo stack order) of a recorded episode's first tick."""
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
    order = [int(i) for i in root["meta/episode_order"][episode]] if "episode_order" in root["meta"] else None
    return np.array(xy), np.array(yaw), order


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

    def act(self, obs):
        rep = self._call({"cmd": "act", "obs": np.asarray(obs, dtype=np.float32)})
        return rep["action"], rep["latency_s"]


def any_order_success(snap):
    return any(stack_success(snap, list(p)) for p in itertools.permutations(range(task.N_CUBES)))


def rollout(sim, env, ex, client, info, xy, yaw, max_time, buf=None, log=print):
    To, Ta = info["n_obs_steps"], info["n_action_steps"]
    ex.move_joints(task.HOME_Q, 2.5)
    ex.set_width(WIDTH_OPEN, force=True)
    sim.sleep(2.8)
    env.reset_world(xy, yaw)
    sim.sleep(0.5)
    snap = env.snapshot()
    ex.sync(snap.q_arm, snap.ee)

    history = deque([spaces.state_vector(snap)] * To, maxlen=To)  # pad like training (repeat first)
    chunk, targets, latencies = None, None, []
    hold, reason, success = 0, "timeout", False
    t_start = t_next = sim.now()
    tick = 0
    while sim.now() - t_start < max_time:
        sim.wait_until(t_next)
        t_next += DT
        snap = env.snapshot()
        history.append(spaces.state_vector(snap))

        if tick % Ta == 0:
            chunk, lat = client.act(np.stack(history))
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
            buf.record(snap, targets[i], width)

        hold = hold + 1 if any_order_success(snap) else 0
        if hold >= SUCCESS_HOLD_TICKS:
            success, reason = True, "ok"
            break
        tick += 1
    return dict(success=success, reason=reason, sim_time=sim.now() - t_start, ticks=tick,
                latency_mean=float(np.mean(latencies)) if latencies else float("nan"),
                latency_max=float(np.max(latencies)) if latencies else float("nan"),
                final_cube_pos=env.snapshot().cube_pos)


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
    writer = ReplayWriter(os.path.expanduser(args.record)) if args.record else None

    results = []
    for ep in args.episode:
        xy, yaw, demo_order = initial_layout(args.dataset, ep, env.model)
        buf = EpisodeBuffer(env.model) if writer else None
        t0 = time.time()
        res = rollout(sim, env, ex, client, info, xy, yaw, args.max_time, buf)
        results.append(res)
        if writer and len(buf):
            writer.append_episode(buf.arrays(), res["success"], ep, demo_order or [-1] * task.N_CUBES)
        print(f"episode {ep} (demo order {demo_order}): {'SUCCESS' if res['success'] else 'FAIL'} "
              f"({res['reason']}) | {res['sim_time']:.1f}s sim, {time.time() - t0:.0f}s wall | "
              f"policy latency mean {res['latency_mean'] * 1000:.0f} ms, max {res['latency_max'] * 1000:.0f} ms",
              flush=True)
        print("  final cube positions:", np.round(res["final_cube_pos"], 3).tolist(), flush=True)
    n_ok = sum(r["success"] for r in results)
    print(f"success {n_ok}/{len(results)}", flush=True)
    sim.shutdown()


if __name__ == "__main__":
    main()
