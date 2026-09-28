"""Collect scripted stacking demonstrations into a zarr ReplayBuffer (sim must be running).

  python3 -m stack_bc.collect --out ~/data/stack_bc/debug.zarr --episodes 200 [--seed 0] [--max-attempts N]

The oracle moves the robot through MoveIt (Pilz LIN trajectories, see motion.py) in the main
thread; a recorder thread samples a fixed 10 Hz sim-time grid meanwhile, logging the observation
and the DESIRED action at each tick: the planned trajectory's pose at t + dt, and the commanded
gripper width (paper Sec. 7.1 / App. D.0.1 - actions are setpoints, not measured next states).

Only SUCCESSFUL episodes are written (--episodes counts successes); failed attempts are reported
and dropped. Each attempt uses the next seed (for cube placement and stack order), and the next
unused seed is stored in the dataset's attrs, so re-running on the same --out resumes without
repeating seeds and any stored episode can be regenerated from meta/episode_seed. An episode
succeeds when, after the oracle finishes, the stack check holds for SUCCESS_HOLD_TICKS consecutive
ticks; recording continues through that window (idle actions at the end teach the policy to stop).
"""

import argparse
import os
import sys
import threading
import time

import numpy as np

from stack_bc import task
from stack_bc.env import StackEnv, stack_success
from stack_bc.executor import EEExecutor
from stack_bc.gripper import WIDTH_OPEN
from stack_bc.motion import MoveItMotion, PlanningFailure
from stack_bc.oracle import OracleFailure, ScriptedStackOracle
from stack_bc.recorder import EpisodeBuffer, ReplayWriter
from stack_bc.runtime import SimNode

DT = 0.1
SUCCESS_HOLD_TICKS = 10
POST_TASK_TIME = 2.0          # s after the oracle finishes to wait for a stable stack
MAX_EPISODE_TIME = 120.0      # s sim time
# Timing guard: the sim runs freely in real time, so if the recorder stalls, ticks land late.
# A tick is irregular when its sim time is more than TICK_TOLERANCE from its 10 Hz slot; episodes
# with more than MAX_IRREGULAR_TICKS are marked failed ("timing") and filtered out of training.
TICK_TOLERANCE = 0.02
MAX_IRREGULAR_TICKS = 3


class TickRecorder(threading.Thread):
    """Samples (observation, desired action) on a fixed 10 Hz sim-time grid until stopped."""

    def __init__(self, sim, env, motion, gripper, buf):
        super().__init__(daemon=True)
        self.sim, self.env, self.motion, self.gripper, self.buf = sim, env, motion, gripper, buf
        self._halt = threading.Event()  # not "_stop": that name is Thread-internal
        self.irregular, self.worst = 0, 0.0
        self.error = None

    def run(self):
        try:
            t_next = self.sim.now()
            while not self._halt.is_set():
                self.sim.wait_until(t_next)
                if self._halt.is_set():
                    break
                snap = self.env.snapshot()
                lag = snap.stamp - t_next
                self.worst = max(self.worst, lag)
                if abs(lag) > TICK_TOLERANCE:
                    self.irregular += 1
                q_des = self.motion.desired_q(t_next + DT)
                T_des = self.env.kin.fk_world(q_des) if q_des is not None else snap.ee
                self.buf.record(snap, T_des, self.gripper.width_cmd)
                t_next += DT
        except Exception as exc:  # surface recorder crashes to the episode loop
            self.error = exc

    def stop(self):
        self._halt.set()
        self.join(timeout=2.0)


def home_and_reset(sim, env, motion, ex, rng):
    snap = env.snapshot()
    if np.max(np.abs(snap.q_arm - task.HOME_Q)) > 1e-3:
        ex.move_joints(task.HOME_Q, 2.5)  # unrecorded; see motion.py on why not MoveIt PTP
        ex.set_width(WIDTH_OPEN, force=True)
        sim.sleep(2.8)
    else:
        ex.set_width(WIDTH_OPEN, force=True)
        sim.sleep(0.5)
    xy, yaw = env.sample_cubes(rng)
    env.reset_world(xy, yaw)
    motion.forget()
    sim.sleep(0.5)  # contacts settle


def run_episode(sim, env, motion, ex, seed):
    rng = np.random.default_rng(seed)
    home_and_reset(sim, env, motion, ex, rng)
    order = [int(i) for i in rng.permutation(task.N_CUBES)]
    buf = EpisodeBuffer(env.model)
    oracle = ScriptedStackOracle(sim, env, motion, ex)
    rec = TickRecorder(sim, env, motion, ex, buf)
    rec.start()
    t_start = sim.now()

    success, reason = False, "ok"
    try:
        oracle.run(order)
        hold, t_end = 0, sim.now() + POST_TASK_TIME
        while sim.now() < t_end:
            sim.sleep(DT)
            hold = hold + 1 if stack_success(env.snapshot(), order) else 0
            if hold >= SUCCESS_HOLD_TICKS:
                success = True
                break
        if not success:
            reason = "stack not stable after release"
    except (OracleFailure, PlanningFailure) as exc:
        reason = f"{exc} (phase {oracle.phase})"
    finally:
        rec.stop()
    if sim.now() - t_start > MAX_EPISODE_TIME:
        success, reason = False, "episode too long"
    if rec.error is not None:
        success, reason = False, f"recorder crashed: {rec.error!r}"
    elif rec.irregular > MAX_IRREGULAR_TICKS:
        success, reason = False, (f"timing: {rec.irregular} ticks off-slot by > {TICK_TOLERANCE * 1000:.0f} ms "
                                  f"(worst {rec.worst * 1000:.0f} ms)")
    return buf, success, reason, order, rec


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=10, help="successful episodes to add")
    parser.add_argument("--max-attempts", type=int, default=None, help="give up after this many attempts "
                        "(default: 2x --episodes)")
    parser.add_argument("--seed", type=int, default=0, help="first seed for a NEW dataset (ignored on resume)")
    args = parser.parse_args()
    max_attempts = args.max_attempts or 2 * args.episodes

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    writer = ReplayWriter(out)
    writer.truncate_partial()
    seed = int(writer.root.attrs.get("next_seed", args.seed))

    sim = SimNode("stack_bc_collect")
    env = StackEnv(sim.node, sim.clock)
    ex = EEExecutor(sim.node, env.kin, dt=DT)
    ex.wait_for_servers()
    if not sim.wait_for(env.ready, 10.0):
        raise RuntimeError("no /joint_states or cube poses")
    motion = MoveItMotion(env.urdf, sim.now)

    added, attempts = 0, 0
    while added < args.episodes and attempts < max_attempts:
        t0 = time.time()
        buf, success, reason, order, rec = run_episode(sim, env, motion, ex, seed)
        attempts += 1
        timing = f" [{rec.irregular} off-slot ticks, worst {rec.worst * 1000:.0f} ms]" if rec.irregular else ""
        if success:
            writer.append_episode(buf.arrays(), True, seed, order)
            added += 1
        writer.root.attrs["next_seed"] = seed + 1  # after the episode is committed (or dropped)
        print(f"seed {seed} order {order}: {'OK  ' if success else 'FAIL'} {len(buf)} steps "
              f"({time.time() - t0:.0f}s wall) {reason}{timing}  "
              f"[added {added}/{args.episodes}, attempts {attempts}, dataset {writer.n_episodes} episodes]",
              flush=True)
        seed += 1
    if added < args.episodes:
        print(f"stopped after {attempts} attempts with {added}/{args.episodes} successes", flush=True)
    # No explicit motion/rclpy shutdown: moveit_py crashes inside it (see __main__ below). Return the
    # ROS/MoveIt objects so they stay referenced - letting them be garbage-collected as main()
    # returns runs MoveItPy's C++ destructor, which segfaults before os._exit is reached.
    return sim, env, motion


if __name__ == "__main__":
    # moveit_py segfaults/aborts on teardown - in MoveItPy.shutdown(), rclpy shutdown, or interpreter
    # finalization, whatever the order (reproduced with a bare init/shutdown script). Every episode
    # is already committed to zarr by then, so skip teardown entirely and exit with a real code.
    code = 0
    keepalive = None
    try:
        keepalive = main()  # noqa: F841 - held until os._exit, see main()
    except BaseException:
        import traceback
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
