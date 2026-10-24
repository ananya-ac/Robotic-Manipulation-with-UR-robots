"""Collect scripted stacking demonstrations into a zarr ReplayBuffer (sim must be running).

  python3 -m stack_bc.collect --out ~/data/stack_bc/debug.zarr --episodes 200 [--seed 0] [--max-attempts N]
      [--oracle obj_unknown]   # perception-driven oracle (see oracle_unknown.py); implies --scene sensors
      [--scene sensors]   # plan against a wrist-depth octomap; needs
                          # ros2 launch ur_gripper_sim_moveit_config ur_moveit.launch.py ur_type:=ur5e gripper:=robotiq_2f85 sensors:=true
      [--transit ompl]    # obj_known: OMPL transits against the ground-truth world (scene_gt.py)
                          # instead of Pilz rise/over/down; no extra launch needed
      [--randomize-colors]   # per-episode cube / table / marker colours (mujoco_dr_plugin)
      [--wrist-images]       # record the live wrist camera (data/wrist_rgb); episodes start at
                             # task.SCAN_Q; needs --transit ompl
      [--wrist-depth]        # + the wrist depth from the same render (data/wrist_depth)
      [--front-images]       # + the station's fixed front camera (data/front_rgb); needs the
                             #   DP-paper station world (world:=dp_station_mujoco.xml) and --wrist-images

Marker world (the tower is built on a randomized stack-location marker) is chosen by the sim:
  ros2 launch ur_bringup_sim robot_mujoco.launch.py world:=tabletop_marker_mujoco.xml
collect detects it, samples the marker with the cubes, and has the oracle place the base on it.
DP-paper station world (marker task + the Diffusion Policy paper's UR5 camera rig; the wrist camera
gets the D415 fovy 42.5 automatically):
  ros2 launch ur_bringup_sim robot_mujoco.launch.py world:=dp_station_mujoco.xml
  python3 -m stack_bc.collect --out ~/data/stack_bc/station.zarr --transit ompl --wrist-images --front-images
Its offline-only cameras (front_b, topdown, side) are rendered from the logged qpos afterwards
(stack_bc.render_views).

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
from stack_bc.oracle_unknown import PerceptionStackOracle
from stack_bc.active_scan import ActiveScanner
from stack_bc.perception_client import PerceptionClient
from stack_bc.camera import CAMERA_SETTLE, DEPTH_TOPIC, FRONT_IMAGE_TOPIC, IMAGE_TOPIC, MAX_IMAGE_AGE, WristCamera
from stack_bc.randomize import ColorRandomizer, sample_colors
from stack_bc.recorder import EpisodeBuffer, ReplayWriter
from stack_bc.runtime import SimNode
from stack_bc.scene_gt import GTScene
from stack_bc.scene_sensing import SceneManager

DT = 0.1
SUCCESS_HOLD_TICKS = 10
POST_TASK_TIME = 2.0          # s after the oracle finishes to wait for a stable stack
MAX_EPISODE_TIME = 120.0      # s sim time
# Timing guard: the sim runs freely in real time, so if the recorder stalls, ticks land late.
# A tick is irregular when its sim time is more than TICK_TOLERANCE from its 10 Hz slot; episodes
# with more than MAX_IRREGULAR_TICKS are marked failed ("timing") and filtered out of training.
TICK_TOLERANCE = 0.02
MAX_IRREGULAR_TICKS = 3
SCAN_MOVE_TIME = task.SCAN_MOVE_TIME   # unrecorded joint moves to / from task.SCAN_Q
# Image guard (--wrist-images): episodes with more than MAX_STALE_IMAGES stale wrist frames
# (camera.MAX_IMAGE_AGE) fail ("timing") like the tick guard above.
MAX_STALE_IMAGES = 3


class TickRecorder(threading.Thread):
    """Samples (observation, desired action) on a fixed 10 Hz sim-time grid until stopped."""

    def __init__(self, sim, env, motion, gripper, buf, perceived=None, camera=None, front=None):
        """perceived: optional callable -> the oracle's perceived object poses (logged per tick).
        camera: optional camera.WristCamera -> the newest wrist frame is logged per tick.
        front: optional camera.WristCamera on FRONT_IMAGE_TOPIC -> the newest front frame too (its
        stale frames count towards the same MAX_STALE_IMAGES guard)."""
        self.front = front
        self._last_front_stamp = None
        super().__init__(daemon=True)
        self.sim, self.env, self.motion, self.gripper, self.buf = sim, env, motion, gripper, buf
        self.perceived = perceived
        self.camera = camera
        self._halt = threading.Event()  # not "_stop": that name is Thread-internal
        self.irregular, self.worst = 0, 0.0
        self.stale_images, self.worst_image_age = 0, 0.0
        self._last_image_stamp = None
        self.error = None

    def _image(self, snap):
        """(stamp, rgb) or, with a depth camera, (stamp, rgb, depth_mm) from one render. A colour /
        depth pair without a shared stamp counts as stale, like an old or repeated frame."""
        if self.camera.depth:
            stamp, rgb, depth, matched = self.camera.latest_rgbd()
        else:
            (stamp, rgb), depth, matched = self.camera.latest(), None, True
        if rgb is None:
            raise RuntimeError("no wrist image received")
        age = snap.stamp - stamp
        self.worst_image_age = max(self.worst_image_age, age)
        if age > MAX_IMAGE_AGE or stamp == self._last_image_stamp or not matched:
            self.stale_images += 1
        self._last_image_stamp = stamp
        return (stamp, rgb) if depth is None else (stamp, rgb, depth)

    def _front(self, snap):
        stamp, rgb = self.front.latest()
        if rgb is None:
            raise RuntimeError("no front image received")
        age = snap.stamp - stamp
        self.worst_image_age = max(self.worst_image_age, age)
        if age > MAX_IMAGE_AGE or stamp == self._last_front_stamp:
            self.stale_images += 1
        self._last_front_stamp = stamp
        return stamp, rgb

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
                self.buf.record(snap, T_des, self.gripper.width_cmd,
                                self.perceived() if self.perceived else None,
                                self._image(snap) if self.camera is not None else None,
                                self._front(snap) if self.front is not None else None)
                t_next += DT
        except Exception as exc:  # surface recorder crashes to the episode loop
            self.error = exc

    def stop(self):
        self._halt.set()
        self.join(timeout=2.0)


def scan_scene(sim, ex, scene):
    """Rebuild the octomap for the freshly reset world from the wrist camera (unrecorded)."""
    scene.detach_cube()   # an episode that failed mid-carry leaves the cube attached
    scene.clear()
    ex.move_joints(task.SCAN_Q, SCAN_MOVE_TIME)
    sim.sleep(SCAN_MOVE_TIME + 0.3)
    scene.wait_for_map()
    ex.move_joints(task.HOME_Q, SCAN_MOVE_TIME)
    sim.sleep(SCAN_MOVE_TIME + 0.3)


def home_and_reset(sim, env, motion, ex, rng, scene=None, scanner=None, colors=None, start_scan=False):
    """Reset the world (unrecorded). Returns (perceived, marker_xy). scanner (oracle_obj_unknown):
    multi-view scan instead of the plain octomap scan; perceived is its {name: Perceived}, else
    None. marker_xy: the stack-location marker's sampled xy (marker world), else None.
    colors: (randomize.ColorRandomizer, rgba) to apply after the reset (colour DR).
    start_scan (--wrist-images): finish at task.SCAN_Q, where the wrist camera sees the whole
    spawn region, so the recorded episode's first frame is that overview."""
    snap = env.snapshot()
    if np.max(np.abs(snap.q_arm - task.HOME_Q)) > 1e-3:
        ex.move_joints(task.HOME_Q, 2.5)  # unrecorded; see motion.py on why not MoveIt PTP
        ex.set_width(WIDTH_OPEN, force=True)
        sim.sleep(2.8)
    else:
        ex.set_width(WIDTH_OPEN, force=True)
        sim.sleep(0.5)
    xy, yaw, marker_xy = env.sample_layout(rng, env.has_marker)
    env.reset_world(xy, yaw, marker_xy)
    if colors is not None:
        colors[0].apply(colors[1])
    motion.forget()
    sim.sleep(0.5)  # contacts settle
    if scanner is not None:
        return scanner.scan(), marker_xy
    if isinstance(scene, SceneManager):
        scan_scene(sim, ex, scene)
    elif scene is not None:
        scene.clear()           # GTScene: nothing held; the world is rebuilt before every plan
    if start_scan:
        ex.move_joints(task.SCAN_Q, SCAN_MOVE_TIME)
        sim.sleep(SCAN_MOVE_TIME + CAMERA_SETTLE)
    return None, marker_xy


def run_episode(sim, env, motion, ex, seed, scene=None, scanner=None, perception=None, transit="lin",
                randomizer=None, camera=None, front=None):
    """randomizer: a randomize.ColorRandomizer -> colour DR, with colours drawn from their own rng
    stream ([seed, 1]), so layouts and stack orders are the same with or without DR.
    camera: a camera.WristCamera -> wrist images recorded every tick; episodes start at SCAN_Q.
    front: a camera.WristCamera on the station's front camera -> also recorded every tick."""
    rng = np.random.default_rng(seed)
    rgba = sample_colors(np.random.default_rng([seed, 1])) if randomizer is not None else None
    scan_error, marker_xy = None, None
    try:
        perceived, marker_xy = home_and_reset(sim, env, motion, ex, rng, scene, scanner,
                                              (randomizer, rgba) if randomizer is not None else None,
                                              start_scan=camera is not None)
    except (RuntimeError, PlanningFailure) as exc:
        if scanner is None:
            raise
        perceived, scan_error = {}, exc        # the episode fails below; the run continues
    order = [int(i) for i in rng.permutation(task.N_CUBES)]   # after sample_layout: same seeds -> same episodes
    extra_meta = {}
    if marker_xy is not None:
        extra_meta["episode_target"] = np.asarray(marker_xy, dtype=np.float64)
    if rgba is not None:
        extra_meta["episode_rgba"] = rgba
    buf = EpisodeBuffer(env.model)
    if scanner is None:
        oracle = ScriptedStackOracle(sim, env, motion, ex, scene, transit=transit)
        rec = TickRecorder(sim, env, motion, ex, buf, camera=camera, front=front)
    else:
        oracle = PerceptionStackOracle(sim, env, motion, ex, scene, perception, perceived)
        rec = TickRecorder(sim, env, motion, ex, buf, perceived=oracle.perceived_state)
    rec.start()
    t_start = sim.now()

    success, reason = False, "ok"
    try:
        if scan_error is not None:
            raise OracleFailure(f"scan: {scan_error}")
        if marker_xy is not None:
            oracle.run(order, target_xy=marker_xy)
        else:
            oracle.run(order)
        hold, t_end = 0, sim.now() + POST_TASK_TIME
        while sim.now() < t_end:
            sim.sleep(DT)
            hold = hold + 1 if stack_success(env.snapshot(), order, target_xy=marker_xy) else 0
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
    elif rec.stale_images > MAX_STALE_IMAGES:
        success, reason = False, (f"timing: {rec.stale_images} stale camera frames (older than "
                                  f"{MAX_IMAGE_AGE * 1000:.0f} ms or repeated; worst age "
                                  f"{rec.worst_image_age * 1000:.0f} ms) - camera rate too low?")
    return buf, success, reason, order, rec, extra_meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=10, help="successful episodes to add")
    parser.add_argument("--max-attempts", type=int, default=None, help="give up after this many attempts "
                        "(default: 2x --episodes)")
    parser.add_argument("--seed", type=int, default=0, help="first seed for a NEW dataset (ignored on resume)")
    parser.add_argument("--oracle", choices=("obj_known", "obj_unknown"), default="obj_known",
                        help="obj_known: ground-truth object poses (the original oracle); obj_unknown: "
                        "multi-view perception + antipodal grasps + OMPL (implies --scene sensors; needs "
                        "`ros2 launch tabletop_perception perception.launch.py`)")
    parser.add_argument("--scene", choices=("none", "sensors"), default="none",
                        help="MoveIt collision world: none (self-collision only) or sensors (wrist-depth "
                        "octomap; needs `ur_moveit.launch.py ur_type:=ur5e gripper:=robotiq_2f85 sensors:=true`)")
    parser.add_argument("--transit", choices=("lin", "ompl"), default="lin",
                        help="obj_known transits: lin (Pilz rise / over / down) or ompl (one OMPL move "
                        "against the ground-truth table + cube boxes; incompatible with --scene sensors)")
    parser.add_argument("--randomize-colors", action="store_true",
                        help="colour domain randomization: per-episode cube / table / marker colours via "
                        "mujoco_dr_plugin (/color_randomizer/set_colors), stored in meta/episode_rgba")
    parser.add_argument("--wrist-images", action="store_true",
                        help="record the live wrist camera (data/wrist_rgb, 240x320 RGB) every tick; episodes "
                        "start at the scan overview pose task.SCAN_Q (needs --transit ompl)")
    parser.add_argument("--wrist-depth", action="store_true",
                        help="also record the wrist depth from the same render (data/wrist_depth, 240x320 uint16 "
                        "mm; needs --wrist-images)")
    parser.add_argument("--front-images", action="store_true",
                        help="also record the DP-paper station's fixed front camera (data/front_rgb, 240x320 RGB; "
                        "needs --wrist-images and the station world, world:=dp_station_mujoco.xml)")
    args = parser.parse_args()
    if args.wrist_depth and not args.wrist_images:
        parser.error("--wrist-depth needs --wrist-images")
    if args.front_images and not args.wrist_images:
        parser.error("--front-images needs --wrist-images")
    if args.wrist_images and (args.oracle != "obj_known" or args.transit != "ompl"):
        # From SCAN_Q (tip z 0.575) the LIN transit's horizontal leg would run at that height,
        # above the IK-feasible ceiling (~0.55) over most of the table; OMPL plans around it.
        parser.error("--wrist-images needs --oracle obj_known --transit ompl")
    max_attempts = args.max_attempts or 2 * args.episodes

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if args.oracle == "obj_unknown":
        args.scene = "sensors"
        if args.transit != "lin":
            parser.error("--transit applies to --oracle obj_known only (obj_unknown always uses OMPL)")
    elif args.transit == "ompl" and args.scene == "sensors":
        parser.error("--transit ompl plans against the ground-truth world; drop --scene sensors")
    sim = SimNode("stack_bc_collect")
    env = StackEnv(sim.node, sim.clock)
    if env.has_marker and (args.oracle != "obj_known" or args.scene == "sensors"):
        parser.error("the marker world (tabletop_marker_mujoco.xml) supports --oracle obj_known without "
                     "--scene sensors only")
    if args.front_images and not env.has_station:
        parser.error("--front-images needs the DP-paper station world (launch with world:=dp_station_mujoco.xml)")
    writer = ReplayWriter(out, oracle=args.oracle, transit=args.transit if args.oracle == "obj_known" else None,
                          marker="on" if env.has_marker else "off",
                          dr="colors" if args.randomize_colors else "none",
                          images=("wrist_rgb,front_rgb" if args.front_images else "wrist_rgb") if args.wrist_images
                          else "none",
                          start="scan" if args.wrist_images else "home",
                          depth="wrist_depth" if args.wrist_depth else "none",
                          station="dp" if env.has_station else "off")
    writer.truncate_partial()
    seed = int(writer.root.attrs.get("next_seed", args.seed))

    ex = EEExecutor(sim.node, env.kin, dt=DT)
    ex.wait_for_servers()
    if not sim.wait_for(env.ready, 10.0):
        raise RuntimeError("no /joint_states or cube poses")
    randomizer = ColorRandomizer(sim.node) if args.randomize_colors else None
    camera = None
    if args.wrist_images:
        camera = WristCamera(sim.node, depth=args.wrist_depth)
        if not sim.wait_for(camera.ready, 10.0):
            raise RuntimeError(f"no wrist images on {IMAGE_TOPIC}" + (f" / {DEPTH_TOPIC}" if args.wrist_depth else ""))
    front = None
    if args.front_images:
        front = WristCamera(sim.node, FRONT_IMAGE_TOPIC, name="front")
        if not sim.wait_for(front.ready, 10.0):
            raise RuntimeError(f"no front images on {FRONT_IMAGE_TOPIC}")
    motion = MoveItMotion(env.urdf, sim.now, mirror_scene=args.scene == "sensors")
    scene = None
    if args.scene == "sensors":
        scene = SceneManager(sim.node, env.urdf, motion.scene_monitor)
    elif args.transit == "ompl":
        scene = GTScene(env, motion.scene_monitor, env.urdf)
    perception = scanner = None
    if args.oracle == "obj_unknown":
        perception = PerceptionClient(sim.node)
        scanner = ActiveScanner(sim, env, motion, ex, scene, perception)

    added, attempts = 0, 0
    while added < args.episodes and attempts < max_attempts:
        t0 = time.time()
        buf, success, reason, order, rec, extra_meta = run_episode(
            sim, env, motion, ex, seed, scene, scanner, perception, transit=args.transit, randomizer=randomizer,
            camera=camera, front=front)
        attempts += 1
        timing = f" [{rec.irregular} off-slot ticks, worst {rec.worst * 1000:.0f} ms]" if rec.irregular else ""
        if camera is not None:
            timing += (f" [{'wrist+front' if front is not None else 'wrist'} frames: {rec.stale_images} stale, "
                       f"worst age {rec.worst_image_age * 1000:.0f} ms]")
        if success:
            writer.append_episode(buf.arrays(), True, seed, order, extra_meta)
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
    except SystemExit as exc:   # argparse errors / --help: already reported
        code = exc.code if isinstance(exc.code, int) else 1
    except BaseException:
        import traceback
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
