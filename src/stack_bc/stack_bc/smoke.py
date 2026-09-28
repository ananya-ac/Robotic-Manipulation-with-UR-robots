"""Live smoke test for StackEnv + EEExecutor (sim must be running).

Homes the arm, does a randomized reset, streams a 10 Hz straight-line top-down EE path over one
cube and back, and reports commanded-vs-measured EE tracking error (at the same tick and one tick
later, since each command takes dt to reach). Then closes/opens the gripper in free air and
reports the measured width.
"""

import numpy as np

from stack_bc import rotations as rot
from stack_bc import task
from stack_bc.env import StackEnv
from stack_bc.executor import EEExecutor
from stack_bc.gripper import WIDTH_CLOSED, WIDTH_OPEN
from stack_bc.runtime import SimNode


def main():
    sim = SimNode("stack_bc_smoke")
    env = StackEnv(sim.node, sim.clock)
    ex = EEExecutor(sim.node, env.kin)
    ex.wait_for_servers()
    assert sim.wait_for(env.ready, 10.0), "no joint_states / cube poses"

    ex.move_joints(task.HOME_Q, 3.0)
    ex.set_width(WIDTH_OPEN, force=True)
    sim.sleep(3.5)
    rng = np.random.default_rng(0)
    xy, yaw = env.sample_cubes(rng)
    env.reset_world(xy, yaw)
    sim.sleep(1.0)
    s = env.snapshot()
    print("after reset: q_arm err vs home", np.round(s.q_arm - task.HOME_Q, 4), "width", round(s.width, 4))
    print("cube pos\n", np.round(s.cube_pos, 4), "\nrequested xy\n", np.round(xy, 4))
    ex.sync(s.q_arm, s.ee)

    # Straight line: home tip pose -> above cube 0 (top-down, cube-aligned yaw) -> back.
    start = s.ee.copy()
    goal = rot.pose_matrix([xy[0, 0], xy[0, 1], task.CUBE_REST_Z + task.HOVER_HEIGHT],
                           rot.top_down_grasp(yaw[0]))
    n = 40
    path = []
    for a in np.linspace(0, 1, n):
        path.append(rot.pose_matrix((1 - a) * start[:3, 3] + a * goal[:3, 3],
                                    rot.slerp(start[:3, :3], goal[:3, :3], a)))
    path += path[::-1]

    errs_now, errs_next, prev_cmd = [], [], None
    t = sim.now()
    for T in path:
        sim.wait_until(t)
        meas = env.snapshot().ee
        if prev_cmd is not None:
            errs_next.append(np.linalg.norm(meas[:3, 3] - prev_cmd[:3, 3]))
        cmd = ex.send(T, WIDTH_OPEN)
        errs_now.append(np.linalg.norm(meas[:3, 3] - cmd[:3, 3]))
        prev_cmd = cmd
        t += ex.dt
    sim.sleep(1.0)
    final = np.linalg.norm(env.snapshot().ee[:3, 3] - prev_cmd[:3, 3])
    print(f"tracking: |meas_t - cmd_(t-1)| mean {np.mean(errs_next)*1000:.1f} mm, max {np.max(errs_next)*1000:.1f} mm; "
          f"settled final err {final*1000:.2f} mm")

    for w in (WIDTH_CLOSED, WIDTH_OPEN):
        ex.set_width(w)
        sim.sleep(2.0)
        print(f"gripper cmd {w*1000:.0f} mm -> measured {env.snapshot().width*1000:.1f} mm")
    sim.shutdown()


if __name__ == "__main__":
    main()
