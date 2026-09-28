"""Shared ROS plumbing: a node spun in a background thread, and a sim clock driven by message stamps.

Sim time comes from the header stamps of /joint_states (published by joint_state_broadcaster in
sim time) rather than from use_sim_time's /clock subscription: /clock arrives at the physics rate
(~500 Hz), and handling it in rclpy on top of the state topics starved this process's callbacks
(measured: 68-106 of ~500 joint_states/s delivered, gaps up to 0.8 s, even on an idle machine).
Waiting is event-driven (a Condition notified per stamp), not a busy poll.

SingleThreadedExecutor, deliberately: with rclpy's MultiThreadedExecutor, callbacks ran hundreds
of ms late even though messages arrived on time (an independent probe saw a smooth 100 Hz
/joint_states in the same window) - clock waits hit p99 671 ms / max 3.2 s and ~30-50% of 10 Hz
ticks were off-slot. Single-threaded: p99 ~100 ms (on schedule); recorded tick spacing std < 1 ms,
worst deviation 10 ms (one /joint_states period), no repeated ticks. All callbacks here are
tiny, so one thread is plenty.
"""

import threading

import rclpy
from rclpy.executors import SingleThreadedExecutor


class StampClock:
    """Sim time = newest stamp fed via update(); waits block on a Condition."""

    def __init__(self):
        self._t = None
        self._cond = threading.Condition()

    def update(self, t):
        with self._cond:
            if self._t is None or t > self._t:
                self._t = t
                self._cond.notify_all()

    def now(self):
        with self._cond:
            return self._t

    def wait_until(self, t_sim, wall_timeout=10.0):
        with self._cond:
            ok = self._cond.wait_for(lambda: self._t is not None and self._t >= t_sim, timeout=wall_timeout)
        if not ok:
            raise TimeoutError(f"sim time stuck at {self._t} waiting for {t_sim} - is the sim running?")


class SimNode:
    """Owns rclpy init, one node, its background executor thread, and the sim StampClock
    (fed by StackEnv's /joint_states callback)."""

    def __init__(self, name):
        rclpy.init()
        self.node = rclpy.create_node(name)
        self.clock = StampClock()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self._thread = threading.Thread(target=self.executor.spin, daemon=True)
        self._thread.start()

    def now(self):
        return self.clock.now()

    def wait_until(self, t_sim):
        self.clock.wait_until(t_sim)

    def sleep(self, dt_sim):
        self.clock.wait_until(self.clock.now() + dt_sim)

    def wait_for(self, predicate, timeout_sim):
        """Check predicate every 10 ms of sim time until true or timeout (needs a live clock)."""
        if self.clock.now() is None:
            self.clock.wait_until(0.0)
        t_end = self.clock.now() + timeout_sim
        while not predicate():
            if self.clock.now() > t_end:
                return False
            self.clock.wait_until(self.clock.now() + 0.01)
        return True

    def shutdown(self):
        # Stop and join the spin thread before tearing down the context; shutting rclpy down under
        # a still-spinning executor aborted the process ("terminate called without an active exception").
        self.executor.shutdown()
        self._thread.join(timeout=5.0)
        self.node.destroy_node()
        rclpy.shutdown()
