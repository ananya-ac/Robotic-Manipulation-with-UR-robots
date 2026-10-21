"""Live wrist-camera capture for image-based policies (collect --wrist-images, evaluate).

Subscribes to the sim's wrist RGB stream (mujoco_ros2_control CameraPlugin, 640x480 rgb8 in
camera_color_toein_frame, stamped in sim time; 20 Hz via ur_bringup_sim/config/mujoco_plugins.yaml)
and keeps only the newest frame, downsampled to IMAGE_SHAPE with an exact 2x2 block mean (the
Diffusion Policy paper's real-world policies use 320x240; no cv2 needed). The recorder takes
latest() at each 10 Hz tick and checks its age against the tick (sim time).

With depth=True it also takes the depth stream (32FC1 metres, rendered by the same camera, so
pixel-aligned with the colour image) and latest_rgbd() returns a colour/depth PAIR from the same
render: depth as uint16 millimetres (0 = invalid), downsampled by a 2x2 minimum over valid pixels.
"""

import threading

import numpy as np
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from stack_bc.imageproc import (DEPTH_SHAPE, IMAGE_SHAPE, SOURCE_SHAPE,  # noqa: F401 (re-exported)
                                depth_frame, downsample2)

IMAGE_TOPIC = "/camera/color/image_raw"
DEPTH_TOPIC = "/camera/depth/image_rect_raw"   # 32FC1 metres, same MuJoCo camera render as colour
# RELIABLE, not the usual sensor-data BEST_EFFORT: a 640x480 rgb8 frame (0.9 MB) travels as many
# UDP fragments, and a best-effort reader drops the whole frame when any fragment is lost (never
# re-sent). Measured on the live sim publishing at 20 Hz: best-effort received 8 Hz with 64% of
# frames lost; reliable received 20.1 Hz with 0 lost. The publisher is RELIABLE too. Depth 2: only
# the newest frame is used, and the callback is a pointer swap, so this reader never throttles the
# publisher (a slow reliable reader would).
IMAGE_QOS = QoSProfile(depth=2, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE)
# A tick's frame is stale if older than this (sim time, tick stamp - frame stamp) or the same frame
# as the previous tick's. At the camera's 20 Hz a fresh frame is <= ~0.05 s old plus publishing
# latency (measured worst ~80 ms with this reader).
MAX_IMAGE_AGE = 0.1
CAMERA_SETTLE = 0.5              # s at the scan pose before an image episode starts (fresh overview)


def depth_to_m(msg):
    """sensor_msgs/Image (32FC1 metres or 16UC1 millimetres) -> (H, W) float32 metres."""
    if msg.encoding == "32FC1":
        rows = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.step // 4)
        return rows[:, :msg.width]
    if msg.encoding == "16UC1":
        rows = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.step // 2)
        return rows[:, :msg.width].astype(np.float32) / 1000.0
    raise ValueError(f"unsupported depth encoding {msg.encoding!r} (expected 32FC1 or 16UC1)")


def image_to_rgb(msg):
    """sensor_msgs/Image (rgb8 / bgr8) -> (H, W, 3) uint8 RGB."""
    if msg.encoding not in ("rgb8", "bgr8"):
        raise ValueError(f"unsupported image encoding {msg.encoding!r} (expected rgb8 or bgr8)")
    rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    rgb = rows[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
    return rgb[..., ::-1] if msg.encoding == "bgr8" else rgb


def _stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def _check_shape(msg, what):
    if (msg.height, msg.width) != SOURCE_SHAPE:
        raise ValueError(f"wrist {what} is {msg.width}x{msg.height}, expected "
                         f"{SOURCE_SHAPE[1]}x{SOURCE_SHAPE[0]} (camera resolution changed?)")


class WristCamera:
    KEEP = 4   # recent messages kept per stream, to find a colour/depth pair from the same render

    def __init__(self, node, topic=IMAGE_TOPIC, depth=False, depth_topic=DEPTH_TOPIC):
        """node must be spun by an executor in another thread (runtime.SimNode).
        depth=True also subscribes to the depth stream (same QoS) for latest_rgbd()."""
        self._lock = threading.Lock()
        self._msg = None
        self._color = {}           # stamp -> msg (recent)
        self._depth = {}
        self.n_received = 0
        self.depth = depth
        node.create_subscription(Image, topic, self._on_image, IMAGE_QOS)
        if depth:
            node.create_subscription(Image, depth_topic, self._on_depth, IMAGE_QOS)

    def _keep(self, store, msg):
        store[_stamp(msg)] = msg
        while len(store) > self.KEEP:
            del store[min(store)]

    def _on_image(self, msg):
        # Keep the raw message only; conversion happens on demand, so a 20 Hz stream costs the
        # executor thread (which also feeds the sim clock) nothing but a pointer swap.
        with self._lock:
            self._msg = msg
            self._keep(self._color, msg)
            self.n_received += 1

    def _on_depth(self, msg):
        with self._lock:
            self._keep(self._depth, msg)

    def ready(self):
        with self._lock:
            return self._msg is not None and (not self.depth or bool(self._depth))

    def latest(self):
        """(stamp in sim seconds, IMAGE_SHAPE uint8 RGB) of the newest frame, or (None, None)."""
        with self._lock:
            msg = self._msg
        if msg is None:
            return None, None
        _check_shape(msg, "image")
        return _stamp(msg), downsample2(image_to_rgb(msg))

    def latest_rgbd(self):
        """(stamp, IMAGE_SHAPE uint8 RGB, DEPTH_SHAPE uint16 mm, matched) for the newest colour and
        depth frames from the SAME render (equal stamps; matched=True). If no stamp is shared yet,
        the newest of each with matched=False (the caller counts it as stale). (None,)*3 + (False,)
        before any frame."""
        with self._lock:
            color, depth = dict(self._color), dict(self._depth)
        if not color or not depth:
            return None, None, None, False
        common = set(color) & set(depth)
        if common:
            stamp = max(common)
            cmsg, dmsg, matched = color[stamp], depth[stamp], True
        else:
            stamp = max(color)
            cmsg, dmsg, matched = color[stamp], depth[max(depth)], False
        _check_shape(cmsg, "image")
        _check_shape(dmsg, "depth")
        return stamp, downsample2(image_to_rgb(cmsg)), depth_frame(depth_to_m(dmsg)), matched
