"""Live wrist-camera capture for image-based policies (collect --wrist-images, evaluate).

Subscribes to the sim's wrist RGB stream (mujoco_ros2_control CameraPlugin, 640x480 rgb8 in
camera_color_toein_frame, stamped in sim time; 20 Hz via ur_bringup_sim/config/mujoco_plugins.yaml)
and keeps only the newest frame, downsampled to IMAGE_SHAPE with an exact 2x2 block mean (the
Diffusion Policy paper's real-world policies use 320x240; no cv2 needed). The recorder takes
latest() at each 10 Hz tick and checks its age against the tick (sim time).
"""

import threading

import numpy as np
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

IMAGE_TOPIC = "/camera/color/image_raw"
# RELIABLE, not the usual sensor-data BEST_EFFORT: a 640x480 rgb8 frame (0.9 MB) travels as many
# UDP fragments, and a best-effort reader drops the whole frame when any fragment is lost (never
# re-sent). Measured on the live sim publishing at 20 Hz: best-effort received 8 Hz with 64% of
# frames lost; reliable received 20.1 Hz with 0 lost. The publisher is RELIABLE too. Depth 2: only
# the newest frame is used, and the callback is a pointer swap, so this reader never throttles the
# publisher (a slow reliable reader would).
IMAGE_QOS = QoSProfile(depth=2, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE)
IMAGE_SHAPE = (240, 320, 3)      # stored H, W, C (uint8 RGB)
SOURCE_SHAPE = (480, 640)        # what the sim publishes; downsampling is an exact factor of 2
# A tick's frame is stale if older than this (sim time, tick stamp - frame stamp) or the same frame
# as the previous tick's. At the camera's 20 Hz a fresh frame is <= ~0.05 s old plus publishing
# latency (measured worst ~80 ms with this reader).
MAX_IMAGE_AGE = 0.1
CAMERA_SETTLE = 0.5              # s at the scan pose before an image episode starts (fresh overview)


def downsample2(rgb):
    """(2H, 2W, 3) uint8 -> (H, W, 3) uint8 by averaging 2x2 blocks."""
    h, w = rgb.shape[0] // 2, rgb.shape[1] // 2
    a = rgb[:2 * h, :2 * w]
    acc = a[0::2, 0::2].astype(np.uint16)   # strided adds: ~10x faster than reshape + sum
    acc += a[1::2, 0::2]
    acc += a[0::2, 1::2]
    acc += a[1::2, 1::2]
    acc += 2                                 # round to nearest
    acc >>= 2
    return acc.astype(np.uint8)


def image_to_rgb(msg):
    """sensor_msgs/Image (rgb8 / bgr8) -> (H, W, 3) uint8 RGB."""
    if msg.encoding not in ("rgb8", "bgr8"):
        raise ValueError(f"unsupported image encoding {msg.encoding!r} (expected rgb8 or bgr8)")
    rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    rgb = rows[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
    return rgb[..., ::-1] if msg.encoding == "bgr8" else rgb


class WristCamera:
    def __init__(self, node, topic=IMAGE_TOPIC):
        """node must be spun by an executor in another thread (runtime.SimNode)."""
        self._lock = threading.Lock()
        self._msg = None
        self.n_received = 0
        node.create_subscription(Image, topic, self._on_image, IMAGE_QOS)

    def _on_image(self, msg):
        # Keep the raw message only; conversion happens on demand, so a 20 Hz stream costs the
        # executor thread (which also feeds the sim clock) nothing but a pointer swap.
        with self._lock:
            self._msg = msg
            self.n_received += 1

    def ready(self):
        with self._lock:
            return self._msg is not None

    def latest(self):
        """(stamp in sim seconds, IMAGE_SHAPE uint8 RGB) of the newest frame, or (None, None)."""
        with self._lock:
            msg = self._msg
        if msg is None:
            return None, None
        if (msg.height, msg.width) != SOURCE_SHAPE:
            raise ValueError(f"wrist image is {msg.width}x{msg.height}, expected "
                             f"{SOURCE_SHAPE[1]}x{SOURCE_SHAPE[0]} (camera resolution changed?)")
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        return stamp, downsample2(image_to_rgb(msg))
