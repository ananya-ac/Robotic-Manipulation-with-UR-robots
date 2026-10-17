"""Visual domain randomization: per-episode colours for the cubes, the table and the marker.

Colours are applied live by mujoco_dr_plugin (/color_randomizer/set_colors, which writes the
materials' mat_rgba in the running sim) and recorded per episode (meta/episode_rgba, rows in
COLOR_SLOTS order), so any episode can be re-rendered exactly. They are drawn from the episode's
own rng, so a seed reproduces its colours.

Sampling: HSV ranges per slot, rejection-sampled until every pair that must stay tellable apart
is far enough apart in CIELAB (Delta E 1976):
  * table vs each cube and table vs marker >= MIN_DE_TABLE - objects never blend into the table
  * marker vs each cube >= MIN_DE_MARKER - the marker stays visible under/next to a cube
Cubes may share colours (the task doesn't care which cube is which). Robot, gripper, floor and
lighting are not randomized.
"""

import colorsys

import numpy as np

# (slot, MJCF material) - materials as in worlds/worlds/tabletop_marker_mujoco.xml; cube_i uses
# its geom's material, so slot order = task.CUBE_NAMES, then table, then marker.
COLOR_SLOTS = (("cube_0", "cube_red"), ("cube_1", "cube_green"), ("cube_2", "cube_blue"),
               ("table", "table_wood"), ("marker", "stack_marker"))
# (hue, saturation, value) ranges per kind
HSV_RANGES = {
    "cube": ((0.0, 1.0), (0.35, 1.0), (0.3, 1.0)),
    "table": ((0.0, 1.0), (0.0, 0.6), (0.25, 0.85)),
    "marker": ((0.0, 1.0), (0.0, 1.0), (0.4, 1.0)),
}
MIN_DE_TABLE = 25.0
MIN_DE_MARKER = 15.0
SET_COLORS_SERVICE = "/color_randomizer/set_colors"


def _srgb_to_lab(rgb):
    """sRGB in [0, 1] -> CIELAB (D65)."""
    rgb = np.asarray(rgb, dtype=float)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = lin @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def delta_e(rgb_a, rgb_b):
    return float(np.linalg.norm(_srgb_to_lab(rgb_a) - _srgb_to_lab(rgb_b)))


def _sample_hsv(rng, kind):
    (h0, h1), (s0, s1), (v0, v1) = HSV_RANGES[kind]
    return np.array(colorsys.hsv_to_rgb(rng.uniform(h0, h1), rng.uniform(s0, s1), rng.uniform(v0, v1)))


def sample_colors(rng, max_tries=1000):
    """(len(COLOR_SLOTS), 4) RGBA, COLOR_SLOTS order."""
    for _ in range(max_tries):
        cubes = [_sample_hsv(rng, "cube") for _ in range(3)]
        table, marker = _sample_hsv(rng, "table"), _sample_hsv(rng, "marker")
        if all(delta_e(c, table) >= MIN_DE_TABLE for c in cubes) and delta_e(marker, table) >= MIN_DE_TABLE \
                and all(delta_e(c, marker) >= MIN_DE_MARKER for c in cubes):
            rgb = np.array(cubes + [table, marker])
            return np.hstack([rgb, np.ones((len(rgb), 1))]).astype(np.float32)
    raise RuntimeError("could not sample colours satisfying the contrast constraints")


class ColorRandomizer:
    """Client for mujoco_dr_plugin's set_colors service."""

    def __init__(self, node):
        from mujoco_dr_plugin.srv import SetColors   # ROS-side only; sampling above stays ROS-free
        self._srv_type = SetColors
        self._client = node.create_client(SetColors, SET_COLORS_SERVICE)

    def apply(self, rgba, timeout=5.0):
        """rgba: (len(COLOR_SLOTS), 4) from sample_colors. Returns once the sim has applied it."""
        import threading
        if not self._client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"{SET_COLORS_SERVICE} unavailable - is mujoco_dr_plugin in mujoco_plugins.yaml "
                               "and the sim (re)started since?")
        req = self._srv_type.Request()
        req.materials = [mat for _, mat in COLOR_SLOTS]
        req.rgba = [float(v) for v in np.asarray(rgba).ravel()]
        done = threading.Event()
        future = self._client.call_async(req)
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout) or not future.result().success:
            raise RuntimeError(f"set_colors failed: {future.result().message if future.done() else 'timeout'}")
