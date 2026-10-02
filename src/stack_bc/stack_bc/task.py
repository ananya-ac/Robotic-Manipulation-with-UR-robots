"""Task constants shared by the feasibility map, randomizer, oracle and success checker.

Geometry mirrors worlds/worlds/tabletop_mujoco.xml; change both together.
"""

import numpy as np

CUBE_NAMES = ("cube_0", "cube_1", "cube_2")
CUBE_SIZE = 0.045                     # full edge length (MJCF half-extent 0.0225)
TABLE_CENTER_XY = np.array([-0.6, 0.0])
TABLE_HALF_EXTENT = 0.3
TABLE_TOP_Z = 0.3
CUBE_REST_Z = TABLE_TOP_Z + CUBE_SIZE / 2   # cube center height resting on the table

# Robot "home" (the MJCF keyframe / launch-time pose) in ARM_JOINTS order.
HOME_Q = np.array([1.57, -1.57, 1.26, -1.57, -1.57, 0.0])

# Wrist-camera scan pose that seeds the sensor-built octomap each episode (--scene sensors only;
# never recorded). The tip points straight down at (-0.415, 0.055, 0.575), the IK branch nearest
# HOME_Q. That puts the camera (camera_color_toein_frame) at (-0.47, 0.09, 0.74), ~0.44 m above
# the table, with its optical axis on the spawn-region center. At fovy 54.9 deg / 640x480 its
# table footprint covers the whole SPAWN_X x SPAWN_Y region plus a 3 cm margin. A top-down view
# over the table center itself is out of reach at a useful height (no IK above tip z ~0.55).
SCAN_Q = np.array([1.1143, -1.5644, 0.8427, -0.8491, -1.5708, -0.456])

# Oracle waypoint offsets, measured on gripper_tip_link relative to the target cube center.
HOVER_HEIGHT = 0.10                   # pre-grasp / pre-place clearance above the target
PLACE_CLEARANCE = 0.004               # release the carried cube this far above resting contact
# Transit between pick and place happens at max(pick hover, place hover): every such point is
# already inside the feasibility-checked set, and with at most a 2-high stack standing while the
# last cube is carried (top face at CUBE_REST_Z + 1.5 * CUBE_SIZE = 0.39 m), a level-2 place hover
# (0.5125 m) keeps the carried cube's bottom face ~10 cm clear of it.

# Randomization. Region bounds come from `feasibility` (IK-checked), not from the table extents:
# its largest all-levels/all-yaws rectangle (hover 10 cm, 4 yaws, 3 cm grid) was x in
# [-0.59, -0.35], y in [-0.23, 0.25], shrunk 1 cm here for tolerance. The binding constraint is
# placing onto a 2-high stack (level 2 reaches only x >= ~-0.59; picking alone reaches ~-0.70), and
# any spawned cube can become the stack base, so the whole region must pass every level.
SPAWN_X = (-0.58, -0.36)
SPAWN_Y = (-0.22, 0.22)
MIN_CUBE_SPACING = 0.08               # center-to-center; leaves finger clearance around each cube
N_CUBES = len(CUBE_NAMES)
