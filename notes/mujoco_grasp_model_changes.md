# Model description changes required for a working MuJoCo pick

Context: getting a real friction-based grasp working on the MuJoCo backend (UR5e + Robotiq
2F-85, `tabletop_mujoco.xml`) took more than getting the robot description to convert and
simulate. This note lists the changes that were actually load-bearing for the pick to work,
grouped by where they live, with why each one mattered. It does not cover camera/vision
changes (orientation, framing) - those affect the D435 feed, not the pick itself, since the
grasp target here comes from ground-truth cube pose, not perception.

## Robot description (`ur_gripper_2f85_mujoco.urdf.xacro`)

- **Self-collision exclusions**: MoveIt's SRDF `disable_collisions` list is a *planning-only*
  ACM - it has zero effect on MuJoCo's own physics. Without translating it into MJCF
  `<contact><exclude>` pairs (post-fixed-joint-fusion body names), a real
  `world<->shoulder_link` contact silently cancelled the shoulder_pan actuator's torque via
  `qfrc_constraint`, and MoveIt-commanded goals were simply never reached. This has to be
  independently re-derived for MuJoCo; it isn't inherited from the SRDF automatically.

- **`armature="0.1"` on all 6 arm joints**: without it, `wrist_3_joint` oscillated
  continuously and never settled, because it carries the full 2F-85+D435 assembly
  cantilevered off-axis - a much larger effective load than the bare-wrist Menagerie config
  this robot's actuator gains were originally sourced from. Applied uniformly (Menagerie's own
  convention), not just to the one joint that first showed symptoms.

- **`gravcomp="1"` on all 12 post-fusion robot bodies** (arm links + gripper links): a plain
  P+D `<position>` actuator has an inherent nonzero steady-state error under any sustained
  load - it needs that error to generate holding torque. Without gravity compensation, the
  arm held poses 0.0001-0.08 rad off target, which is enough to make a repeatable, precise
  pick unreliable. `gravcomp` cancels each body's own weight directly in the physics engine,
  removing the need for an integral term (or any compensation) in the controller at all.

- **Per-joint `<position>` actuator `kp`/`kv` retuned away from Menagerie's stock values**:
  Menagerie's own gains were tuned for a *bare* wrist/arm. With this robot's actual payload,
  the stock `kv` on the wrist joints saturated into a genuine torque-limit-cycle (actuator
  force pinned at exactly the joint's `actuatorfrcrange` bound, flipping sign every physics
  step) - a real chattering instability, not just suboptimal tracking. Retuned per joint
  against this robot's own combined multi-joint dynamics (Gauss-Seidel coordinate descent
  against a realistic simultaneous 6-joint move, not isolated single-joint step tests, which
  underestimated real coupling badly - an isolated-joint search that looked fine produced
  40-55% overshoot once all 6 joints moved together for real).

- Gripper joint `damping`/`armature` (knuckle: `damping=0.1 armature=0.005`; the 5 passive
  mimic joints: `armature=0.001`, `damping=0.00125` on the ones that need it): the vendor
  URDF's mimic joints have no damping at all. Undamped, the mimic-equality-constrained
  mechanism either drifted to a different equality-satisfying configuration under a step
  command, or oscillated indefinitely instead of settling.

## World / scene (`worlds/worlds/tabletop_mujoco.xml`)

- **`<option noslip_iterations="10" impratio="10"/>`** - the single most important fix for
  the pick actually *holding*. MuJoCo's default regularized friction model has no true
  static-friction "stick" state: a pinched object creeps out of a grip at a steady rate even
  when the load is well inside the Coulomb friction cone (this is a documented MuJoCo
  characteristic, not a config bug - see google-deepmind/mujoco#3328). The gripper would
  close and correctly stall against the cube, but the cube would slip back out during any
  lift - confirmed even via a manual rviz-driven MoveIt command, ruling out anything
  trajectory-related. Raising the friction coefficient barely helps (~3x at best); these two
  solver options are what actually fixes it (~110x and ~10x respectively in the referenced
  measurements). This is a global `<option>`, not a per-object property.

- **Table height** (top surface z=0.4 -> 0.3): a real reachability limit, not a preference.
  With the gripper's approach constrained close to vertical (needed for a clean, non-sliding
  grasp - see below), the cube's specific horizontal distance from the base left only ~1.5cm
  of vertical workspace above grasp height before the arm's elbow neared full extension
  (confirmed both by an offline IK solver and by MoveIt's own OMPL planner failing outright
  on the hover waypoint). Lowering the table moved grasp height further from that same fixed
  ceiling, raising the usable margin to ~11.5cm.

- **Keyframe `qpos` padding for the tabletop's free-joint cubes**: MuJoCo pads a keyframe's
  `qpos` with *zeros* (not each body's own default pose) when it's shorter than the compiled
  model's `nq`. The robot's own generated MJCF only ever lists its own 12 DOF in its `home`
  keyframe; once `<include>`d into a scene with 3 more free-joint cube bodies (21 more `qpos`
  values), every cube collapsed to world origin on reset unless that keyframe was extended
  with the cubes' actual spawn pose. Fixed in `robot_mujoco.launch.py` (reads the scene
  template's free-joint bodies and appends their pose to the generated keyframe), not in the
  xacro, since the robot description is deliberately scene-agnostic.

## Grasp approach geometry (not the model file itself, but load-bearing for the above)

The gripper's approach needs to be genuinely close to vertical - not a tuning nicety. An
off-vertical approach (even the arm's own already-validated "home" orientation, ~17.7 degrees
off vertical) produces a net *sideways* force as the fingers close, which was confirmed
offline to slide the cube continuously out from under the fingers (~17mm drift, never
settling) rather than gripping it. Switching to an exactly vertical approach fixed this
immediately in the same offline reproduction. This constraint is what makes the table-height
reachability limit above load-bearing in the first place.

## ros2_control config (not model description, but necessary alongside it)

- **`gripper_controller`'s `stall_velocity_threshold`/`stall_timeout`** (defaults 0.001 rad/s /
  1.0s, loosened to 0.01 / 0.5): with `allow_stalling: true` but the defaults, the
  `GripperCommand` action hung indefinitely after a real physical stall against the cube -
  the gripper's mimic-linkage mechanism has enough residual softness/oscillation under load
  that measured velocity apparently never sat below 0.001 rad/s continuously for a full
  second, so stall detection never fired.

## Dead end worth noting

Descend-to-grasp and lift were briefly switched from OMPL to the Pilz industrial motion
planner's "LIN" primitive (a forced straight Cartesian line), because an OMPL-planned descent
was seen swinging the gripper into the cube off-axis and knocking it over. That symptom
disappeared once the *actual* root cause (the friction-creep fix above) was found - re-tested
with plain OMPL afterward and it now works correctly and noticeably faster, so Pilz was
solving a symptom of the friction bug, not an independent problem. Current code uses OMPL for
all three arm waypoints (`USE_PILZ_FOR_DESCEND_AND_LIFT = False` in
`mujoco_grasp_test.py`), with the Pilz path kept as a toggle rather than removed.
