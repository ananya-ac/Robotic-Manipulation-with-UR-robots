# Open problem: EAIK URDF loading breaks on comments (local patch in ur_python_utilities)

Status: **`stack_bc` no longer depends on the fix** - `stack_bc/kinematics.py` sanitizes the URDF
with a real XML parser (`sanitize_urdf`) before handing it to `EAIKKinematics`, verified against an
unmodified upstream `ur_control` (FK vs MuJoCo 1.6e-7 m, IK round-trip 2e-12 rad). The bug itself
is still present upstream and still affects `ur_control.Arm()` (see below); on this machine a
local, uncommitted vendor patch covers that.

## Symptom

Building `ur_control.eaik_kinematics.EAIKKinematics` from our robot description fails with

```
lxml.etree.XMLSyntaxError: Comment not terminated, line 30, column 1
```

`stack_bc/kinematics.py` (`ArmKinematics`) uses `EAIKKinematics` for its analytical IK, so every
`stack_bc` entry point (`collect`, `feasibility`, `smoke`) fails at startup.

## Cause

`_strip_non_urdf_tags()` in `src/ur_python_utilities/ur_control/ur_control/eaik_kinematics.py`
removes extension blocks with regexes before `urchin` parses the URDF:

```python
xml = re.sub(r"<ros2_control[\s\S]*?</ros2_control>", "", xml)
```

A regex does not know what a comment is. The xacro output of
`ur5e_2f85_camera_mujoco.urdf.xacro` contains comments that mention the tag by name (e.g.
"... inside the SAME <mujoco_inputs>/<ros2_control> blocks ..."). The match starts at that text
inside the comment and deletes everything up to the next real `</ros2_control>`, including the
comment's `-->`, so the result is no longer valid XML.

## Current workaround (local only)

Two lines added to `_strip_non_urdf_tags()` in the vendor clone, before the existing regexes:

```python
xml = re.sub(r"<!--[\s\S]*?-->", "", xml)                        # strip comments first
xml = re.sub(r"<mujoco_inputs[\s\S]*?</mujoco_inputs>", "", xml) # not strictly needed
```

Only the comment line is required; urchin merely warns about `<mujoco_inputs>`.

The change is **uncommitted** in `src/ur_python_utilities` (a separate git clone of
`cambel/ur3`, branch `ros2-jazzy`, gitignored by this repo and already carrying 2 unpushed local
commits), so it is not in any pushed history.

## Wider impact: `ur_control.Arm`

Before EAIK was installed in `~/venvs/ros-jazzy`, `Arm` silently fell back to KDL IK. With EAIK
installed, `Arm()` on this robot description would now crash instead: its fallback only catches
`ImportError`/`ValueError`, not the XML syntax error. This affects anything that constructs an
`Arm` (e.g. `ur_control_examples` scripts) unless the patch is present.

## Resolution

1. **Decouple `stack_bc` - DONE.** `sanitize_urdf()` in `stack_bc/kinematics.py` (stdlib
   ElementTree: comments dropped at parse time, `ros2_control`/`mujoco_inputs`/`gazebo` elements
   removed) runs before `EAIKKinematics`. `stack_bc` works from this repo + upstream `cambel/ur3`.
2. **Fix `ur_control` properly - still open** (separate, for `Arm`): make `_strip_non_urdf_tags()` parse XML
   instead of using regexes (or at least strip comments first), commit it on a fork of
   `cambel/ur3`, point `dependencies.repos` at the fork, and optionally send it upstream.
