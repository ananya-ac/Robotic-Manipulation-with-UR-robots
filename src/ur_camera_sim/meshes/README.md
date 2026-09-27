# meshes/d435i/

9 per-part D435 visual meshes, one per `part_N/d435i_N.obj` (N = 0..8), vendored
from Google DeepMind's [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(`realsense_d435i` package, Apache-2.0, see `d435i/LICENSE`).

Each `part_N/` also has its own `material_0.mtl`, added here (not part of the
vendored files). Every `d435i_N.obj` references the identical generic
`mtllib material_0.mtl` / `usemtl material_0` -- an artifact of the `obj2mjcf`
conversion tool used to prep these for MuJoCo, not real per-part material data
(Menagerie sources color from its own MJCF file, not `.mtl`, so a real `.mtl`
was never shipped). gz-sim's OBJ loader resolves `material_0.mtl` relative to
each `.obj`'s own directory, so giving each part its own subdirectory + its own
`material_0.mtl` is what makes 9 distinct colors possible -- a single shared
`material_0.mtl` next to all 9 `.obj` files would apply one color to all of them.

These are derived by that project directly from `realsense2_description`'s own
`meshes/d435.dae` (imported into Blender, split and colored per-part, exported
as OBJ) -- confirmed by comparing vertex bounding boxes: `d435i_8.obj`'s extent
matches `d435.dae`'s raw vertex bounding box exactly, so no axis/scale
conversion happened during that export. This is why `urdf/inc/realsense_d435.xacro`
applies the *same* visual origin/rotation to these meshes that
`realsense2_description`'s own `sensor_d435` macro applies to the single `.dae`
mesh -- they share the same coordinate frame.

Per-part colors (from Menagerie's `d435i.xml`), set in both `part_N/material_0.mtl`
(`Kd`/`Ka`) and as a URDF `<material>` override per mesh in `realsense_d435.xacro`
(belt-and-suspenders, since it wasn't obvious which one gz-sim's OBJ loader prefers):

| mesh | color name | rgba |
|---|---|---|
| part_0, part_3 | IR_Lens | 0.035601 0.035601 0.035601 1 |
| part_1 | IR_Emitter_Lens | 0.287440 0.665387 0.327778 1 |
| part_2 | IR_Rim | 0.799102 0.806952 0.799103 1 |
| part_4 | Cameras_Gray | 0.296138 0.296138 0.296138 1 |
| part_5, part_6 | Black_Acrylic | 0.070360 0.070360 0.070360 1 |
| part_7 | RGB_Pupil | 0.087140 0.002866 0.009346 1 |
| part_8 | Metal_Casing | 1 1 1 1 |

Note: `part_4/d435i_4.obj` (~11MB) and `part_8/d435i_8.obj` (~31MB) are large for
such small parts -- likely un-decimated Blender exports. Fine for occasional sim
use; worth decimating if this ever becomes a real-time performance bottleneck.
