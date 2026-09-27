#!/usr/bin/env bash
# Fetches and prepares the D435 visual meshes vendored from Google DeepMind's
# MuJoCo Menagerie (realsense_d435i package, Apache-2.0 -- see LICENSE in this
# directory). Not committed directly (~45MB, largely un-decimated Blender
# exports -- see README.md); reproduced here instead.
#
# Writes into meshes/d435i/part_N/ (one subdirectory per part, each with its
# own material_0.mtl): every d435i_N.obj references the identical generic
# "mtllib material_0.mtl", so per-part subdirectories are what makes 9 distinct
# colors possible (see README.md for why, and the color table each part uses).
#
# Run once after cloning, before building ur_camera_sim. Re-running is safe.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${HERE}/d435i"
BASE_URL="https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/realsense_d435i/assets"

# part index -> Kd/Ka color, from Menagerie's d435i.xml <asset><material> entries
# (see README.md's color table for which named material each index corresponds to)
declare -A COLORS=(
  [0]="0.035601 0.035601 0.035601"
  [1]="0.287440 0.665387 0.327778"
  [2]="0.799102 0.806952 0.799103"
  [3]="0.035601 0.035601 0.035601"
  [4]="0.296138 0.296138 0.296138"
  [5]="0.070360 0.070360 0.070360"
  [6]="0.070360 0.070360 0.070360"
  [7]="0.087140 0.002866 0.009346"
  [8]="1 1 1"
)

for i in 0 1 2 3 4 5 6 7 8; do
  part_dir="${OUT}/part_${i}"
  mkdir -p "${part_dir}"
  echo "Fetching d435i_${i}.obj..."
  curl -sL "${BASE_URL}/d435i_${i}.obj" -o "${part_dir}/d435i_${i}.obj"
  if [ ! -s "${part_dir}/d435i_${i}.obj" ]; then
    echo "ERROR: d435i_${i}.obj download failed or is empty" >&2
    exit 1
  fi
  cat > "${part_dir}/material_0.mtl" <<EOF
newmtl material_0
Kd ${COLORS[$i]}
Ka ${COLORS[$i]}
Ks 0.0 0.0 0.0
d 1.0
illum 1
EOF
done

echo "Done: meshes written to ${OUT}/part_*/"
