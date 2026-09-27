# Vendored meshes

`hande/` and `robotiq_85_coupler.stl` are copied from
[omron-sinicx/robotiq-cri](https://github.com/omron-sinicx/robotiq-cri)
(`robotiq_description/meshes/`, `ros2-jazzy` branch, BSD-licensed). They cover
the Robotiq Hand-E gripper, which the upstream `robotiq_description` (PickNik)
package does not model.

Vendored directly instead of pulled in via CMake at a relative `../../robotiq-cri`
path so `ur_gripper_sim` builds standalone (outside the `osx_robot_env` parent
workspace layout it was written for), including under the pixi environment
in this repo.
