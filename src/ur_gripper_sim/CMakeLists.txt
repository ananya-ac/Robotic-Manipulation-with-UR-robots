cmake_minimum_required(VERSION 3.8)
project(ur_gripper_sim)

find_package(ament_cmake REQUIRED)

# Hand-E meshes vendored from robotiq-cri (PickNik vendor stack has no
# Hand-E model); see meshes/README.md for provenance.
install(DIRECTORY config launch urdf
  DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY meshes/hande
  DESTINATION share/${PROJECT_NAME}/meshes
)
install(FILES meshes/robotiq_85_coupler.stl
  DESTINATION share/${PROJECT_NAME}/meshes
)

ament_package()
