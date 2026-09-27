cmake_minimum_required(VERSION 3.8)
project(ur_camera_sim)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch urdf meshes
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
