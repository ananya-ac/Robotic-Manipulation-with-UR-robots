cmake_minimum_required(VERSION 3.8)
project(ur_camera_sim)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch urdf meshes
  DESTINATION share/${PROJECT_NAME}
)
install(PROGRAMS scripts/depth_to_cloud.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
