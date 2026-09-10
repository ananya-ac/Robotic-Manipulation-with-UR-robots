cmake_minimum_required(VERSION 3.8)
project(ur_bringup_gz)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
