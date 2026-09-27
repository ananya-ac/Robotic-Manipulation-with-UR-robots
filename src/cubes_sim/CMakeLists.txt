cmake_minimum_required(VERSION 3.8)
project(cubes_sim)

find_package(ament_cmake REQUIRED)

install(DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
