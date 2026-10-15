cmake_minimum_required(VERSION 3.8)
project(tabletop_perception_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(vision_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/GraspCandidate.msg"
  "srv/DetectObjects.srv"
  "srv/PlanGrasps.srv"
  DEPENDENCIES geometry_msgs vision_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
