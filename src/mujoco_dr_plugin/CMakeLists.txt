cmake_minimum_required(VERSION 3.16)
project(mujoco_dr_plugin)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(pluginlib REQUIRED)
find_package(mujoco_vendor REQUIRED)
find_package(mujoco_ros2_control_plugins REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME} "srv/SetColors.srv")
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} rosidl_typesupport_cpp)

# Not ${PROJECT_NAME}: rosidl_generate_interfaces already owns that target name.
add_library(color_randomizer_plugin SHARED src/color_randomizer_plugin.cpp)
target_link_libraries(color_randomizer_plugin
  mujoco_ros2_control_plugins::mujoco_ros2_control_plugins
  mujoco::mujoco
  rclcpp::rclcpp
  pluginlib::pluginlib
  "${cpp_typesupport_target}")

pluginlib_export_plugin_description_file(mujoco_ros2_control_plugins plugins.xml)

install(TARGETS color_randomizer_plugin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
