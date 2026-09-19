cmake_minimum_required(VERSION 3.8)
project(cv3_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RingCenter.msg"
  "msg/SurfaceTrackingResult.msg"
  "msg/RingCalibrationResult.msg"
  "msg/VisionStatus.msg"
  "srv/GetVisionStatus.srv"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
