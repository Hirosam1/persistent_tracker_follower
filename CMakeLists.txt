cmake_minimum_required(VERSION 3.8)
project(persistent_tracker)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(scripts)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  FILES
    yolo11n.pt
    osnet_x0_25_market1501.pth
    osnet_x0_25_msmt17.pth
  DESTINATION share/${PROJECT_NAME}/models
)

install(
  PROGRAMS scripts/main_ros.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME main_ros
)

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
