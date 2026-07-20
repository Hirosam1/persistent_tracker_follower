cmake_minimum_required(VERSION 3.8) # min cmake version
project(persistent_tracker)                    # package name

# Compiler Options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
#find_package(<dependency> REQUIRED) add more here

# Install files
install(
  DIRECTORY config description launch meshes rviz worlds models nodes media
  DESTINATION share/${PROJECT_NAME}
)

# Install Python executables to lib/<pkg> so ros2 run can find them
install(
  PROGRAMS nodes/dock_pose_estimator.py nodes/odom_to_tf.py nodes/charging_manager.py nodes/alignment_test.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install base_carregamento.stl into the Gazebo model directory so that
# model://charging_dock/meshes/... resolves when GZ_SIM_RESOURCE_PATH
# points to share/<pkg>/models.
install(
  FILES meshes/base_carregamento.stl
  DESTINATION share/${PROJECT_NAME}/models/charging_dock/meshes
)

# Install ArUco tag PNG as the model texture (rename to drop special chars).
install(
  FILES "media/aruco-marker-ID=771(maior).png"
  DESTINATION share/${PROJECT_NAME}/models/charging_dock/materials/textures
  RENAME aruco_marker_771.png
)

ament_package()
