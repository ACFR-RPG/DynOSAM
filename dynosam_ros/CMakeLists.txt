cmake_minimum_required(VERSION 3.11)

project(dynosam_ros)

# set(CMAKE_C_FLAGS "-std=gnu11 -Wall -Wextra -O3 -march=sandybridge -flto")
set(CMAKE_C_FLAGS "-Wall -Wextra")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)


option(ENABLE_DYNAMIC_SLAM_INTERFACES "Use dynamic_slam_interfaces" ON)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(dynosam REQUIRED)
find_package(PCL REQUIRED conversions)
find_package(rclpy REQUIRED)
find_package(message_filters REQUIRED)


# setup targets
include_directories(
  include
  ${rclcpp_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${dynosam_INCLUDE_DIRS}
)


set(SRC_FILES
  src/PipelineRos.cc
  src/RosUtils.cc
  src/Utils.cc
  src/DataProviderRos.cc
  src/OnlineDataProviderRos.cc

  src/displays/backend_displays/HybridBackendDisplay.cc
  src/displays/DisplaysCommon.cc

)

set(LINK_LIBS
  ${geometry_msgs_TARGETS}
  ${nav_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
  ${std_msgs_TARGETS}
  ${visualization_msgs_TARGETS}
  cv_bridge::cv_bridge
  dynamic_slam_interfaces::dynamic_slam_interfaces_library
  dynosam::dynosam
  image_transport::image_transport
  message_filters::message_filters
  pcl_conversions::pcl_conversions
  rclcpp::rclcpp
  sensor_msgs::sensor_msgs_library
  tf2_ros::static_transform_broadcaster_node
  tf2_ros::tf2_ros

)

set(TARGET_EXPORT_DEPS
  rclcpp
  std_msgs
  sensor_msgs
  dynosam
  nav_msgs
  message_filters
)


if(ENABLE_DYNAMIC_SLAM_INTERFACES)
  message(WARNING "dynamic_slam_interfaces will be used")
  add_definitions(-DUSE_DYNAMIC_SLAM_INTERFACES)

  find_package(dynamic_slam_interfaces REQUIRED)
  list(APPEND LINK_LIBS   ${dynamic_slam_interfaces_TARGETS})
  list(APPEND TARGET_EXPORT_DEPS dynamic_slam_interfaces)

  list(APPEND SRC_FILES
    src/displays/dynamic_slam_displays/DSDCommonRos.cc
    src/displays/dynamic_slam_displays/FrontendDSDRos.cc
    src/displays/dynamic_slam_displays/BackendDSDRos.cc)

else()
  message(WARNING "dynamic_slam_interfaces will be not used")
  list(APPEND SRC_FILES
    src/displays/inbuilt_displays/BackendInbuiltDisplayRos.cc
    src/displays/inbuilt_displays/InbuiltDisplayCommon.cc
    src/displays/inbuilt_displays/FrontendInbuiltDisplayRos.cc)


endif()

# build library
add_library(${PROJECT_NAME} SHARED ${SRC_FILES})

target_link_libraries(${PROJECT_NAME}
  ${LINK_LIBS}
  glog
  gflags
)

ament_export_libraries(${PROJECT_NAME}
  glog
  gflags
)

ament_export_dependencies(${TARGET_EXPORT_DEPS})


target_include_directories(${PROJECT_NAME}
  PUBLIC
  ${rclcpp_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})

install(
  DIRECTORY include/
  DESTINATION include
)
install(
  TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)


#install python scripts/modules
ament_python_install_package(${PROJECT_NAME})

#####exec test#########
add_executable(dynosam_node nodes/dynosam_node.cc)
target_link_libraries(dynosam_node
  ${PROJECT_NAME}
  ${PCL_LIBRARIES}

)
install(TARGETS
  dynosam_node
  DESTINATION lib/${PROJECT_NAME})



add_executable(dynosam_tracking_demo_node nodes/dynosam_tracking_demo_node.cc)
target_link_libraries(dynosam_tracking_demo_node
  ${PROJECT_NAME}
  ${PCL_LIBRARIES}

)
install(TARGETS
  dynosam_tracking_demo_node
  DESTINATION lib/${PROJECT_NAME})




    # Install Python executables
install(PROGRAMS
scripts/run_dynosam_gtest.py
DESTINATION lib/${PROJECT_NAME}
)


if(BUILD_TESTING)
  # find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_gmock REQUIRED)
  find_package(ament_lint_auto REQUIRED)
    find_package(ament_index_cpp REQUIRED)


  ament_lint_auto_find_test_dependencies()

  # Run all lint tests in package.xml except those listed above
  ament_lint_auto_find_test_dependencies()
  ament_add_gmock(${PROJECT_NAME}_test
    test/test_main.cc
    test/test_ros_utils.cc
    test/test_adaptors.cc
    test/test_online_dataprovider_ros.cc
)

target_include_directories(${PROJECT_NAME}_test
    PRIVATE
     ${ament_index_cpp_INCLUDE_DIRS}
  )

target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME} ${rclcpp_LIBRARIES} ${ament_index_cpp_TARGETS})


  install(TARGETS
  ${PROJECT_NAME}_test
      DESTINATION test/)

endif()


ament_package()
