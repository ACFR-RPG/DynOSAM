cmake_minimum_required(VERSION 3.11)

project(dynosam2)

set(CMAKE_C_FLAGS "-Wall -Wextra  -O3")
set(CMAKE_CXX_FLAGS "-Wall -Wextra -O3")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -O3
  )
endif()


find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(dynosam_common REQUIRED)
find_package(dynosam_cv REQUIRED)
find_package(dynosam_core REQUIRED)
find_package(dynosam_sensors REQUIRED)
find_package(dynosam_opt REQUIRED)
find_package(dynosam_processing REQUIRED)
find_package(dynosam_nn REQUIRED)



# setup targets
include_directories(
  include
  ${dynosam_common_INCLUDE_DIRS}
  ${dynosam_core_INCLUDE_DIRS}
  ${dynosam_cv_INCLUDE_DIRS}
  ${dynosam_sensors_INCLUDE_DIRS}
  ${dynosam_opt_INCLUDE_DIRS}
  ${dynosam_processing_INCLUDE_DIRS}
  ${dynosam_nn_INCLUDE_DIRS}
  ${rclcpp_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

# build library
add_library(${PROJECT_NAME} SHARED
  src/frontend/FeatureBlockContainer.cc
  src/frontend/SparseFlow.cc
  src/frontend/FeatureTracker.cc
  src/frontend/GeometryUpdater.cc
  src/frontend/MatchingAdaptor.cc
  src/frontend/FlowRefinement.cc
  src/frontend/Frontend.cc

  src/common/Types.cc
  src/common/DynamicSlamInterface.cc
  src/common/Parameters.cc

  src/estimator/BaGraph.cc
  src/estimator/VioGraph.cc
  src/estimator/DoGraph.cc
  src/estimator/FullGraph.cc
  src/estimator/SlamBackend.cc

  src/DynamicSlam.cc
  src/DynamicSlamVisualizer.cc
)




target_link_libraries(${PROJECT_NAME}
  PUBLIC
  ${std_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
  dynosam_common::dynosam_common
  dynosam_sensors::dynosam_sensors
  dynosam_opt::dynosam_opt
  dynosam_nn::dynosam_nn
  dynosam_processing::dynosam_processing
  dynosam_core::dynosam_core
  dynosam_cv::dynosam_cv
)


ament_export_libraries(${PROJECT_NAME}
  dynosam_common::dynosam_common
  dynosam_sensors::dynosam_sensors
  dynosam_opt::dynosam_opt
  dynosam_nn::dynosam_nn
  dynosam_processing::dynosam_processing
)


set(DYNO_EXPORT_DEPS
  dynosam_common
  dynosam_sensors
  dynosam_opt
  dynosam_nn
  dynosam_processing
)


ament_export_dependencies(${DYNO_EXPORT_DEPS})


target_include_directories(${PROJECT_NAME}
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${GFLAGS_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
    ${GTSAM_INCLUDE_DIR}
    ${OPENGV_INCLUDE_DIR}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

ament_export_include_directories(
  include
)

ament_export_libraries(${PROJECT_NAME})

install(
  DIRECTORY include/
  DESTINATION include
)


install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

## This is VERY IMPORTANT so that dynosam2_ros can find the params folder using ament indexing!!
install(DIRECTORY
  params
  DESTINATION share/${PROJECT_NAME}
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)


# #####exec test#########
add_executable(dynosam2_feature_tracking example/dynosam2_feature_tracking.cc)
target_link_libraries(dynosam2_feature_tracking
  ${PROJECT_NAME}
)

install(TARGETS
  dynosam2_feature_tracking
  DESTINATION lib/${PROJECT_NAME})



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
    test/test_feature_block_container.cc
    test/test_optical_flow_benchmark.cc
    test/test_concepts.cc
)

  target_include_directories(${PROJECT_NAME}_test
    PRIVATE
     ${ament_index_cpp_INCLUDE_DIRS}
  )

  target_link_libraries(
    ${PROJECT_NAME}_test
    ${PROJECT_NAME}
    ${rclcpp_LIBRARIES}
    ${ament_index_cpp_TARGETS}
    config_utilities::config_utilities
    gtsam
    ${OpenCV_LIBRARIES}
  )



  install(TARGETS
  ${PROJECT_NAME}_test
      DESTINATION test/)


endif()


ament_package()
