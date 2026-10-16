cmake_minimum_required(VERSION 3.11)

project(dynosam_processing)

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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(dynosam_common REQUIRED)
find_package(dynosam_sensors REQUIRED)
find_package(dynosam_cv REQUIRED)
find_package(dynosam_core REQUIRED)

# setup targets
include_directories(
  include
  ${dynosam_common_INCLUDE_DIRS}
  ${dynosam_sensors_INCLUDE_DIRS}
  ${dynosam_cv_INCLUDE_DIRS}
  ${dynosam_core_INCLUDE_DIRS}
)

# build library
add_library(${PROJECT_NAME} SHARED
  src/PipelineBase.cc
  src/DataReader.cc
  src/DataReaderUtils.cc
  src/DatasetReader.cc
  src/DataInterfacePipeline.cc
  src/ThreadSafeImuBuffer.cc
  src/ImageViewer.cc

  src/datasets/DataProviderFactory.cc
  src/datasets/VirtualKittidataProvider.cc
  src/datasets/ClusterSlamDataProvider.cc
  src/datasets/OMDDataProvider.cc
  src/datasets/ProjectAriaDataProvider.cc
  src/datasets/TartanAirShibuya.cc
  src/datasets/ViodeDataProvider.cc
  src/datasets/DynopetsDataProvider.cc


)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    dynosam_common::dynosam_common
    dynosam_sensors::dynosam_sensors
    dynosam_cv::dynosam_cv
    dynosam_core::dynosam_core
)


ament_export_dependencies(
  dynosam_common
  dynosam_sensors
  dynosam_cv
  dynosam_core
)




target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)


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

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_include_directories(include)

if(BUILD_TESTING)
  # find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_gmock REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_index_cpp REQUIRED)


  ament_lint_auto_find_test_dependencies()

  # Run all lint tests in package.xml except those listed above
  ament_lint_auto_find_test_dependencies()
  ament_add_gmock(${PROJECT_NAME}_test
    test/test_pipelines.cc
    test/test_main.cc
    test/thread_safe_queue_tests.cc
    test/thread_safe_temporal_buffer_test.cc
    test/test_dataset_provider.cc
    test/test_threadsafe_imu_buffer.cc

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
