cmake_minimum_required(VERSION 3.11)

project(dynosam_common)

set(CMAKE_C_FLAGS "-Wall -Wextra  -O3")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

# Must be set **before** find_package(PCL)
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)



set(WITH_MPI OFF CACHE BOOL "Disable MPI in PCL")

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(opengv REQUIRED)
find_package(GTSAM REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters features)
find_package(TBB REQUIRED) # should be packaged with GTSAM
find_package(nlohmann_json 3.11.3 REQUIRED)
find_package(config_utilities REQUIRED)


find_package(Boost REQUIRED)
if(NOT TARGET Boost::boost)
  add_library(Boost::boost INTERFACE IMPORTED)
  set_target_properties(Boost::boost PROPERTIES
  INTERFACE_LINK_LIBRARIES "${Boost_LIBRARIES}"
  INTERFACE_INCLUDE_DIRECTORIES "${Boost_INCLUDE_DIRS}")
endif()


include(VerifyGtsamConfig)
option(DYNO_VERIFY_GTSAM_CONFIG "Check that GTSAM was compiled with the right options" ON)
if (DYNO_VERIFY_GTSAM_CONFIG)
    verify_gtsam_config()
endif()


# setup targets
include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

# build library
add_library(${PROJECT_NAME} SHARED
    src/Types.cc
    src/GroundTruthPacket.cc
    src/StructuredContainers.cc
    src/DynamicObjects.cc
    src/PointCloudProcess.cc
    src/Algorithms.cc
    src/SharedModuleInfo.cc
    src/SensorModels.cc

    src/byte_tracker/ByteTracker.cc
    src/byte_tracker/Detection.cc
    src/byte_tracker/KalmanFilter.cc
    src/byte_tracker/Lapjv.cc
    src/byte_tracker/Rect.cc
    src/byte_tracker/Track.cc

    src/utils/OpenCVUtils.cc
    src/utils/GtsamUtils.cc
    src/utils/Spinner.cc
    src/utils/Statistics.cc
    src/utils/TimingStats.cc
    src/utils/Metrics.cc
    src/utils/CsvParser.cc
    src/utils/Histogram.cc
    src/utils/FileSystem.cc

    src/logger/Logger.cc

    src/viz/Colour.cc

)


target_link_libraries(${PROJECT_NAME}
  PUBLIC
  opengv
  nlohmann_json::nlohmann_json
  config_utilities::config_utilities
  glog
  gflags
  gtsam
  gtsam_unstable
  Boost::boost
  png # We need png++ (libpng-dev) for png manipulation in VirtualKittiDatasetLoader
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  TBB::tbb
  TBB::tbbmalloc
  # PRIVATE
)


ament_export_libraries(${PROJECT_NAME}
  glog
  gflags
  opengv
  Boost::boost
  config_utilities::config_utilities
  nlohmann_json::nlohmann_json
  gtsam
  gtsam_unstable
  # gtsamDebug
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  png
  TBB::tbb
  TBB::tbbmalloc
)


set(DYNO_EXPORT_DEPS
  rclcpp
  std_msgs
  sensor_msgs
  nlohmann_json
  config_utilities
  OpenCV
  opengv
  PCL
  GTSAM
  eigen3_cmake_module
  Eigen3
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
ament_export_include_directories(include  ${GTSAM_INCLUDE_DIR} ${OPENGV_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})




ament_package()
