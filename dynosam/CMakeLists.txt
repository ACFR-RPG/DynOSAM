cmake_minimum_required(VERSION 3.8)

project(dynosam)

# set(CMAKE_C_FLAGS "-std=gnu11 -Wall -Wextra -O3 -march=sandybridge -flto")
set(CMAKE_C_FLAGS "-Wall -Wextra")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)


list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(diagnostic_updater REQUIRED)
find_package(OpenCV REQUIRED)
find_package(opengv REQUIRED)
find_package(GTSAM REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED) #nlohmann-json3-dev


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
  ${rclcpp_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

# build library
add_library(${PROJECT_NAME} SHARED

  src/common/Types.cc
  src/common/CameraParams.cc
  src/common/Camera.cc
  src/common/GroundTruthPacket.cc
  src/common/ImageContainer.cc
  src/common/ImageTypes.cc
  src/common/StructuredContainers.cc
  src/common/DynamicObjects.cc
  src/common/Exceptions.cc
  src/common/Map.cc
  src/common/MapNodes.cc
  src/common/PointCloudProcess.cc
  src/common/Algorithms.cc
  src/common/StereoCamera.cc

  src/common/byte_tracker/ByteTracker.cc
  src/common/byte_tracker/Detection.cc
  src/common/byte_tracker/KalmanFilter.cc
  src/common/byte_tracker/Lapjv.cc
  src/common/byte_tracker/Rect.cc
  src/common/byte_tracker/Track.cc

  src/dataprovider/DataInterfacePipeline.cc
  src/dataprovider/DataProvider.cc
  src/dataprovider/DatasetLoader.cc
  src/dataprovider/DataProviderUtils.cc
  src/dataprovider/DataProviderFactory.cc
  src/dataprovider/VirtualKittidataProvider.cc
  src/dataprovider/ClusterSlamDataProvider.cc
  src/dataprovider/OMDDataProvider.cc
  src/dataprovider/DatasetProvider.cc
  src/dataprovider/ProjectAriaDataProvider.cc

  src/frontend/imu/ThreadSafeImuBuffer.cc
  src/frontend/vision/Feature.cc
  src/frontend/vision/UndistortRectifier.cc
  src/frontend/vision/Frame.cc
  src/frontend/FrontendPipeline.cc
  src/frontend/FrontendModule.cc
  src/frontend/FrontendParams.cc
  src/frontend/RGBDInstanceFrontendModule.cc
  src/frontend/MonoInstanceFrontendModule.cc
  src/frontend/vision/ORBextractor.cc
  src/frontend/vision/FeatureTracker.cc
  src/frontend/vision/ObjectTracker.cc
  src/frontend/vision/VisionTools.cc
  src/frontend/vision/MotionSolver.cc
  src/frontend/vision/StereoMatcher.cc
  src/frontend/RGBDInstance-Definitions.cc
  src/frontend/Frontend-Definitions.cc

  src/backend/BackendDefinitions.cc
  src/backend/BackendPipeline.cc
  src/backend/BackendModule.cc
  src/backend/BackendParams.cc
  src/backend/FactorGraphTools.cc
  #src/backend/MonoBackendModule.cc
  #src/backend/MonoBatchBackendModule.cc
  src/backend/MonoBackendTools.cc
  src/backend/DynamicPointSymbol.cc
  src/backend/RGBDBackendModule.cc
  src/backend/DynoISAM2.cc
  src/backend/DynoISAM2Params.cc
  src/backend/DynoISAM2-impl.cc
  src/backend/DynoISAM2Clique.cc
  src/backend/ObjectCentricUpdater.cc

  src/factors/LandmarkMotionTernaryFactor.cc
  src/factors/Pose3FlowProjectionFactor.cc
  src/factors/LandmarkQuadricFactor.cc
  src/factors/ObjectKinematicFactor.cc
  src/factors/LandmarkMotionPoseFactor.cc
  src/factors/LandmarkPoseSmoothingFactor.cc
  src/factors/ObjectCentricMotionFactor.cc

  src/pipeline/PipelineManager.cc
  src/pipeline/PipelineDefinitions.cc
  src/pipeline/PipelineParams.cc
  src/pipeline/PipelineBase.cc
  src/pipeline/PipelineSpinner.cc

  src/logger/Logger.cc

  src/utils/OpenCVUtils.cc
  src/utils/GtsamUtils.cc
  src/utils/Spinner.cc
  src/utils/Statistics.cc
  src/utils/TimingStats.cc
  src/utils/Metrics.cc
  src/utils/CsvParser.cc
  src/utils/JsonUtils.cc

  src/visualizer/ColourMap.cc
  src/visualizer/VisualizerPipelines.cc
  src/visualizer/OpenCVFrontendDisplay.cc
  src/visualizer/Visualizer-Definitions.cc
)

# target_compile_definitions(${PROJECT_NAME}
#   PRIVATE "MINIMAL_COMPOSITION_DLL")

# eigen3_cmake_module
# Eigen3

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  std_msgs
  sensor_msgs
  diagnostic_updater
  opengv
  GTSAM
  OpenCV)

target_link_libraries(${PROJECT_NAME}
  # PRIVATE
  opengv
  nlohmann_json::nlohmann_json
  # PUBLIC
  ${rclcpp_LIBRARIES}
  glog
  gflags
  gtsam
  gtsam_unstable
  # gtsamDebug
  # gtsam_unstableDebug
  Boost::boost
  png # We need png++ (libpng-dev) for png manipulation in VirtualKittiDatasetLoader
  # Eigen3::Eigen
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  # PRIVATE
)


ament_export_libraries(${PROJECT_NAME}
  glog
  gflags
  opengv
  Boost::boost
  gtsam
  # gtsamDebug
  ${OpenCV_LIBRARIES}
  png
)


set(DYNO_EXPORT_DEPS
  rclcpp
  std_msgs
  sensor_msgs
  OpenCV opengv
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

ament_export_include_directories(include  ${GTSAM_INCLUDE_DIR} ${OPENGV_INCLUDE_DIR})
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

## This is VERY IMPORTANT so that dynosam_ros can find the params folder using ament indexing!!
install(DIRECTORY
  params
  DESTINATION share/${PROJECT_NAME}
)


#####exec test#########
add_executable(dyno_sam example/dyno_sam.cc)
target_link_libraries(dyno_sam
  ${PROJECT_NAME}
)
ament_target_dependencies(dyno_sam "rclcpp" "std_msgs")



install(TARGETS
  dyno_sam
  DESTINATION lib/${PROJECT_NAME})



if(BUILD_TESTING)
  # find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_gmock REQUIRED)
  find_package(ament_lint_auto REQUIRED)

  ament_lint_auto_find_test_dependencies()

  # Run all lint tests in package.xml except those listed above
  ament_lint_auto_find_test_dependencies()
  ament_add_gmock(${PROJECT_NAME}_test
    test/test_main.cc
    test/thread_safe_queue_tests.cc
    test/thread_safe_temporal_buffer_test.cc
    test/test_threadsafe_imu_buffer.cc
    test/test_dataset_provider.cc
    test/test_types.cc
    test/test_code_concepts.cc
    test/test_camera_params.cc
    test/test_camera.cc
    test/test_dynamic_point_symbol.cc
    test/internal/tmp_file.cc
    test/internal/simulator.cc
    test/test_landmark_quadric_factor.cc
    test/test_tools.cc
    test/test_map.cc
    test/test_pipelines.cc
    test/test_structured_containers.cc
    test/test_triangulation.cc
    test/test_backend_structures.cc
    test/test_csv.cc
    test/test_rgbd_backend.cc
    test/test_algorithms.cc
    test/test_viz.cc
    test/test_factors.cc
    test/test_numerical.cc
)

  target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME} ${rclcpp_LIBRARIES})

  install(TARGETS
  ${PROJECT_NAME}_test
      DESTINATION test/)

  #install data so it can be found on the share package path
  install(
    DIRECTORY test/data/
    DESTINATION test/data/
  )

endif()


ament_package()
