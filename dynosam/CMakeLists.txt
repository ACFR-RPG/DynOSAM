cmake_minimum_required(VERSION 3.11)

project(dynosam)

# set(CMAKE_C_FLAGS "-std=gnu11 -Wall -Wextra -O3 -march=sandybridge -flto")
set(CMAKE_C_FLAGS "-Wall -Wextra  -O3")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

# Must be set **before** find_package(PCL)
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)

include(FetchContent)



set(WITH_MPI OFF CACHE BOOL "Disable MPI in PCL")


find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(dynosam_common REQUIRED)
find_package(dynosam_cv REQUIRED)
find_package(dynosam_opt REQUIRED)
find_package(dynosam_nn REQUIRED)



include(VerifyGtsamConfig)
option(DYNO_VERIFY_GTSAM_CONFIG "Check that GTSAM was compiled with the right options" ON)
if (DYNO_VERIFY_GTSAM_CONFIG)
    verify_gtsam_config()
endif()


# setup targets
include_directories(
  include
  ${dynosam_common_INCLUDE_DIRS}
  ${dynosam_cv_INCLUDE_DIRS}
  ${dynosam_opt_INCLUDE_DIRS}
  ${dynosam_nn_INCLUDE_DIRS}
  ${rclcpp_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

# build library
add_library(${PROJECT_NAME} SHARED
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
  src/dataprovider/TartanAirShibuya.cc
  src/dataprovider/ViodeDataProvider.cc

  src/frontend/imu/ThreadSafeImuBuffer.cc
  src/frontend/imu/ImuFrontend.cc
  src/frontend/imu/ImuParams.cc

  src/frontend/vision/Frame.cc
  src/frontend/FrontendPipeline.cc
  src/frontend/FrontendModule.cc
  src/frontend/FrontendParams.cc
  src/frontend/RGBDInstanceFrontendModule.cc
  src/frontend/VisionImuOutputPacket.cc
  src/frontend/vision/ORBextractor.cc
  src/frontend/vision/FeatureTracker.cc
  src/frontend/vision/VisionTools.cc
  src/frontend/vision/MotionSolver.cc
  src/frontend/vision/StereoMatcher.cc
  src/frontend/vision/StaticFeatureTracker.cc
  src/frontend/vision/TrackerParams.cc
  src/frontend/vision/FeatureTrackerBase.cc
  src/frontend/vision/FeatureDetector.cc
  src/frontend/RGBDInstance-Definitions.cc
  src/frontend/Frontend-Definitions.cc
  src/frontend/anms/NonMaximumSupression.cc
  src/frontend/anms/anms.cc

  src/backend/BackendDefinitions.cc
  src/backend/BackendPipeline.cc
  src/backend/BackendModule.cc
  src/backend/BackendParams.cc
  src/backend/RegularBackendModule.cc
  src/backend/rgbd/WorldMotionEstimator.cc
  src/backend/rgbd/WorldPoseEstimator.cc
  src/backend/rgbd/HybridEstimator.cc
  src/backend/rgbd/impl/test_HybridFormulations.cc
  src/backend/ParallelObjectISAM.cc
  src/backend/ParallelHybridBackendModule.cc

  # src/backend/ISAM2Agent.cc
  #src/backend/rgbd/impl/ObjectTrackingIsamAgent.cc

  src/factors/LandmarkMotionTernaryFactor.cc
  src/factors/Pose3FlowProjectionFactor.cc
  src/factors/ObjectKinematicFactor.cc
  src/factors/LandmarkMotionPoseFactor.cc
  src/factors/LandmarkPoseSmoothingFactor.cc
  src/factors/HybridFormulationFactors.cc

  src/pipeline/PipelineManager.cc
  src/pipeline/PipelineParams.cc
  src/pipeline/PipelineBase.cc
  src/pipeline/PipelineSpinner.cc

  src/visualizer/VisualizerPipelines.cc
  src/visualizer/OpenCVFrontendDisplay.cc
  src/visualizer/Visualizer-Definitions.cc
)

# target_compile_definitions(${PROJECT_NAME}
#   PRIVATE "MINIMAL_COMPOSITION_DLL")

# eigen3_cmake_module
# Eigen3


target_link_libraries(${PROJECT_NAME}
  PUBLIC
  ${std_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
  dynosam_common::dynosam_common
  dynosam_cv::dynosam_cv
  dynosam_opt::dynosam_opt
  dynosam_nn::dynosam_nn
  # rclcpp::rclcpp
  # opengv
  # nlohmann_json::nlohmann_json
  # config_utilities::config_utilities
  # # PUBLIC
  # ${rclcpp_LIBRARIES}
  # glog
  # gflags
  # gtsam
  # gtsam_unstable
  # Boost::boost
  # png # We need png++ (libpng-dev) for png manipulation in VirtualKittiDatasetLoader
  # # Eigen3::Eigen
  # ${OpenCV_LIBRARIES}
  # ${PCL_LIBRARIES}
  # TBB::tbb
  # TBB::tbbmalloc
  # PRIVATE
)


ament_export_libraries(${PROJECT_NAME}
  # glog
  # gflags
  # opengv
  # Boost::boost
  # config_utilities::config_utilities
  # nlohmann_json::nlohmann_json
  # gtsam
  # # gtsamDebug
  # ${OpenCV_LIBRARIES}
  # ${PCL_LIBRARIES}
  # png
  # TBB::tbb
  # TBB::tbbmalloc
  dynosam_common::dynosam_common
  dynosam_cv::dynosam_cv
  dynosam_opt::dynosam_opt
  dynosam_nn::dynosam_nn
)


set(DYNO_EXPORT_DEPS
  # rclcpp
  # std_msgs
  # sensor_msgs
  # nlohmann_json
  # config_utilities
  # OpenCV
  # opengv
  # PCL
  # GTSAM
  # eigen3_cmake_module
  # Eigen3
  dynosam_common
  dynosam_cv
  dynosam_opt
  dynosam_nn
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

# install test include files for usage in other packges
# header only files will work since the .cc files are not included in the exported library
install(
  DIRECTORY test/internal/
  DESTINATION include/dynosam/test
  FILES_MATCHING PATTERN "*.hpp"
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

## This is VERY IMPORTANT so that dynosam_ros can find the params folder using ament indexing!!
install(DIRECTORY
  params
  DESTINATION share/${PROJECT_NAME}
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)


#####exec test#########
add_executable(dyno_sam example/dyno_sam.cc)
target_link_libraries(dyno_sam
  ${PROJECT_NAME}
)



install(TARGETS
  dyno_sam
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
    test/test_hybrid_motion.cc
    # test/test_landmark_quadric_factor.cc
    test/test_tools.cc
    test/test_map.cc
    test/test_histogram.cc
    test/test_pipelines.cc
    test/test_structured_containers.cc
    # test/test_triangulation.cc
    test/test_backend_structures.cc
    test/test_csv.cc
    test/test_rgbd_backend.cc
    test/test_algorithms.cc
    test/test_viz.cc
    test/test_factors.cc
    test/test_numerical.cc
    test/test_params.cc
    test/test_optimizations.cc
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

  #install data so it can be found on the share package path
  install(
    DIRECTORY test/data/
    DESTINATION test/data/
  )

endif()


ament_package()
