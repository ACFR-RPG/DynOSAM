cmake_minimum_required(VERSION 3.11)

project(dynosam_opt)

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
include(FetchContent)


#Prefer Boost >= 1.81 for regular flat hash containers.
find_package(Boost 1.81 QUIET MODULE)

if(Boost_FOUND)
    message(WARNING "DynoSAM containers: Using Boost flat containers")
    set(DYNOSAM_USE_BOOST_FLAT_CONTAINERS ON)
    set(DYNOSAM_HAS_PHMAP OFF)
else()
    message(WARNING "DynoSAM containers: Boost >= 1.81 unavailable; using phmap")

    set(PHMAP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(PHMAP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(PHMAP_INSTALL OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        parallel-hashmap
        GIT_REPOSITORY https://github.com/greg7mdp/parallel-hashmap.git
        GIT_TAG        v2.1.0
    )

    FetchContent_MakeAvailable(parallel-hashmap)

    # target_link_libraries(${PROJECT_NAME} PUBLIC phmap)
    include_directories(${parallel-hashmap_SOURCE_DIR})

    set(DYNOSAM_USE_BOOST_FLAT_CONTAINERS OFF)
    set(DYNOSAM_HAS_PHMAP ON)
endif()



# setup targets
include_directories(
  include
  ${dynosam_common_INCLUDE_DIRS}
  ${dynosam_sensors_INCLUDE_DIRS}
)

# build library
add_library(${PROJECT_NAME} SHARED
    src/FactorGraphTools.cc
    src/Symbols.cc

    src/SlidingWindowOptimization.cc
    src/ISAM2-impl.cc
    src/ISAM2.cc
    src/ISAM2Clique.cc
    src/ISAM2Params.cc
    src/IncrementalOptimization.cc
    src/NonlinearOptimizer.cc
    src/StateQuery.cc

    src/factors/LandmarkMotionTernaryFactor.cc
    src/factors/Pose3FlowProjectionFactor.cc
    src/factors/ObjectKinematicFactor.cc
    src/factors/LandmarkMotionPoseFactor.cc
    src/factors/LandmarkPoseSmoothingFactor.cc
    src/factors/HybridFormulationFactors.cc

)

target_compile_definitions(${PROJECT_NAME} PUBLIC
  DYNOSAM_USE_BOOST_FLAT_CONTAINERS=$<BOOL:${DYNOSAM_USE_BOOST_FLAT_CONTAINERS}>
  DYNOSAM_HAS_PHMAP=$<BOOL:${DYNOSAM_HAS_PHMAP}>
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
  dynosam_common::dynosam_common
  dynosam_sensors::dynosam_sensors
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
    test/test_main.cc
    test/test_hybrid_motion.cc
    test/test_map.cc
    test/test_factors.cc


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
