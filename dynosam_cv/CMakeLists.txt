cmake_minimum_required(VERSION 3.11)

project(dynosam_cv)

set(CMAKE_C_FLAGS "-Wall -Wextra  -O3")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

find_package(dynosam_common REQUIRED)

# setup targets
include_directories(
  include
  ${dynosam_common_INCLUDE_DIRS}
)

# build library
add_library(${PROJECT_NAME} SHARED
    src/CameraParams.cc
    src/Camera.cc
    src/ImageContainer.cc
    src/ImageTypes.cc
    src/StereoCamera.cc
    src/RGBDCamera.cc
    src/UndistortRectifier.cc
    src/Feature.cc

)


target_link_libraries(${PROJECT_NAME}
  PUBLIC
  dynosam_common::dynosam_common
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

  ament_lint_auto_find_test_dependencies()

  # Run all lint tests in package.xml except those listed above
  ament_lint_auto_find_test_dependencies()
  ament_add_gmock(${PROJECT_NAME}_test
    test/test_main.cc
    test/test_cuda_cache.cc
)

  target_include_directories(${PROJECT_NAME}_test
    PRIVATE
     ${ament_index_cpp_INCLUDE_DIRS}
  )

  target_link_libraries(
    ${PROJECT_NAME}_test
    ${PROJECT_NAME}
    dynosam_common::dynosam_common
  )



  install(TARGETS
  ${PROJECT_NAME}_test
      DESTINATION test/)

endif()



ament_package()
