cmake_minimum_required(VERSION 3.11)

project(dynosam_vision_common)

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



ament_package()
