cmake_minimum_required(VERSION 3.11)

project(dynosam_nn)

set(CMAKE_C_FLAGS "-Wall -Wextra  -O3")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

find_package(dynosam_common REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Python3 3.10 COMPONENTS Interpreter Development REQUIRED)
find_package(Boost COMPONENTS python3 numpy3 REQUIRED)

find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)



# setup targets
include_directories(
  include
  ${dynosam_common_INCLUDE_DIRS}
  ${ament_index_cpp_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED
  src/PyObjectDetector.cc
  src/ObjectDetector.cc
  src/PyWrapper.cc)



target_link_libraries(${PROJECT_NAME}
  PUBLIC
   ${Boost_LIBRARIES}
   ${Python3_LIBRARIES}
   ${ament_index_cpp_TARGETS}
  dynosam_common::dynosam_common
)


target_include_directories(${PROJECT_NAME}
  PUBLIC
    ${Boost_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${ament_index_cpp_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)


install(
  DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY
  weights
  DESTINATION share/${PROJECT_NAME}
)



install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

add_executable(test_nn nodes/test_nn.cc)
target_link_libraries(test_nn
  ${PROJECT_NAME}
)

install(TARGETS
  test_nn
  DESTINATION lib/${PROJECT_NAME})


add_executable(image_segmenter_node nodes/ImageSegmenterNode.cc)
target_link_libraries(image_segmenter_node
  ${PROJECT_NAME}
  rclcpp::rclcpp
  image_transport::image_transport
  sensor_msgs::sensor_msgs_library
  cv_bridge::cv_bridge
)

install(TARGETS
  image_segmenter_node
  DESTINATION lib/${PROJECT_NAME})

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_include_directories(include)



pybind11_add_module(_core MODULE src/PyBindWrapper.cc)
target_link_libraries(_core
  PUBLIC
   ${Boost_LIBRARIES}
   ${Python3_LIBRARIES}
   ${PROJECT_NAME}
  dynosam_common::dynosam_common
)

# Install the compiled extension into the Python package folder
install(
  TARGETS _core
  DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}_py"
)


#install python scripts/modules
ament_python_install_package(${PROJECT_NAME}_py)

ament_package()
