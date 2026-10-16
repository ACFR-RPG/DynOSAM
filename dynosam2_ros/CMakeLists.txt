cmake_minimum_required(VERSION 3.11)

project(dynosam2_ros)

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


find_package(dynosam_ros_common)
find_package(dynosam2)

# setup targets
include_directories(
  include
  ${dynosam_ros_common_INCLUDE_DIRS}
  ${dynosam2_INCLUDE_DIRS}
)



# build library
add_library(${PROJECT_NAME} SHARED
  src/Dynosam2Node.cc
)

target_link_libraries(${PROJECT_NAME}
  ${LINK_LIBS}
  glog
  gflags
  rclcpp_components::component
  rclcpp_components::component_manager
  dynosam_ros_common::dynosam_ros_common
  dynosam2::dynosam2
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
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)

add_executable(dynosam2_node nodes/dynosam2_node_main.cc)
target_link_libraries(dynosam2_node
  ${PROJECT_NAME}
)
install(TARGETS
  dynosam2_node
  DESTINATION lib/${PROJECT_NAME})


# rclcpp_components_register_nodes(
#     ${PROJECT_NAME}
#     PLUGIN "dyno::DynosamComposableNode"
# )


install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# install(DIRECTORY
#   config
#   DESTINATION share/${PROJECT_NAME}
# )






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
