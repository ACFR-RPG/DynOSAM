cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(dynosam_nn LANGUAGES C CXX CUDA)

set(CMAKE_C_FLAGS "-Wall -Wextra  -O3")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(MPI REQUIRED)
find_package(dynosam_common REQUIRED)
find_package(dynosam_cv REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Python3 3.10 COMPONENTS Interpreter Development REQUIRED)
find_package(Boost REQUIRED python3 numpy3)

find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)


option(DYNOSAM_NN_USE_TRT "Build with TensorRT" ON)

find_package(CUDAToolkit)
find_package(TensorRT COMPONENTS onnxparser infer_plugin)

set(ENABLE_TENSORRT_CXX_VALUE 1)
if(DYNOSAM_NN_USE_TRT AND (NOT CUDAToolkit_FOUND OR NOT TensorRT_FOUND))
  message(WARNING "TensorRT is required for inference to run!")
  if(NOT CUDAToolkit_FOUND)
    message(WARNING "Missing CUDAToolkit")
  endif()
  if(NOT TensorRT_FOUND)
    message(WARNING "Missing TensorRT")
  endif()
  set(ENABLE_TENSORRT_CXX_VALUE 0)
elseif(NOT DYNOSAM_NN_USE_TRT)
  message(WARNING "Disabling TensorRT! Inference will not work!")
  set(ENABLE_TENSORRT_CXX_VALUE 0)
endif()

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" SYSTEM_ARCH_LOWER)

## --- 1. Detect and Set CUDA Architecture ---
if (SYSTEM_ARCH_LOWER MATCHES "^(x86_64|amd64)$")
    # Case 1: Standard Desktop/Server (x86_64)
    # Use 'all' to target the maximum number of modern GPUs, relying on the user
    # or the CI system to set a more specific list if needed.
    message("Detected x86_64/AMD64. Setting CUDA Architectures to 'all' (flexible build).")
    set(CMAKE_CUDA_ARCHITECTURES all)

elseif (SYSTEM_ARCH_LOWER MATCHES "^(aarch64|arm64)$")
    # Case 2: 64-bit Embedded/ARM (NVIDIA ORIN, Jetson)
    # The NVIDIA ORIN/Jetson series typically uses the Volta (72) or Ampere (87) generation.
    # We will use 72/87 as a robust default.
    # NOTE: Check your specific Jetson model! 75/87 are common targets.
    message("Detected AARCH64/ARM64. Setting CUDA Architectures to 87 (e.g., Orin/Ampere target).")
    set(CMAKE_CUDA_ARCHITECTURES 75 87)

    # Optional: If you support older Jetsons (e.g., Xavier/TX2), you might add:
    # list(APPEND CMAKE_CUDA_ARCHITECTURES 72)

else()
    message(FATAL_ERROR
        "Unsupported System Processor Architecture: ${SYSTEM_ARCH_LOWER}\n"
        "This project only supports x86_64/AMD64 or AARCH64/ARM64. "
        "Please check CMAKE_SYSTEM_PROCESSOR or adjust the build script."
    )

endif()

set(nvcc_options --device-w --ftz=true --prec-div=false --prec-sqrt=false --fmad=true --default-stream per-thread -O3)

# --- 2. Final Output ---
message("Final CMAKE_CUDA_ARCHITECTURES set to: ${CMAKE_CUDA_ARCHITECTURES}")

# setup targets
include_directories(
  include
  ${dynosam_common_INCLUDE_DIRS}
  ${dynosam_cv_INCLUDE_DIRS}
  ${ament_index_cpp_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${CUDAToolkit_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED
  src/PyObjectDetector.cc
  src/ObjectDetector.cc
  src/TrtUtilities.cc
  src/ModelConfig.cc
  src/YoloV8ObjectDetector.cc

  src/CudaUtils.cc
  src/YoloV8CudaUtils.cu


  src/trackers/byte_tracker/ByteTracker.cc
  src/trackers/byte_tracker/KalmanFilter.cc
  src/trackers/byte_tracker/lapjv.cc
  src/trackers/byte_tracker/Rect.cc
  src/trackers/byte_tracker/STrack.cc
  src/trackers/ObjectTracker.cc

  src/bindings/PyBoostWrapper.cc
)

target_link_libraries(${PROJECT_NAME}
PUBLIC
  #  ${Boost_LIBRARIES}
  Boost::python3
  Boost::numpy3
   ${Python3_LIBRARIES}
   ${ament_index_cpp_TARGETS}
  dynosam_common::dynosam_common
  dynosam_cv::dynosam_cv
PRIVATE
  TensorRT::TensorRT
  CUDA::cudart
)

# Enable separable compilation for CUDA (optional but recommended)
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(${PROJECT_NAME}
  PRIVATE
    ${TENSORRT_INCLUDE_DIRS}
  PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
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

install(DIRECTORY
  resources
  DESTINATION share/${PROJECT_NAME}
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
ament_export_dependencies(Boost Python3)

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
  CUDA::cudart
)

install(TARGETS
  image_segmenter_node
  DESTINATION lib/${PROJECT_NAME})

pybind11_add_module(_core MODULE
src/bindings/PyBindings.cc
src/bindings/cvnp.cc
src/bindings/cvnp_synonyms.cc)
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

# so boost components can be found correctly
# https://robotics.stackexchange.com/questions/93010/ament-export-dependenciesboost-not-working
# TODO: honestly should try and make boost and Python private: there is no need to export them!!
ament_package(CONFIG_EXTRAS "cmake/ConfigExtras.cmake")
