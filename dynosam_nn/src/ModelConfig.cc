#include "dynosam_nn/ModelConfig.hpp"

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "dynosam_common/utils/FileSystem.hpp"

namespace fs = std::filesystem;

namespace dyno {

fs::path getNNWeightsPath() {
  return fs::path(ament_index_cpp::get_package_share_directory("dynosam_nn")) /
         "weights";
}

std::filesystem::path getModelDirectory() {
  const auto weights_path = getNNWeightsPath();
  utils::throwExceptionIfPathInvalid(weights_path);
  return weights_path;
}

std::filesystem::path ModelConfig::modelPath() const {
  return getModelDirectory() / model_file;
}

std::filesystem::path ModelConfig::enginePath() const {
  return modelPath().replace_extension(".trt");
}

std::filesystem::path ModelConfig::onnxPath() const {
  return modelPath().replace_extension(".onnx");
}

}  // namespace dyno
