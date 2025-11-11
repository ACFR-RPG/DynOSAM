#include "dynosam_nn/ModelConfig.hpp"

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "dynosam_common/utils/FileSystem.hpp"

namespace fs = std::filesystem;

namespace dyno {

constexpr static auto PackageName = "dynosam_nn";

fs::path getDynosamNNShareDirectory() {
  return fs::path(ament_index_cpp::get_package_share_directory(PackageName));
}

fs::path getNNWeightsPath() { return getDynosamNNShareDirectory() / "weights"; }

// not using the ament resource index things
// (https://github.com/ament/ament_cmake/blob/foxy/ament_cmake_core/doc/resource_index.md)
// becuase it appears to be unnessary here (maybe in future)
std::filesystem::path getNNResourcesPath() {
  return getDynosamNNShareDirectory() / "resources";
}

std::filesystem::path getModelDirectory() {
  const auto weights_path = getNNWeightsPath();
  utils::throwExceptionIfPathInvalid(weights_path);
  return weights_path;
}

std::filesystem::path ModelConfig::getResouce(const std::string& file_name) {
  const auto resources_path = getNNResourcesPath() / file_name;
  utils::throwExceptionIfPathInvalid(resources_path);
  return resources_path;
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
