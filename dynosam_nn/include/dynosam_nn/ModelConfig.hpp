#pragma once

#include <filesystem>

namespace dyno {

std::filesystem::path getNNWeightsPath();
std::filesystem::path getNNResourcesPath();

struct ModelConfig {
  std::filesystem::path model_file;
  std::string log_severity = "INFO";
  //! If true, then the model path will be
  //! loaded directly from the environment variable DYNOSAM_NN_MODEL_DIR
  // bool use_model_path_from_env = false;
  bool force_rebuild = false;
  size_t min_optimization_size = 100;
  size_t max_optimization_size = 2000;
  size_t target_optimization_size = 500;

  /**
   * @brief Safetly get the full file path of a resource using
   * getNNResourcesPath() / file_name
   *
   * DynosamException will be thrown if the full path does not exist.
   *
   * @param file_name
   * @return std::filesystem::path
   */
  static std::filesystem::path getResouce(const std::string& file_name);
  std::filesystem::path modelPath() const;
  std::filesystem::path enginePath() const;
  std::filesystem::path onnxPath() const;
};

}  // namespace dyno
