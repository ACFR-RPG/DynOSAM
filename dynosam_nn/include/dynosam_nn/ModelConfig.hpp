#pragma once

#include <filesystem>

namespace dyno {

/**
 * @brief Specifies the default location of all weights used by dynosam_nn.
 * This should in the dynosam_nn share package location + weights (e.g.
 * /dynosam_nn/share/dynosam_nn/weights)
 *
 * @return std::filesystem::path
 */
std::filesystem::path getNNWeightsPath();

/**
 * @brief Specifies the default location of all weights used by dynosam_nn.
 * This should in the dynosam_nn share package location + resources (e.g.
 * /dynosam_nn/share/dynosam_nn/resouces)
 *
 * @return std::filesystem::path
 */
std::filesystem::path getNNResourcesPath();

struct ModelConfig {
  //! The file name (only) of the models weights. Should have some suffix
  //! (likely .pt e.g. from YOLO)
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
