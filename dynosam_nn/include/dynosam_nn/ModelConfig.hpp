#pragma once

#include <filesystem>

namespace dyno {

std::filesystem::path getNNWeightsPath();

struct ModelConfig {
  std::filesystem::path model_file;
  std::string log_severity = "INFO";
  bool force_rebuild = false;
  size_t min_optimization_size = 100;
  size_t max_optimization_size = 2000;
  size_t target_optimization_size = 500;

  std::filesystem::path modelPath() const;
  std::filesystem::path enginePath() const;
  std::filesystem::path onnxPath() const;
};

}  // namespace dyno
