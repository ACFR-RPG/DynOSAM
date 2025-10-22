/* -----------------------------------------------------------------------------
 * BSD 3-Clause License
 *
 * Copyright (c) 2021-2024, Massachusetts Institute of Technology.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * * --------------------------------------------------------------------------
 */

#pragma once

#include <NvInfer.h>

#include <filesystem>
#include <memory>
#include <numeric>
#include <opencv4/opencv2/core/mat.hpp>
#include <optional>
#include <string>

#include "dynosam_nn/ModelConfig.hpp"

namespace dyno {

using Severity = nvinfer1::ILogger::Severity;
using EnginePtr = std::unique_ptr<nvinfer1::ICudaEngine>;
using RuntimePtr = std::unique_ptr<nvinfer1::IRuntime>;

struct CudaMemoryManager {
  static void* alloc(size_t size);

  struct Delete {
    void operator()(void* object);
  };
};

struct Shape {
  int width = -1;
  int height = -1;
  std::optional<int> channels;
  bool chw_order = false;

  Shape updateFrom(const cv::Mat& input);

  nvinfer1::Dims dims() const;

  size_t numel() const;

  bool operator==(const Shape& other) const;
};

struct ImageMemoryPair {
  std::unique_ptr<float, CudaMemoryManager::Delete> device_image;
  cv::Mat host_image;
  Shape shape;

  inline operator bool() const {
    return !host_image.empty() && device_image != nullptr;
  }

  bool updateShape(const Shape& shape);

  size_t size() const;
};

std::string toString(const nvinfer1::Dims& dims);

std::string toString(nvinfer1::DataType dtype);

std::string toString(nvinfer1::TensorIOMode mode);

RuntimePtr getRuntime(const std::string& verbosity);

EnginePtr deserializeEngine(nvinfer1::IRuntime& runtime,
                            const std::filesystem::path& engine_path);

// EnginePtr buildEngineFromEngine(const ModelConfig& model_config,
//                               nvinfer1::IRuntime& runtime);

EnginePtr buildEngineFromOnnx(const ModelConfig& model_config,
                              nvinfer1::IRuntime& runtime);

}  // namespace dyno
