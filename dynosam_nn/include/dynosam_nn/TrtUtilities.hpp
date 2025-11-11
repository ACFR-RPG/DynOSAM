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
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>

#include <filesystem>
#include <memory>
#include <numeric>
#include <opencv4/opencv2/core/mat.hpp>
#include <optional>
#include <string>

#include "dynosam_common/Exceptions.hpp"
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

template <typename T>
using unique_cuda_ptr = std::unique_ptr<T, CudaMemoryManager::Delete>;

struct TensorInfo {
  std::string name;
  nvinfer1::Dims dims;
  nvinfer1::DataType dtype;

  /**
   * @brief Calculates total size of the tensor (ie. product of all dimensions)
   * If any of the dimensions are dynamic function returns -1
   *
   * @return int
   */
  int size() const;
  int isDynamic() const;
};

std::ostream& operator<<(std::ostream& out, const TensorInfo& info);

class ImageTensorInfo : public TensorInfo {
 public:
  struct Shape {
    int width = -1;
    int height = -1;
    std::optional<int> channels;
    bool chw_order = false;

    /**
     * @brief Casting operator to cv::Size
     *
     * @return cv::Size
     */
    inline operator cv::Size() const { return cv::Size(width, height); }
  };

  ImageTensorInfo(const TensorInfo& info);
  ImageTensorInfo& operator=(const TensorInfo& info);

  Shape shape() const { return shape_; }

 private:
  Shape shape_;
};

std::ostream& operator<<(std::ostream& out,
                         const ImageTensorInfo::Shape& shape);
std::ostream& operator<<(std::ostream& out, const ImageTensorInfo& info);

bool isDynamic(const nvinfer1::Dims& dims);

template <typename T>
struct DeviceMemory {
  using DataType = T;
  using DataTypePtr = T*;
  using UniqueDevicePtr = unique_cuda_ptr<T>;

  UniqueDevicePtr device_pointer;
  //! Size of requested tensor
  size_t tensor_size{0};
  //! Memory (in bytes) allocated to the device pointer.
  //! This in effect is tensor_size * sizeof(T)
  size_t allocated_size{0};

  bool allocate(const TensorInfo& info) {
    try {
      // TODO: check info type is same as T!

      // returns total memory needed for allocation and the size of the tensor
      //  total memory is effectively t_size * size of datatype
      auto [memory_needed, t_size] = DeviceMemory<T>::memorySize(info);

      if (allocated_size == memory_needed) {
        CHECK_EQ(tensor_size, t_size);
        return true;
      }

      LOG(INFO) << "Allocating " << memory_needed << " from tensor info "
                << info << " and tensor size " << t_size;
      device_pointer.reset(reinterpret_cast<DataTypePtr>(
          CudaMemoryManager::alloc(memory_needed)));
      allocated_size = memory_needed;
      tensor_size = t_size;
      return true;

    } catch (const DynosamException& e) {
      LOG(WARNING) << "Failed to allocate device memory: " << e.what();
      return false;
    }
  }

  static std::pair<size_t, size_t> memorySize(const TensorInfo& info) {
    if (info.isDynamic()) {
      DYNO_THROW_MSG(DynosamException)
          << "Cannot determine memory size from dynamically sized tensor info"
          << info;
      throw;
    }
    size_t tensor_size = static_cast<size_t>(info.size());
    size_t total_memory = sizeof(DataType) * tensor_size;
    return {total_memory, tensor_size};
  }

  bool getFromDevice(std::vector<DataType>& data, cudaStream_t stream = 0) {
    data.resize(tensor_size);
    auto error =
        cudaMemcpyAsync(data.data(), device_pointer.get(),
                        allocated_size,  // note allocated size not tensor size,
                        cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess) {
      LOG(ERROR) << "Error copying host -> device: "
                 << cudaGetErrorString(error);
      return false;
    }
    return true;
  }

  // data should really be const here
  bool pushFromHost(std::vector<DataType>& data, cudaStream_t stream = 0) {
    CHECK_EQ(data.size(), tensor_size);
    auto error =
        cudaMemcpyAsync(device_pointer.get(), data.data(), allocated_size,
                        cudaMemcpyHostToDevice, stream);

    if (error != cudaSuccess) {
      LOG(ERROR) << "Error copying device -> host: "
                 << cudaGetErrorString(error);
      return false;
    }
    return true;
  }
};

struct ImageMemoryPair {
  std::unique_ptr<float, CudaMemoryManager::Delete> device_image;
  cv::Mat host_image;
  // Shape shape;

  inline operator bool() const {
    return !host_image.empty() && device_image != nullptr;
  }

  // bool updateShape(const Shape& shape);

  // size_t size() const;
};

class TRTEngine {
 public:
  TRTEngine(const ModelConfig& config);
  virtual ~TRTEngine();

  inline bool isInitalized() const { return initialized_; }

 private:
  bool initialized_ = false;

 protected:
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  cudaStream_t stream_;
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
