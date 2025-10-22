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

#include "dynosam_nn/TrtUtilities.hpp"

#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>

#include <fstream>
#include <map>

// #include "semantic_inference/logging.h"

namespace dyno {

using nvinfer1::IHostMemory;
using nvinfer1::INetworkDefinition;
using nvinfer1::IRuntime;
using nvinfer1::NetworkDefinitionCreationFlag;
using nvonnxparser::IParser;

Severity stringToSeverity(const std::string& severity) {
  std::map<std::string, Severity> level_map{
      {"INTERNAL_ERROR", Severity::kINTERNAL_ERROR},
      {"ERROR", Severity::kERROR},
      {"WARNING", Severity::kWARNING},
      {"INFO", Severity::kINFO},
      {"VERBOSE", Severity::kVERBOSE}};
  auto iter = level_map.find(severity);
  return iter == level_map.end() ? Severity::kINFO : iter->second;
}

class LoggingShim : public nvinfer1::ILogger {
 public:
  static LoggingShim& instance(const std::string& severity = "INFO") {
    if (!s_instance_) {
      s_instance_.reset(new LoggingShim());
    }

    s_instance_->setSeverity(severity);
    return *s_instance_;
  }

  void log(Severity severity, const char* msg) noexcept override {
    if (severity > min_severity_) {
      return;
    }

    // logging::Level level;
    // switch (severity) {
    //   case Severity::kINTERNAL_ERROR:
    //     level = logging::Level::FATAL;
    //     break;
    //   case Severity::kERROR:
    //     level = logging::Level::ERROR;
    //     break;
    //   case Severity::kWARNING:
    //     level = logging::Level::WARNING;
    //     break;
    //   case Severity::kINFO:
    //     level = logging::Level::INFO;
    //     break;
    //   case Severity::kVERBOSE:
    //   default:
    //     level = logging::Level::DEBUG;
    //     break;
    // }

    LOG(INFO) << msg;

    // // disable file-like formatting by specifying negative line number
    // logging::LogEntry entry(level, "TensorRT", -1);
    // entry << msg;
  }

  void setSeverity(const std::string& severity) {
    min_severity_ = stringToSeverity(severity);
  }

 private:
  Severity min_severity_{};
  inline static std::unique_ptr<LoggingShim> s_instance_ = nullptr;
  LoggingShim() : min_severity_(Severity::kVERBOSE) {}
};

void* CudaMemoryManager::alloc(size_t size) {
  void* raw_ptr = nullptr;
  auto error = cudaMalloc(&raw_ptr, size);
  if (error != cudaSuccess) {
    // LOG(ERROR) << "Failed to allocate " << size << " bytes on device";
  }

  return raw_ptr;
}

void CudaMemoryManager::Delete::operator()(void* object) {
  if (object != nullptr) {
    cudaFree(object);
  }
}

Shape Shape::updateFrom(const cv::Mat& input) {
  int new_width = width == -1 ? input.cols : width;
  int new_height = height == -1 ? input.rows : height;
  return {new_width, new_height, channels, chw_order};
}

nvinfer1::Dims Shape::dims() const {
  return chw_order ? nvinfer1::Dims3(channels.value_or(1), height, width)
                   : nvinfer1::Dims3(height, width, channels.value_or(1));
}

size_t Shape::numel() const {
  if (width == -1 || height == -1) {
    return 0;
  }

  return width * height * channels.value_or(1);
}

bool Shape::operator==(const Shape& other) const {
  return other.width == width && other.height == height &&
         other.channels == channels;
}

bool ImageMemoryPair::updateShape(const Shape& new_shape) {
  if (new_shape.width == -1 || new_shape.height == -1) {
    // LOG(ERROR) << "Cannot create matrices with unspecified sizes!";
    return false;
  }

  if (shape == new_shape) {
    return true;
  }

  shape = new_shape;

  std::vector<int> dims;
  if (shape.chw_order) {
    dims = {shape.channels.value_or(1), shape.height, shape.width};
  } else {
    dims = {shape.height, shape.width, shape.channels.value_or(1)};
  }

  // this is not relevant for us!!
  host_image = cv::Mat(dims, CV_32FC1);
  device_image.reset(
      reinterpret_cast<float*>(CudaMemoryManager::alloc(size())));
  return true;
}

size_t ImageMemoryPair::size() const { return shape.numel() * sizeof(float); }

std::string toString(const nvinfer1::Dims& dims) {
  std::stringstream ss;
  ss << "[";
  for (int32_t i = 0; i < dims.nbDims - 1; ++i) {
    ss << dims.d[i] << " x ";
  }
  ss << dims.d[dims.nbDims - 1] << "]";
  return ss.str();
}

std::string toString(nvinfer1::DataType dtype) {
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
      return "FLOAT";
    case nvinfer1::DataType::kHALF:
      return "HALF";
    case nvinfer1::DataType::kINT8:
      return "INT8";
    case nvinfer1::DataType::kINT32:
      return "INT32";
    case nvinfer1::DataType::kBOOL:
      return "BOOL";
    case nvinfer1::DataType::kUINT8:
      return "UINT8";
    case nvinfer1::DataType::kFP8:
      return "FP8";
#if NV_TENSORRT_MAJOR >= 10
    case nvinfer1::DataType::kBF16:
      return "BF16";
    case nvinfer1::DataType::kINT64:
      return "INT64";
    case nvinfer1::DataType::kINT4:
      return "INT4";
#endif
    default:
      return "UNKNOWN: '" + std::to_string(static_cast<int>(dtype)) + "'";
  }
}

std::string toString(nvinfer1::TensorIOMode mode) {
  switch (mode) {
    case nvinfer1::TensorIOMode::kINPUT:
      return "in";
    case nvinfer1::TensorIOMode::kOUTPUT:
      return "out";
    default:
    case nvinfer1::TensorIOMode::kNONE:
      return "none";
  }
}

RuntimePtr getRuntime(const std::string& log_severity) {
  auto& logger = LoggingShim::instance(log_severity);
  return RuntimePtr(nvinfer1::createInferRuntime(logger));
}

EnginePtr deserializeEngine(IRuntime& runtime,
                            const std::filesystem::path& engine_path) {
  std::ifstream engine_file(engine_path, std::ios::binary);
  if (engine_file.fail()) {
    LOG(INFO) << "Engine file: " << engine_path << " not found!";
    return nullptr;
  }

  const auto engine_size = std::filesystem::file_size(engine_path);
  std::vector<char> engine_data(engine_size);
  engine_file.read(engine_data.data(), engine_size);

  EnginePtr engine(
      runtime.deserializeCudaEngine(engine_data.data(), engine_size));
  if (!engine) {
    LOG(ERROR) << "Engine deserialization failed for " << engine_path;
    return nullptr;
  }

  return engine;
}

inline bool isDynamic(const nvinfer1::Dims& dims) {
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0) {
      return true;
    }
  }

  return false;
}

inline nvinfer1::Dims replaceDynamic(const nvinfer1::Dims& dims,
                                     int64_t new_value) {
  auto new_dims = dims;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0) {
      new_dims.d[i] = new_value;
    }
  }

  return new_dims;
}

// EnginePtr buildEngineFromEngine(const ModelConfig& model_config,
// nvinfer1::IRuntime& runtime) {
//   const auto model_path = model_config.enginePath();
//   LOG(INFO) << "Building engine from " << model_path << "...";

//   // 2. Read the engine file into a buffer
//   std::ifstream engineFile(model_path, std::ios::binary);
//   if (!engineFile) {
//       LOG(ERROR) << "Failed to open engine file: " << model_path;
//       return nullptr;
//   }
//   model_path.seekg(0, std::ios::end);
//   long int fsize = model_path.tellg();
//   model_path.seekg(0, std::ios::beg);
//   std::vector<char> engineData(fsize);
//   model_path.read(engineData.data(), fsize);
//   model_path.close();

//   // 3. Deserialize the engine
//   EnginePtr engine(runtime.deserializeCudaEngine(engineData.data(), fsize));
//   if (!engine) {
//       LOG(ERROR) << "Failed to deserialize CUDA engine.";
//       // runtime->destroy();
//       return nullptr;
//   }
//   return engine;
// }

EnginePtr buildEngineFromOnnx(const ModelConfig& model_config,
                              IRuntime& runtime) {
  using nvinfer1::Dims3;
  using nvinfer1::OptProfileSelector;

  const auto model_path = model_config.onnxPath();
  LOG(INFO) << "Building onnx engine from " << model_path << "...";
  auto& logger = LoggingShim::instance(model_config.log_severity);
  std::unique_ptr<nvinfer1::IBuilder> builder(
      nvinfer1::createInferBuilder(logger));
  int flags = 0;
#if NV_TENSORRT_MAJOR < 10
  // required for exported models previously to tensorrt 10
  flags = (1U << static_cast<uint32_t>(
               NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
#endif

  std::unique_ptr<INetworkDefinition> net(builder->createNetworkV2(flags));
  std::unique_ptr<IParser> parser(nvonnxparser::createParser(*net, logger));
  parser->parseFromFile(model_path.c_str(),
                        static_cast<int>(Severity::kVERBOSE));
  for (int i = 0; i < parser->getNbErrors(); ++i) {
    LOG(ERROR) << "Parser Error #" << i << ": " << parser->getError(i)->desc();
  }

  // these are details specific the the YOLO model...
  if (net->getNbOutputs() != 2) {
    LOG(ERROR) << "Network does not have a single output: "
               << net->getNbOutputs() << " != 2";
    return nullptr;
  }

  auto output = net->getOutput(0);
  auto output_dtype = output->getType();
  if (output_dtype != nvinfer1::DataType::kFLOAT) {
    LOG(WARNING) << "Add extra output cast layer to INT32";
    // auto layer = net->addCast(*output, nvinfer1::DataType::kINT32);
    // net->unmarkOutput(*output);
    // layer->getOutput(0)->setName("output_int32");
    // net->markOutput(*layer->getOutput(0));
  }

  std::unique_ptr<nvinfer1::IBuilderConfig> config(
      builder->createBuilderConfig());
  auto* profile = builder->createOptimizationProfile();
  for (int i = 0; i < net->getNbInputs(); ++i) {
    const auto input_tensor = net->getInput(i);
    const auto name = input_tensor->getName();
    const auto dims = input_tensor->getDimensions();
    // if (!isDynamic(dims)) {
    //   continue;
    // }

    profile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN, dims);
    profile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT, dims);
    profile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX, dims);

    // profile->setDimensions(name,
    //                        OptProfileSelector::kMIN,
    //                        replaceDynamic(dims,
    //                        model_config.min_optimization_size));
    // profile->setDimensions(name,
    //                        OptProfileSelector::kOPT,
    //                        replaceDynamic(dims,
    //                        model_config.target_optimization_size));
    // profile->setDimensions(name,
    //                        OptProfileSelector::kMAX,
    //                        replaceDynamic(dims,
    //                        model_config.max_optimization_size));
    config->addOptimizationProfile(profile);
  }

  std::unique_ptr<IHostMemory> memory(
      builder->buildSerializedNetwork(*net, *config));
  if (!memory) {
    LOG(FATAL) << "Failed to build network for " << model_path;
    return nullptr;
  }

  // will ovewride engine path from YOLO!
  std::ofstream fout(model_config.enginePath(), std::ios::binary);
  fout.write(reinterpret_cast<char*>(memory->data()), memory->size());

  EnginePtr engine(
      runtime.deserializeCudaEngine(memory->data(), memory->size()));
  if (!engine) {
    LOG(FATAL) << "Engine creation failed from " << model_path;
    return nullptr;
  }

  return engine;
}

}  // namespace dyno
