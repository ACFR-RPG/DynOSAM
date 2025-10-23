#pragma once

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <opencv4/opencv2/opencv.hpp>

#include "dynosam_nn/ModelConfig.hpp"
#include "dynosam_nn/ObjectDetector.hpp"
#include "dynosam_nn/TrtUtilities.hpp"

namespace dyno {

class YoloV8ModelInfo {
 public:
  YoloV8ModelInfo() {}
  explicit YoloV8ModelInfo(const nvinfer1::ICudaEngine& engine);

  operator bool() const { return images_ && output0_ && output1_; }
  const ImageTensorInfo& input() const { return images_.value(); }
  const TensorInfo& output0() const { return output0_.value(); }
  const TensorInfo& output1() const { return output1_.value(); }

 private:
  // T is constructable from TensorInfo
  template <typename T>
  inline bool setIfUnset(const TensorInfo& info, std::optional<T>& field) {
    if (!field) {
      field = info;
      return true;
    }
    return false;
  }
  //! Input image: shape(1, 3, 640, 640) DataType.FLOAT where 640,640 is a
  //! static image size
  std::optional<ImageTensorInfo> images_;
  //! First output: shape(1, 116, 8400) DataType.FLOAT
  //! [1, 116, num_detections] where 80 class + 4 bbox parms + 32 seg masks =
  //! 116
  std::optional<TensorInfo> output0_;
  //! Second output shape(1, 32, 160, 160) DataType.FLOAT
  //! [1, 32, maskH, maskW]
  std::optional<TensorInfo> output1_;
};

std::ostream& operator<<(std::ostream& out, const YoloV8ModelInfo& info);

class YoloV8ObjectDetector : public ObjectDetectionEngine, public TRTEngine {
 public:
  YoloV8ObjectDetector(const ModelConfig& config);
  ~YoloV8ObjectDetector();

  ObjectDetectionResult process(const cv::Mat& image) override;
  bool onDestruction() override;
  ObjectDetectionResult result() const override;

 private:
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  YoloV8ModelInfo model_info_;

  DeviceMemory<float> input_device_ptr_;
  DeviceMemory<float> output0_device_ptr_;
  DeviceMemory<float> output1_device_ptr_;
};

}  // namespace dyno
