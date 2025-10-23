#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <boost/python.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "cv_bridge/cv_bridge.hpp"
#include "dynosam_nn/ModelConfig.hpp"
#include "dynosam_nn/PyObjectDetector.hpp"
#include "dynosam_nn/TrtUtilities.hpp"
#include "dynosam_nn/YoloObjectDetector.hpp"
#include "image_transport/image_transport.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/image.hpp"

using namespace nvinfer1;

// Simple logger for TensorRT messages
class Logger : public ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kINFO)  // filter warnings/info
      std::cout << "[TRT] " << msg << std::endl;
  }
};

#ifdef FAKE

// trt_utilities
namespace dyno {

struct TensorInfo {
  std::string name;
  nvinfer1::Dims dims;
  nvinfer1::DataType dtype;

  bool isCHWOrder() const;
  bool isDynamic() const;
  // this only supports CHW form?
  Shape shape() const;
  nvinfer1::Dims replaceDynamic(const cv::Mat& mat) const;

  // does not work if is dynamic!!
  size_t size() const {
    size_t total_size = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
      CHECK_GT(dims.d[i], 0);
      total_size *= static_cast<size_t>(dims.d[i]);
    }
    return total_size;
  }
};

std::ostream& operator<<(std::ostream& out, const TensorInfo& info);

struct DimInfo {
  bool is_image = false;
  bool is_color = false;
  int start = 0;
};

DimInfo getDimInfo(const nvinfer1::Dims& dims) {
  if (dims.nbDims < 3) {
    return {};
  }

  const auto start = dims.nbDims - 3;
  const auto end = dims.nbDims - 1;
  const bool is_color = dims.d[start] == 3 || dims.d[end] == 3;
  return {true, is_color, start};
}

bool areDimsCHWOrder(const nvinfer1::Dims& dims) {
  const auto info = getDimInfo(dims);
  if (!info.is_image) {
    throw std::runtime_error("invalid tensor for image method");
  }

  const int expected_channels = info.is_color ? 3 : 1;
  return dims.d[info.start] == expected_channels;
}

// does not support dims > 3 (that is only support CHW?)
Shape getShapeFromDims(const nvinfer1::Dims& dims) {
  const auto info = getDimInfo(dims);
  if (dims.nbDims < 2) {
    // TODO(nathan) fix
    // SLOG(ERROR) << "invalid tensor: " << *this;
    throw std::runtime_error("unsupported layout!");
  }

  Shape shape;
  if (!info.is_image) {
    shape.height = dims.d[0];
    shape.width = dims.d[1];
    return shape;
  }

  shape.chw_order = areDimsCHWOrder(dims);
  if (shape.chw_order) {
    shape.height = dims.d[info.start + 1];
    shape.width = dims.d[info.start + 2];
    shape.channels = dims.d[info.start];
  } else {
    shape.height = dims.d[info.start];
    shape.width = dims.d[info.start + 1];
    shape.channels = dims.d[info.start + 2];
  }

  return shape;
}

bool TensorInfo::isCHWOrder() const { return areDimsCHWOrder(dims); }

bool TensorInfo::isDynamic() const {
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] == -1) {
      return true;
    }
  }
  return false;
}

nvinfer1::Dims TensorInfo::replaceDynamic(const cv::Mat& mat) const {
  nvinfer1::Dims new_dims = dims;
  const auto info = getDimInfo(dims);
  const auto chw_order = isCHWOrder();
  size_t h_index = chw_order ? info.start + 1 : info.start;
  size_t w_index = chw_order ? info.start + 2 : info.start + 1;
  new_dims.d[h_index] =
      new_dims.d[h_index] < 0 ? mat.rows : new_dims.d[h_index];
  new_dims.d[w_index] =
      new_dims.d[w_index] < 0 ? mat.cols : new_dims.d[w_index];
  return new_dims;
}

Shape TensorInfo::shape() const { return getShapeFromDims(dims); }

std::ostream& operator<<(std::ostream& out, const TensorInfo& info) {
  out << "<name=" << info.name << ", layout=" << toString(info.dims) << " ("
      << toString(info.dtype) << ")>";
  return out;
}

class ModelInfo {
 public:
  ModelInfo() {}

  explicit ModelInfo(const nvinfer1::ICudaEngine& engine) {
    auto num_tensors = engine.getNbIOTensors();
    for (int i = 0; i < num_tensors; ++i) {
      std::string tname(engine.getIOTensorName(i));
      const auto tmode = engine.getTensorIOMode(tname.c_str());
      if (tmode == nvinfer1::TensorIOMode::kNONE) {
        continue;
      }

      LOG(INFO) << tname;

      const auto dims = engine.getTensorShape(tname.c_str());
      const auto dtype = engine.getTensorDataType(tname.c_str());
      TensorInfo info{tname, dims, dtype};

      if (tname == "images") {
        if (tmode == nvinfer1::TensorIOMode::kINPUT) {
          if (!setIfUnset(info, images_)) {
            LOG(ERROR) << "Multiple outputs detected! Rejecting " << tname;
          } else {
            LOG(INFO) << "info " << info << " for images";
          }
        }
      }

      if (tname == "output0") {
        if (tmode == nvinfer1::TensorIOMode::kOUTPUT) {
          if (!setIfUnset(info, output0_)) {
            LOG(ERROR) << "Multiple outputs0's detected! Rejecting " << tname;
          } else {
            LOG(INFO) << "info " << info << " for outputs0";
          }
        }
      }

      if (tname == "output1") {
        if (tmode == nvinfer1::TensorIOMode::kOUTPUT) {
          if (!setIfUnset(info, output1_)) {
            LOG(ERROR) << "Multiple output1's detected! Rejecting " << tname;
          } else {
            LOG(INFO) << "info " << info << " for output1";
          }
        }
      }
    }
  }

  operator bool() const { return images_ && output0_ && output1_; }
  const TensorInfo& input() const { return images_.value(); }
  const TensorInfo& output0() const { return output0_.value(); }
  const TensorInfo& output1() const { return output1_.value(); }

 private:
  bool setIfUnset(const TensorInfo& info, std::optional<TensorInfo>& field) {
    if (!field) {
      field = info;
      return true;
    }

    return false;
  }

  //! Input image: shape(1, 3, 640, 640) DataType.FLOAT where 640,640 is a
  //! static image size
  std::optional<TensorInfo> images_;
  //! First output: shape(1, 116, 8400) DataType.FLOAT
  //! [1, 116, num_detections] where 80 class + 4 bbox parms + 32 seg masks =
  //! 116
  std::optional<TensorInfo> output0_;
  //! Second output shape(1, 32, 160, 160) DataType.FLOAT
  //! [1, 32, maskH, maskW]
  std::optional<TensorInfo> output1_;
};

std::ostream& operator<<(std::ostream& out, const ModelInfo& info) {
  out << "Model: ";
  if (!info) {
    out << "(uninitialized)";
    return out;
  }

  out << "input=" << info.input();

  out << " , output0=" << info.output0();
  out << " , output1=" << info.output1();
  return out;
}

struct Model {
  Model(const ModelConfig& config)
      : runtime_(getRuntime(config.log_severity)),
        engine_(deserializeEngine(*runtime_, config.enginePath())) {
    if (!engine_) {
      LOG(WARNING) << "Failed buildong engine file rebuilding...";
      engine_ = buildEngineFromOnnx(config, *runtime_);
      LOG(INFO) << "Finished building engine";
      CHECK(engine_);
    } else {
      LOG(INFO) << "Loaded engine file";
    }

    // LOG(INFO) << "TensorRT runtime version: "
    //           << getInferLibVersion();
    std::cout << "TensorRT version: " << NV_TENSORRT_MAJOR << "."
              << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH << std::endl;

    if (!engine_) {
      LOG(ERROR) << "Building engine from onnx failed!";
      throw std::runtime_error("failed to load or build engine");
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) {
      LOG(ERROR) << "Failed to create execution context";
      throw std::runtime_error("failed to set up trt context");
    }

    info_ = ModelInfo(*engine_);
    LOG(INFO) << info_;
    if (!info_) {
      LOG(ERROR) << "Invalid engine for segmentation!";
      throw std::runtime_error("invalid model");
    }

    LOG(INFO) << "Execution context started";

    if (cudaStreamCreate(&stream_) != cudaSuccess) {
      LOG(ERROR) << "Creating cuda stream failed!";
      throw std::runtime_error("failed to set up cuda stream");
    } else {
      LOG(INFO) << "CUDA stream started";
    }

    initialized_ = true;
  }

  ~Model() {
    if (initialized_) {
      cudaStreamDestroy(stream_);
    }
  }

  void infer(const cv::Mat& rgb) {
    const TensorInfo& input_info = info_.input();
    const auto required_shape = input_info.shape();
    LOG(INFO) << "w=" << required_shape.width << " h=" << required_shape.height;

    // if (!input_.updateShape(required_shape) || !input_) {
    //   LOG(ERROR) << "Failed to reshape input_!";
    //   return;
    // }
    cv::Size original_size = rgb.size();
    cv::Size required_size(required_shape.width, required_shape.height);

    // // // cv::Mat resized;
    // cv::Mat letter_box;
    // // // cv::resize(rgb, resized, required_size);
    // letterbox(rgb, letter_box, {required_shape.height,
    // required_shape.width});

    // //just use bblobFromImage to scale the image, set output as CV_32F and
    // convert structure to NCHW
    // //image resizing etc is done by letterbox as this is the same as YOLO
    // preprocessing
    // https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-OpenCV-ONNX-Python/main.py
    // just do dnn from blob cv::dnn::blobFromImage(rgb, input_.host_image, 1.0
    // / 255.0, required_size,
    //                        cv::Scalar(), true, false);

    // cv::Mat blob;
    // // // swapRB=true → BGR to RGB (YOLO expects RGB)
    // // // crop=false → keeps aspect ratio
    // cv::Mat resized;
    // cv::resize(rgb, resized, required_size);
    // cv::dnn::blobFromImage(resized, blob, 1.0 / 255.0, required_size,
    //                        cv::Scalar(), true, false);
    // input_.host_image = blob;

    cv::Mat letterboxImage;
    // assume not dynamic
    letterBox(rgb, letterboxImage, required_size, cv::Scalar(114, 114, 114),
              /*auto_=*/false,
              /*scaleFill=*/false, /*scaleUp=*/true, /*stride=*/32);

    letterboxImage.convertTo(letterboxImage, CV_32FC3, 1.0f / 255.0f);

    LOG(INFO) << "Lbox size " << letterboxImage.rows << " "
              << letterboxImage.cols;

    size_t letter_box_size = static_cast<size_t>(letterboxImage.rows) *
                             static_cast<size_t>(letterboxImage.cols) * 3;
    CHECK_EQ(letter_box_size, input_info.size());
    float* blobPtr = new float[letter_box_size];

    std::vector<cv::Mat> channels(3);
    for (int c = 0; c < 3; ++c) {
      channels[c] =
          cv::Mat(letterboxImage.rows, letterboxImage.cols, CV_32FC1,
                  blobPtr + c * (letterboxImage.rows * letterboxImage.cols));
    }
    cv::split(letterboxImage, channels);
    std::vector<float> input_vector(blobPtr, blobPtr + letter_box_size);

    delete[] blobPtr;

    // now allocate gpu memory
    size_t gpu_size = sizeof(float) * letter_box_size;
    input_.device_image.reset(
        reinterpret_cast<float*>(CudaMemoryManager::alloc(gpu_size)));

    // should be 1 x 3 x 640 x 640
    LOG(INFO) << "input info size: " << input_info.size();
    // cv::Mat letter_box;
    // letterbox(rgb, letter_box, {required_shape.height,
    // required_shape.width});

    // letter_box.convertTo(letter_box, CV_32FC3, 1.0f/255.0f);
    // input_.host_image =
    // // Now blob is 1x3x640x640 (NCHW)
    // cv::dnn::blobFromImage(letter_box, input_.host_image);

    // input_.host_image = preprocessImage(
    //   rgb,
    //   required_shape.height,
    //   required_shape.width
    // );

    // LOG(INFO) << "Mat shape: ("
    //       << input_.host_image.rows << ", "        // height
    //       << input_.host_image.cols << ", "        // width
    //       << input_.host_image.channels() << ")";   // number of channels

    // put input into gpu memory at input_.device_image location
    context_->setInputTensorAddress(input_info.name.c_str(),
                                    input_.device_image.get());
    auto error =
        cudaMemcpyAsync(input_.device_image.get(), input_vector.data(),
                        letter_box_size, cudaMemcpyHostToDevice, stream_);
    if (error != cudaSuccess) {
      LOG(ERROR) << "Copying color input failed: " << cudaGetErrorString(error);
      return;
    }

    // set address of output tensors
    const TensorInfo& output0_info = info_.output0();
    // const auto required_output_0shape = output0_info.shape();
    auto required_output_0size = sizeof(float) * output0_info.size();
    output_0_.reset(reinterpret_cast<float*>(
        CudaMemoryManager::alloc(required_output_0size)));
    context_->setTensorAddress(output0_info.name.c_str(), output_0_.get());

    // should be 1 * 116 x 8400
    LOG(INFO) << output0_info.size();

    const TensorInfo& output1_info = info_.output1();
    // const auto required_output_1shape = output1_info.shape();
    auto required_output_1size = sizeof(float) * output1_info.size();
    output_1_.reset(reinterpret_cast<float*>(
        CudaMemoryManager::alloc(required_output_1size)));
    context_->setTensorAddress(output1_info.name.c_str(), output_1_.get());
    // should be 1 * 32 * 160 * 160
    LOG(INFO) << output1_info.size();

    cudaStreamSynchronize(stream_);
    bool status = context_->enqueueV3(stream_);
    if (!status) {
      LOG(ERROR) << "initializing inference failed!";
      return;
    }

    const auto actual_output0_dims =
        context_->getTensorShape(output0_info.name.c_str());
    const auto actual_output1_dims =
        context_->getTensorShape(output1_info.name.c_str());

    if (actual_output1_dims.nbDims != 4 || actual_output1_dims.d[0] != 1 ||
        actual_output1_dims.d[1] != 32)
      throw std::runtime_error(
          "Unexpected output1 shape. Expected [1, 32, maskH, maskW].");

    LOG(INFO) << "actual_output0_dims " << toString(actual_output0_dims);
    LOG(INFO) << "actual_output1_dims " << toString(actual_output1_dims);

    // //get data of GPU
    float output0_data[output0_info.size()];
    float output1_data[output1_info.size()];

    // NOTE: size is in bytes so must include data type!!
    error =
        cudaMemcpyAsync(&output0_data, output_0_.get(), required_output_0size,
                        cudaMemcpyDeviceToHost, stream_);
    if (error != cudaSuccess) {
      LOG(ERROR) << "Copying output0 failed: " << cudaGetErrorString(error);
      return;
    }

    error =
        cudaMemcpyAsync(&output1_data, output_1_.get(), required_output_1size,
                        cudaMemcpyDeviceToHost, stream_);
    if (error != cudaSuccess) {
      LOG(ERROR) << "Copying output1 failed: " << cudaGetErrorString(error);
      return;
    }

    cudaStreamSynchronize(stream_);

    const size_t num_features =
        actual_output0_dims
            .d[1];  // e.g 80 class + 4 bbox parms + 32 seg masks = 116
    const size_t num_detections = actual_output0_dims.d[2];

    LOG(INFO) << "num features" << num_features;

    const int numBoxes = static_cast<int>(num_detections);
    const int maskH = static_cast<int>(actual_output1_dims.d[2]);
    const int maskW = static_cast<int>(actual_output1_dims.d[3]);

    LOG(INFO) << "numBoxes " << numBoxes;

    LOG(INFO) << maskH << " " << maskW;

    const int numClasses =
        static_cast<int>(num_features - 4 - 32);  // Corrected number of classes
    LOG(INFO) << "numClasses " << numClasses;

    // Constants from model architecture
    constexpr int BOX_OFFSET = 0;
    constexpr int CLASS_CONF_OFFSET = 4;
    const int MASK_COEFF_OFFSET = numClasses + CLASS_CONF_OFFSET;

    // 1. Process prototype masks
    // Store all prototype masks in a vector for easy access
    std::vector<cv::Mat> prototypeMasks;
    prototypeMasks.reserve(32);
    for (int m = 0; m < 32; ++m) {
      // Each mask is maskH x maskW
      cv::Mat proto(maskH, maskW, CV_32F,
                    const_cast<float*>(output1_data + m * maskH * maskW));
      prototypeMasks.emplace_back(
          proto.clone());  // Clone to ensure data integrity
    }

    // 2. Process detections
    std::vector<cv::Rect> boxes;
    boxes.reserve(numBoxes);
    std::vector<float> confidences;
    confidences.reserve(numBoxes);
    std::vector<int> classIds;
    classIds.reserve(numBoxes);
    std::vector<std::vector<float>> maskCoefficientsList;
    maskCoefficientsList.reserve(numBoxes);

    float confThreshold = 0.2;

    for (int i = 0; i < numBoxes; ++i) {
      // Extract box coordinates
      float xc = output0_data[BOX_OFFSET * numBoxes + i];
      float yc = output0_data[(BOX_OFFSET + 1) * numBoxes + i];
      float w = output0_data[(BOX_OFFSET + 2) * numBoxes + i];
      float h = output0_data[(BOX_OFFSET + 3) * numBoxes + i];

      // cv::Rect box_cv(
      //     static_cast<int>(std::round(xc - w / 2.0f)),  // top-left x
      //     static_cast<int>(std::round(yc - h / 2.0f)),  // top-left y
      //     static_cast<int>(std::round(w)),             // width
      //     static_cast<int>(std::round(h))              // height
      // );
      cv::Rect box_cv(
          static_cast<int>(std::round(xc - w / 2.0f)),  // top-left x
          static_cast<int>(std::round(yc - h / 2.0f)),  // top-left y
          static_cast<int>(std::round(w)),              // width
          static_cast<int>(std::round(h))               // height
      );

      // Get class confidence
      float maxConf = 0.0f;
      int classId = -1;
      for (int c = 0; c < numClasses; ++c) {
        float conf = output0_data[(CLASS_CONF_OFFSET + c) * numBoxes + i];
        if (conf > maxConf) {
          maxConf = conf;
          classId = c;
        }
      }

      if (maxConf < confThreshold) continue;

      // Store detection
      boxes.push_back(box_cv);
      confidences.push_back(maxConf);
      classIds.push_back(classId);
      // LOG(INFO) << "Found box with id " << classId;

      // Store mask coefficients
      std::vector<float> maskCoeffs(32);
      for (int m = 0; m < 32; ++m) {
        maskCoeffs[m] = output0_data[(MASK_COEFF_OFFSET + m) * numBoxes + i];
      }
      maskCoefficientsList.emplace_back(std::move(maskCoeffs));
    }

    // Early exit if no boxes after confidence threshold
    if (boxes.empty()) {
      return;
    }

    float scoreThreshold = 0.5;  // Minimum confidence score to consider a box
    float nmsThreshold =
        0.4;  // IoU (Intersection over Union) threshold for NMS

    // 3. Apply NMS
    std::vector<int> nmsIndices;
    cv::dnn::NMSBoxes(boxes, confidences, scoreThreshold, nmsThreshold,
                      nmsIndices);
    if (nmsIndices.empty()) {
      return;
    }

    // 4. Prepare final results
    // results.reserve(nmsIndices.size());

    // Calculate letterbox parameters
    const float gain = std::min(
        static_cast<float>(required_size.height) / original_size.height,
        static_cast<float>(required_size.width) / original_size.width);
    const int scaledW = static_cast<int>(original_size.width * gain);
    const int scaledH = static_cast<int>(original_size.height * gain);
    const float padW = (required_size.width - scaledW) / 2.0f;
    const float padH = (required_size.height - scaledH) / 2.0f;

    // Precompute mask scaling factors
    const float maskScaleX = static_cast<float>(maskW) / required_size.width;
    const float maskScaleY = static_cast<float>(maskH) / required_size.height;

    for (const int idx : nmsIndices) {
      // Segmentation seg;
      auto box = boxes[idx];
      auto conf = confidences[idx];
      auto classId = classIds[idx];

      // 5. Scale box to original image
      box = scaleCoords(required_size, box, original_size, true);

      cv::rectangle(rgb, box, cv::Scalar(0, 255, 0),
                    2);  // Green box, 2 px thick

      // 6. Process mask
      const auto& maskCoeffs = maskCoefficientsList[idx];

      // Linear combination of prototype masks
      cv::Mat finalMask = cv::Mat::zeros(maskH, maskW, CV_32F);
      for (int m = 0; m < 32; ++m) {
        // LOG(INFO) << "maskCoeffs " <<  maskCoeffs[m];
        finalMask += maskCoeffs[m] * prototypeMasks[m];
      }

      // Apply sigmoid activation
      finalMask = sigmoid(finalMask);

      // LOG(INFO) << "Final mask" << finalMask.rows << " " << finalMask.cols;

      // // Convert to 8-bit for display
      // cv::Mat mask_rgb;
      // finalMask.convertTo(mask_rgb, CV_8U, 255.0);

      // // Optional: colorize mask
      // cv::applyColorMap(mask_rgb, mask_rgb, cv::COLORMAP_JET);

      // cv::imshow("Masl", mask_rgb);

      // Crop mask to letterbox area with a slight padding to avoid border
      // issues
      int x1 = static_cast<int>(std::round((padW - 0.1f) * maskScaleX));
      int y1 = static_cast<int>(std::round((padH - 0.1f) * maskScaleY));
      int x2 = static_cast<int>(
          std::round((required_size.width - padW + 0.1f) * maskScaleX));
      int y2 = static_cast<int>(
          std::round((required_size.height - padH + 0.1f) * maskScaleY));

      // Ensure coordinates are within mask bounds
      x1 = std::max(0, std::min(x1, maskW - 1));
      y1 = std::max(0, std::min(y1, maskH - 1));
      x2 = std::max(x1, std::min(x2, maskW));
      y2 = std::max(y1, std::min(y2, maskH));

      // Handle cases where cropping might result in zero area
      if (x2 <= x1 || y2 <= y1) {
        // Skip this mask as cropping is invalid
        LOG(INFO) << "Mask invalud?";
        continue;
      }

      cv::Rect cropRect(x1, y1, x2 - x1, y2 - y1);
      cv::Mat croppedMask =
          finalMask(cropRect).clone();  // Clone to ensure data integrity

      // Resize to original dimensions
      cv::Mat resizedMask;
      cv::resize(croppedMask, resizedMask, original_size, 0, 0,
                 cv::INTER_LINEAR);

      // Threshold and convert to binary
      cv::Mat binaryMask;
      cv::threshold(resizedMask, binaryMask, 0.5, 255.0, cv::THRESH_BINARY);
      binaryMask.convertTo(binaryMask, CV_8U);

      // Crop to bounding box
      cv::Mat finalBinaryMask = cv::Mat::zeros(original_size, CV_8U);
      cv::Rect roi(box.x, box.y, box.width, box.height);
      roi &= cv::Rect(0, 0, binaryMask.cols,
                      binaryMask.rows);  // Ensure ROI is within mask
      if (roi.area() > 0) {
        binaryMask(roi).copyTo(finalBinaryMask(roi));
      }

      auto mask = finalBinaryMask;

      if (!mask.empty()) {
        // Ensure the mask is single-channel
        cv::Mat mask_gray;
        if (mask.channels() == 3) {
          cv::cvtColor(mask, mask_gray, cv::COLOR_BGR2GRAY);
        } else {
          mask_gray = mask.clone();
        }

        // Threshold the mask to binary (object: 255, background: 0)
        cv::Mat mask_binary;
        cv::threshold(mask_gray, mask_binary, 127, 255, cv::THRESH_BINARY);

        // Create a colored version of the mask
        cv::Mat colored_mask;
        cv::cvtColor(mask_binary, colored_mask, cv::COLOR_GRAY2BGR);
        colored_mask.setTo(cv::Scalar(255, 0, 0),
                           mask_binary);  // Apply color where mask is present

        // Blend the colored mask with the original image
        cv::addWeighted(rgb, 1.0, colored_mask, 0.5f, 0, rgb);
      }
    }

    // scaleBoxesToOriginal(boxes, required_size, rgb.size());

    // for (const auto& box : boxes) {
    //     cv::rectangle(rgb, box, cv::Scalar(0, 255, 0), 2); // Green box, 2 px
    //     thick
    // }

    cv::imshow("Detectops", rgb);
    cv::waitKey(1);
  }

  float generate_scale(const cv::Mat& image,
                       const std::vector<int>& target_size) {
    int origin_w = image.cols;
    int origin_h = image.rows;

    int target_h = target_size[0];
    int target_w = target_size[1];

    float ratio_h = static_cast<float>(target_h) / static_cast<float>(origin_h);
    float ratio_w = static_cast<float>(target_w) / static_cast<float>(origin_w);
    float resize_scale = std::min(ratio_h, ratio_w);
    return resize_scale;
  }

  inline void letterBox(const cv::Mat& image, cv::Mat& outImage,
                        const cv::Size& newShape,
                        const cv::Scalar& color = cv::Scalar(114, 114, 114),
                        bool auto_ = true, bool scaleFill = false,
                        bool scaleUp = true, int stride = 32) {
    float r = std::min((float)newShape.height / (float)image.rows,
                       (float)newShape.width / (float)image.cols);
    if (!scaleUp) {
      r = std::min(r, 1.0f);
    }

    int newW = static_cast<int>(std::round(image.cols * r));
    int newH = static_cast<int>(std::round(image.rows * r));

    int dw = newShape.width - newW;
    int dh = newShape.height - newH;

    if (auto_) {
      dw = dw % stride;
      dh = dh % stride;
    } else if (scaleFill) {
      newW = newShape.width;
      newH = newShape.height;
      dw = 0;
      dh = 0;
    }

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

    int top = dh / 2;
    int bottom = dh - top;
    int left = dw / 2;
    int right = dw - left;
    cv::copyMakeBorder(resized, outImage, top, bottom, left, right,
                       cv::BORDER_CONSTANT, color);
  }

  cv::Mat sigmoid(const cv::Mat& src) {
    cv::Mat dst;
    cv::exp(-src, dst);
    dst = 1.0 / (1.0 + dst);
    return dst;
  }

  template <typename T>
  T clamp(const T& val, const T& low, const T& high) {
    return std::max(low, std::min(val, high));
  }

  inline cv::Rect scaleCoords(const cv::Size& required_shape,
                              const cv::Rect& coords,
                              const cv::Size& originalShape,
                              bool p_Clip = true) {
    float gain =
        std::min((float)required_shape.height / (float)originalShape.height,
                 (float)required_shape.width / (float)originalShape.width);

    int padW = static_cast<int>(std::round(
        ((float)required_shape.width - (float)originalShape.width * gain) /
        2.f));
    int padH = static_cast<int>(std::round(
        ((float)required_shape.height - (float)originalShape.height * gain) /
        2.f));

    cv::Rect ret;
    ret.x =
        static_cast<int>(std::round(((float)coords.x - (float)padW) / gain));
    ret.y =
        static_cast<int>(std::round(((float)coords.y - (float)padH) / gain));
    ret.width = static_cast<int>(std::round((float)coords.width / gain));
    ret.height = static_cast<int>(std::round((float)coords.height / gain));

    if (p_Clip) {
      ret.x = clamp(ret.x, 0, originalShape.width);
      ret.y = clamp(ret.y, 0, originalShape.height);
      ret.width = clamp(ret.width, 0, originalShape.width - ret.x);
      ret.height = clamp(ret.height, 0, originalShape.height - ret.y);
    }

    return ret;
  }

  //   void scaleBoxesToOriginal(
  //     std::vector<cv::Rect>& boxes,
  //     const cv::Size& resized_size,
  //     const cv::Size& original_size)
  // {
  //     float x_scale = static_cast<float>(original_size.width) /
  //     resized_size.width; float y_scale =
  //     static_cast<float>(original_size.height) / resized_size.height;

  //     for (auto& box : boxes)
  //     {
  //         int new_x = static_cast<int>(std::round(box.x * x_scale));
  //         int new_y = static_cast<int>(std::round(box.y * y_scale));
  //         int new_w = static_cast<int>(std::round(box.width * x_scale));
  //         int new_h = static_cast<int>(std::round(box.height * y_scale));

  //         // Clamp to image bounds
  //         new_x = std::max(0, std::min(new_x, original_size.width - 1));
  //         new_y = std::max(0, std::min(new_y, original_size.height - 1));
  //         if (new_x + new_w > original_size.width)
  //             new_w = original_size.width - new_x;
  //         if (new_y + new_h > original_size.height)
  //             new_h = original_size.height - new_y;

  //         box = cv::Rect(new_x, new_y, new_w, new_h);
  //     }
  // }

  float letterbox(const cv::Mat& input_image, cv::Mat& output_image,
                  const std::vector<int>& target_size) {
    if (input_image.cols == target_size[1] &&
        input_image.rows == target_size[0]) {
      if (input_image.data == output_image.data) {
        return 1.;
      } else {
        output_image = input_image.clone();
        return 1.;
      }
    }

    float resize_scale = generate_scale(input_image, target_size);
    int new_shape_w = std::round(input_image.cols * resize_scale);
    int new_shape_h = std::round(input_image.rows * resize_scale);
    float padw = (target_size[1] - new_shape_w) / 2.;
    float padh = (target_size[0] - new_shape_h) / 2.;

    int top = std::round(padh - 0.1);
    int bottom = std::round(padh + 0.1);
    int left = std::round(padw - 0.1);
    int right = std::round(padw + 0.1);

    cv::resize(input_image, output_image, cv::Size(new_shape_w, new_shape_h), 0,
               0, cv::INTER_AREA);

    cv::copyMakeBorder(output_image, output_image, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114., 114., 114));
    return resize_scale;
  }

  //   //TODO: maybe need letter boxing
  //   //https://github.com/Geekgineer/YOLOs-CPP/blob/main/include/seg/YOLO8Seg.hpp#L339
  //   cv::Mat preprocessImage(const cv::Mat& img, int target_h, int target_w) {
  //     cv::Mat resized;

  //     // 1. Resize to target size
  //     cv::resize(img, resized, cv::Size(target_w, target_h));

  //     // 2. Ensure 3 channels
  //     cv::Mat rgb;
  //     if (resized.channels() == 1) {
  //         cv::cvtColor(resized, rgb, cv::COLOR_GRAY2RGB);
  //     } else if (resized.channels() == 4) {
  //         cv::cvtColor(resized, rgb, cv::COLOR_BGRA2RGB);
  //     } else if (resized.channels() == 3) {
  //         cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB); // OpenCV default is
  //         BGR
  //     } else {
  //         throw std::runtime_error("Unsupported number of channels in input
  //         image");
  //     }

  //     // 3. Convert to float
  //     rgb.convertTo(rgb, CV_32F, 1.0 / 255.0f); // Normalize to [0,1] if
  //     needed

  //     // 4. HWC → CHW
  //     std::vector<cv::Mat> channels(3);
  //     cv::split(rgb, channels);

  //     // std::vector<float> chw_data(3 * target_h * target_w);
  //     // for (int c = 0; c < 3; ++c) {
  //     //     memcpy(chw_data.data() + c * target_h * target_w,
  //     //            channels[c].data,
  //     //            target_h * target_w * sizeof(float));
  //     // }
  //     cv::Mat chw(3, target_h * target_w, CV_32F);
  //     for (int c = 0; c < 3; ++c) {
  //         memcpy(chw.ptr<float>(c), channels[c].data, target_h * target_w *
  //         sizeof(float));
  //     }

  //     // 5. Reshape to (3, H, W)
  //     chw.reshape(1, {3, target_h, target_w});

  //     // 5. Batch dimension of 1 is implicit: TensorRT expects (1,3,640,640)
  //     return chw;
  // }

  bool initialized_ = false;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  cudaStream_t stream_;

  ModelInfo info_;

  ImageMemoryPair input_;
  std::unique_ptr<float, CudaMemoryManager::Delete> output_0_;
  std::unique_ptr<float, CudaMemoryManager::Delete> output_1_;
};

}  // namespace dyno

#endif

class ImageSegmenterNode : public rclcpp::Node {
 public:
  ImageSegmenterNode() : Node("image_subscriber_node") {
    // engine_ = dyno::PyObjectDetectorWrapper::CreateYoloDetector();
    dyno::ModelConfig config;
    config.model_file = "yolov8n-seg.pt";
    engine_ = std::make_unique<dyno::YoloV8ObjectDetector>(config);
    // model_ = std::make_unique<dyno::Model>(config);
    // Use image_transport for efficiency (handles compressed images too)
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/camera/color/image_rect_color", 10,
        std::bind(&ImageSegmenterNode::imageCallback, this,
                  std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(),
                "Image subscriber initialized and listening on "
                "/camera/color/image_rect_color");
  }

 private:
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    try {
      // Convert to OpenCV image (BGR8)
      cv::Mat frame = cv_bridge::toCvShare(msg, "bgr8")->image;

      // if (model_) model_->infer(frame);
      engine_->process(frame);

      // cv::Mat resized;
      // cv::resize(frame, resized, cv::Size(640, 480), 0, 0, cv::INTER_LINEAR);

      // // Print image info
      // RCLCPP_INFO(this->get_logger(), "Received image %dx%d", resized.cols,
      //             resized.rows);
      // auto r = engine_->process(resized);

      // // LOG(INFO) << r;

      // // // // Optional: visualize (disable in headless mode)
      // cv::imshow("View", r.colouredMask());
      // cv::waitKey(1);

    } catch (const cv_bridge::Exception& e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  dyno::ObjectDetectionEngine::UniquePtr engine_;
  // std::unique_ptr<dyno::Model> model_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  FLAGS_log_prefix = 1;

  // const std::string enginePath = dyno::getNNWeightsPath() /
  // "yolov8n-seg.engine";

  // dyno::ModelConfig config;
  // config.model_file = "yolov8n-seg.pt";
  // dyno::Model model(config);

  // Py_Initialize();
  // {
  auto node = std::make_shared<ImageSegmenterNode>();
  //   FLAGS_logtostderr = 1;
  //   FLAGS_colorlogtostderr = 1;
  //   FLAGS_log_prefix = 1;

  rclcpp::spin(node);
  // }
  rclcpp::shutdown();
  // Finalize the Python interpreter.
  // Py_FinalizeEx();
  return 0;
}
