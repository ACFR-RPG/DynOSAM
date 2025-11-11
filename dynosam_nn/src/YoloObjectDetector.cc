#include "dynosam_nn/YoloObjectDetector.hpp"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "dynosam_common/viz/Colour.hpp"
#include "dynosam_nn/trackers/ObjectTracker.hpp"

namespace dyno {

/**
 * @brief Much of this code is developed from:
 * https://github.dev/Geekgineer/YOLOs-CPP/blob/main/include/seg/YOLO8Seg.hpp
 *
 */

struct YoloV8ObjectDetector::Impl {
  Impl(const YoloConfig& yolo_config)
      : yolo_config_(yolo_config),
        tracker_(std::make_unique<ByteObjectTracker>()) {
    // load
    const std::filesystem::path file_names_resouce =
        ModelConfig::getResouce("coco.names");
    // set up a mapping of class ids (provided by the detection) and the class
    // labels (from the resource) only included class from the yolo config will
    // be included, making it easy to check which class ids we want to track
    setIncludedClassMapping(file_names_resouce, yolo_config);
  }

  bool preprocess(const ImageTensorInfo& input_info, const cv::Mat& rgb,
                  std::vector<float>& processed_vector) {
    const cv::Size required_size = input_info.shape();

    // preprocess image according to YOLO training pre-procesing
    cv::Mat letterbox_image;
    // assume not dynamic
    letterBox(rgb, letterbox_image, required_size, cv::Scalar(114, 114, 114),
              /*auto_=*/false,
              /*scaleFill=*/false, /*scaleUp=*/true, /*stride=*/32);
    letterbox_image.convertTo(letterbox_image, CV_32FC3, 1.0f / 255.0f);

    size_t letter_box_size = static_cast<size_t>(letterbox_image.rows) *
                             static_cast<size_t>(letterbox_image.cols) * 3;
    CHECK_EQ(letter_box_size, input_info.size());
    float* blobPtr = new float[letter_box_size];

    std::vector<cv::Mat> channels(3);
    for (int c = 0; c < 3; ++c) {
      channels[c] =
          cv::Mat(letterbox_image.rows, letterbox_image.cols, CV_32FC1,
                  blobPtr + c * (letterbox_image.rows * letterbox_image.cols));
    }
    cv::split(letterbox_image, channels);
    std::vector<float> input_vector(blobPtr, blobPtr + letter_box_size);

    delete[] blobPtr;
    processed_vector = input_vector;
    return true;
  }

  bool postprocess(const ImageTensorInfo& input_info, const cv::Mat& rgb,
                   const std::vector<float>& output0,
                   const std::vector<float>& output1,
                   const nvinfer1::Dims& output0_dims,
                   const nvinfer1::Dims& output1_dims,
                   ObjectDetectionResult& result) {
    if (output1_dims.nbDims != 4 || output1_dims.d[0] != 1 ||
        output1_dims.d[1] != 32)
      throw std::runtime_error(
          "Unexpected output1 shape. Expected [1, 32, mask_h, mask_w].");

    const cv::Size required_size = input_info.shape();
    const cv::Size original_size = rgb.size();

    const float* output0_data = output0.data();
    const float* output1_data = output1.data();

    const size_t num_features =
        output0_dims.d[1];  // e.g 80 class + 4 bbox parms + 32 seg masks = 116
    const size_t num_detections = output0_dims.d[2];

    const int num_boxes = static_cast<int>(num_detections);
    const int mask_h = static_cast<int>(output1_dims.d[2]);
    const int mask_w = static_cast<int>(output1_dims.d[3]);

    const int num_classes =
        static_cast<int>(num_features - 4 - 32);  // Corrected number of classes

    // Constants from model architecture
    constexpr int BoxOffset = YoloV8ModelInfo::Constants::BoxOffset;
    constexpr int ClassConfOffset = YoloV8ModelInfo::Constants::ClassConfOffset;
    const int MaskCoeffOffset =
        YoloV8ModelInfo::Constants::MaskCoeffOffset(num_classes);

    // 1. Process prototype masks
    // Store all prototype masks in a vector for easy access
    std::vector<cv::Mat> prototypeMasks;
    prototypeMasks.reserve(32);
    for (int m = 0; m < 32; ++m) {
      // Each mask is mask_h x mask_w
      cv::Mat proto(mask_h, mask_w, CV_32F,
                    const_cast<float*>(output1_data + m * mask_h * mask_w));
      prototypeMasks.emplace_back(
          proto.clone());  // Clone to ensure data integrity
    }

    // 2. Process detections
    std::vector<cv::Rect> boxes;
    boxes.reserve(num_boxes);
    std::vector<float> confidences;
    confidences.reserve(num_boxes);
    std::vector<int> class_ids;
    class_ids.reserve(num_boxes);
    std::vector<std::vector<float>> mask_coefficients;
    mask_coefficients.reserve(num_boxes);

    for (int i = 0; i < num_boxes; ++i) {
      // Extract box coordinates
      float xc = output0_data[BoxOffset * num_boxes + i];
      float yc = output0_data[(BoxOffset + 1) * num_boxes + i];
      float w = output0_data[(BoxOffset + 2) * num_boxes + i];
      float h = output0_data[(BoxOffset + 3) * num_boxes + i];

      cv::Rect box_cv(
          static_cast<int>(std::round(xc - w / 2.0f)),  // top-left x
          static_cast<int>(std::round(yc - h / 2.0f)),  // top-left y
          static_cast<int>(std::round(w)),              // width
          static_cast<int>(std::round(h))               // height
      );

      // Get class confidence
      float maxConf = 0.0f;
      int classId = -1;
      for (int c = 0; c < num_classes; ++c) {
        float conf = output0_data[(ClassConfOffset + c) * num_boxes + i];
        if (conf > maxConf) {
          maxConf = conf;
          classId = c;
        }
      }

      if (maxConf < yolo_config_.conf_threshold) continue;

      // Store detection
      boxes.push_back(box_cv);
      confidences.push_back(maxConf);
      class_ids.push_back(classId);
      // LOG(INFO) << "Found box with id " << classId;

      // Store mask coefficients
      std::vector<float> mask_coeffs(32);
      for (int m = 0; m < 32; ++m) {
        mask_coeffs[m] = output0_data[(MaskCoeffOffset + m) * num_boxes + i];
      }
      mask_coefficients.emplace_back(std::move(mask_coeffs));
    }

    // Early exit if no boxes after confidence threshold
    if (boxes.empty()) {
      return false;
    }

    // 3. Apply NMS
    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes, confidences, yolo_config_.conf_threshold,
                      yolo_config_.nms_threshold, nms_indices);
    if (nms_indices.empty()) {
      return false;
    }

    // Calculate letterbox parameters
    const float gain = std::min(
        static_cast<float>(required_size.height) / original_size.height,
        static_cast<float>(required_size.width) / original_size.width);
    const int scaled_w = static_cast<int>(original_size.width * gain);
    const int scaled_h = static_cast<int>(original_size.height * gain);
    const float pad_w = (required_size.width - scaled_w) / 2.0f;
    const float pad_h = (required_size.height - scaled_h) / 2.0f;

    // Precompute mask scaling factors
    const float mask_scale_x = static_cast<float>(mask_w) / required_size.width;
    const float mask_scale_y =
        static_cast<float>(mask_h) / required_size.height;

    std::vector<ObjectDetection> detections;
    std::vector<cv::Mat> binary_detection_masks;
    detections.reserve(nms_indices.size());
    binary_detection_masks.reserve(nms_indices.size());

    for (const int idx : nms_indices) {
      const float confidence = confidences[idx];
      const int class_id = class_ids[idx];

      // skip object if not in the set of included class labels
      std::string class_label;
      if (!safeGetClassLabel(class_id, class_label)) {
        continue;
      }

      // 5. Scale box to original image
      const cv::Rect bounding_box =
          scaleCoords(required_size, boxes[idx], original_size, true);

      // 6. Process mask
      const auto& mask_coeffs = mask_coefficients[idx];

      // Linear combination of prototype masks
      cv::Mat final_mask = cv::Mat::zeros(mask_h, mask_w, CV_32F);
      for (int m = 0; m < 32; ++m) {
        final_mask += mask_coeffs[m] * prototypeMasks[m];
      }
      // Apply sigmoid activation
      final_mask = sigmoid(final_mask);

      // Crop mask to letterbox area with a slight padding to avoid border
      // issues
      int x1 = static_cast<int>(std::round((pad_w - 0.1f) * mask_scale_x));
      int y1 = static_cast<int>(std::round((pad_h - 0.1f) * mask_scale_y));
      int x2 = static_cast<int>(
          std::round((required_size.width - pad_w + 0.1f) * mask_scale_x));
      int y2 = static_cast<int>(
          std::round((required_size.height - pad_h + 0.1f) * mask_scale_y));

      // Ensure coordinates are within mask bounds
      x1 = std::max(0, std::min(x1, mask_w - 1));
      y1 = std::max(0, std::min(y1, mask_h - 1));
      x2 = std::max(x1, std::min(x2, mask_w));
      y2 = std::max(y1, std::min(y2, mask_h));

      // Handle cases where cropping might result in zero area
      if (x2 <= x1 || y2 <= y1) {
        // Skip this mask as cropping is invalid
        LOG(INFO) << "Mask invalud?";
        continue;
      }

      cv::Rect cropRect(x1, y1, x2 - x1, y2 - y1);
      cv::Mat croppedMask =
          final_mask(cropRect).clone();  // Clone to ensure data integrity

      // Resize to original dimensions
      cv::Mat resized_mask;
      cv::resize(croppedMask, resized_mask, original_size, 0, 0,
                 cv::INTER_LINEAR);

      // Threshold and convert to binary
      cv::Mat binary_mask;
      cv::threshold(resized_mask, binary_mask, 0.5, 255.0, cv::THRESH_BINARY);
      binary_mask.convertTo(binary_mask, CV_8U);

      // Crop to bounding box
      cv::Mat final_binary_mask = cv::Mat::zeros(original_size, CV_8U);
      cv::Rect roi = bounding_box;
      roi &= cv::Rect(0, 0, binary_mask.cols,
                      binary_mask.rows);  // Ensure ROI is within mask
      if (roi.area() > 0) {
        binary_mask(roi).copyTo(final_binary_mask(roi));
      }

      binary_detection_masks.push_back(final_binary_mask);

      ObjectDetection detection{final_binary_mask, bounding_box, class_label,
                                confidence};
      detections.push_back(detection);
    }

    cv::Mat labelled_mask =
        cv::Mat::zeros(original_size, ObjectDetectionEngine::MaskDType);

    std::vector<SingleDetectionResult> tracking_result =
        tracker_->track(detections);

    // //construct label mask from tracked result
    for (size_t i = 0; i < tracking_result.size(); i++) {
      const SingleDetectionResult& single_result = tracking_result.at(i);

      // this may happen if the object was not well tracked
      if (!single_result.isValid()) {
        continue;
      }

      cv::Mat single_label_mask =
          cv::Mat::zeros(original_size, ObjectDetectionEngine::MaskDType);
      single_label_mask.setTo(single_result.object_id, single_result.mask);
      // set pixel values to object label and update full labelled mask
      // cv::Mat binary_mask = single_result.mask * single_result.object_id;
      labelled_mask += single_label_mask;
    }

    result.detections = tracking_result;
    result.labelled_mask = labelled_mask;
    result.input_image = rgb;

    return false;
  }

  inline cv::Mat sigmoid(const cv::Mat& src) {
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

    int pad_w = static_cast<int>(std::round(
        ((float)required_shape.width - (float)originalShape.width * gain) /
        2.f));
    int pad_h = static_cast<int>(std::round(
        ((float)required_shape.height - (float)originalShape.height * gain) /
        2.f));

    cv::Rect ret;
    ret.x =
        static_cast<int>(std::round(((float)coords.x - (float)pad_w) / gain));
    ret.y =
        static_cast<int>(std::round(((float)coords.y - (float)pad_h) / gain));
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

  void letterBox(const cv::Mat& image, cv::Mat& outImage,
                 const cv::Size& newShape,
                 const cv::Scalar& color = cv::Scalar(114, 114, 114),
                 bool auto_ = true, bool scaleFill = false, bool scaleUp = true,
                 int stride = 32) {
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

  bool safeGetClassLabel(int class_id, std::string& label) {
    if (included_class_names_.find(class_id) != included_class_names_.end()) {
      label = included_class_names_.at(class_id);
      return true;
    } else {
      label = kUnknownClassLabel;
      return false;
    }
  }

  void setIncludedClassMapping(const std::filesystem::path& file_path,
                               const YoloConfig& yolo_config) {
    std::vector<std::string> class_names;
    VLOG(1) << "Attempting to load YOLO file names from path";
    class_names.clear();

    std::ifstream file(file_path);
    if (!file) {
      // TODO: maybe should be fatal
      LOG(ERROR) << "Could not open class names file: " << file_path;
      return;
    }
    std::string line;
    while (std::getline(file, line)) {
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      class_names.push_back(line);
    }
    LOG(INFO) << "Loaded " << class_names.size()
              << " YOLO class names from path: " << file_path;

    const std::vector<std::string>& included_classes =
        yolo_config.included_classes;
    included_class_names_.clear();
    // if empty create a 1-to-1 mapping for all objects
    if (included_classes.empty()) {
      LOG(INFO) << "No included classes specified with the given config. All "
                   "detected object classes are valid!";
      for (size_t i = 0; i < class_names.size(); i++) {
        included_class_names_.insert({static_cast<int>(i), class_names.at(i)});
      }
    } else {
      for (const auto& included_class : included_classes) {
        auto it =
            std::find(class_names.begin(), class_names.end(), included_class);
        if (it != class_names.end()) {
          int index = static_cast<int>(std::distance(class_names.begin(), it));
          LOG(INFO) << "Found mapping for included class " << included_class
                    << " -> " << index;
          included_class_names_.insert({index, included_class});
        } else {
          LOG(WARNING) << "Requested included class " << included_class
                       << " is not a known class";
        }
      }
    }
  }

  YoloConfig yolo_config_;
  ObjectTracker::UniquePtr tracker_;
  //! Mapping of class ids (as provided by the detection) to class labels (i.e 0
  //! -> "person") but only for the included classes as requested by the
  //! YoloConfig
  std::unordered_map<int, std::string> included_class_names_;

  static constexpr char kUnknownClassLabel[] = "unknown";
};

YoloV8ObjectDetector::YoloV8ObjectDetector(const ModelConfig& config,
                                           const YoloConfig& yolo_config)
    : ObjectDetectionEngine(), TRTEngine(config) {
  model_info_ = YoloV8ModelInfo(*engine_);
  LOG(INFO) << model_info_;
  if (!model_info_) {
    LOG(ERROR) << "Invalid engine for segmentation!";
    throw std::runtime_error("invalid model");
  }
  impl_ = std::make_unique<Impl>(yolo_config);
}

YoloV8ObjectDetector::~YoloV8ObjectDetector() = default;

ObjectDetectionResult YoloV8ObjectDetector::process(const cv::Mat& image) {
  const auto& input_info = model_info_.input();
  const auto& output0_info = model_info_.output0();
  const auto& output1_info = model_info_.output1();

  std::vector<float> input_data;
  impl_->preprocess(input_info, image, input_data);
  // allocate input data
  CHECK(input_device_ptr_.allocate(input_info));
  CHECK_EQ(input_device_ptr_.tensor_size, input_data.size());

  context_->setInputTensorAddress(input_info.name.c_str(),
                                  input_device_ptr_.device_pointer.get());
  // put image data onto gpu
  CHECK(input_device_ptr_.pushFromHost(input_data, stream_));

  // prepare output data
  CHECK(output0_device_ptr_.allocate(output0_info));
  CHECK(output1_device_ptr_.allocate(output1_info));

  // set output address tensors
  context_->setTensorAddress(output0_info.name.c_str(),
                             output0_device_ptr_.device_pointer.get());

  context_->setTensorAddress(output1_info.name.c_str(),
                             output1_device_ptr_.device_pointer.get());

  cudaStreamSynchronize(stream_);
  bool status = context_->enqueueV3(stream_);
  if (!status) {
    LOG(ERROR) << "initializing inference failed!";
    return ObjectDetectionResult{};
  }

  std::vector<float> output0_data, output1_data;
  CHECK(output0_device_ptr_.getFromDevice(output0_data, stream_));
  CHECK(output1_device_ptr_.getFromDevice(output1_data, stream_));

  cudaStreamSynchronize(stream_);

  const auto output0_dims = context_->getTensorShape(output0_info.name.c_str());
  const auto output1_dims = context_->getTensorShape(output1_info.name.c_str());

  ObjectDetectionResult result;
  impl_->postprocess(input_info, image, output0_data, output1_data,
                     output0_dims, output1_dims, result);

  result_ = result;
  return result_;
}

ObjectDetectionResult YoloV8ObjectDetector::result() const { return result_; }

YoloV8ModelInfo::YoloV8ModelInfo(const nvinfer1::ICudaEngine& engine) {
  auto num_tensors = engine.getNbIOTensors();
  for (int i = 0; i < num_tensors; ++i) {
    std::string tname(engine.getIOTensorName(i));
    const auto tmode = engine.getTensorIOMode(tname.c_str());
    if (tmode == nvinfer1::TensorIOMode::kNONE) {
      continue;
    }

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

std::ostream& operator<<(std::ostream& out, const YoloV8ModelInfo& info) {
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

}  // namespace dyno
