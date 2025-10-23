#include "dynosam_nn/YoloObjectDetector.hpp"

namespace dyno {

struct YoloV8ObjectDetector::Impl {
  bool preprocess(const ImageTensorInfo& input_info, const cv::Mat& rgb,
                  std::vector<float>& processed_vector) {
    const cv::Size required_size = input_info.shape();
    LOG(INFO) << "w=" << required_size.width << " h=" << required_size.height;

    cv::Size original_size = rgb.size();

    // preprocess image according to YOLO training pre-procesing
    cv::Mat letterboxImage;
    // assume not dynamic
    letterBox(rgb, letterboxImage, required_size, cv::Scalar(114, 114, 114),
              /*auto_=*/false,
              /*scaleFill=*/false, /*scaleUp=*/true, /*stride=*/32);
    letterboxImage.convertTo(letterboxImage, CV_32FC3, 1.0f / 255.0f);

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
          "Unexpected output1 shape. Expected [1, 32, maskH, maskW].");

    LOG(INFO) << "actual_output0_dims " << toString(output0_dims);
    LOG(INFO) << "actual_output1_dims " << toString(output1_dims);

    const cv::Size required_size = input_info.shape();
    const cv::Size original_size = rgb.size();

    const float* output0_data = output0.data();
    const float* output1_data = output1.data();

    const size_t num_features =
        output0_dims.d[1];  // e.g 80 class + 4 bbox parms + 32 seg masks = 116
    const size_t num_detections = output0_dims.d[2];

    LOG(INFO) << "num features" << num_features;

    const int numBoxes = static_cast<int>(num_detections);
    const int maskH = static_cast<int>(output1_dims.d[2]);
    const int maskW = static_cast<int>(output1_dims.d[3]);

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
      return false;
    }

    float scoreThreshold = 0.5;  // Minimum confidence score to consider a box
    float nmsThreshold =
        0.4;  // IoU (Intersection over Union) threshold for NMS

    // 3. Apply NMS
    std::vector<int> nmsIndices;
    cv::dnn::NMSBoxes(boxes, confidences, scoreThreshold, nmsThreshold,
                      nmsIndices);
    if (nmsIndices.empty()) {
      return false;
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
};

YoloV8ObjectDetector::YoloV8ObjectDetector(const ModelConfig& config)
    : ObjectDetectionEngine(), TRTEngine(config) {
  model_info_ = YoloV8ModelInfo(*engine_);
  LOG(INFO) << model_info_;
  if (!model_info_) {
    LOG(ERROR) << "Invalid engine for segmentation!";
    throw std::runtime_error("invalid model");
  }
  impl_ = std::make_unique<Impl>();
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

  return result;
}
bool YoloV8ObjectDetector::onDestruction() {}
ObjectDetectionResult YoloV8ObjectDetector::result() const {}

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
