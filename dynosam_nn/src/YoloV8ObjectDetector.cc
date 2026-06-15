#include "dynosam_nn/YoloV8ObjectDetector.hpp"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "dynosam_common/utils/OpenCVUtils.hpp"
#include "dynosam_common/utils/TimingStats.hpp"
#include "dynosam_common/viz/Colour.hpp"
#include "dynosam_nn/CudaUtils.hpp"
#include "dynosam_nn/YoloV8CudaUtils.hpp"
#include "dynosam_nn/trackers/ObjectTracker.hpp"

namespace dyno {

namespace internal {

YoloDetectionGpuMatDevice::YoloDetectionGpuMatDevice(
    AlignedYoloDetection* d_detection) {
  constexpr static size_t struct_pitch_bytes =
      sizeof(AlignedYoloDetection);  // 152 bytes
  float* d_detections_ptr = reinterpret_cast<float*>(d_detection);

  // The column type is always CV_32FC1 (float) for all wrappers.

  // -------------------------------------------------------------------------
  // 1. Boxes (X, Y, W, H) - 4 columns
  // -------------------------------------------------------------------------
  // Start address: d_detection (offset 0)
  boxes = cv::cuda::GpuMat(
      1,                  // rows
      4,                  // cols
      CV_32FC1,           // type (float, 1 channel)
      d_detections_ptr,   // data pointer (points to the 'x' field)
      struct_pitch_bytes  // step (stride from row to row)
  );

  // -------------------------------------------------------------------------
  // 2. Scores and Class IDs - 2 columns
  // -------------------------------------------------------------------------
  // Start address: &d_detection[0].confidence
  // Offset in floats: 4 (x, y, w, h)
  // Offset in bytes: 4 * sizeof(float) = 16 bytes
  float* d_scores_ptr = reinterpret_cast<float*>(d_detection) + 4;
  scores_and_classes = cv::cuda::GpuMat(
      1,                  // rows
      2,                  // cols
      CV_32FC1,           // type
      d_scores_ptr,       // data pointer (points to 'confidence' field)
      struct_pitch_bytes  // step
  );

  // -------------------------------------------------------------------------
  // 3. Mask Coefficients - 32 columns
  // -------------------------------------------------------------------------
  // Start address: &d_detections[0].mask
  // Offset in floats: 6 (4 box + 1 confidence + 1 class_id)
  // Offset in bytes: 6 * sizeof(float) = 24 bytes
  float* d_masks_ptr = reinterpret_cast<float*>(d_detection) + 6;
  mask_coeffs =
      cv::cuda::GpuMat(1,            // rows
                       32,           // cols
                       CV_32FC1,     // type
                       d_masks_ptr,  // data pointer (points to 'mask[0]' field)
                       struct_pitch_bytes  // step
      );
}

}  // namespace internal

/**
 * @brief Much of this code is developed from:
 * https://github.dev/Geekgineer/YOLOs-CPP/blob/main/include/seg/YOLO8Seg.hpp
 *
 */

using namespace internal;

inline void getScalePad(const cv::Size& originalSize,
                        const cv::Size& letterboxSize, float& scale,
                        float& padX, float& padY) {
  scale =
      std::min(static_cast<float>(letterboxSize.height) / originalSize.height,
               static_cast<float>(letterboxSize.width) / originalSize.width);

  // Use round() for new dimensions (matches Ultralytics)
  int newW = static_cast<int>(std::round(originalSize.width * scale));
  int newH = static_cast<int>(std::round(originalSize.height * scale));

  // For descaling, use UNROUNDED padding values (matches Ultralytics behavior)
  padX = (letterboxSize.width - newW) / 2.0f;
  padY = (letterboxSize.height - newH) / 2.0f;
}

struct YoloV8ObjectDetector::Impl {
  const YoloConfig yolo_config_;
  // Information about the tensor to be provided as input directly to the model
  const ImageTensorInfo input_info_;

  bool is_first{true};
  ObjectTracker::UniquePtr tracker_;
  // // Device (GPU) detection buffer
  // YoloDetection* d_buffer_;
  // // Host (CPU) detection buffer
  // YoloDetection* h_buffer_;
  // // Device (GPU) detection count
  // int* d_counter_;

  // Device (GPU) detection buffer
  AlignedYoloDetection* d_indir_buffer_;
  // Host (CPU) detection buffer
  AlignedYoloDetection* h_indir_buffer_;
  // Device (GPU) detection count
  int* d_indir_counter_;
  int* h_indir_counter_;

  // Stores the input image on the GPU prior to pre-processing kernel
  // Avoids constant reallocation
  DeviceMemory<uchar> input_image_device_;

  //! Size of the input image prior to preprocessing (ie. size of the camera
  //! image)
  cv::Size original_size_;
  cv::Size letter_box_size_;

  CudaStreamPool stream_pool_;
  //! Mapping of class ids (as provided by the detection) to class labels (i.e 0
  //! -> "person") but only for the included classes as requested by the
  //! YoloConfig
  std::unordered_map<int, std::string> included_class_names_;

  static constexpr char kUnknownClassLabel[] = "unknown";

  Impl(const YoloConfig& yolo_config, const ImageTensorInfo& input_info)
      : yolo_config_(yolo_config),
        input_info_(input_info),
        is_first{true},
        tracker_(std::make_unique<ByteObjectTracker>()) {
    // load
    const std::filesystem::path file_names_resouce =
        ModelConfig::getResouce("coco.names");
    // set up a mapping of class ids (provided by the detection) and the class
    // labels (from the resource) only included class from the yolo config will
    // be included, making it easy to check which class ids we want to track
    setIncludedClassMapping(file_names_resouce, yolo_config);

    const size_t det_size = YoloV8ModelInfo::Constants::MaxDetections *
                            sizeof(AlignedYoloDetection);
    const size_t count_size = sizeof(int);

    // cudaMalloc(&d_buffer_, det_size);
    cudaHostAlloc(&h_indir_buffer_, det_size, cudaHostAllocMapped);
    cudaHostAlloc(&h_indir_counter_, count_size, cudaHostAllocMapped);

    // cudaMallocHost(&h_buffer_, det_size);
    cudaHostGetDevicePointer(&d_indir_buffer_, h_indir_buffer_, 0);
    cudaHostGetDevicePointer(&d_indir_counter_, h_indir_counter_, 0);

    *h_indir_counter_ = 0;
  }

  ~Impl() {
    if (h_indir_buffer_) {
      cudaFreeHost(h_indir_buffer_);
    }

    if (h_indir_counter_) {
      cudaFreeHost(h_indir_counter_);
    }
  }

  // input to the model
  inline const cv::Size requiredInputSize() const {
    return input_info_.shape();
  }

  const cv::Size& originalSize(const cv::Mat& rgb) {
    //! is the first call OR
    //! if dynamic input dimensions, update original size at each pre-processing
    if (is_first || !yolo_config_.assume_static_input_dimensions) {
      original_size_ = rgb.size();
    } else {
      CHECK(utils::cvSizeEqual(original_size_, rgb.size()));
    }
    return original_size_;
  }

  bool preprocess(const ImageTensorInfo& input_info, const cv::Mat& rgb,
                  HostMemory<float>& input_vector) {
    // set input image size
    originalSize(rgb);

    cv::Size actual_size;
    // Assuming target size logic is pre-determined or based on a dry run if
    // dynamic
    cv::Size target_size = requiredInputSize();
    // input_vector.allocate(input_info);
    // float* input_data = input_vector.get(); // Pre-allocated memory pointer
    // (Unified/Device memory)

    // Pass a CUDA stream if your framework provides one to prevent CPU blocking
    // cudaStream_t stream = 0;

    // Run entirely on GPU
    // letterBoxToBlobGPU(rgb, input_data, 3, target_size, actual_size, false,
    // stream);

    // letter_box_size_ = actual_size;
    std::vector<float> blob;
    letterBoxToBlob(rgb, blob, 3, requiredInputSize(), actual_size);
    letter_box_size_ = actual_size;

    // CHECK_EQ(blob.size(), input_info.size());

    // // float* blobPtr = new float[letter_box_size];
    input_vector.allocate(input_info);
    float* input_data = input_vector.get();

    // //copy data to GPU
    std::copy(blob.begin(), blob.end(), input_data);

    is_first = false;
    return true;
  }

  bool postprocess(const cv::Mat& rgb, const float* d_output0,
                   const float* d_output1, const nvinfer1::Dims& output0_dims,
                   const nvinfer1::Dims& output1_dims,
                   ObjectDetectionResult& result) {
    utils::ChronoTimingStats timing_all("yolov8_detection.post_process.run", 5);

    if (output1_dims.nbDims != 4 || output1_dims.d[0] != 1 ||
        output1_dims.d[1] != 32)
      throw std::runtime_error(
          "Unexpected output1 shape. Expected [1, 32, mask_h, mask_w].");

    const cv::Size& original_size = originalSize(rgb);

    utils::ChronoTimingStats timing_setup("yolov8_detection.post_process.setup",
                                          5);

    // result result regardless
    result.labelled_mask =
        cv::Mat::zeros(original_size, ObjectDetectionEngine::MaskDType);
    result.input_image = rgb;

    // const float* output0_data = output0;
    // const float* output1_data = output1;

    const size_t numChannels =
        output0_dims.d[1];  // e.g 80 class + 4 bbox parms + 32 seg masks = 116
    // const size_t numAnchors = output0_dims.d[2];

    // const int num_boxes = static_cast<int>(num_detections);
    const int SEG_H = static_cast<int>(output1_dims.d[2]);
    const int SEG_W = static_cast<int>(output1_dims.d[3]);

    const int numClasses =
        static_cast<int>(numChannels - 32 - 4);  // Corrected number of classes

    const int MaskCoeffOffset =
        YoloV8ModelInfo::Constants::MaskCoeffOffset(numClasses);

    const float* d_output1_void = d_output1;
    std::vector<cv::Mat> prototypeMasks;
    prototypeMasks.reserve(32);

    for (int m = 0; m < 32; ++m) {
      cv::cuda::GpuMat proto_d(
          SEG_H, SEG_W, CV_32F,
          const_cast<float*>(d_output1_void + m * SEG_H * SEG_W));

      cv::Mat proto_h;
      proto_d.download(proto_h);
      prototypeMasks.emplace_back(proto_h.clone());
    }

    timing_setup.stop();

    YoloKernelConfig config;
    config.num_classes = numClasses;
    config.mask_coeff_offset = MaskCoeffOffset;
    config.conf_threshold = yolo_config_.conf_threshold;

    utils::ChronoTimingStats timing_detection(
        "yolov8_detection.post_process.compute_detections", 5);

    int count = internal::YoloOutputToDetections(
        d_output0,  // The raw GPU pointer from TensorRT/ONNX
        config,
        h_indir_buffer_,   // Destination on CPU
        d_indir_buffer_,   // Temp storage on GPU
        d_indir_counter_,  // Temp counter on GPU,
        h_indir_counter_, stream_pool_.getCudaStream());

    timing_detection.stop();

    std::vector<int> labels;
    std::vector<float> scores;
    std::vector<cv::Rect> bboxes;
    std::vector<cv::Mat> maskConfs;
    std::vector<int> indices;

    for (int i = 0; i < count; ++i) {
      // 1. Get a reference to the detection data in the contiguous array
      const AlignedYoloDetection& det = h_indir_buffer_[i];

      float score = det.confidence;

      if (score < yolo_config_.conf_threshold) {
        continue;
      }

      // class id which maps to the class name
      const int label = static_cast<int>(det.class_id);

      float* mask = const_cast<float*>(det.mask);
      cv::Mat maskConf = cv::Mat(1, 32, CV_32F, mask);

      bboxes.push_back(det.toCvRect());
      labels.push_back(label);
      scores.push_back(score);
      maskConfs.push_back(maskConf);
    }

    // // Early exit if no boxes after confidence threshold
    if (bboxes.empty()) {
      return false;
    }

    cv::dnn::NMSBoxesBatched(bboxes, scores, labels,
                             yolo_config_.conf_threshold,
                             yolo_config_.nms_threshold, indices);

    float gain, padW, padH;
    getScalePad(original_size, letter_box_size_, gain, padW, padH);
    const float invGain = 1.0f / gain;

    const float maskScaleX = static_cast<float>(SEG_W) / letter_box_size_.width;
    const float maskScaleY =
        static_cast<float>(SEG_H) / letter_box_size_.height;

    std::vector<ObjectDetection> detections;
    detections.reserve(indices.size());
    for (int idx : indices) {
      // NOW descale box coordinates from letterbox to original
      const cv::Rect2f& lbBox = bboxes[idx];
      const int class_id = labels[idx];
      const float confidence = scores[idx];

      std::string class_label;
      if (!safeGetClassLabel(class_id, class_label)) {
        continue;
      }

      const float left = (lbBox.x - padW) * invGain;
      const float top = (lbBox.y - padH) * invGain;
      const float scaledW = lbBox.width * invGain;
      const float scaledH = lbBox.height * invGain;
      // cv::Rect_<float> box;
      cv::Rect box;
      box.x = dyno::clamp(static_cast<int>(left), 0, original_size.width - 1);
      box.y = dyno::clamp(static_cast<int>(top), 0, original_size.height - 1);
      box.width = dyno::clamp(static_cast<int>(scaledW), 1,
                              original_size.width - box.x);
      box.height = dyno::clamp(static_cast<int>(scaledH), 1,
                               original_size.height - box.y);

      // Compute mask from prototype masks and coefficients
      cv::Mat finalMask = cv::Mat::zeros(SEG_H, SEG_W, CV_32F);
      float* mask_ptr = maskConfs[idx].ptr<float>();
      for (int m = 0; m < 32; ++m) {
        float conf = *(mask_ptr + m);
        finalMask += conf * prototypeMasks[m];
      }

      // Apply sigmoid activation
      cv::exp(-finalMask, finalMask);
      finalMask = 1.0 / (1.0 + finalMask);

      // Crop to letterbox area
      int x1 = static_cast<int>(std::round((padW - 0.1f) * maskScaleX));
      int y1 = static_cast<int>(std::round((padH - 0.1f) * maskScaleY));
      int x2 = static_cast<int>(
          std::round((letter_box_size_.width - padW + 0.1f) * maskScaleX));
      int y2 = static_cast<int>(
          std::round((letter_box_size_.height - padH + 0.1f) * maskScaleY));

      x1 = std::max(0, std::min(x1, SEG_W - 1));
      y1 = std::max(0, std::min(y1, SEG_H - 1));
      x2 = std::max(x1, std::min(x2, SEG_W));
      y2 = std::max(y1, std::min(y2, SEG_H));

      if (x2 <= x1 || y2 <= y1) continue;

      cv::Mat croppedMask =
          finalMask(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();

      // Resize to original image size
      cv::Mat resizedMask;
      cv::resize(croppedMask, resizedMask, original_size, 0, 0,
                 cv::INTER_LINEAR);

      // Threshold and convert to binary
      cv::Mat binaryMask;
      cv::threshold(resizedMask, binaryMask, 0.5f, 255.0, cv::THRESH_BINARY);
      binaryMask.convertTo(binaryMask, CV_8U);

      // Crop to bounding box
      cv::Mat finalBinaryMask = cv::Mat::zeros(original_size, CV_8U);
      cv::Rect roi(box.x, box.y, box.width, box.height);
      roi &= cv::Rect(0, 0, binaryMask.cols, binaryMask.rows);
      if (roi.area() > 0) {
        binaryMask(roi).copyTo(finalBinaryMask(roi));
      }

      // viz.setTo(cv::Scalar(0, 255, 0), finalBinaryMask);
      ObjectDetection detection;
      detection.mask = finalBinaryMask;
      detection.bounding_box = box;
      detection.class_name = class_label;
      detection.confidence = confidence;

      detections.push_back(detection);
    }

    std::vector<SingleDetectionResult> tracking_result =
        tracker_->track(detections);

    for (const SingleDetectionResult& single_result : tracking_result) {
      // this may happen if the object was not well tracked
      if (!single_result.isValid()) {
        continue;
      }

      cv::Mat single_label_mask =
          cv::Mat::zeros(original_size, ObjectDetectionEngine::MaskDType);
      single_label_mask.setTo(single_result.object_id, single_result.mask);
      // set pixel values to object label and update full labelled mask
      // cv::Mat binary_mask = single_result.mask * single_result.object_id;
      result.labelled_mask += single_label_mask;
    }
    // timing_finalise.stop();

    result.detections = tracking_result;

    return true;
  }

  /// @brief Fast letterbox with buffer reuse
  /// @param image Input BGR image
  /// @param buffer Pre-allocated inference buffer
  /// @param targetChannels Target channels for inference
  /// @param targetSize Target size for inference
  /// @param[out] actualSize Actual output size
  /// @param dynamicShape Whether to use dynamic shape
  inline void letterBoxToBlob(const cv::Mat& image, std::vector<float>& blob,
                              int targetChannels, const cv::Size& targetSize,
                              cv::Size& actualSize, bool dynamicShape = false) {
    const int srcH = image.rows;
    const int srcW = image.cols;
    int dstH = targetSize.height;
    int dstW = targetSize.width;

    // Calculate scale (match Ultralytics exactly)
    const float scale = std::min(static_cast<float>(dstH) / srcH,
                                 static_cast<float>(dstW) / srcW);

    // Ultralytics uses round() for new dimensions
    int newH = static_cast<int>(std::round(srcH * scale));
    int newW = static_cast<int>(std::round(srcW * scale));

    // For dynamic shape, adjust to stride-aligned minimum size
    if (dynamicShape) {
      constexpr int stride = 32;
      dstH = ((newH + stride - 1) / stride) * stride;
      dstW = ((newW + stride - 1) / stride) * stride;
    }

    actualSize = cv::Size(dstW, dstH);
    // buffer.ensureCapacity(dstH, dstW, targetChannels);
    size_t required = static_cast<size_t>(dstH * dstW * targetChannels);
    blob.resize(required);

    // Ultralytics uses asymmetric padding with -0.1/+0.1 adjustment
    const float dh = (dstH - newH) / 2.0f;
    const float dw = (dstW - newW) / 2.0f;
    const int padTop = static_cast<int>(std::round(dh - 0.1f));
    const int padLeft = static_cast<int>(std::round(dw - 0.1f));

    // Fill with padding (normalized 114/255)
    constexpr float padNorm = 114.0f / 255.0f;
    std::fill(blob.begin(), blob.begin() + dstH * dstW * targetChannels,
              padNorm);

    // Resize if needed
    cv::Mat resized;
    if (newW != srcW || newH != srcH) {
      cv::resize(image, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);
    } else {
      resized = image;  // Reference, no copy
    }

    constexpr float scale255 = 1.0f / 255.0f;
    if (targetChannels == 3) {
      // Direct BGR->RGB + normalize to CHW blob
      float* rChannel = blob.data();
      float* gChannel = blob.data() + dstH * dstW;
      float* bChannel = blob.data() + 2 * dstH * dstW;

      for (int y = 0; y < newH; ++y) {
        const int dstY = y + padTop;
        const uchar* row = resized.ptr<uchar>(y);
        const int rowOffset = dstY * dstW + padLeft;

        for (int x = 0; x < newW; ++x) {
          const int dstIdx = rowOffset + x;
          const int srcIdx = x * 3;

          bChannel[dstIdx] = row[srcIdx + 0] * scale255;
          gChannel[dstIdx] = row[srcIdx + 1] * scale255;
          rChannel[dstIdx] = row[srcIdx + 2] * scale255;
        }
      }
    } else {
      // normalize directly into blob (single channel)
      float* blobPtr = blob.data();
      for (int y = 0; y < newH; ++y) {
        const int dstY = y + padTop;
        const uchar* row = resized.ptr<uchar>(y);
        const int rowOffset = dstY * dstW + padLeft;

        for (int x = 0; x < newW; ++x) {
          blobPtr[rowOffset + x] = static_cast<float>(row[x]) * scale255;
        }
      }
    }
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
  impl_ = std::make_unique<Impl>(yolo_config, model_info_.input());
}

YoloV8ObjectDetector::~YoloV8ObjectDetector() = default;

ObjectDetectionResult YoloV8ObjectDetector::process(const cv::Mat& image) {
  utils::ChronoTimingStats timing("yolov8_detection.process");
  static constexpr int kTimingVerbosityLevel = 5;

  const auto& input_info = model_info_.input();
  const auto& output0_info = model_info_.output0();
  const auto& output1_info = model_info_.output1();

  {
    utils::ChronoTimingStats timing("yolov8_detection.pre_process",
                                    kTimingVerbosityLevel);
    impl_->preprocess(input_info, image, preprocessed_host_ptr_);
  }

  {
    // allocate input data
    utils::ChronoTimingStats timing("yolov8_detection.allocInput",
                                    kTimingVerbosityLevel);
    bool allocated = input_device_ptr_.allocate(input_info);
    CHECK(
        input_device_ptr_.checkTensorSize(preprocessed_host_ptr_.tensor_size));

    if (allocated) {
      context_->setInputTensorAddress(input_info.name.c_str(),
                                      input_device_ptr_.get());
    }
  }

  // put image data onto gpu
  {
    utils::ChronoTimingStats timing("yolov8_detection.push_from_host",
                                    kTimingVerbosityLevel);
    CHECK(
        input_device_ptr_.pushFromHost(preprocessed_host_ptr_.get(), stream_));
  }

  // prepare output data
  {
    utils::ChronoTimingStats timing("yolov8_detection.allocOutput",
                                    kTimingVerbosityLevel);
    bool output0_allocated = output0_device_ptr_.allocate(output0_info);
    bool output1_allocated = output1_device_ptr_.allocate(output1_info);

    if (output0_allocated) {
      context_->setTensorAddress(output0_info.name.c_str(),
                                 output0_device_ptr_.get());
    }

    if (output1_allocated) {
      context_->setTensorAddress(output1_info.name.c_str(),
                                 output1_device_ptr_.get());
    }
  }

  // set output address tensors
  {
    utils::ChronoTimingStats timing("yolov8_detection.infer");
    cudaStreamSynchronize(stream_);
    bool status = context_->enqueueV3(stream_);
    if (!status) {
      LOG(ERROR) << "initializing inference failed!";
      return ObjectDetectionResult{};
    }
  }

  const auto output0_dims = context_->getTensorShape(output0_info.name.c_str());
  const auto output1_dims = context_->getTensorShape(output1_info.name.c_str());

  ObjectDetectionResult result;
  {
    const float* d_output0_data = output0_device_ptr_.get();
    const float* d_output1_data = output1_device_ptr_.get();
    utils::ChronoTimingStats timing("yolov8_detection.post_process",
                                    kTimingVerbosityLevel);
    impl_->postprocess(image, d_output0_data, d_output1_data, output0_dims,
                       output1_dims, result);
  }

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
