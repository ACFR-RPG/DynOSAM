#include "dynosam_nn/YoloObjectDetector.hpp"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "dynosam_common/utils/OpenCVUtils.hpp"
#include "dynosam_common/utils/TimingStats.hpp"
#include "dynosam_common/viz/Colour.hpp"
#include "dynosam_nn/YoloV8Utils.hpp"
#include "dynosam_nn/cuda_utils.hpp"
#include "dynosam_nn/trackers/ObjectTracker.hpp"

// TODO: to remove and from cmake!
#include <omp.h>

DetectionGpuMats createDetectionGpuMatWrappers(YoloDetection* d_detections) {
  DetectionGpuMats wrappers;

  // --- Pitch Calculation ---
  // The stride (pitch) is the distance in bytes from the start of one struct
  // to the start of the next struct.
  const size_t struct_pitch_bytes = sizeof(YoloDetection);  // 152 bytes

  float* d_detections_ptr = reinterpret_cast<float*>(d_detections);

  // The column type is always CV_32FC1 (float) for all wrappers.

  // -------------------------------------------------------------------------
  // 1. Boxes (X, Y, W, H) - 4 columns
  // -------------------------------------------------------------------------
  // Start address: d_detections (offset 0)
  wrappers.boxes = cv::cuda::GpuMat(
      1,                  // rows
      4,                  // cols
      CV_32FC1,           // type (float, 1 channel)
      d_detections_ptr,   // data pointer (points to the 'x' field)
      struct_pitch_bytes  // step (stride from row to row)
  );

  // -------------------------------------------------------------------------
  // 2. Scores and Class IDs - 2 columns
  // -------------------------------------------------------------------------
  // Start address: &d_detections[0].confidence
  // Offset in floats: 4 (x, y, w, h)
  // Offset in bytes: 4 * sizeof(float) = 16 bytes
  float* d_scores_ptr = reinterpret_cast<float*>(d_detections) + 4;

  wrappers.scores_and_classes = cv::cuda::GpuMat(
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
  float* d_masks_ptr = reinterpret_cast<float*>(d_detections) + 6;

  wrappers.mask_coeffs =
      cv::cuda::GpuMat(1,            // rows
                       32,           // cols
                       CV_32FC1,     // type
                       d_masks_ptr,  // data pointer (points to 'mask[0]' field)
                       struct_pitch_bytes  // step
      );

  return wrappers;
}

namespace dyno {

/**
 * @brief Much of this code is developed from:
 * https://github.dev/Geekgineer/YOLOs-CPP/blob/main/include/seg/YOLO8Seg.hpp
 *
 */

struct YoloV8ObjectDetector::Impl {
  const YoloConfig yolo_config_;
  // Information about the tensor to be provided as input directly to the model
  const ImageTensorInfo input_info_;

  bool is_first{true};
  ObjectTracker::UniquePtr tracker_;
  // Device (GPU) detection buffer
  YoloDetection* d_buffer_;
  // Host (CPU) detection buffer
  YoloDetection* h_buffer_;
  // Device (GPU) detection count
  int* d_counter_;

  //! Size of the input image prior to preprocessing (ie. size of the camera
  //! image)
  cv::Size original_size_;

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

    size_t det_size = MaxDetections * sizeof(YoloDetection);
    cudaMalloc(&d_buffer_, det_size);

    cudaMallocHost(&h_buffer_, det_size);

    size_t count_size = sizeof(int);
    cudaMalloc(&d_counter_, count_size);
    cudaMemset(d_counter_, 0, count_size);
  }

  ~Impl() {
    if (d_buffer_) {
      cudaFree(d_buffer_);
    }
    if (h_buffer_) {
      cudaFreeHost(h_buffer_);
    }
    if (d_counter_) {
      cudaFree(d_counter_);
    }
  }

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
    // preprocess image according to YOLO training pre-procesing
    cv::Mat letterbox_image;
    // assume not dynamic
    letterBox(rgb, letterbox_image, requiredInputSize(),
              cv::Scalar(114, 114, 114),
              /*auto_=*/false,
              /*scaleFill=*/false, /*scaleUp=*/true, /*stride=*/32);
    letterbox_image.convertTo(letterbox_image, CV_32FC3, 1.0f / 255.0f);

    size_t letter_box_size = static_cast<size_t>(letterbox_image.rows) *
                             static_cast<size_t>(letterbox_image.cols) * 3;
    CHECK_EQ(letter_box_size, input_info.size());
    // float* blobPtr = new float[letter_box_size];
    input_vector.allocate(input_info);

    float* input_data = input_vector.get();

    std::vector<cv::Mat> channels(3);
    for (int c = 0; c < 3; ++c) {
      channels[c] = cv::Mat(
          letterbox_image.rows, letterbox_image.cols, CV_32FC1,
          input_data + c * (letterbox_image.rows * letterbox_image.cols));
    }
    cv::split(letterbox_image, channels);
    // std::vector<float> input_vector(blobPtr, blobPtr + letter_box_size);

    // delete[] blobPtr;
    // processed_vector = input_vector;
    is_first = false;
    return true;
  }

  bool postprocess(const cv::Mat& rgb, const float* d_output0,
                   const float* d_output1, const nvinfer1::Dims& output0_dims,
                   const nvinfer1::Dims& output1_dims,
                   ObjectDetectionResult& result) {
    utils::TimingStatsCollector timing_all("yolov8_detection.post_process.run",
                                           5);

    if (output1_dims.nbDims != 4 || output1_dims.d[0] != 1 ||
        output1_dims.d[1] != 32)
      throw std::runtime_error(
          "Unexpected output1 shape. Expected [1, 32, mask_h, mask_w].");

    const cv::Size& required_size = requiredInputSize();
    const cv::Size& original_size = originalSize(rgb);

    utils::TimingStatsCollector timing_setup(
        "yolov8_detection.post_process.setup", 5);

    // result result regardless
    result.labelled_mask =
        cv::Mat::zeros(original_size, ObjectDetectionEngine::MaskDType);
    result.input_image = rgb;

    // const float* output0_data = output0;
    // const float* output1_data = output1;

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

    timing_setup.stop();

    // 1. Process prototype masks
    // Store all prototype masks in a vector for easy access
    utils::TimingStatsCollector timing_proto(
        "yolov8_detection.post_process.proto", 5);

    // prototypeMasks.reserve(32);
    // for (int m = 0; m < 32; ++m) {
    //   // Each mask is mask_h x mask_w
    //   cv::Mat proto(mask_h, mask_w, CV_32F,
    //                 const_cast<float*>(output1_data + m * mask_h * mask_w));
    //   prototypeMasks.emplace_back(
    //       proto.clone());  // Clone to ensure data integrity
    // }

    // mat does not take const pointer!!
    const void* d_output1_void = d_output1;

    cv::cuda::GpuMat d_prototype_masks(
        YoloV8ModelInfo::Constants::NumMasks *
            mask_h,  // rows = stacked vertically
        mask_w,      // cols
        CV_32F, const_cast<void*>(d_output1_void), mask_w * sizeof(float));

    timing_proto.stop();

    // 2. Process detections
    // std::vector<cv::Rect> boxes;
    // boxes.reserve(num_boxes);
    // std::vector<float> confidences;
    // confidences.reserve(num_boxes);
    // std::vector<int> class_ids;
    // class_ids.reserve(num_boxes);
    // std::vector<std::vector<float>> mask_coefficients;

    CHECK_EQ(num_boxes, MaxDetections);
    CHECK_EQ(num_classes, 80);

    YoloKernelConfig config;
    config.num_boxes = MaxDetections;
    config.num_classes = num_classes;
    config.conf_threshold = yolo_config_.conf_threshold;
    config.box_offset = BoxOffset;
    config.class_conf_offset = ClassConfOffset;
    config.mask_coeff_offset = MaskCoeffOffset;

    utils::TimingStatsCollector timing_boxes(
        "yolov8_detection.post_process.boxes", 5);
    int count = LaunchYoloPostProcess(
        d_output0,  // The raw GPU pointer from TensorRT/ONNX
        config,
        h_buffer_,  // Destination on CPU
        d_buffer_,  // Temp storage on GPU
        d_counter_  // Temp counter on GPU
    );

    // LOG(INFO) << "yolo boxes post process " << count;
    // return false;
    // mask_coefficients.reserve(num_boxes);

    // Define the temporary storage variables for each thread.
    // These are created privately for each thread that executes the parallel
    // region. thread_local std::vector<cv::Rect> private_boxes; thread_local
    // std::vector<float> private_confidences; thread_local std::vector<int>
    // private_class_ids; thread_local std::vector<std::vector<float>>
    // private_mask_coefficients;
    //   for (int i = 0; i < num_boxes; ++i) {
    //     // // Extract box coordinates
    //     float xc = output0_data[BoxOffset * num_boxes + i];
    //     float yc = output0_data[(BoxOffset + 1) * num_boxes + i];
    //     float w = output0_data[(BoxOffset + 2) * num_boxes + i];
    //     float h = output0_data[(BoxOffset + 3) * num_boxes + i];

    //     // Computational Optimization: Calculate half-width/height once.
    //     float half_w = w * 0.5f;
    //     float half_h = h * 0.5f;

    //     // Get class confidence
    //     float maxConf = 0.0f;
    //     int classId = -1;
    //     for (int c = 0; c < num_classes; ++c) {
    //       float conf = output0_data[(ClassConfOffset + c) * num_boxes + i];
    //       // const float conf = output0_data[base_conf_idx + c * num_boxes +
    //       i]; if (conf > maxConf) {
    //         maxConf = conf;
    //         classId = c;
    //       }
    //     }

    //     if (maxConf < yolo_config_.conf_threshold) continue;

    //     {
    //       cv::Rect box_cv(static_cast<int>(std::round(xc - half_w)),  //
    //       top-left x
    //                       static_cast<int>(std::round(yc - half_h)),  //
    //                       top-left y static_cast<int>(std::round(w)), //
    //                       width static_cast<int>(std::round(h)) // height
    //       );

    //       // Store detection
    //       boxes.push_back(box_cv);
    //       confidences.push_back(maxConf);
    //       class_ids.push_back(classId);

    //       // LOG(INFO) << "Found box with id " << classId;

    //       // Store mask coefficients
    //       std::vector<float> mask_coeffs(32);
    //       for (int m = 0; m < 32; ++m) {
    //         // mask_coeffs[m] = output0_data[base_mask_idx + m * num_boxes +
    //         i]; mask_coeffs[m] = output0_data[(MaskCoeffOffset + m) *
    //         num_boxes + i];
    //       }
    //       mask_coefficients.emplace_back(std::move(mask_coeffs));
    //     }
    //   }
    // #pragma omp parallel for default(none) \
    //   shared(output0_data, num_boxes, num_classes, yolo_config_, BoxOffset,
    //   ClassConfOffset, MaskCoeffOffset) \ schedule(static) //
    // for (int i = 0; i < num_boxes; ++i) {

    //     // ------------------------------------------------
    //     // A. Extract and Transform Box Coordinates
    //     // ------------------------------------------------
    //     // Large stride access pattern (slow but unavoidable given input
    //     layout) float xc = output0_data[BoxOffset * num_boxes + i]; float yc
    //     = output0_data[(BoxOffset + 1) * num_boxes + i]; float w =
    //     output0_data[(BoxOffset + 2) * num_boxes + i]; float h =
    //     output0_data[(BoxOffset + 3) * num_boxes + i];

    //     float half_w = w * 0.5f;
    //     float half_h = h * 0.5f;

    //     // We defer the creation of cv::Rect until we know the box is kept.
    //     // This saves creating an object that might be immediately thrown
    //     away.

    //     // ------------------------------------------------
    //     // B. Find Max Class Confidence
    //     // ------------------------------------------------
    //     float maxConf = 0.0f;
    //     int classId = -1;

    //     // Inner loop for class confidence search
    //     for (int c = 0; c < num_classes; ++c) {
    //         float conf = output0_data[(ClassConfOffset + c) * num_boxes + i];
    //         if (conf > maxConf) {
    //             maxConf = conf;
    //             classId = c;
    //         }
    //     }

    //     // ------------------------------------------------
    //     // C. Filtering and Storage
    //     // ------------------------------------------------
    //     if (maxConf < yolo_config_.conf_threshold) {
    //         // Skip this box immediately if below threshold
    //         continue;
    //     }

    //     // --- Only if the box is kept, perform object creation/vector write
    //     ---

    //     // Create the cv::Rect now
    //     cv::Rect box_cv(static_cast<int>(std::round(xc - half_w)),  //
    //     top-left x
    //                     static_cast<int>(std::round(yc - half_h)),  //
    //                     top-left y static_cast<int>(std::round(w)), // width
    //                     static_cast<int>(std::round(h)));           // height

    //     // Store detection in thread-local vectors
    //     private_boxes.push_back(box_cv);
    //     private_confidences.push_back(maxConf);
    //     private_class_ids.push_back(classId);

    //     // Store mask coefficients
    //     std::vector<float> mask_coeffs(32);
    //     for (int m = 0; m < 32; ++m) {
    //         mask_coeffs[m] = output0_data[(MaskCoeffOffset + m) * num_boxes +
    //         i];
    //     }
    //     private_mask_coefficients.emplace_back(std::move(mask_coeffs));
    // }

    // // ----------------------------------------------------------------------
    // // 3. Merging Private Results into Shared Vectors
    // // ----------------------------------------------------------------------

    // // The merging step must be sequential, but it only involves fast memory
    // copies
    // // and runs much faster than the computation loop.
    // // Note: This merging must happen AFTER the parallel region (outside the
    // #pragma).

    // size_t total_detections = 0;

    // // Step 1: Calculate the total number of detections across all threads
    // // This is required to pre-allocate the main vectors, avoiding expensive
    // reallocations. #pragma omp parallel reduction(+:total_detections)
    // {
    //     total_detections += private_boxes.size();
    // }

    // // Step 2: Resize the main vectors
    // //TODO!!!
    // boxes.reserve(boxes.size() + total_detections);
    // confidences.reserve(confidences.size() + total_detections);
    // class_ids.reserve(class_ids.size() + total_detections);
    // mask_coefficients.reserve(mask_coefficients.size() + total_detections);

    // // Step 3: Append data from all thread-local vectors to the main vectors
    // // This can be done efficiently using another parallel loop or
    // sequentially.
    // // Since the vectors are not being modified within the loop, the append
    // operation
    // // is sequential, but fast. The `reduction(merge)` is not used here for
    // custom types. #pragma omp parallel
    // {
    //     // Append the local vectors to the shared vectors within a critical
    //     section
    //     // using move semantics to avoid copy overhead.
    //     // Critical is necessary here because writing to the shared vector's
    //     capacity
    //     // or size must be atomic.
    //     #pragma omp critical
    //     {
    //         if (!private_boxes.empty()) {
    //             boxes.insert(boxes.end(),
    //                         std::make_move_iterator(private_boxes.begin()),
    //                         std::make_move_iterator(private_boxes.end()));

    //             confidences.insert(confidences.end(),
    //                               std::make_move_iterator(private_confidences.begin()),
    //                               std::make_move_iterator(private_confidences.end()));

    //             class_ids.insert(class_ids.end(),
    //                             std::make_move_iterator(private_class_ids.begin()),
    //                             std::make_move_iterator(private_class_ids.end()));

    //             // mask_coefficients.insert(mask_coefficients.end(),
    //             //
    //             std::make_move_iterator(private_mask_coefficients.begin()),
    //             // std::make_move_iterator(private_mask_coefficients.end()));
    //             mask_coefficients.insert(mask_coefficients.end(),
    //                                     private_mask_coefficients.begin(),
    //                                     private_mask_coefficients.end());
    //         }
    //     }
    // }
    timing_boxes.stop();

    std::vector<cv::Rect> boxes;
    boxes.reserve(num_boxes);
    std::vector<float> confidences;
    confidences.reserve(num_boxes);

    for (int i = 0; i < count; ++i) {
      // 1. Get a reference to the detection data in the contiguous array
      const YoloDetection& det = h_buffer_[i];

      // 2. Convert and Store Box/Confidence/Class ID

      // Use the struct helper to calculate the cv::Rect efficiently
      boxes.push_back(det.toCvRect());

      // Confidence is a straight copy
      confidences.push_back(det.confidence);
    }
    // Early exit if no boxes after confidence threshold
    if (boxes.empty()) {
      return false;
    }

    // 3. Apply NMS
    utils::TimingStatsCollector timing_nms("yolov8_detection.post_process.nms",
                                           5);
    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes, confidences, yolo_config_.conf_threshold,
                      yolo_config_.nms_threshold, nms_indices);
    timing_nms.stop();
    if (nms_indices.empty()) {
      return false;
    }

    utils::TimingStatsCollector timing_gain(
        "yolov8_detection.post_process.gain", 5);
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

    // --- Crop Coordinates (Calculated once) ---
    int x1_crop = static_cast<int>(std::round((pad_w - 0.1f) * mask_scale_x));
    int y1_crop = static_cast<int>(std::round((pad_h - 0.1f) * mask_scale_y));
    int x2_crop = static_cast<int>(
        std::round((required_size.width - pad_w + 0.1f) * mask_scale_x));
    int y2_crop = static_cast<int>(
        std::round((required_size.height - pad_h + 0.1f) * mask_scale_y));
    // Clamping is done inside the kernel for safety, but can be done here too
    // x1_crop = std::max(0, std::min(x1_crop, mask_w - 1)); // etc.

    const cv::Rect prototype_crop_rect(x1_crop, y1_crop, x2_crop - x1_crop,
                                       y2_crop - y1_crop);

    timing_gain.stop();

    // --- Output Buffer ---
    // Allocate a single GpuMat to hold ALL final, full-resolution masks.
    // Size: NMS_COUNT * Original_H * Original_W
    // cv::cuda::GpuMat d_final_masks_buffer(
    //   wrappers.boxes.rows,
    //   original_size.height * original_size.width, // Flat storage for HxW
    //   mask CV_8UC1
    // );

    utils::TimingStatsCollector timing_detections(
        "yolov8_detection.post_process.detections", 5);
    std::vector<ObjectDetection> detections;
    detections.reserve(nms_indices.size());
    for (const int idx : nms_indices) {
      YoloDetection* d_det = d_buffer_ + idx;
      const YoloDetection* h_det = h_buffer_ + idx;

      const int class_id = static_cast<int>(h_det->class_id);

      std::string class_label;
      if (!safeGetClassLabel(class_id, class_label)) {
        continue;
      }

      DetectionGpuMats detection_gpu_mats =
          createDetectionGpuMatWrappers(d_det);

      cv::cuda::Stream stream = stream_pool_.getCvStream();

      ObjectDetection detection;
      RunFullMaskPipelineGPU(detection_gpu_mats, d_prototype_masks,
                             prototype_crop_rect, h_det, required_size,
                             original_size, mask_h, mask_w, stream, detection);
      detections.push_back(detection);
    }
    //   const float confidence = confidences[idx];
    //   const int class_id = class_ids[idx];

    //    std::string class_label;
    //   if (!safeGetClassLabel(class_id, class_label)) {
    //     continue;
    //   }

    //   // detection still on the GPU
    //   YoloDetection* det = d_buffer_[idx];
    //   DetectionGpuMats detection_gpu_mats =
    //   createDetectionGpuMatWrappers(det);

    //   const auto& mask_coeffs = mask_coefficients[idx];

    //   // // Linear combination of prototype masks
    //   // cv::cuda::GpuMat final_mask = cv::cuda::GpuMat::zeros(mask_h,
    //   mask_w, CV_32F);
    //   // for (int m = 0; m < 32; ++m) {
    //   //   final_mask += mask_coeffs[m] * prototypeMasks[m];
    //   // }

    // }
    // std::vector<ObjectDetection> detections;
    // for (const int idx : nms_indices) {
    //   const float confidence = confidences[idx];
    //   const int class_id = class_ids[idx];

    //   // skip object if not in the set of included class labels
    //   std::string class_label;
    //   if (!safeGetClassLabel(class_id, class_label)) {
    //     continue;
    //   }

    // 5. Scale box to original image
    //   const cv::Rect bounding_box =
    //       scaleCoords(required_size, boxes[idx], original_size, true);

    //   // 6. Process mask
    //   const auto& mask_coeffs = mask_coefficients[idx];

    //   // Linear combination of prototype masks
    //   cv::Mat final_mask = cv::Mat::zeros(mask_h, mask_w, CV_32F);
    //   for (int m = 0; m < 32; ++m) {
    //     final_mask += mask_coeffs[m] * prototypeMasks[m];
    //   }
    //   // Apply sigmoid activation
    //   final_mask = sigmoid(final_mask);

    //   // Crop mask to letterbox area with a slight padding to avoid border
    //   // issues
    //   int x1 = static_cast<int>(std::round((pad_w - 0.1f) * mask_scale_x));
    //   int y1 = static_cast<int>(std::round((pad_h - 0.1f) * mask_scale_y));
    //   int x2 = static_cast<int>(
    //       std::round((required_size.width - pad_w + 0.1f) * mask_scale_x));
    //   int y2 = static_cast<int>(
    //       std::round((required_size.height - pad_h + 0.1f) * mask_scale_y));

    //   // Ensure coordinates are within mask bounds
    //   x1 = std::max(0, std::min(x1, mask_w - 1));
    //   y1 = std::max(0, std::min(y1, mask_h - 1));
    //   x2 = std::max(x1, std::min(x2, mask_w));
    //   y2 = std::max(y1, std::min(y2, mask_h));

    //   // Handle cases where cropping might result in zero area
    //   if (x2 <= x1 || y2 <= y1) {
    //     // Skip this mask as cropping is invalid
    //     LOG(INFO) << "Mask invalud?";
    //     continue;
    //   }

    //   cv::Rect cropRect(x1, y1, x2 - x1, y2 - y1);
    //   cv::Mat croppedMask =
    //       final_mask(cropRect).clone();  // Clone to ensure data integrity

    //   // Resize to original dimensions
    //   cv::Mat resized_mask;
    //   cv::resize(croppedMask, resized_mask, original_size, 0, 0,
    //              cv::INTER_LINEAR);

    //   // Threshold and convert to binary
    //   cv::Mat binary_mask;
    //   cv::threshold(resized_mask, binary_mask, 0.5, 255.0,
    //   cv::THRESH_BINARY); binary_mask.convertTo(binary_mask, CV_8U);

    //   // Crop to bounding box
    //   cv::Mat final_binary_mask = cv::Mat::zeros(original_size, CV_8U);
    //   cv::Rect roi = bounding_box;
    //   roi &= cv::Rect(0, 0, binary_mask.cols,
    //                   binary_mask.rows);  // Ensure ROI is within mask
    //   if (roi.area() > 0) {
    //     binary_mask(roi).copyTo(final_binary_mask(roi));
    //   }

    //   ObjectDetection detection{final_binary_mask, bounding_box, class_label,
    //                             confidence};
    //   detections.push_back(detection);
    // }
    timing_detections.stop();

    // return false;

    utils::TimingStatsCollector timing_track(
        "yolov8_detection.post_process.track", 5);
    std::vector<SingleDetectionResult> tracking_result =
        tracker_->track(detections);
    timing_track.stop();

    // //construct label mask from tracked result
    utils::TimingStatsCollector timing_finalise(
        "yolov8_detection.post_process.finalise", 5);
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

  inline cv::Mat sigmoid(const cv::Mat& src) {
    cv::Mat dst;
    cv::exp(-src, dst);
    dst = 1.0 / (1.0 + dst);
    return dst;
  }

  // template <typename T>
  // T clamp(const T& val, const T& low, const T& high) {
  //   return std::max(low, std::min(val, high));
  // }

  // inline cv::Rect scaleCoords(const cv::Size& required_shape,
  //                             const cv::Rect& coords,
  //                             const cv::Size& originalShape,
  //                             bool p_Clip = true) {
  //   float gain =
  //       std::min((float)required_shape.height / (float)originalShape.height,
  //                (float)required_shape.width / (float)originalShape.width);

  //   int pad_w = static_cast<int>(std::round(
  //       ((float)required_shape.width - (float)originalShape.width * gain) /
  //       2.f));
  //   int pad_h = static_cast<int>(std::round(
  //       ((float)required_shape.height - (float)originalShape.height * gain) /
  //       2.f));

  //   cv::Rect ret;
  //   ret.x =
  //       static_cast<int>(std::round(((float)coords.x - (float)pad_w) /
  //       gain));
  //   ret.y =
  //       static_cast<int>(std::round(((float)coords.y - (float)pad_h) /
  //       gain));
  //   ret.width = static_cast<int>(std::round((float)coords.width / gain));
  //   ret.height = static_cast<int>(std::round((float)coords.height / gain));

  //   if (p_Clip) {
  //     ret.x = clamp(ret.x, 0, originalShape.width);
  //     ret.y = clamp(ret.y, 0, originalShape.height);
  //     ret.width = clamp(ret.width, 0, originalShape.width - ret.x);
  //     ret.height = clamp(ret.height, 0, originalShape.height - ret.y);
  //   }

  //   return ret;
  // }

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
};

void ConstructPlanarGpuMat(float* d_input_ptr, int H, int W,
                           std::vector<cv::cuda::GpuMat>& channel_wrappers) {
  // The number of elements per channel (Height * Width)
  const size_t channel_size_elements = static_cast<size_t>(H) * W;

  // The stride (pitch) in bytes for a single row.
  // Since the data is linear/non-pitched (CHW), the row stride is simply
  // the width (W) times the size of a float (4 bytes).
  const size_t step_bytes = static_cast<size_t>(W) * sizeof(float);

  // The memory offset for the next channel (in bytes)
  const size_t channel_offset_bytes = channel_size_elements * sizeof(float);

  // --- Allocate the vector to hold the wrappers ---
  channel_wrappers.resize(3);

  for (int c = 0; c < 3; ++c) {
    // 1. Calculate the starting address for this channel (R, G, or B)
    float* d_channel_ptr = d_input_ptr + (c * channel_size_elements);

    // 2. Wrap the memory with cv::cuda::GpuMat
    // GpuMat(rows, cols, type, data_ptr, step_bytes)
    channel_wrappers[c] = cv::cuda::GpuMat(
        H,              // rows
        W,              // cols
        CV_32FC1,       // type (float, 1 channel)
        d_channel_ptr,  // raw pointer to the start of this channel
        step_bytes      // step (W * sizeof(float))
    );

    std::cout << "Channel " << c
              << " wrapped. Start address: " << (void*)d_channel_ptr
              << ", Step: " << step_bytes << " bytes." << std::endl;
  }

  // Now channel_wrappers[0] points to R, [1] to G, [2] to B.
  // They can be used in your subsequent CUDA kernels or OpenCV CUDA functions.
}

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
  utils::TimingStatsCollector timing("yolov8_detection.process");
  static constexpr int kTimingVerbosityLevel = 5;

  const auto& input_info = model_info_.input();
  const auto& output0_info = model_info_.output0();
  const auto& output1_info = model_info_.output1();

  // TODO: should cuda MallocHost!!
  // std::vector<float> input_data;
  {
    utils::TimingStatsCollector timing("yolov8_detection.pre_process",
                                       kTimingVerbosityLevel);
    impl_->preprocess(input_info, image, preprocessed_host_ptr_);
  }

  {
    // allocate input data
    utils::TimingStatsCollector timing("yolov8_detection.allocInput",
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
    utils::TimingStatsCollector timing("yolov8_detection.push_from_host",
                                       kTimingVerbosityLevel);
    CHECK(
        input_device_ptr_.pushFromHost(preprocessed_host_ptr_.get(), stream_));
  }

  // prepare output data
  {
    utils::TimingStatsCollector timing("yolov8_detection.allocOutput",
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

  // if (output0_data_ == nullptr) {
  //   LOG(INFO) << "Alloc 0";
  //   auto error =
  //       cudaMallocHost(&output0_data_, output0_device_ptr_.allocated_size);
  //   if (error != cudaSuccess) {
  //     LOG(ERROR) << "Failed to allocate output0_data_";
  //   }
  // }

  // if (output1_data_ == nullptr) {
  //   LOG(INFO) << "Alloc 1";
  //   auto error =
  //       cudaMallocHost(&output1_data_, output1_device_ptr_.allocated_size);
  //   if (error != cudaSuccess) {
  //     LOG(ERROR) << "Failed to allocate output1_data_";
  //   }
  // }

  // set output address tensors
  {
    utils::TimingStatsCollector timing("yolov8_detection.infer");
    cudaStreamSynchronize(stream_);
    bool status = context_->enqueueV3(stream_);
    if (!status) {
      LOG(ERROR) << "initializing inference failed!";
      return ObjectDetectionResult{};
    }
  }

  // std::vector<float> output0_data, output1_data;
  {
      // utils::TimingStatsCollector timing("yolov8_detection.get_from_device",
      //                                    kTimingVerbosityLevel);
      // CHECK(output0_device_ptr_.getFromDevice(output0_data_, stream_));
      // CHECK(output1_device_ptr_.getFromDevice(output1_data_, stream_));
  }

  {
    // now we do need to synchronize since post-processing has its own stream
    // pool
    utils::TimingStatsCollector timing("yolov8_detection.synchronize",
                                       kTimingVerbosityLevel);
    cudaStreamSynchronize(stream_);
  }

  const auto output0_dims = context_->getTensorShape(output0_info.name.c_str());
  const auto output1_dims = context_->getTensorShape(output1_info.name.c_str());

  ObjectDetectionResult result;
  {
    const float* d_output0_data = output0_device_ptr_.get();
    const float* d_output1_data = output1_device_ptr_.get();
    utils::TimingStatsCollector timing("yolov8_detection.post_process",
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
