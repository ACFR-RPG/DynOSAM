#pragma once

#include <opencv4/opencv2/core/cuda.hpp>
#include <opencv4/opencv2/core/types.hpp>

#include "dynosam_common/DynamicObjects.hpp"

// A single detection struct that overlays the raw float data
struct alignas(float) YoloDetection {
  // 0-3: Box Coordinates (Center X, Center Y, Width, Height)
  float x, y, w, h;

  // 4: Confidence
  float confidence;

  // 5: Class ID (Stored as float, cast to int when reading)
  float class_id;

  // 6-37: Mask Coefficients
  float mask[32];

// Helper to get cv::Rect on the fly (Zero overhead until called)
// Note: Assuming you have access to OpenCV headers here.
// If not, remove this helper and do the math at the usage site.
#ifdef __cplusplus
  inline cv::Rect toCvRect() const {
    return cv::Rect(static_cast<int>(std::round(x - w / 2.0f)),  // top-left x
                    static_cast<int>(std::round(y - h / 2.0f)),  // top-left y
                    static_cast<int>(std::round(w)),             // width
                    static_cast<int>(std::round(h))              // height);
    );
  }
#endif
};

struct DetectionGpuMats {
  // Wrapper for all box coordinates (count rows x 4 cols, float)
  cv::cuda::GpuMat boxes;

  // Wrapper for all confidence and class IDs (count rows x 2 cols, float)
  cv::cuda::GpuMat scores_and_classes;

  // Wrapper for all mask coefficients (count rows x 32 cols, float)
  cv::cuda::GpuMat mask_coeffs;
};

DetectionGpuMats createDetectionGpuMatWrappers(YoloDetection* d_detections);

struct YoloKernelConfig {
  int num_boxes;    // e.g., 8400
  int num_classes;  // e.g., 80
  float conf_threshold;
  int box_offset;         // Index offset for Box data
  int class_conf_offset;  // Index offset for Class data
  int mask_coeff_offset;  // Index offset for Mask data
};

/**
 * @brief Runs YOLO post-processing on the GPU.
 * @param d_model_output    Pointer to the raw YOLO output on Device (float*).
 * @param config            Configuration struct for offsets and thresholds.
 * @param h_output_buffer   Pointer to Host memory where results will be copied.
 * Must be large enough: num_boxes * sizeof(YoloDetection).
 * @param d_output_buffer   Pointer to Device memory for temporary storage. Must
 * be large enough: num_boxes * sizeof(YoloDetection).
 * @param d_count_buffer    Pointer to Device memory for the atomic counter
 * (sizeof(int)).
 * @param stream            (Optional) CUDA stream to launch kernels on.
 * @return int              The number of valid detections found.
 */
int YoloOutputToDetections(const float* d_model_output,
                           const YoloKernelConfig& config,
                           YoloDetection* h_output_buffer,
                           YoloDetection* d_output_buffer, int* d_count_buffer,
                           int* h_count_buffer, void* stream = nullptr);

void YoloDetectionsToObjects(
    const DetectionGpuMats& wrappers, const cv::cuda::GpuMat& d_prototype_masks,
    const cv::Rect& prototype_crop_rect,  // Pre-calculated crop area
    const YoloDetection* h_detection, const cv::Size& required_size,
    const cv::Size& original_size, const std::string& label, const int mask_h,
    const int mask_w, cv::cuda::Stream stream,
    dyno::ObjectDetection& detection);
