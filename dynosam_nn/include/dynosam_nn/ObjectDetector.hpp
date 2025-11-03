#pragma once

#include <opencv4/opencv2/opencv.hpp>

#include "dynosam_common/DynamicObjects.hpp"
#include "dynosam_common/utils/Macros.hpp"
#include "dynosam_cv/ImageTypes.hpp"

namespace dyno {

/**
 * @brief Base class for a ObjectDetection network
 *
 */
class ObjectDetectionEngine {
 public:
  DYNO_POINTER_TYPEDEFS(ObjectDetectionEngine)

  /// @brief Expected output mask (OpenCV) type
  constexpr static int MaskDType = ImageType::MotionMask::OpenCVType;

  ObjectDetectionEngine() {}
  virtual ~ObjectDetectionEngine() {}

  virtual ObjectDetectionResult process(const cv::Mat& image) = 0;
  virtual ObjectDetectionResult result() const = 0;

  cv::Mat mask() const;
  cv::Mat colouredMask() const;
  cv::Mat inputImage() const;
};

}  // namespace dyno
