#pragma once

#include <opencv4/opencv2/opencv.hpp>

#include "dynosam_common/DynamicObjects.hpp"
#include "dynosam_common/utils/Macros.hpp"

namespace dyno {

/**
 * @brief Base class for a ObjectDetection network
 *
 */
class ObjectDetectionEngine {
 public:
  DYNO_POINTER_TYPEDEFS(ObjectDetectionEngine)

  ObjectDetectionEngine() {}
  virtual ~ObjectDetectionEngine() {}

  virtual ObjectDetectionResult process(const cv::Mat& image) = 0;
  virtual bool onDestruction() = 0;
  virtual ObjectDetectionResult result() const = 0;

  cv::Mat mask() const;
  cv::Mat colouredMask() const;
  cv::Mat inputImage() const;
};

}  // namespace dyno
