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

  /**
   * @brief Get the generated labelled object mask.
   * All pixels == 0 are background and all pixels >0 represents a unique object
   * label $j$ which is the tracked obejct instance.
   *
   * Mask must be of type MaskDType.
   * Mask will be empty if result() is invalid
   *
   * @return cv::Mat
   */
  cv::Mat mask() const;

  /**
   * @brief Visualisation of mask where objects are uniquely coloured by their
   * object ids and overalyed on the background rgb image
   *
   * @return cv::Mat
   */
  cv::Mat colouredMask() const;

  /**
   * @brief The RGB input image that was used for inference
   *
   * @return cv::Mat
   */
  cv::Mat inputImage() const;
};

}  // namespace dyno
