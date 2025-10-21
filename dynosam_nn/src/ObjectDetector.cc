#include "dynosam_nn/ObjectDetector.hpp"

#include <iostream>

#include "dynosam_common/Types.hpp"

namespace dyno {

cv::Mat ObjectDetectionEngine::colouredMask() const {
  const ObjectDetectionResult result = this->result();
  return result.colouredMask();
}

cv::Mat ObjectDetectionEngine::mask() const {
  const ObjectDetectionResult result = this->result();
  return result.labelled_mask;
}

cv::Mat ObjectDetectionEngine::inputImage() const {
  const ObjectDetectionResult result = this->result();
  return result.input_image;
}

}  // namespace dyno
