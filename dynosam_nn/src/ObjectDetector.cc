#include "dynosam_nn/ObjectDetector.hpp"

#include <iostream>

#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/OpenCVUtils.hpp"

namespace dyno {

cv::Mat ObjectDetectionResult::colouredMask() const {
  // input image should never be empty
  if (num() == 0 || labelled_mask.empty()) {
    return input_image;
  }
  return utils::labelMaskToRGB(labelled_mask,
                               background_label,  // from dynosam_common
                               input_image);
}

ObjectIds ObjectDetectionResult::objectIds() const {
  ObjectIds object_ids;
  std::transform(
      detections.begin(), detections.end(), std::back_inserter(object_ids),
      [](const SingleDetectionResult& result) { return result.object_id; });
  return object_ids;
}

std::ostream& operator<<(std::ostream& os,
                         const dyno::ObjectDetectionResult& res) {
  os << "ObjectDetectionResult:\n";
  os << "  detections (" << res.detections.size() << "):\n";
  for (const auto& det : res.detections) {
    os << "    id=" << det.object_id << ", class=" << det.class_name
       << ", confidence=" << det.confidence << ", bbox=(" << det.bounding_box.x
       << "," << det.bounding_box.y << "," << det.bounding_box.width << ","
       << det.bounding_box.height << ")\n";
  }
  // if (!res.labelled_mask.empty()) {
  //     os << "  labelled_mask: (" << res.labelled_mask.rows
  //        << "x" << res.labelled_mask.cols
  //        << ", channels=" << res.labelled_mask.channels() << ")\n";
  // } else {
  //     os << "  labelled_mask: empty\n";
  // }
  return os;
}

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
