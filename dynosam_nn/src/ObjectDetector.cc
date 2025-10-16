#include "dynosam_nn/ObjectDetector.hpp"

#include <iostream>

#include "dynosam_common/viz/Colour.hpp"

namespace dyno {

std::ostream& operator<<(std::ostream& os,
                         const dyno::ObjectDetectionResult& res) {
  os << "ObjectDetectionResult:\n";
  os << " success: " << res.success << "\n";
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

cv::Mat ObjectDetectionEngine::colouredMask() {
  const cv::Mat mask = this->mask();
  if (mask.empty()) {
    return cv::Mat();
  }
}

cv::Mat ObjectDetectionEngine::mask() {
  const ObjectDetectionResult result = this->result();
  if (!result.success) {
    return cv::Mat();
  }
  return result.labelled_mask;
}

}  // namespace dyno
