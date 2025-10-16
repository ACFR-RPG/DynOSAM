#pragma once

#include <opencv4/opencv2/opencv.hpp>

#include "dynosam_common/DynamicObjects.hpp"

namespace dyno {

struct SingleDetectionResult : public ObjectDetection {
  std::string class_name;
  float confidence;
};

struct ObjectDetectionResult {
  std::vector<SingleDetectionResult> detections;
  cv::Mat labelled_mask;
  bool success = false;

  friend std::ostream& operator<<(std::ostream& os,
                                  const dyno::ObjectDetectionResult& res);
};

class ObjectDetectionEngine {
 public:
  ObjectDetectionEngine() {}
  virtual ~ObjectDetectionEngine() {}

  virtual ObjectDetectionResult process(const cv::Mat& image) = 0;
  virtual bool loadModel() = 0;
  virtual bool onDestruction() = 0;
  virtual ObjectDetectionResult result() = 0;

  cv::Mat mask();
  cv::Mat colouredMask();
};

}  // namespace dyno
