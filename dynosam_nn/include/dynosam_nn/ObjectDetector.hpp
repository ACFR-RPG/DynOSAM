#pragma once

#include <opencv4/opencv2/opencv.hpp>

#include "dynosam_common/DynamicObjects.hpp"
#include "dynosam_common/utils/Macros.hpp"

namespace dyno {

struct SingleDetectionResult : public ObjectDetection {
  std::string class_name;
  float confidence;
};

struct ObjectDetectionResult {
  std::vector<SingleDetectionResult> detections;
  cv::Mat labelled_mask;
  cv::Mat input_image;  // Should be a 3 channel RGB image. Should always be set

  cv::Mat colouredMask() const;
  //! number of detections
  inline size_t num() const { return detections.size(); }
  ObjectIds objectIds() const;

  friend std::ostream& operator<<(std::ostream& os,
                                  const dyno::ObjectDetectionResult& res);
};

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
