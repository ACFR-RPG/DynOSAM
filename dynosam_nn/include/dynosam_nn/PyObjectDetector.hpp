#pragma once

#include <opencv4/opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "dynosam_nn/ObjectDetector.hpp"
#include "dynosam_nn/PyObjectDetector.hpp"
#include "dynosam_nn/PyWrapper.hpp"

namespace dyno {

/**
 * @brief A C++ wrapper for the Python Engine.
 *
 * This class sets up the Python environment and wraps the Engine class.
 */
class PyObjectDetectorWrapper : public ObjectDetectionEngine {
 public:
  PyObjectDetectorWrapper();

  ~PyObjectDetectorWrapper();

  ObjectDetectionResult process(const cv::Mat& image) override;
  bool onDestruction() override;
  ObjectDetectionResult result() override;

 private:
  bp::object engine_;  // The wrapped Python Engine instance.
};

}  // namespace dyno
