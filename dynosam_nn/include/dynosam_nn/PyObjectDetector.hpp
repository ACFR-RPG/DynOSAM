#pragma once

#include <any>
#include <map>
#include <opencv4/opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "dynosam_common/utils/Macros.hpp"
#include "dynosam_nn/ObjectDetector.hpp"
#include "dynosam_nn/PyObjectDetector.hpp"
#include "dynosam_nn/bindings/PyBoostWrapper.hpp"

namespace dyno {

/**
 * @brief A C++ wrapper for the a ObjectDetectionEngine written in python!
 *
 * This class sets up the Python environment and wraps the Engine class.
 */
class PyObjectDetectorWrapper : public ObjectDetectionEngine {
 public:
  DYNO_POINTER_TYPEDEFS(PyObjectDetectorWrapper)
  using Kwargs = std::map<std::string, std::any>;

  PyObjectDetectorWrapper(const std::string& package,
                          const std::string& engine_class,
                          const Kwargs& args = Kwargs());

  ~PyObjectDetectorWrapper();

  static ObjectDetectionEngine::Ptr CreateYoloDetector();
  static ObjectDetectionEngine::Ptr CreateRTDETRDetector();

  ObjectDetectionResult process(const cv::Mat& image) override;
  bool onDestruction() override;
  ObjectDetectionResult result() const override;

 private:
  bp::object
      engine_;  // The wrapped ObjectDetectionEngine Python Engine instance.
};

}  // namespace dyno
