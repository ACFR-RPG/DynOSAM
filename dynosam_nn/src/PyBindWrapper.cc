#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dynosam_nn/ObjectDetector.hpp"
#include "dynosam_nn/PyWrapper.hpp"

namespace dyno {
/**
 * @brief Python Wrapper for ObjectDetectionEngine allowing implementation in
 * pure python. Python classes should inherit directly from this class (i.e not
 * from ObjectDetectionEngine)
 *
 */
class ObjectDetectorEnginePy : public ObjectDetectionEngine {
 public:
  using ObjectDetectionEngine::ObjectDetectionEngine;

  ObjectDetectionResult process(const cv::Mat &image) override {
    PYBIND11_OVERRIDE_PURE(ObjectDetectionResult, ObjectDetectionEngine,
                           process, image);
  }

  bool loadModel() override {
    PYBIND11_OVERRIDE_PURE(bool, ObjectDetectionEngine, loadModel);
  }

  bool onDestruction() override {
    PYBIND11_OVERRIDE_PURE(bool, ObjectDetectionEngine, onDestruction);
  }

  ObjectDetectionResult result() override {
    PYBIND11_OVERRIDE_PURE(ObjectDetectionResult, ObjectDetectionEngine,
                           result);
  }
};

}  // namespace dyno

PYBIND11_MODULE(_core, m) {
  pybind11::class_<dyno::ObjectDetectionEngine, dyno::ObjectDetectorEnginePy>(
      m, "ObjectDetectionEngine")
      .def(pybind11::init<>())  // <-- now fine, trampoline makes it concrete
      .def("process", &dyno::ObjectDetectionEngine::process)
      .def("load_model", &dyno::ObjectDetectionEngine::loadModel)
      .def("on_destruction", &dyno::ObjectDetectionEngine::onDestruction)
      .def("mask", &dyno::ObjectDetectionEngine::mask);

  // ObjectDetection
  // NOTE: this class is actaully from the dynosam_common package...
  pybind11::class_<dyno::ObjectDetection>(m, "ObjectDetection")
      .def(pybind11::init<>())
      .def_readwrite("object_id", &dyno::ObjectDetection::object_id)
      .def_readwrite("bounding_box", &dyno::ObjectDetection::bounding_box);

  // cv::Rect binding
  pybind11::class_<cv::Rect>(m, "Rect")
      .def(pybind11::init<>())
      .def_readwrite("x", &cv::Rect::x)
      .def_readwrite("y", &cv::Rect::y)
      .def_readwrite("width", &cv::Rect::width)
      .def_readwrite("height", &cv::Rect::height);

  // SingleDetectionResult
  pybind11::class_<dyno::SingleDetectionResult, dyno::ObjectDetection>(
      m, "SingleDetectionResult")
      .def(pybind11::init<>())
      .def_readwrite("class_name", &dyno::SingleDetectionResult::class_name)
      .def_readwrite("confidence", &dyno::SingleDetectionResult::confidence);

  // ObjectDetectionResult
  pybind11::class_<dyno::ObjectDetectionResult>(m, "ObjectDetectionResult")
      .def(pybind11::init<>())
      .def_readwrite("detections", &dyno::ObjectDetectionResult::detections)
      .def_readwrite("success", &dyno::ObjectDetectionResult::success)
      // convert labelled_mask to numpy on access
      .def_property(
          "labelled_mask",
          [](const dyno::ObjectDetectionResult &r) {
            return dyno::ConvertMatToNDArray(r.labelled_mask);
          },
          [](dyno::ObjectDetectionResult &r, const cv::Mat &m) {
            r.labelled_mask = m;
          })
      .def("__repr__",
           [](const dyno::ObjectDetectionResult &r) {
             std::ostringstream oss;
             oss << r;
             return oss.str();
           })
      .def("__str__", [](const dyno::ObjectDetectionResult &r) {
        std::ostringstream oss;
        oss << r;
        return oss.str();
      });
}
