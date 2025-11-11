#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>  // important!
#include <pybind11/stl_bind.h>

#include <vector>

#include "dynosam_common/utils/OpenCVUtils.hpp"
#include "dynosam_nn/ModelConfig.hpp"
#include "dynosam_nn/ObjectDetector.hpp"
#include "dynosam_nn/bindings/PyBoostWrapper.hpp"
#include "dynosam_nn/bindings/cvnp.hpp"

namespace dyno {

/**
 * @brief Trampoline class allowing additional python specific virtual functions
 * to be defined
 *
 */
class PyObjectDetectionEngineTrampoline : public ObjectDetectionEngine {
 public:
  using ObjectDetectionEngine::ObjectDetectionEngine;
  virtual bool onDestruction() = 0;
};

/**
 * @brief Python Wrapper for ObjectDetectionEngine allowing implementation in
 * pure python. Python classes should inherit directly from this class (i.e not
 * from ObjectDetectionEngine)
 *
 */
class ObjectDetectorEnginePy : public PyObjectDetectionEngineTrampoline {
 public:
  using Base = PyObjectDetectionEngineTrampoline;
  using Base::Base;

  ObjectDetectionResult process(const cv::Mat &image) override {
    PYBIND11_OVERRIDE_PURE(ObjectDetectionResult, Base, process, image);
  }

  ObjectDetectionResult result() const override {
    PYBIND11_OVERRIDE_PURE(ObjectDetectionResult, Base, result);
  }

  bool onDestruction() override {
    PYBIND11_OVERRIDE_PURE(bool, Base, onDestruction);
  }
};

}  // namespace dyno

// See https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html
PYBIND11_MAKE_OPAQUE(std::vector<dyno::SingleDetectionResult>);

PYBIND11_MODULE(_core, m) {
  pybind11::class_<dyno::PyObjectDetectionEngineTrampoline,
                   dyno::ObjectDetectorEnginePy>(m, "ObjectDetectionEngine")
      .def(pybind11::init<>())  // <-- now fine, trampoline makes it concrete
      // we need to translate this to a np.dtype (ie. np.int32!!)
      //  .def_readonly_static("MaskDType",
      //  &dyno::ObjectDetectionEngine::MaskDType)
      .def("process", &dyno::ObjectDetectionEngine::process)
      .def("on_destruction",
           &dyno::PyObjectDetectionEngineTrampoline::onDestruction)
      .def("mask", &dyno::ObjectDetectionEngine::mask)
      .def("coloured_mask", &dyno::ObjectDetectionEngine::colouredMask)
      .def("result", &dyno::ObjectDetectionEngine::result);

  // ObjectDetection
  // NOTE: this class is actaully from the dynosam_common package...
  pybind11::class_<dyno::ObjectDetection>(m, "ObjectDetection")
      .def(pybind11::init<>())
      .def_readwrite("bounding_box", &dyno::ObjectDetection::bounding_box)
      .def_readwrite("class_name", &dyno::ObjectDetection::class_name)
      .def_readwrite("confidence", &dyno::ObjectDetection::confidence);

  // SingleDetectionResult
  pybind11::class_<dyno::SingleDetectionResult, dyno::ObjectDetection>(
      m, "SingleDetectionResult")
      .def(pybind11::init<>())
      .def_readwrite("object_id", &dyno::SingleDetectionResult::object_id)
      .def_readwrite("well_tracked", &dyno::SingleDetectionResult::well_tracked)
      .def("is_valid", &dyno::SingleDetectionResult::isValid);

  pybind11::bind_vector<std::vector<dyno::SingleDetectionResult>>(
      m, "SingleDetectionResults");

  // ObjectDetectionResult
  pybind11::class_<dyno::ObjectDetectionResult>(m, "ObjectDetectionResult")
      .def(pybind11::init<>())
      .def_readwrite("detections", &dyno::ObjectDetectionResult::detections)
      .def_readwrite("labelled_mask",
                     &dyno::ObjectDetectionResult::labelled_mask)
      .def_readwrite("input_image", &dyno::ObjectDetectionResult::input_image)
      .def("coloured_mask", &dyno::ObjectDetectionResult::colouredMask)
      .def("num", &dyno::ObjectDetectionResult::num)
      .def("object_ids", &dyno::ObjectDetectionResult::objectIds)
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

  m.def("get_nn_weights_path", &dyno::getNNWeightsPath);
  // mask must be CV_32SC1!!
  m.def("mask_to_rgb", [](const cv::Mat &mask, const cv::Mat &rgb) -> cv::Mat {
    return dyno::utils::labelMaskToRGB(
        mask,
        dyno::background_label,  // from dynosam_common
        rgb);
  });
}
