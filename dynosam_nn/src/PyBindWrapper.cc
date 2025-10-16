#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>  // important!

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

  bool onDestruction() override {
    PYBIND11_OVERRIDE_PURE(bool, ObjectDetectionEngine, onDestruction);
  }

  ObjectDetectionResult result() override {
    PYBIND11_OVERRIDE_PURE(ObjectDetectionResult, ObjectDetectionEngine,
                           result);
  }
};

}  // namespace dyno

namespace py = pybind11;
// Convert cv::Mat → NumPy array (shares memory)
inline py::array MatToNDArray(const cv::Mat &mat) {
  std::vector<size_t> shape;
  std::vector<size_t> strides;

  int channels = mat.channels();
  if (channels == 1) {
    shape = {(size_t)mat.rows, (size_t)mat.cols};
    strides = {(size_t)mat.step, (size_t)mat.elemSize()};
  } else {
    shape = {(size_t)mat.rows, (size_t)mat.cols, (size_t)channels};
    strides = {(size_t)mat.step, (size_t)(mat.elemSize() * channels),
               (size_t)mat.elemSize1()};
  }

  std::string format;
  switch (mat.depth()) {
    case CV_8U:
      format = py::format_descriptor<uint8_t>::format();
      break;
    case CV_8S:
      format = py::format_descriptor<int8_t>::format();
      break;
    case CV_16U:
      format = py::format_descriptor<uint16_t>::format();
      break;
    case CV_16S:
      format = py::format_descriptor<int16_t>::format();
      break;
    case CV_32S:
      format = py::format_descriptor<int32_t>::format();
      break;
    case CV_32F:
      format = py::format_descriptor<float>::format();
      break;
    case CV_64F:
      format = py::format_descriptor<double>::format();
      break;
    default:
      throw std::runtime_error("Unsupported cv::Mat type");
  }

  return py::array(py::buffer_info(mat.data, mat.elemSize1(), format,
                                   shape.size(), shape, strides));
}

// Convert NumPy array → cv::Mat
inline cv::Mat NDArrayToMat(py::array arr) {
  py::buffer_info info = arr.request();

  int rows = static_cast<int>(info.shape[0]);
  int cols = static_cast<int>(info.shape[1]);
  int channels = info.ndim == 3 ? static_cast<int>(info.shape[2]) : 1;

  int cv_type = 0;
  if (info.format == py::format_descriptor<uint8_t>::format())
    cv_type = CV_8UC(channels);
  else if (info.format == py::format_descriptor<int8_t>::format())
    cv_type = CV_8SC(channels);
  else if (info.format == py::format_descriptor<uint16_t>::format())
    cv_type = CV_16UC(channels);
  else if (info.format == py::format_descriptor<int16_t>::format())
    cv_type = CV_16SC(channels);
  else if (info.format == py::format_descriptor<int32_t>::format())
    cv_type = CV_32SC(channels);
  else if (info.format == py::format_descriptor<float>::format())
    cv_type = CV_32FC(channels);
  else if (info.format == py::format_descriptor<double>::format())
    cv_type = CV_64FC(channels);
  else
    throw std::runtime_error("Unsupported numpy dtype for cv::Mat");

  return cv::Mat(rows, cols, cv_type, info.ptr)
      .clone();  // clone ensures ownership
}

PYBIND11_MODULE(_core, m) {
  pybind11::class_<dyno::ObjectDetectionEngine, dyno::ObjectDetectorEnginePy>(
      m, "ObjectDetectionEngine")
      .def(pybind11::init<>())  // <-- now fine, trampoline makes it concrete
      .def("process", &dyno::ObjectDetectionEngine::process)
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
            // return dyno::ConvertMatToNDArray(r.labelled_mask);
            return MatToNDArray(r.labelled_mask);
          },
          [](dyno::ObjectDetectionResult &r, pybind11::array arr) {
            r.labelled_mask = NDArrayToMat(arr);
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

  m.def("get_nn_weights_path", &dyno::getNNWeightsPath);
}
