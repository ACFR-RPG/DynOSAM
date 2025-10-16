#include "dynosam_nn/PyObjectDetector.hpp"

#include <glog/logging.h>
#include <pybind11/pybind11.h>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>

#include "dynosam_nn/PyWrapper.hpp"

namespace bp = boost::python;
namespace np = boost::python::numpy;

namespace dyno {

// boost python doesnt know anything about the pybind stuff so we need to
// convert!!
dyno::ObjectDetectionResult convertDetectionResult(bp::object py_obj) {
  // extract the C++ object using pybind11
  // pybind11::object py_objcast(py_obj.ptr());  // wrap Boost.Python object in
  // pybind11 return extract_object_detection_result(py_objcast); return
  // py_objcast.cast<ObjectDetectionResult>();
  // 2. Create a pybind11::object from the PyObject* (borrowing the reference)
  pybind11::object py_obj_borrowed =
      pybind11::reinterpret_borrow<pybind11::object>(py_obj.ptr());
  return py_obj_borrowed.cast<ObjectDetectionResult>();
}

PyObjectDetectorWrapper::PyObjectDetectorWrapper() {
  PyGILGuard guard;
  try {
    // Initialize Python and NumPy if needed.
    if (!Py_IsInitialized()) {
      Py_Initialize();
    }
    np::initialize();

    google::InitGoogleLogging("YOLODetectionEngine");

    // // Import the 'src' module and get the Engine class.
    bp::object src_module = bp::import("dynosam_nn_py");
    bp::object engine_cls = src_module.attr("YOLODetectionEngine");

    engine_ = engine_cls();

    // // Call build and start_inference on the engine instance.
    // engine_.attr("build")();
    // engine_.attr("start_inference")();
    // engine_.attr("test")(10.0);
  } catch (bp::error_already_set&) {
    PyErr_Print();
  }
}

PyObjectDetectorWrapper::~PyObjectDetectorWrapper() {
  PyGILGuard guard;
  try {
    engine_.attr("on_destruction")();
  } catch (bp::error_already_set&) {
    PyErr_Print();
  }
}

ObjectDetectionResult PyObjectDetectorWrapper::process(const cv::Mat& image) {
  return executePythonCall([&]() -> ObjectDetectionResult {
    np::ndarray np_img = ConvertMatToNDArray(image);
    bp::object result = engine_.attr("process")(np_img);
    return convertDetectionResult(result);
  });
}

bool PyObjectDetectorWrapper::onDestruction() {
  return executePythonCall([&]() -> bool {
    bp::object result = engine_.attr("on_destruction")();
    return bp::extract<bool>(result);
  });
}

ObjectDetectionResult PyObjectDetectorWrapper::result() {
  return executePythonCall([&]() -> ObjectDetectionResult {
    bp::object result = engine_.attr("result")();
    return convertDetectionResult(result);
  });
}

}  // namespace dyno
