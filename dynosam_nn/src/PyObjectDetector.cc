#include "dynosam_nn/PyObjectDetector.hpp"

#include <glog/logging.h>
#include <pybind11/pybind11.h>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <typeinfo>

#include "dynosam_nn/bindings/PyBoostWrapper.hpp"

namespace bp = boost::python;
namespace np = boost::python::numpy;

namespace dyno {

namespace {
// boost python doesnt know anything about the pybind stuff so we need to
// convert!!
dyno::ObjectDetectionResult convertDetectionResult(bp::object py_obj) {
  pybind11::object py_obj_borrowed =
      pybind11::reinterpret_borrow<pybind11::object>(py_obj.ptr());
  return py_obj_borrowed.cast<ObjectDetectionResult>();
}

bp::object anyVectorToBPList(const std::any& value) {
  bp::list py_list;

  if (value.type() == typeid(std::vector<int>)) {
    for (int v : std::any_cast<std::vector<int>>(value)) py_list.append(v);
  } else if (value.type() == typeid(std::vector<double>)) {
    for (double v : std::any_cast<std::vector<double>>(value))
      py_list.append(v);
  } else if (value.type() == typeid(std::vector<float>)) {
    for (float v : std::any_cast<std::vector<float>>(value)) py_list.append(v);
  } else if (value.type() == typeid(std::vector<bool>)) {
    for (bool v : std::any_cast<std::vector<bool>>(value)) py_list.append(v);
  } else if (value.type() == typeid(std::vector<std::string>)) {
    for (const std::string& v : std::any_cast<std::vector<std::string>>(value))
      py_list.append(v);
  } else {
    std::cerr << "Unsupported vector type: " << value.type().name()
              << std::endl;
  }

  return py_list;
}

bp::object anyToBPobject(const std::any& value) {
  if (value.type() == typeid(int)) {
    return bp::object(std::any_cast<int>(value));
  } else if (value.type() == typeid(double)) {
    return bp::object(std::any_cast<double>(value));
  } else if (value.type() == typeid(float)) {
    return bp::object(std::any_cast<float>(value));
  } else if (value.type() == typeid(bool)) {
    return bp::object(std::any_cast<bool>(value));
  } else if (value.type() == typeid(std::string)) {
    return bp::object(std::any_cast<std::string>(value));
  }
  // vector types
  else if (value.type() == typeid(std::vector<int>) ||
           value.type() == typeid(std::vector<double>) ||
           value.type() == typeid(std::vector<float>) ||
           value.type() == typeid(std::vector<bool>) ||
           value.type() == typeid(std::vector<std::string>)) {
    return anyVectorToBPList(value);
  } else {
    LOG(FATAL) << "Unsupported type: " << value.type().name();
    return bp::object();  // None
  }
}

}  // namespace

bp::dict makeKwargs(const PyObjectDetectorWrapper::Kwargs& args) {
  bp::dict kwargs;
  for (const auto& [key, value] : args) {
    kwargs[key.c_str()] = anyToBPobject(value);
  }
  return kwargs;
}

PyObjectDetectorWrapper::PyObjectDetectorWrapper(
    const std::string& package, const std::string& engine_class,
    const Kwargs& args) {
  // NOTE: not wrapped in PyGILGuard becuase we initalise the PythonInterepter
  // inside this constructor and PyGILGuard can only be used after initalisation
  //  PyGILGuard guard;
  try {
    // Initialize Python and NumPy if needed.
    if (!Py_IsInitialized()) {
      Py_Initialize();
    }
    np::initialize();

    // google::InitGoogleLogging("YOLODetectionEngine");

    LOG(INFO) << "Initalising PyObjectDetectorWarpper with package: " << package
              << ", engine class: " << engine_class;

    // // // Import the 'src' module and get the Engine class.
    // bp::object src_module = bp::import("dynosam_nn_py");
    // bp::object engine_cls = src_module.attr("YOLODetectionEngine");
    bp::object src_module = bp::import(package.c_str());
    bp::object engine_cls = src_module.attr(engine_class.c_str());

    // Convert to kwargs
    bp::dict kwargs = makeKwargs(args);
    // bp::object must take *args and **kwargs!
    bp::tuple args;
    engine_ = engine_cls(*args, **kwargs);

    // // Call build and start_inference on the engine instance.
    // engine_.attr("build")();
    // engine_.attr("start_inference")();
    // engine_.attr("test")(10.0);
  } catch (bp::error_already_set&) {
    PyErr_Print();
  }
}

PyObjectDetectorWrapper::~PyObjectDetectorWrapper() {
  {
    PyGILGuard guard;
    try {
      engine_.attr("on_destruction")();
    } catch (bp::error_already_set&) {
      PyErr_Print();
    }
  }
  Py_FinalizeEx();
}

ObjectDetectionEngine::Ptr PyObjectDetectorWrapper::CreateYoloDetector() {
  Kwargs kwargs;
  kwargs["verbose"] = false;
  kwargs["agnostic_nms"] = true;
  kwargs["half"] = true;
  return std::make_shared<PyObjectDetectorWrapper>(
      "dynosam_nn_py", "YOLODetectionEngine", kwargs);
}

ObjectDetectionEngine::Ptr PyObjectDetectorWrapper::CreateRTDETRDetector() {
  Kwargs kwargs;
  kwargs["verbose"] = false;
  kwargs["agnostic_nms"] = true;
  kwargs["half"] = true;
  return std::make_shared<PyObjectDetectorWrapper>(
      "dynosam_nn_py", "RTDETRDetectionEngine", kwargs);
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

ObjectDetectionResult PyObjectDetectorWrapper::result() const {
  return executePythonCall([&]() -> ObjectDetectionResult {
    bp::object result = engine_.attr("result")();
    return convertDetectionResult(result);
  });
}

}  // namespace dyno
