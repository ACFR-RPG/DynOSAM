#pragma once

#include <Python.h>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <opencv2/core.hpp>

namespace bp = boost::python;
namespace np = boost::python::numpy;
namespace dyno {

// Helper function to convert a cv::Mat to a NumPy array.
// NOTE: this is for interfacing with boost python (and not with PyBind!)
np::ndarray ConvertMatToNDArray(const cv::Mat& mat);
// Helper function to convert a NumPy array to a cv::Mat.
// NOTE: this is for interfacing with boost python (and not with PyBind!)
cv::Mat ConvertNDArrayToMat(const np::ndarray& ndarr);

// RAII guard for Python GIL management.
class PyGILGuard {
 public:
  PyGILGuard();
  ~PyGILGuard();

 private:
  PyGILState_STATE gstate_;
};

// Helper function template that wraps a callable with GIL management and error
// handling.
template <typename Func>
auto executePythonCall(Func&& func) -> decltype(func()) {
  PyGILGuard guard;
  try {
    return func();
  } catch (bp::error_already_set&) {
    PyErr_Print();
    throw;
  }
}

}  // namespace dyno
