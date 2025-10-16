#include "dynosam_nn/PyWrapper.hpp"

#include <glog/logging.h>

#include <opencv4/opencv2/opencv.hpp>

#include "ament_index_cpp/get_package_share_directory.hpp"

namespace bp = boost::python;
namespace np = boost::python::numpy;
namespace fs = std::filesystem;

namespace dyno {

fs::path getNNWeightsPath() {
  return fs::path(ament_index_cpp::get_package_share_directory("dynosam_nn")) /
         "weights";
}

// Conversion functions taken from:
// https://gist.github.com/aFewThings/c79e124f649ea9928bfc7bb8827f1a1c

// Helper function to convert a cv::Mat to a NumPy array.
np::ndarray ConvertMatToNDArray(const cv::Mat& mat) {
  bp::tuple shape = bp::make_tuple(mat.rows, mat.cols, mat.channels());
  bp::tuple stride =
      bp::make_tuple(mat.channels() * mat.cols * sizeof(uchar),
                     mat.channels() * sizeof(uchar), sizeof(uchar));
  np::dtype dt = np::dtype::get_builtin<uchar>();
  np::ndarray ndImg = np::from_data(mat.data, dt, shape, stride, bp::object());

  return ndImg;
}

// Helper function to convert a NumPy array to a cv::Mat.
// Support uint8 type images with 2 or 3 dimensions
cv::Mat ConvertNDArrayToMat(const np::ndarray& ndarr) {
  const Py_intptr_t* shape = ndarr.get_shape();
  char* dtype_str = bp::extract<char*>(bp::str(ndarr.get_dtype()));

  assert(dtype_str != nullptr);
  assert(shape != nullptr);

  int rows = shape[0];
  int cols = shape[1];

  int channel = 1;  // default to 1 (grayscale)

  // determine number of channels from dimensions
  int ndim = ndarr.get_nd();
  if (ndim == 3) {
    channel = shape[2];
  } else if (ndim != 2) {
    std::cerr << "Unsupported ndarray dimensions: " << ndim << std::endl;
    return cv::Mat();
  }

  int depth;
  // determine the depth type
  if (!strcmp(dtype_str, "uint8")) {
    depth = CV_8U;
  } else {
    std::cerr << "Unsupported dtype: " << dtype_str << std::endl;
    return cv::Mat();
  }

  // verify channel count
  if (channel != 1 && channel != 3) {
    std::cerr << "Unsupported number of channels: " << channel << std::endl;
    return cv::Mat();
  }

  int type = CV_MAKETYPE(depth, channel);

  cv::Mat mat(rows, cols, type);
  memcpy(mat.data, ndarr.get_data(), sizeof(uchar) * rows * cols * channel);

  return mat;
}

// Implementation of PyGILGuard
PyGILGuard::PyGILGuard() { gstate_ = PyGILState_Ensure(); }

PyGILGuard::~PyGILGuard() { PyGILState_Release(gstate_); }

}  // namespace dyno
