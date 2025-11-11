#include "dynosam_nn/bindings/cvnp.hpp"

// Thanks to Dan Mašek who gave me some inspiration here:
// https://stackoverflow.com/questions/60949451/how-to-send-a-cvmat-to-python-over-shared-memory

namespace cvnp {

namespace detail {
// #define DEBUG_ALLOCATOR

#ifdef DEBUG_ALLOCATOR
int nbAllocations = 0;
#endif

namespace py = pybind11;

// Translated from cv2_numpy.cpp in OpenCV source code
class CvnpAllocator : public cv::MatAllocator {
 public:
  CvnpAllocator() = default;
  ~CvnpAllocator() = default;

  // Attaches a numpy array object to a cv::Mat
  static void attach_nparray(cv::Mat& m, pybind11::array& a) {
    static CvnpAllocator instance;

    cv::UMatData* u = new cv::UMatData(&instance);
    u->data = u->origdata = (uchar*)a.mutable_data(0);
    u->size = a.size();
    u->userdata = a.inc_ref().ptr();
    u->refcount = 1;

#ifdef DEBUG_ALLOCATOR
    ++nbAllocations;
    printf("CvnpAllocator::attach_nparray(py::array) nbAllocations=%d\n",
           nbAllocations);
#endif

    m.u = u;
    m.allocator = &instance;
  }

  cv::UMatData* allocate(int dims0, const int* sizes, int type, void* data,
                         size_t* step, cv::AccessFlag flags,
                         cv::UMatUsageFlags usageFlags) const override {
    throw py::value_error(
        "CvnpAllocator::allocate \"standard\" should never happen");
    // return stdAllocator->allocate(dims0, sizes, type, data, step, flags,
    // usageFlags);
  }

  bool allocate(cv::UMatData* u, cv::AccessFlag accessFlags,
                cv::UMatUsageFlags usageFlags) const override {
    throw py::value_error(
        "CvnpAllocator::allocate \"copy\" should never happen");
    // return stdAllocator->allocate(u, accessFlags, usageFlags);
  }

  void deallocate(cv::UMatData* u) const override {
    if (!u) {
#ifdef DEBUG_ALLOCATOR
      printf("CvnpAllocator::deallocate() with null ptr!!! nbAllocations=%d\n",
             nbAllocations);
#endif
      return;
    }

    // This function can be called from anywhere, so need the GIL
    py::gil_scoped_acquire gil;
    assert(u->urefcount >= 0);
    assert(u->refcount >= 0);
    if (u->refcount == 0) {
      PyObject* o = (PyObject*)u->userdata;
      Py_XDECREF(o);
      delete u;
#ifdef DEBUG_ALLOCATOR
      --nbAllocations;
      printf("CvnpAllocator::deallocate() nbAllocations=%d\n", nbAllocations);
#endif
    } else {
#ifdef DEBUG_ALLOCATOR
      printf(
          "CvnpAllocator::deallocate() - not doing anything since urefcount=%d "
          "nbAllocations=%d\n",
          u->urefcount, nbAllocations);
#endif
    }
  }
};

py::dtype determine_np_dtype(int cv_depth) {
  for (auto format_synonym : cvnp::sTypeSynonyms)
    if (format_synonym.cv_depth == cv_depth) return format_synonym.dtype();

  std::string msg =
      "numpy does not support this OpenCV depth: " + std::to_string(cv_depth) +
      " (in determine_np_dtype)";
  throw std::invalid_argument(msg.c_str());
}

int determine_cv_depth(pybind11::dtype dt) {
  for (auto format_synonym : cvnp::sTypeSynonyms)
    if (format_synonym.np_format[0] == dt.char_())
      return format_synonym.cv_depth;

  std::string msg = std::string("OpenCV does not support this numpy format: ") +
                    dt.char_() + " (in determine_np_dtype)";
  throw std::invalid_argument(msg.c_str());
}

int determine_cv_type(const py::array& a, int depth) {
  if (a.ndim() < 2)
    throw std::invalid_argument(
        "determine_cv_type needs at least two dimensions");
  if (a.ndim() > 3)
    throw std::invalid_argument(
        "determine_cv_type needs at most three dimensions");
  if (a.ndim() == 2) return CV_MAKETYPE(depth, 1);
  // We now know that shape.size() == 3
  return CV_MAKETYPE(depth, a.shape()[2]);
}

cv::Size determine_cv_size(const py::array& a) {
  if (a.ndim() < 2)
    throw std::invalid_argument(
        "determine_cv_size needs at least two dimensions");
  return cv::Size(static_cast<int>(a.shape()[1]),
                  static_cast<int>(a.shape()[0]));
}

std::vector<std::size_t> determine_shape(const cv::Mat& m) {
  if (m.channels() == 1) {
    return {static_cast<size_t>(m.rows), static_cast<size_t>(m.cols)};
  }
  return {static_cast<size_t>(m.rows), static_cast<size_t>(m.cols),
          static_cast<size_t>(m.channels())};
}

std::vector<std::size_t> determine_strides(const cv::Mat& m) {
  if (m.channels() == 1) {
    return {
        static_cast<size_t>(m.step[0]),  // row stride (in bytes)
        static_cast<size_t>(m.step[1])   // column stride (in bytes)
    };
  }
  return {
      static_cast<size_t>(m.step[0]),     // row stride (in bytes)
      static_cast<size_t>(m.step[1]),     // column stride (in bytes)
      static_cast<size_t>(m.elemSize1())  // channel stride (in bytes)
  };
}

py::capsule make_capsule_mat(const cv::Mat& m) {
  return py::capsule(new cv::Mat(m),
                     [](void* v) { delete reinterpret_cast<cv::Mat*>(v); });
}

}  // namespace detail

pybind11::array mat_to_nparray(const cv::Mat& m) {
  return pybind11::array(
      detail::determine_np_dtype(m.depth()), detail::determine_shape(m),
      detail::determine_strides(m), m.data, detail::make_capsule_mat(m));
}

bool is_array_contiguous(const pybind11::array& a) {
  return a.flags() & pybind11::array::c_style;
}

cv::Mat nparray_to_mat(pybind11::array& a) {
  // note: empty arrays are not contiguous, but that's fine. Just
  //       make sure to not access mutable_data
  bool is_contiguous = is_array_contiguous(a);
  bool is_not_empty = a.size() != 0;
  if (!is_contiguous && is_not_empty) {
    throw std::invalid_argument(
        "cvnp::nparray_to_mat / Only contiguous numpy arrays are supported. / "
        "Please use np.ascontiguousarray() to convert your matrix");
  }

  int depth = detail::determine_cv_depth(a.dtype());
  int type = detail::determine_cv_type(a, depth);
  cv::Size size = detail::determine_cv_size(a);
  cv::Mat m(size, type, is_not_empty ? a.mutable_data(0) : nullptr);

  if (is_not_empty) {
    detail::CvnpAllocator::attach_nparray(m, a);  //, ndims, size, type, step);
  }

  return m;
}

// this version tries to handle strides and sub-matrices
// this is WIP, currently broken, and not used
cv::Mat nparray_to_mat_with_strides_broken(pybind11::array& a) {
  int depth = detail::determine_cv_depth(a.dtype());
  int type = detail::determine_cv_type(a, depth);
  cv::Size size = detail::determine_cv_size(a);

  auto buffer_info = a.request();

  // Get the array strides (convert from pybind11::ssize_t to size_t)
  std::vector<size_t> strides;
  for (auto v : buffer_info.strides) strides.push_back(static_cast<size_t>(v));

  // Get the number of dimensions
  int ndims = static_cast<int>(buffer_info.ndim);
  // if ((ndims != 2) && (ndims != 3))
  //     throw std::invalid_argument("nparray_to_mat needs support only 2 or 3
  //     dimension matrices");

  // Convert the shape (sizes) to a vector of int
  std::vector<int> sizes;
  for (auto v : buffer_info.shape) sizes.push_back(static_cast<int>(v));

  // Create the cv::Mat with the specified strides (steps)
  // We are calling this Mat constructor:
  //     Mat(const std::vector<int>& sizes, int type, void* data, const size_t*
  //     steps=0)
  cv::Mat m(sizes, type, a.mutable_data(0), strides.data());
  return m;
}

}  // namespace cvnp
