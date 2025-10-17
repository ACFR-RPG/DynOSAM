#pragma once
#include <pybind11/numpy.h>
#include <stdlib.h>

#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <stdexcept>
#include <vector>

#include "dynosam_nn/bindings/cvnp_synonyms.hpp"
//
// Explicit transformers between cv::Mat / cv::Matx and numpy.ndarray, with
// *possibly*/ shared memory Also see automatic casters in the namespace
// pybind11:detail below
//
namespace cvnp {
//
// Public interface
//

// For cv::Mat (*with* shared memory)
pybind11::array mat_to_nparray(const cv::Mat &m);
cv::Mat nparray_to_mat(pybind11::array &a);

// For cv::Matx (*without* shared memory)
template <typename _Tp, int _rows, int _cols>
pybind11::array matx_to_nparray(const cv::Matx<_Tp, _rows, _cols> &m);
template <typename _Tp, int _rows, int _cols>
void nparray_to_matx(pybind11::array &a,
                     cv::Matx<_Tp, _rows, _cols> &out_matrix);

//
// Private details and implementations below
//
namespace detail {
template <typename _Tp, int _rows, int _cols>
pybind11::capsule make_capsule_matx(const cv::Matx<_Tp, _rows, _cols> &m) {
  return pybind11::capsule(new cv::Matx<_Tp, _rows, _cols>(m), [](void *v) {
    delete reinterpret_cast<cv::Matx<_Tp, _rows, _cols> *>(v);
  });
}
}  // namespace detail

template <typename _Tp, int _rows, int _cols>
pybind11::array matx_to_nparray(const cv::Matx<_Tp, _rows, _cols> &m) {
  return pybind11::array(
      pybind11::dtype::of<_Tp>(), std::vector<std::size_t>{_rows, _cols},
      std::vector<std::size_t>{_cols * sizeof(_Tp), sizeof(_Tp)}  // Strides
      ,
      m.val);
}

template <typename _Tp, int _rows, int _cols>
void nparray_to_matx(pybind11::array &a,
                     cv::Matx<_Tp, _rows, _cols> &out_matrix) {
  size_t mat_size = (size_t)(_rows * _cols);
  if (a.size() != mat_size) throw std::runtime_error("Bad size");

  _Tp *arrayValues = (_Tp *)a.data(0);
  for (size_t i = 0; i < mat_size; ++i) out_matrix.val[i] = arrayValues[i];
}
}  // namespace cvnp

//
// 1. Casts with shared memory between {cv::Mat, cv::Matx, cv::Vec} and
// numpy.ndarray
//
// 2. Casts without shared memory between {cv::Size, cv::Point, cv::Point3} and
// python tuples
//
namespace pybind11 {
namespace detail {
//
// Cast between cv::Mat and numpy.ndarray
// The cast between cv::Mat and numpy.ndarray works
//   - *with* shared memory when going from C++ to Python
//   - *with* shared memory when going from Python to C++
//   any modification to the Matrix size, type, and values is immediately
//   impacted on both sides.
template <>
struct type_caster<cv::Mat> {
 public:
  PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

  /**
   * Conversion part 1 (Python->C++):
   * Return false upon failure.
   * The second argument indicates whether implicit conversions should be
   * applied.
   */
  bool load(handle src, bool) {
    if (!isinstance<array>(src)) return false;

    auto a = reinterpret_borrow<array>(src);
    auto new_mat = cvnp::nparray_to_mat(a);
    value = new_mat;
    return true;
  }

  /**
   * Conversion part 2 (C++ -> Python):
   * The second and third arguments are used to indicate the return value policy
   * and parent object (for ``return_value_policy::reference_internal``) and are
   * generally ignored by implicit casters.
   */
  static handle cast(const cv::Mat &m, return_value_policy, handle defval) {
    auto a = cvnp::mat_to_nparray(m);
    return a.release();
  }
};

//
// Cast between cv::Mat_ and numpy.ndarray
// The cast between cv::Mat_ and numpy.ndarray works
//   - *with* shared memory when going from C++ to Python
//   - *with* shared memory when going from Python to C++
//   any modification to the Matrix size, type, and values is immediately
//   impacted on both sides.
template <typename _Tp>
struct type_caster<cv::Mat_<_Tp>> {
  using MatTp = cv::Mat_<_Tp>;

 public:
  PYBIND11_TYPE_CASTER(MatTp, _("numpy.ndarray"));

  /**
   * Conversion part 1 (Python->C++):
   * Return false upon failure.
   * The second argument indicates whether implicit conversions should be
   * applied.
   */
  bool load(handle src, bool) {
    if (!isinstance<array>(src)) return false;

    auto a = reinterpret_borrow<array>(src);
    auto new_mat = cvnp::nparray_to_mat(a);
    value = new_mat;
    return true;
  }

  /**
   * Conversion part 2 (C++ -> Python):
   * The second and third arguments are used to indicate the return value policy
   * and parent object (for ``return_value_policy::reference_internal``) and are
   * generally ignored by implicit casters.
   */
  static handle cast(const MatTp &m, return_value_policy, handle defval) {
    auto a = cvnp::mat_to_nparray(m);
    return a.release();
  }
};

// Cast between cv::Matx<_rows,_cols> (aka Matx33d, Matx21d, etc) and
// numpy.ndarray *without* shared memory.
template <typename _Tp, int _rows, int _cols>
struct type_caster<cv::Matx<_Tp, _rows, _cols>> {
  using Matxxx = cv::Matx<_Tp, _rows, _cols>;

 public:
  PYBIND11_TYPE_CASTER(Matxxx, _("numpy.ndarray"));

  // Conversion part 1 (Python->C++)
  bool load(handle src, bool) {
    if (!isinstance<array>(src)) return false;

    auto a = reinterpret_borrow<array>(src);
    cvnp::nparray_to_matx<_Tp, _rows, _cols>(a, value);
    return true;
  }

  // Conversion part 2 (C++ -> Python)
  static handle cast(const Matxxx &m, return_value_policy, handle defval) {
    auto a = cvnp::matx_to_nparray<_Tp, _rows, _cols>(m);
    return a.release();
  }
};

// Cast between cv::Vec<_rows> and numpy.ndarray
// *without* shared memory.
template <typename _Tp, int _rows>
struct type_caster<cv::Vec<_Tp, _rows>> {
  using Vecxxx = cv::Vec<_Tp, _rows>;

 public:
  PYBIND11_TYPE_CASTER(Vecxxx, _("numpy.ndarray"));

  // Conversion part 1 (Python->C++)
  bool load(handle src, bool) {
    if (!isinstance<array>(src)) return false;

    auto a = reinterpret_borrow<array>(src);
    cvnp::nparray_to_matx<_Tp, _rows, 1>(a, value);
    return true;
  }

  // Conversion part 2 (C++ -> Python)
  static handle cast(const Vecxxx &m, return_value_policy, handle defval) {
    auto a = cvnp::matx_to_nparray<_Tp, _rows, 1>(m);
    return a.release();
  }
};

//
// Cast between cv::Size and a simple python tuple.
// No shared memory, you cannot modify the width or the height without
// transferring the whole Size from C++, or the tuple from python
//
template <typename _Tp>
struct type_caster<cv::Size_<_Tp>> {
  using SizeTp = cv::Size_<_Tp>;

 public:
  PYBIND11_TYPE_CASTER(SizeTp, _("tuple"));

  // Conversion part 1 (Python->C++, i.e tuple -> Size)
  bool load(handle src, bool) {
    if (!isinstance<pybind11::tuple>(src)) return false;

    auto tuple = pybind11::reinterpret_borrow<pybind11::tuple>(src);
    if (tuple.size() != 2)
      throw std::invalid_argument("Size should be in a tuple of size 2");

    SizeTp r;
    r.width = tuple[0].cast<_Tp>();
    r.height = tuple[1].cast<_Tp>();

    value = r;
    return true;
  }

  // Conversion part 2 (C++ -> Python, i.e Size -> tuple)
  static handle cast(const SizeTp &value, return_value_policy, handle defval) {
    auto result = pybind11::make_tuple(value.width, value.height);
    return result.release();
  }
};

//
// Cast between cv::Point and a simple python tuple
// No shared memory, you cannot modify x or y without
// transferring the whole Point from C++, or the tuple from python
//
template <typename _Tp>
struct type_caster<cv::Point_<_Tp>> {
  using PointTp = cv::Point_<_Tp>;

 public:
  PYBIND11_TYPE_CASTER(PointTp, _("tuple"));

  // Conversion part 1 (Python->C++)
  bool load(handle src, bool) {
    if (!isinstance<pybind11::tuple>(src)) return false;

    auto tuple = pybind11::reinterpret_borrow<pybind11::tuple>(src);
    if (tuple.size() != 2)
      throw std::invalid_argument("Point should be in a tuple of size 2");

    PointTp r;
    r.x = tuple[0].cast<_Tp>();
    r.y = tuple[1].cast<_Tp>();

    value = r;
    return true;
  }

  // Conversion part 2 (C++ -> Python)
  static handle cast(const PointTp &value, return_value_policy, handle defval) {
    auto result = pybind11::make_tuple(value.x, value.y);
    return result.release();
  }
};

//
// Point3
// No shared memory
//
template <typename _Tp>
struct type_caster<cv::Point3_<_Tp>> {
  using PointTp = cv::Point3_<_Tp>;

 public:
  PYBIND11_TYPE_CASTER(PointTp, _("tuple"));

  // Conversion part 1 (Python->C++)
  bool load(handle src, bool) {
    if (!isinstance<pybind11::tuple>(src)) return false;

    auto tuple = pybind11::reinterpret_borrow<pybind11::tuple>(src);
    if (tuple.size() != 3)
      throw std::invalid_argument("Point3 should be in a tuple of size 3");

    PointTp r;
    r.x = tuple[0].cast<_Tp>();
    r.y = tuple[1].cast<_Tp>();
    r.z = tuple[2].cast<_Tp>();

    value = r;
    return true;
  }

  // Conversion part 2 (C++ -> Python)
  static handle cast(const PointTp &value, return_value_policy, handle defval) {
    auto result = pybind11::make_tuple(value.x, value.y, value.z);
    return result.release();
  }
};

//
// Scalar
// No shared memory
//
template <typename _Tp>
struct type_caster<cv::Scalar_<_Tp>> {
  using ScalarTp = cv::Scalar_<_Tp>;

 public:
  PYBIND11_TYPE_CASTER(ScalarTp, _("tuple"));

  // Conversion part 1 (Python->C++)
  bool load(handle src, bool) {
    if (!isinstance<pybind11::tuple>(src)) return false;

    auto tuple = pybind11::reinterpret_borrow<pybind11::tuple>(src);
    const auto tupleSize = tuple.size();
    if (tupleSize > 4)
      throw std::invalid_argument(
          "Scalar should be a tuple with at most 4 elements. Got " +
          std::to_string(tupleSize));

    ScalarTp r;
    if (tupleSize == 1)
      r = ScalarTp(tuple[0].cast<_Tp>());
    else if (tupleSize == 2)
      r = ScalarTp(tuple[0].cast<_Tp>(), tuple[1].cast<_Tp>());
    else if (tupleSize == 3)
      r = ScalarTp(tuple[0].cast<_Tp>(), tuple[1].cast<_Tp>(),
                   tuple[2].cast<_Tp>());
    else if (tupleSize == 4)
      r = ScalarTp(tuple[0].cast<_Tp>(), tuple[1].cast<_Tp>(),
                   tuple[2].cast<_Tp>(), tuple[3].cast<_Tp>());

    value = r;
    return true;
  }

  // Conversion part 2 (C++ -> Python)
  static handle cast(const ScalarTp &value, return_value_policy,
                     handle defval) {
    auto result = pybind11::make_tuple(value[0], value[1], value[2], value[3]);
    return result.release();
  }
};

//
// Rect_
// No shared memory
//
template <typename _Tp>
struct type_caster<cv::Rect_<_Tp>> {
  using RectTp = cv::Rect_<_Tp>;

 public:
  PYBIND11_TYPE_CASTER(RectTp, _("tuple"));

  // Conversion part 1 (Python->C++)
  bool load(handle src, bool) {
    if (!isinstance<pybind11::tuple>(src)) return false;

    auto tuple = pybind11::reinterpret_borrow<pybind11::tuple>(src);
    const auto tupleSize = tuple.size();
    if (tupleSize != 4)
      throw std::invalid_argument(
          "Rect should be a tuple with 4 elements. Got " +
          std::to_string(tupleSize));

    RectTp r;
    r.x = tuple[0].cast<_Tp>();
    r.y = tuple[1].cast<_Tp>();
    r.width = tuple[2].cast<_Tp>();
    r.height = tuple[3].cast<_Tp>();

    value = r;
    return true;
  }

  // Conversion part 2 (C++ -> Python)
  static handle cast(const RectTp &value, return_value_policy, handle defval) {
    auto result =
        pybind11::make_tuple(value.x, value.y, value.width, value.height);
    return result.release();
  }
};

}  // namespace detail
}  // namespace pybind11
