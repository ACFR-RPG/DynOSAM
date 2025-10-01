/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a
 copy
 *   of this software and associated documentation files (the "Software"), to
 deal
 *   in the Software without restriction, including without limitation the
 rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in
 all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE
 *   SOFTWARE.
 */

#pragma once

#include <eigen3/Eigen/Dense>

#include "dynosam_common/byte_tracker/Rect.hpp"

namespace dyno {
namespace byte_track {
class KalmanFilter {
 public:
  template <int rows, int cols>
  using Matrix = Eigen::Matrix<float, rows, cols, Eigen::RowMajor>;

  KalmanFilter(float std_weight_position = 1. / 20,
               float std_weight_velocity = 1. / 160);

  void initiate(const RectBase& measurement);

  TlwhRect predict(bool mean_eight_to_zero);

  TlwhRect update(const RectBase& measurement);

 private:
  float std_weight_position_;
  float std_weight_velocity_;

  Matrix<8, 8> motion_mat_;
  Matrix<4, 8> update_mat_;

  Matrix<1, 8> mean_;
  Matrix<8, 8> covariance_;

  void project(Matrix<1, 4>& projected_mean,
               Matrix<4, 4>& projected_covariance);

  Matrix<1, 4> rect_to_xyah(const RectBase& rect) const;

  TlwhRect xyah_to_tlwh(const Matrix<1, 4>& xyah) const;
};
}  // namespace byte_track
}  // namespace dyno
