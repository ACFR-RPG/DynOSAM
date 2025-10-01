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

#include "dynosam_common/byte_tracker/KalmanFilter.hpp"

#include <cstddef>
#include <eigen3/Eigen/Dense>

#include "dynosam_common/byte_tracker/Rect.hpp"

namespace dyno {
namespace byte_track {
KalmanFilter::KalmanFilter(float std_weight_position, float std_weight_velocity)
    : std_weight_position_(std_weight_position),
      std_weight_velocity_(std_weight_velocity) {
  motion_mat_ = Eigen::MatrixXf::Identity(8, 8);
  motion_mat_.block<4, 4>(0, 4) = Eigen::Matrix4f::Identity();
  update_mat_ = Eigen::MatrixXf::Identity(4, 8);
}

void KalmanFilter::initiate(const RectBase &measurement) {
  mean_.block<1, 4>(0, 0) = rect_to_xyah(measurement);
  mean_.block<1, 4>(0, 4) = Eigen::Vector4f::Zero();

  Matrix<1, 8> std;
  std(0) = 2 * std_weight_position_ * measurement.height();
  std(1) = 2 * std_weight_position_ * measurement.height();
  std(2) = 1e-2;
  std(3) = 2 * std_weight_position_ * measurement.height();
  std(4) = 10 * std_weight_velocity_ * measurement.height();
  std(5) = 10 * std_weight_velocity_ * measurement.height();
  std(6) = 1e-5;
  std(7) = 10 * std_weight_velocity_ * measurement.height();

  Matrix<1, 8> tmp = std.array().square();
  covariance_ = tmp.asDiagonal();
}

TlwhRect KalmanFilter::predict(bool mean_eight_to_zero) {
  if (mean_eight_to_zero) mean_[7] = 0;
  Matrix<1, 8> std;
  std(0) = std_weight_position_ * mean_(3);
  std(1) = std_weight_position_ * mean_(3);
  std(2) = 1e-2;
  std(3) = std_weight_position_ * mean_(3);
  std(4) = std_weight_velocity_ * mean_(3);
  std(5) = std_weight_velocity_ * mean_(3);
  std(6) = 1e-5;
  std(7) = std_weight_velocity_ * mean_(3);

  Matrix<1, 8> tmp = std.array().square();
  Matrix<8, 8> motion_cov = tmp.asDiagonal();

  mean_ = motion_mat_ * mean_.transpose();
  covariance_ =
      motion_mat_ * covariance_ * motion_mat_.transpose() + motion_cov;
  return xyah_to_tlwh(mean_.block<1, 4>(0, 0));
}

TlwhRect KalmanFilter::update(const RectBase &measurement) {
  Matrix<1, 4> projected_mean;
  Matrix<4, 4> projected_cov;
  project(projected_mean, projected_cov);

  Eigen::Matrix<float, 4, 8> B =
      (covariance_ * update_mat_.transpose()).transpose();
  Eigen::Matrix<float, 8, 4> kalman_gain =
      (projected_cov.llt().solve(B)).transpose();
  Eigen::Matrix<float, 1, 4> innovation =
      rect_to_xyah(measurement) - projected_mean;

  const auto tmp = innovation * kalman_gain.transpose();
  mean_ = (mean_.array() + tmp.array()).matrix();
  covariance_ =
      covariance_ - kalman_gain * projected_cov * kalman_gain.transpose();
  return xyah_to_tlwh(mean_.block<1, 4>(0, 0));
}

void KalmanFilter::project(Matrix<1, 4> &projected_mean,
                           Matrix<4, 4> &projected_covariance) {
  Matrix<1, 4> std{std_weight_position_ * mean_(3),
                   std_weight_position_ * mean_(3), 1e-1,
                   std_weight_position_ * mean_(3)};

  projected_mean = update_mat_ * mean_.transpose();
  projected_covariance = update_mat_ * covariance_ * update_mat_.transpose();

  Eigen::Matrix<float, 4, 4> diag = std.asDiagonal();
  projected_covariance += diag.array().square().matrix();
}

KalmanFilter::Matrix<1, 4> KalmanFilter::rect_to_xyah(
    const RectBase &rect) const {
  return Matrix<1, 4>{rect.left() + rect.width() / 2,
                      rect.top() + rect.height() / 2,
                      rect.width() / rect.height(), rect.height()};
}

TlwhRect KalmanFilter::xyah_to_tlwh(const Matrix<1, 4> &xyah) const {
  float xyah_width = xyah(2) * xyah(3);
  return TlwhRect{xyah(1) - xyah(3) / 2, xyah(0) - xyah_width / 2, xyah_width,
                  xyah(3)};
}

}  // namespace byte_track
}  // namespace dyno
