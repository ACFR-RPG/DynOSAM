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

#include "dynosam_common/utils/Metrics.hpp"

namespace dyno::utils {

TRErrorPair::TRErrorPair(double translation, double rot)
    : translation_(translation), rot_(rot) {}

double TRErrorPair::getRotationErrorDegrees() const {
  return dyno::rads2Deg(rot_);
}

TRErrorPair TRErrorPair::CalculatePoseError(const gtsam::Pose3& M,
                                            const gtsam::Pose3& M_hat) {
  const gtsam::Pose3 E = M.inverse() * M_hat;

  // L2 norm - ie. magnitude
  const double t_e = E.translation().norm();

  const gtsam::Matrix33 rotation_error = E.rotation().matrix();
  std::pair<gtsam::Unit3, double> axis_angle =
      gtsam::Rot3(rotation_error).axisAngle();

  // need to wrap?
  const double r_e = axis_angle.second;

  return TRErrorPair(t_e, r_e);
}

TRErrorPair TRErrorPair::CalculateRelativePoseError(
    const gtsam::Pose3& M_ref, const gtsam::Pose3& M_curr,
    const gtsam::Pose3& M_hat_ref, const gtsam::Pose3& M_hat_curr) {
  // pose change between ref and current
  const gtsam::Pose3 ref_T_curr = M_ref.inverse() * M_curr;
  const gtsam::Pose3 ref_T_hat_curr = M_hat_ref.inverse() * M_hat_curr;
  return CalculatePoseError(ref_T_curr, ref_T_hat_curr);
}

void TRErrorPairVector::push_back(double translation, double rot) {
  Base::push_back(TRErrorPair(translation, rot));
}

TRErrorPair TRErrorPairVector::average() const {
  if (Base::empty()) {
    return TRErrorPair();
  }

  TRErrorPair error_pair;

  double x = 0, y = 0;
  for (const TRErrorPair& pair : *this) {
    error_pair.translation_ += pair.translation_;

    y += std::sin(pair.rot_);
    x += std::cos(pair.rot_);
  }

  const double length = static_cast<double>(this->size());
  x /= length;
  y /= length;
  error_pair.rot_ = dyno::wrapTwoPi(std::atan2(y, x));
  error_pair.translation_ /= length;

  return error_pair;
}

}  // namespace dyno::utils
