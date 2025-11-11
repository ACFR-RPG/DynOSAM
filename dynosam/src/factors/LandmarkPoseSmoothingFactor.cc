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

#include "dynosam/factors/LandmarkPoseSmoothingFactor.hpp"

#include <gtsam/geometry/Pose3.h>

namespace dyno {

gtsam::Vector LandmarkPoseSmoothingFactor::evaluateError(
    const gtsam::Pose3& pose_k_2, const gtsam::Pose3& pose_k_1,
    const gtsam::Pose3& pose_k, boost::optional<gtsam::Matrix&> J1,
    boost::optional<gtsam::Matrix&> J2,
    boost::optional<gtsam::Matrix&> J3) const {
  // TODO: do analytically!!
  if (J1) {
    // error.w.r.t pose at k-2
    Eigen::Matrix<double, 6, 6> df_dp_k_2 =
        gtsam::numericalDerivative31<gtsam::Vector6, gtsam::Pose3, gtsam::Pose3,
                                     gtsam::Pose3>(
            std::bind(&LandmarkPoseSmoothingFactor::residual,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3),
            pose_k_2, pose_k_1, pose_k);
    *J1 = df_dp_k_2;
  }

  if (J2) {
    // error.w.r.t pose at k-1
    Eigen::Matrix<double, 6, 6> df_dp_k_1 =
        gtsam::numericalDerivative32<gtsam::Vector6, gtsam::Pose3, gtsam::Pose3,
                                     gtsam::Pose3>(
            std::bind(&LandmarkPoseSmoothingFactor::residual,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3),
            pose_k_2, pose_k_1, pose_k);
    *J2 = df_dp_k_1;
  }

  if (J3) {
    // error.w.r.t pose at k
    Eigen::Matrix<double, 6, 6> df_dp_k =
        gtsam::numericalDerivative33<gtsam::Vector6, gtsam::Pose3, gtsam::Pose3,
                                     gtsam::Pose3>(
            std::bind(&LandmarkPoseSmoothingFactor::residual,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3),
            pose_k_2, pose_k_1, pose_k);
    *J3 = df_dp_k;
  }

  return residual(pose_k_2, pose_k_1, pose_k);
}

gtsam::Vector LandmarkPoseSmoothingFactor::residual(
    const gtsam::Pose3& pose_k_2, const gtsam::Pose3& pose_k_1,
    const gtsam::Pose3& pose_k) {
  const gtsam::Pose3 k_2_H_k_1 = pose_k_1 * pose_k_2.inverse();
  const gtsam::Pose3 k_1_H_k = pose_k * pose_k_1.inverse();

  gtsam::Pose3 hx = gtsam::traits<gtsam::Pose3>::Between(
      k_2_H_k_1, k_1_H_k, boost::none, boost::none);  // h(x)
  static const gtsam::Pose3 Z = gtsam::Pose3::Identity();

  return gtsam::traits<gtsam::Pose3>::Local(Z, hx);
}

}  // namespace dyno
