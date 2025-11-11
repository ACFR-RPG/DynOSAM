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

#include "dynosam/factors/LandmarkMotionPoseFactor.hpp"

namespace dyno {

LandmarkMotionPoseFactor::LandmarkMotionPoseFactor(
    gtsam::Key previousPointKey, gtsam::Key currentPointKey,
    gtsam::Key previousPoseKey, gtsam::Key currentPoseKey,
    gtsam::SharedNoiseModel model)
    : Base(model, previousPointKey, currentPointKey, previousPoseKey,
           currentPoseKey) {}

gtsam::Vector LandmarkMotionPoseFactor::evaluateError(
    const gtsam::Point3& previousPoint, const gtsam::Point3& currentPoint,
    const gtsam::Pose3& previousPose, const gtsam::Pose3& currentPose,
    boost::optional<gtsam::Matrix&> J1, boost::optional<gtsam::Matrix&> J2,
    boost::optional<gtsam::Matrix&> J3,
    boost::optional<gtsam::Matrix&> J4) const {
  if (J1) {
    // error w.r.t to previous point in world
    Eigen::Matrix<double, 3, 3> df_dp_k_1 =
        gtsam::numericalDerivative41<gtsam::Vector3, gtsam::Point3,
                                     gtsam::Point3, gtsam::Pose3, gtsam::Pose3>(
            std::bind(&LandmarkMotionPoseFactor::residual,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, std::placeholders::_4),
            previousPoint, currentPoint, previousPose, currentPose);
    *J1 = df_dp_k_1;
  }

  if (J2) {
    // error w.r.t to current point in world
    Eigen::Matrix<double, 3, 3> df_dp_k =
        gtsam::numericalDerivative42<gtsam::Vector3, gtsam::Point3,
                                     gtsam::Point3, gtsam::Pose3, gtsam::Pose3>(
            std::bind(&LandmarkMotionPoseFactor::residual,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, std::placeholders::_4),
            previousPoint, currentPoint, previousPose, currentPose);
    *J2 = df_dp_k;
  }

  if (J3) {
    // error w.r.t to previous pose in world
    Eigen::Matrix<double, 3, 6> df_dL_k_1 =
        gtsam::numericalDerivative43<gtsam::Vector3, gtsam::Point3,
                                     gtsam::Point3, gtsam::Pose3, gtsam::Pose3>(
            std::bind(&LandmarkMotionPoseFactor::residual,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, std::placeholders::_4),
            previousPoint, currentPoint, previousPose, currentPose);
    *J3 = df_dL_k_1;
  }

  if (J4) {
    // error w.r.t to current pose in world
    Eigen::Matrix<double, 3, 6> df_dL_k =
        gtsam::numericalDerivative44<gtsam::Vector3, gtsam::Point3,
                                     gtsam::Point3, gtsam::Pose3, gtsam::Pose3>(
            std::bind(&LandmarkMotionPoseFactor::residual,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, std::placeholders::_4),
            previousPoint, currentPoint, previousPose, currentPose);
    *J4 = df_dL_k;
  }

  return residual(previousPoint, currentPoint, previousPose, currentPose);
}

gtsam::Vector LandmarkMotionPoseFactor::residual(
    const gtsam::Point3& previousPoint, const gtsam::Point3& currentPoint,
    const gtsam::Pose3& previousPose, const gtsam::Pose3& currentPose) {
  return gtsam::Vector3(currentPoint -
                        (currentPose * previousPose.inverse() * previousPoint));
}

}  // namespace dyno
