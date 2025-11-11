/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam/factors/LandmarkMotionTernaryFactor.hpp"

namespace dyno {

LandmarkMotionTernaryFactor::LandmarkMotionTernaryFactor(
    gtsam::Key previousPointKey, gtsam::Key currentPointKey,
    gtsam::Key motionKey, gtsam::SharedNoiseModel model)
    : gtsam::NoiseModelFactor3<gtsam::Point3, gtsam::Point3, gtsam::Pose3>(
          model, previousPointKey, currentPointKey, motionKey) {}

gtsam::Vector LandmarkMotionTernaryFactor::evaluateError(
    const gtsam::Point3& previousPoint, const gtsam::Point3& currentPoint,
    const gtsam::Pose3& H, boost::optional<gtsam::Matrix&> J1,
    boost::optional<gtsam::Matrix&> J2,
    boost::optional<gtsam::Matrix&> J3) const {
  gtsam::Vector3 l2H = H.inverse() * currentPoint;
  gtsam::Vector3 expected = previousPoint - l2H;

  if (J1) {
    *J1 = (gtsam::Matrix33() << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
              .finished();
  }

  if (J2) {
    *J2 = -H.inverse().rotation().matrix();
  }

  if (J3) {
    Eigen::Matrix<double, 3, 6, Eigen::ColMajor> J;
    J.fill(0);
    J.block<3, 3>(0, 3) = gtsam::Matrix33::Identity();
    gtsam::Vector3 invHl2 = H.inverse() * currentPoint;
    J(0, 1) = invHl2(2);
    J(0, 2) = -invHl2(1);
    J(1, 0) = -invHl2(2);
    J(1, 2) = invHl2(0);
    J(2, 0) = invHl2(1);
    J(2, 1) = -invHl2(0);
    *J3 = J;
  }

  // return error vector
  return expected;
}

}  // namespace dyno
