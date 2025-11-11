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

#include "dynosam/factors/ObjectKinematicFactor.hpp"

#include <gtsam/base/numericalDerivative.h>

namespace dyno {

ObjectKinematicFactor::ObjectKinematicFactor(gtsam::Key motionKey,
                                             gtsam::Key previousObjectPoseKey,
                                             gtsam::Key currentObjectPoseKey,
                                             gtsam::SharedNoiseModel model)
    : gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>(
          model, motionKey, previousObjectPoseKey, currentObjectPoseKey) {}

gtsam::Vector6 ObjectKinematicFactor::calculateResidual(
    const gtsam::Pose3& H_w, const gtsam::Pose3& L_previous_world,
    const gtsam::Pose3& L_current_world) {
  gtsam::Pose3 propogated = H_w * L_previous_world;
  gtsam::Pose3 Hx = L_current_world.between(propogated);
  return gtsam::Pose3::Identity().localCoordinates(Hx);
}

gtsam::Vector ObjectKinematicFactor::evaluateError(
    const gtsam::Pose3& H_w, const gtsam::Pose3& L_previous_world,
    const gtsam::Pose3& L_current_world, boost::optional<gtsam::Matrix&> J1,
    boost::optional<gtsam::Matrix&> J2,
    boost::optional<gtsam::Matrix&> J3) const {
  if (J1) {
    *J1 = gtsam::numericalDerivative31<gtsam::Vector6, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Pose3>(
        std::bind(&ObjectKinematicFactor::calculateResidual,
                  std::placeholders::_1, std::placeholders::_2,
                  std::placeholders::_3),
        H_w, L_previous_world, L_current_world);
  }

  if (J2) {
    *J2 = gtsam::numericalDerivative32<gtsam::Vector6, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Pose3>(
        std::bind(&ObjectKinematicFactor::calculateResidual,
                  std::placeholders::_1, std::placeholders::_2,
                  std::placeholders::_3),
        H_w, L_previous_world, L_current_world);
  }

  if (J3) {
    *J3 = gtsam::numericalDerivative33<gtsam::Vector6, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Pose3>(
        std::bind(&ObjectKinematicFactor::calculateResidual,
                  std::placeholders::_1, std::placeholders::_2,
                  std::placeholders::_3),
        H_w, L_previous_world, L_current_world);
  }

  return calculateResidual(H_w, L_previous_world, L_current_world);
}

}  // namespace dyno
