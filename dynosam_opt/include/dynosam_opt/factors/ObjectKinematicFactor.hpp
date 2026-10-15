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

#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/Expression.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace dyno {

[[deprecated("Holder over ICRA2024 paper and not used elsewhere!")]] class
    ObjectKinematicFactor
    : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3,
                                      gtsam::Pose3> {
 private:
  static gtsam::Vector6 calculateResidual(const gtsam::Pose3& H_w,
                                          const gtsam::Pose3& L_previous_world,
                                          const gtsam::Pose3& L_current_world);

 public:
  typedef boost::shared_ptr<ObjectKinematicFactor> shared_ptr;

  ObjectKinematicFactor(gtsam::Key motionKey, gtsam::Key previousObjectPoseKey,
                        gtsam::Key currentObjectPoseKey,
                        gtsam::SharedNoiseModel model);

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new ObjectKinematicFactor(*this)));
  }

  gtsam::Vector evaluateError(
      const gtsam::Pose3& H_w, const gtsam::Pose3& L_previous_world,
      const gtsam::Pose3& L_current_world,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none,
      boost::optional<gtsam::Matrix&> J3 = boost::none) const override;
};

}  // namespace dyno
