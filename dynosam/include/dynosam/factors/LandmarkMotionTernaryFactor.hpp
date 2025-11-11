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

#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace dyno {

/**
 * @brief Implements the landmark motion factor that models the displacement of
 * a tracked point, i, on a rigid body, j with the absolute motion H.
 *
 * Cost resdidual is defined as: ^wm_k - ^w_{k-1}H_k * ^wm_{k-1}
 *
 */
class LandmarkMotionTernaryFactor
    : public gtsam::NoiseModelFactor3<gtsam::Point3, gtsam::Point3,
                                      gtsam::Pose3> {
 public:
  typedef boost::shared_ptr<LandmarkMotionTernaryFactor> shared_ptr;
  typedef LandmarkMotionTernaryFactor This;

  /**
   * @brief Constructs a new factor
   *
   * @param previousPointKey gtsam::Key for point ^wm_{k-1}
   * @param currentPointKey gtsam::Key for ^wm_k
   * @param motionKey gtsam::Key for absolute motion ^w_{k-1}H_k
   * @param model gtsam::SharedNoiseModel with dimension 3.
   */
  LandmarkMotionTernaryFactor(gtsam::Key previousPointKey,
                              gtsam::Key currentPointKey, gtsam::Key motionKey,
                              gtsam::SharedNoiseModel model);

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }

  gtsam::Vector evaluateError(
      const gtsam::Point3& previousPoint, const gtsam::Point3& currentPoint,
      const gtsam::Pose3& H, boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none,
      boost::optional<gtsam::Matrix&> J3 = boost::none) const override;

  inline gtsam::Key PreviousPointKey() const { return key1(); }
  inline gtsam::Key CurrentPointKey() const { return key2(); }
  inline gtsam::Key MotionKey() const { return key3(); }
};

}  // namespace dyno
