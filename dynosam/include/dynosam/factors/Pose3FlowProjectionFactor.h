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
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace dyno {

// expects the camera to have a calibration()
template <class CALIBRATION = gtsam::Cal3_S2>
class Pose3FlowProjectionFactor
    : public gtsam::NoiseModelFactor2<gtsam::Point2, gtsam::Pose3> {
 public:
  using Calibration = CALIBRATION;
  using This = Pose3FlowProjectionFactor<Calibration>;
  using Base =
      gtsam::NoiseModelFactor2<gtsam::Point2,
                               gtsam::Pose3>;  // keypoint to camera pose

  using shared_ptr = boost::shared_ptr<This>;

  Pose3FlowProjectionFactor(gtsam::Key optical_flow_key,
                            gtsam::Key camera_pose_key,
                            const gtsam::Point2& keypoint_previous,
                            double depth, const gtsam::Pose3& pose_previous,
                            const Calibration& calibration,
                            gtsam::SharedNoiseModel model)
      : Base(model, optical_flow_key, camera_pose_key),
        keypoint_previous_(keypoint_previous),
        depth_(depth),
        pose_previous_(pose_previous),
        calibration_(calibration) {}

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }

  gtsam::Vector evaluateError(
      const gtsam::Point2& optical_flow, const gtsam::Pose3& camera_pose,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none) const override {
    auto I = gtsam::traits<gtsam::Pose3>::Identity();
    gtsam::PinholeCamera<Calibration> previous_camera(I, calibration_);

    const double fx = calibration_.fx();
    const double fy = calibration_.fy();
    // project the ref keypoint into the world frame given the camera at k-1
    gtsam::Point3 P_W = pose_previous_ *
                        previous_camera.backproject(keypoint_previous_, depth_);

    // camera at k using the same calibration
    gtsam::PinholeCamera<Calibration> camera_current(I, calibration_);

    gtsam::Point2 predicted_keypoint = keypoint_previous_ + optical_flow;
    // project P_w into the current frame frame which should be the predicted
    // one
    gtsam::Point3 P_curr = camera_pose.inverse() * P_W;

    try {
      gtsam::Point2 predicted_keypoint_from_projection =
          camera_current.project2(P_curr);

      if (J1) {
        *J1 = gtsam::Matrix22::Identity();
      }

      if (J2) {
        // point in the current camera frame
        const double x = P_curr(0);
        const double y = P_curr(1);
        const double z = P_curr(2);
        const double z_2 = z * z;

        gtsam::Matrix26 H;
        H(0, 0) = x * y / z_2 * fx;
        H(0, 1) = -(1 + (x * x / z_2)) * fx;
        H(0, 2) = y / z * fx;
        H(0, 3) = -1. / z * fx;
        H(0, 4) = 0;
        H(0, 5) = x / z_2 * fx;

        H(1, 0) = (1 + y * y / z_2) * fy;
        H(1, 1) = -x * y / z_2 * fy;
        H(1, 2) = -x / z * fy;
        H(1, 3) = 0;
        H(1, 4) = -1. / z * fy;
        H(1, 5) = y / z_2 * fy;

        *J2 = -1.0 * H;
      }

      return predicted_keypoint - predicted_keypoint_from_projection;
    } catch (gtsam::CheiralityException&) {
      if (J1) *J1 = gtsam::Matrix::Zero(2, 2);
      if (J2) *J2 = gtsam::Matrix::Zero(2, 6);
      return gtsam::Vector2::Constant(2.0 * calibration_.fx());
    }
  }

 private:
  gtsam::Point2 keypoint_previous_;  //! Observed keypoint, the origin of the
                                     //! optical flow (in the ref frame)
  double
      depth_;  //! Observed depth measurement of the keypoint (in the ref frame)
  gtsam::Pose3 pose_previous_;  //! Pose of the reference frame
  Calibration calibration_;
};

}  // namespace dyno
