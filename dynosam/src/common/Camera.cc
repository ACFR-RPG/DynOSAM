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

#include "dynosam/common/Camera.hpp"

#include <glog/logging.h>

namespace dyno {

Camera::Camera(const CameraParams& camera_params)
    : camera_params_(camera_params) {
  camera_impl_ = std::make_unique<CameraImpl>(
      gtsam::Pose3::Identity(),  // this should be the body pose cam from the
                                 // calibration but currently we want everything
                                 // in camera frame,
      camera_params_.constructGtsamCalibration<CalibrationType>());

  CHECK(camera_impl_);
}

void Camera::project(const Landmark& lmk, Keypoint* kpt) const {
  CHECK_NOTNULL(kpt);
  // I believe this project call in gtsam is quite inefficient as it goes
  // through a cascade of calls... Only useful if you want gradients I guess...
  *kpt = camera_impl_->project2(lmk);
}

void Camera::project(const Landmarks& lmks, Keypoints* kpts) const {
  CHECK_NOTNULL(kpts)->clear();
  const auto& n_lmks = lmks.size();
  kpts->resize(n_lmks);
  // Can be greatly optimized with matrix mult or vectorization
  for (size_t i = 0u; i < n_lmks; i++) {
    project(lmks[i], &(*kpts)[i]);
  }
}

bool Camera::isKeypointContained(const Keypoint& kpt, Depth depth) const {
  return isKeypointContained(kpt) && depth > 0.0;
}

bool Camera::isKeypointContained(const Keypoint& kpt) const {
  return kpt(0) >= 0.0 && kpt(0) < camera_params_.ImageWidth() &&
         kpt(1) >= 0.0 && kpt(1) < camera_params_.ImageHeight();
}

void Camera::backProject(const Keypoints& kps, const Depths& depths,
                         Landmarks* lmks) const {
  CHECK_NOTNULL(lmks)->clear();
  lmks->reserve(kps.size());
  CHECK_EQ(kps.size(), depths.size());
  CHECK(camera_impl_);
  for (size_t i = 0u; i < kps.size(); i++) {
    Landmark lmk;
    backProject(kps[i], depths[i], &lmk);
    lmks->push_back(lmk);
  }
}
void Camera::backProject(const Keypoint& kp, const Depth& depth,
                         Landmark* lmk) const {
  CHECK(lmk);
  backProject(kp, depth, lmk, gtsam::Pose3::Identity());
}

// TODO: no tests
void Camera::backProject(const Keypoint& kp, const Depth& depth, Landmark* lmk,
                         const gtsam::Pose3& X_world) const {
  CHECK(lmk);
  *lmk = X_world * camera_impl_->backproject(kp, depth);
}

void Camera::backProjectFromZ(const Keypoint& kp, const double Z, Landmark* lmk,
                              gtsam::OptionalJacobian<3, 3> Dp) const {
  CHECK(lmk);
  backProjectFromZ(kp, Z, lmk, gtsam::Pose3::Identity(), Dp);
}

void Camera::backProjectFromZ(const Keypoint& kp, const double Z, Landmark* lmk,
                              const gtsam::Pose3& X_world,
                              gtsam::OptionalJacobian<3, 3> Dp) const {
  CHECK(lmk);
  const auto calibration = camera_impl_->calibration();
  // Convert (distorted) image coordinates uv to intrinsic coordinates xy
  Eigen::Matrix<double, 2, decltype(calibration)::dimension> Dcal;  // unused
  gtsam::Matrix22 Dcal_point;
  gtsam::Point2 pc = calibration.calibrate(kp, Dcal, Dcal_point);

  // projection equation x = f(X/Z). The f is used to normalize the image
  // cooridinates and since we already have normalized cordinates xy, we can
  // omit the f
  const double X = pc(0) * Z;
  const double Y = pc(1) * Z;

  const gtsam::Point3 P_camera(X, Y, Z);

  // jacobian of the transforFrom function w.r.t P_camera
  gtsam::Matrix33 H_point;
  *lmk = X_world.transformFrom(P_camera, boost::none, H_point);

  if (Dp) {
    // TODO: test!!!
    // cal1 should be top row of Dcal_point which determines how the pc(0)
    // changes with input u, v
    const double dcal1_du = Dcal_point(0, 0);
    const double dcal1_dv = Dcal_point(0, 1);

    // cal1 should be top row of Dcal_point which determines how the pc(1)
    // changes with input u, v
    const double dcal2_du = Dcal_point(1, 0);
    const double dcal2_dv = Dcal_point(1, 1);

    gtsam::Matrix33 J;
    // top row is how X changes with u, v and Z
    J(0, 0) = dcal1_du * Z;
    J(0, 1) = dcal1_dv * Z;
    J(0, 2) = pc(0);

    // how Y changes with u, v and Z
    J(1, 0) = dcal2_du * Z;
    J(1, 1) = dcal2_dv * Z;
    J(1, 2) = pc(1);

    // how Z changes with Z
    J(2, 0) = 0;
    J(2, 1) = 0;
    J(2, 2) = 1;

    // J is computed as the jacobian without the affect of X_world
    //  f = X_world * C(u, v, Z)
    //    = X_world * P_camera
    // df/dfuvz = df/dC * dC/duvz
    //          = df/dC * J
    // since C -> C(u, v, z) = point in R^3, df/dC is the Jacobian of X_world *
    // P_camera -> Rotation component of X_world, or H_point
    *Dp = H_point * J;
  }
}

Landmark Camera::cameraToWorldConvention(const Landmark& lmk) {
  // expecting camera convention to be z forward, x, right and y down
  return Landmark(lmk(2), lmk(0), -lmk(2));
}

bool Camera::isLandmarkContained(const Landmark& lmk,
                                 Keypoint* keypoint) const {
  Keypoint kp;

  try {
    project(lmk, &kp);
  } catch (const gtsam::CheiralityException&) {
    // if CheiralityException then point is behind camera
    return false;
  }

  const bool result = isKeypointContained(kp);
  if (keypoint) *keypoint = kp;
  return result;
}

}  // namespace dyno
