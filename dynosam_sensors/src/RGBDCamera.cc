/*
 *   Copyright (c) 2025 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam_sensors/RGBDCamera.hpp"

namespace dyno {

namespace {

double tryGetBaseline(const CameraParams& camera_param) {
  if (camera_param.hasDepthParams()) {
    return camera_param.depthParams().virtual_baseline;
  }
  throw InvalidRGBDCameraParams();
}

StereoCalibPtr constructCalibrationHelper(const CameraParams& camera_params,
                                          double baseline) {
  const auto calibration =
      camera_params.constructGtsamCalibration<gtsam::Cal3_S2>();
  return StereoCalibPtr(new gtsam::Cal3_S2Stereo(
      calibration.fx(), calibration.fy(), calibration.skew(), calibration.px(),
      calibration.py(), baseline));
}

}  // namespace

RGBDCamera::RGBDCamera(const CameraParams& camera_params)
    : Camera(camera_params),
      fx_b_(camera_params.fx() * tryGetBaseline(camera_params)) {
  stereo_calibration_ = constructCalibrationHelper(getParams(), baseline());
  stereo_camera_ = gtsam::StereoCamera(gtsam::Pose3(), stereo_calibration_);
}

double RGBDCamera::depthFromDisparity(double disparity) const {
  return fx_b_ / disparity;
}

Baseline RGBDCamera::baseline() const {
  return getParams().depthParams().virtual_baseline;
}

bool RGBDCamera::safeProjectRight(double depth, const Keypoint& left_keypoint,
                                  Keypoint& right_keypoint) const {
  if (depth < 0) {
    return false;
  }

  const double uL = left_keypoint(0);
  const double v = left_keypoint(1);
  const double uR = rightKeypoint(depth, uL);

  // out of image bounds
  if (uR < 0.0 || uR > this->getParams().ImageWidth()) {
    return false;
  }

  right_keypoint = Keypoint(uR, v);
  return true;
}

// bool RGBDCamera::projectRight(Feature::Ptr feature) const {
//   CHECK(feature);
//   if (feature->hasDepth()) {
//     double uL = feature->keypoint()(0);
//     double v = feature->keypoint()(1);
//     double uR = rightKeypoint(feature->depth(), uL);

//     // out of image bounds
//     if (uR < 0.0 || uR > this->getParams().ImageWidth()) {
//       return false;
//     }

//     feature->rightKeypoint(Keypoint(uR, v));
//     return true;
//   }
//   return false;
// }

// std::pair<bool, gtsam::StereoPoint2> RGBDCamera::getStereo(
//     Feature::Ptr feature, const bool force_recalculation) {
//   gtsam::StereoPoint2 stereo_point;
//   // if no right feature or recalculation flag is true
//   if (!feature->hasRightKeypoint() || force_recalculation) {
//     // update the right keypoint of feature
//     if (!this->projectRight(feature)) {
//       return {false, stereo_point};
//     }
//   }

//   CHECK(feature->stereoPoint(stereo_point));
//   return {true, stereo_point};
// }

double RGBDCamera::fxb() const { return fx_b_; }

StereoCalibPtr RGBDCamera::getFakeStereoCalib() const {
  return stereo_calibration_;
}

const gtsam::StereoCamera& RGBDCamera::getFakeStereoCamera() const {
  return stereo_camera_;
}

gtsam::Pose3 RGBDCamera::StereoExtrinsics(const dyno::Camera::Ptr& camera) {
  const dyno::RGBDCamera& rgbd_camera = camera->getRGBDCamera();
  return gtsam::Pose3(gtsam::Rot3(),
                      gtsam::Point3(rgbd_camera.baseline(), 0.0, 0.0));
}

StereoCalibPtr RGBDCamera::Calibration(const dyno::Camera::Ptr& camera) {
  const StereoCalibPtr K_stereo = camera->getRGBDCamera().getFakeStereoCalib();
  return K_stereo;
}

gtsam::Cal3_S2 MonoCalibration(const dyno::Camera::Ptr& camera) {
  return Camera::Calibration(camera);
}

}  // namespace dyno
