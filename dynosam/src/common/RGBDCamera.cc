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

#include "dynosam/common/RGBDCamera.hpp"

namespace dyno {

RGBDCamera::RGBDCamera(const CameraParams& camera_params,
                       const RGBDCameraParams& rgbd_params)
    : Camera(camera_params),
      rgbd_params_(rgbd_params),
      fx_b_(camera_params.fx() * rgbd_params.virtual_baseline) {}

double RGBDCamera::depthFromDisparity(double disparity) const {
  return fx_b_ / disparity;
}

Baseline RGBDCamera::baseline() const { return rgbd_params_.virtual_baseline; }

bool RGBDCamera::projectRight(Feature::Ptr feature) const {
  CHECK(feature);
  if (feature->hasDepth()) {
    double uL = feature->keypoint()(0);
    double uR = rightKeypoint(feature->depth(), uL);

    // out of image bounds
    if (uR < 0.0 || uR > this->getParams().ImageWidth()) {
      return false;
    }

    feature->rightKeypoint(Keypoint(uL, uR));
    return true;
  }
  return false;
}

Keypoint RGBDCamera::rightKeypoint(double depth,
                                   const Keypoint& left_keypoint) const {
  return Keypoint(rightKeypoint(depth, left_keypoint(0)), left_keypoint(1));
}

double RGBDCamera::rightKeypoint(double depth, double uL) const {
  const double disparity = fx_b_ / depth;
  return uL - disparity;
}

double RGBDCamera::fxb() const { return fx_b_; }

StereoCalibPtr RGBDCamera::getFakeStereoCalib() const {
  const auto calibration =
      getParams().constructGtsamCalibration<gtsam::Cal3_S2>();
  return StereoCalibPtr(new gtsam::Cal3_S2Stereo(
      calibration.fx(), calibration.fy(), calibration.skew(), calibration.px(),
      calibration.py(), baseline()));
}
gtsam::StereoCamera RGBDCamera::getFakeStereoCamera() const {
  return {gtsam::Pose3(), getFakeStereoCalib()};
}

}  // namespace dyno
