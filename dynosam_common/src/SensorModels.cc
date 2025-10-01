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

#include "dynosam_common/SensorModels.hpp"

#include "dynosam_common/Exceptions.hpp"

namespace dyno {

CameraMeasurement::CameraMeasurement(
    const MeasurementWithCovariance<Keypoint>& keypoint)
    : keypoint_(keypoint) {}

CameraMeasurement& CameraMeasurement::keypoint(
    const MeasurementWithCovariance<Keypoint>& keypoint) {
  keypoint_ = keypoint;
  return *this;
}

CameraMeasurement& CameraMeasurement::landmark(
    const MeasurementWithCovariance<Landmark>& landmark) {
  landmark_ = landmark;
  return *this;
}
CameraMeasurement& CameraMeasurement::depth(
    const MeasurementWithCovariance<Depth>& depth) {
  depth_ = depth;
  return *this;
}

CameraMeasurement& CameraMeasurement::rightKeypoint(
    const MeasurementWithCovariance<Keypoint>& right_keypoint) {
  right_keypoint_ = right_keypoint;
  return *this;
}

bool CameraMeasurement::monocular() const { return !rgbd() || !stereo(); }

bool CameraMeasurement::rgbd() const { return depth_.has_value(); }
bool CameraMeasurement::stereo() const { return right_keypoint_.has_value(); }

bool CameraMeasurement::hasLandmark() const { return landmark_.has_value(); }

const MeasurementWithCovariance<Keypoint>& CameraMeasurement::keypoint() const {
  return keypoint_;
}

const MeasurementWithCovariance<Landmark>& CameraMeasurement::landmark() const {
  if (landmark_) return landmark_.value();
  DYNO_THROW_MSG(DynosamException)
      << "Landmark measurement missing from CameraMeasurement";
}
const MeasurementWithCovariance<Depth>& CameraMeasurement::depth() const {
  if (depth_) return depth_.value();
  DYNO_THROW_MSG(DynosamException)
      << "Depth measurement missing from CameraMeasurement";
}
const MeasurementWithCovariance<Keypoint>& CameraMeasurement::rightKeypoint()
    const {
  if (right_keypoint_) return right_keypoint_.value();
  DYNO_THROW_MSG(DynosamException)
      << "Right Keypoint measurement missing from CameraMeasurement";
}

StereoMeasurement::Optional CameraMeasurement::stereoMeasurement() const {
  if (!stereo()) {
    return {};
  }

  gtsam::Vector3 sigmas;
  if (keypoint_.hasModel() && right_keypoint_->hasModel()) {
    auto left_model_sigmas = keypoint_.model()->sigmas();
    auto right_model_sigmas = right_keypoint_->model()->sigmas();

    sigmas << left_model_sigmas(0), left_model_sigmas(1), right_model_sigmas(0);
  } else {
    LOG(WARNING)
        << "Creating stereo measurement with default noise model (1px)";
    sigmas << 1, 1, 1;
  }

  const Keypoint& L = keypoint_.measurement();
  const Keypoint& R = right_keypoint_->measurement();

  return StereoMeasurement::FromSigmas(gtsam::StereoPoint2(L(0), R(0), L(1)),
                                       sigmas);
}

}  // namespace dyno
