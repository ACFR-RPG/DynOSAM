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

#include <gtest/gtest.h>

#include "ament_index_cpp/get_package_prefix.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_vision_common/Camera.hpp"
// #include "simulator.hpp"

/**
 * @brief gets the full path to the installation directory of the test data
 * which is expected to be at dynosam/test/data
 *
 * The full path will be the ROS install directory of this data after building
 *
 * @return std::string
 */
inline std::string getTestDataPath() {
  return ament_index_cpp::get_package_prefix("dynosam") + "/test/data";
}

namespace dyno_testing {

using namespace dyno;

inline KeypointStatus makeStatusKeypointMeasurement(
    TrackletId tracklet_id, ObjectId object_id, FrameId frame_id,
    const Keypoint& keypoint = Keypoint(), double sigma = 2.0) {
  gtsam::Vector2 kp_sigmas;
  kp_sigmas << sigma, sigma;
  MeasurementWithCovariance<Keypoint> kp_measurement =
      MeasurementWithCovariance<Keypoint>::FromSigmas(keypoint, kp_sigmas);
  return KeypointStatus(kp_measurement, frame_id, tracklet_id, object_id,
                        ReferenceFrame::LOCAL);
}

inline void compareLandmarks(const Landmarks& lmks_1, const Landmarks& lmks_2,
                             const float& tol = 1e-9) {
  ASSERT_EQ(lmks_1.size(), lmks_2.size());
  for (size_t i = 0u; i < lmks_1.size(); i++) {
    const auto& lmk_1 = lmks_1[i];
    const auto& lmk_2 = lmks_2[i];
    EXPECT_TRUE(gtsam::assert_equal(lmk_1, lmk_2, tol));
  }
}

inline void compareKeypoints(const Keypoints& lmks_1, const Keypoints& lmks_2,
                             const float& tol = 1e-9) {
  ASSERT_EQ(lmks_1.size(), lmks_2.size());
  for (size_t i = 0u; i < lmks_1.size(); i++) {
    const auto& lmk_1 = lmks_1[i];
    const auto& lmk_2 = lmks_2[i];
    EXPECT_TRUE(gtsam::assert_equal(lmk_1, lmk_2, tol));
  }
}

inline CameraParams makeDefaultCameraParams() {
  CameraParams::IntrinsicsCoeffs intrinsics(4);
  CameraParams::DistortionCoeffs distortion(4);
  intrinsics.at(0) = 554.256;  // fx
  intrinsics.at(1) = 554.256;  // fy
  intrinsics.at(2) = 640 / 2;  // u0
  intrinsics.at(3) = 480 / 2;  // v0
  return CameraParams(intrinsics, distortion, cv::Size(640, 480), "radtan");
}

inline Camera makeDefaultCamera() { return Camera(makeDefaultCameraParams()); }

inline Camera::Ptr makeDefaultCameraPtr() {
  return std::make_shared<Camera>(makeDefaultCameraParams());
}

}  // namespace dyno_testing
