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

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <gtsam/base/numericalDerivative.h>

#include "dynosam_common/utils/Numerical.hpp"
#include "dynosam_cv/Camera.hpp"
#include "internal/helpers.hpp"

using namespace dyno;

TEST(Camera, project) {
  Landmarks lmks;
  lmks.push_back(Landmark(0.0, 0.0, 1.0));
  lmks.push_back(Landmark(0.0, 0.0, 2.0));
  lmks.push_back(Landmark(0.0, 1.0, 2.0));
  lmks.push_back(Landmark(0.0, 10.0, 20.0));
  lmks.push_back(Landmark(1.0, 0.0, 2.0));

  CameraParams::IntrinsicsCoeffs intrinsics(4);
  CameraParams::DistortionCoeffs distortion(4);

  intrinsics.at(0) = 1.0;  // fx
  intrinsics.at(1) = 1.0;  // fy
  intrinsics.at(2) = 3.0;  // u0
  intrinsics.at(3) = 2.0;  // v0
  Keypoints expected_kpts;
  expected_kpts.push_back(Keypoint(intrinsics.at(2), intrinsics.at(3)));
  expected_kpts.push_back(Keypoint(intrinsics.at(2), intrinsics.at(3)));
  expected_kpts.push_back(Keypoint(3.0, 1.0 / 2.0 + 2.0));
  expected_kpts.push_back(Keypoint(3.0, 1.0 / 2.0 + 2.0));
  expected_kpts.push_back(Keypoint(1.0 / 2.0 + 3.0, 2.0));

  CameraParams camera_params(intrinsics, distortion, cv::Size(640, 480),
                             "radtan");

  Camera camera(camera_params);

  Keypoints actual_kpts;
  EXPECT_NO_THROW(camera.project(lmks, &actual_kpts));
  dyno_testing::compareKeypoints(actual_kpts, expected_kpts);
}

TEST(Camera, backProjectSingleSimple) {
  // Easy test first, back-project keypoint at the center of the image with
  // a given depth.
  CameraParams camera_params = dyno_testing::makeDefaultCameraParams();
  Camera camera(camera_params);

  Keypoint kpt(camera_params.cu(), camera_params.cv());
  Landmark actual_lmk;
  double depth = 2.0;
  camera.backProject(kpt, depth, &actual_lmk);

  Landmark expected_lmk(0.0, 0.0, depth);
  EXPECT_NEAR(expected_lmk.x(), actual_lmk.x(), 0.0001);
  EXPECT_NEAR(expected_lmk.y(), actual_lmk.y(), 0.0001);
  EXPECT_NEAR(expected_lmk.z(), actual_lmk.z(), 0.0001);
}

TEST(Camera, backProjectMultipleSimple) {
  // Easy test first, back-project keypoints at the center of the image with
  // different depths.
  CameraParams camera_params = dyno_testing::makeDefaultCameraParams();
  Camera camera(camera_params);

  Keypoint kpt(camera_params.cu(), camera_params.cv());
  // Create 3 keypoints centered at image with different depths
  Keypoints kpts(3, kpt);
  Depths depths = {2.0, 3.0, 4.5};
  Landmarks actual_lmks;
  camera.backProject(kpts, depths, &actual_lmks);

  Landmarks expected_lmks;
  for (const auto& depth : depths) {
    expected_lmks.push_back(Landmark(0.0, 0.0, depth));
  }

  dyno_testing::compareLandmarks(actual_lmks, expected_lmks);
}

TEST(Camera, backProjectSingleTopLeft) {
  // Back-project keypoint at the center of the image with a given depth.
  CameraParams::IntrinsicsCoeffs intrinsics(4);
  CameraParams::DistortionCoeffs distortion(4);

  double fx = 30.9 / 2.2;
  double fy = 12.0 / 23.0;
  double cu = 390.8;
  double cv = 142.2;

  intrinsics.at(0) = fx;  // fx
  intrinsics.at(1) = fy;  // fy
  intrinsics.at(2) = cu;  // u0
  intrinsics.at(3) = cv;  // v0
  CameraParams camera_params(intrinsics, distortion, cv::Size(640, 480),
                             "radtan");

  Camera camera(camera_params);

  Landmark actual_lmk;
  double depth = 2.0;
  Keypoint kpt(0.0, 0.0);  // Top-left corner
  camera.backProject(kpt, depth, &actual_lmk);

  Landmark expected_lmk(depth / fx * (-cu), depth / fy * (-cv), depth);
  EXPECT_NEAR(expected_lmk.x(), actual_lmk.x(), 0.0001);
  EXPECT_NEAR(expected_lmk.y(), actual_lmk.y(), 0.0001);
  EXPECT_NEAR(expected_lmk.z(), actual_lmk.z(), 0.0001);
}

TEST(Camera, backProjectToZ) {
  CameraParams camera_params = dyno_testing::makeDefaultCameraParams();
  Camera camera(camera_params);

  Keypoint kpt(camera_params.cu() / 2.0, camera_params.cv() / 2.0);
  double depth = 2.0;

  Landmark actual_lmk;
  camera.backProject(kpt, depth, &actual_lmk);

  const double Z = actual_lmk(2);
  Landmark z_projected_lmk;
  camera.backProjectFromZ(kpt, Z, &z_projected_lmk);

  // check that the keypoint is the same as the actual one (we just change the
  // Z) but the "measurement" shold remain the same
  Keypoint calculated_kp;
  camera.project(z_projected_lmk, &calculated_kp);
  EXPECT_TRUE(gtsam::assert_equal(kpt, calculated_kp));
}

TEST(Camera, backProjectToZJacobian) {
  CameraParams camera_params = dyno_testing::makeDefaultCameraParams();
  Camera camera(camera_params);

  Keypoint kpt(camera_params.cu() / 2.0, camera_params.cv() / 2.0);
  double depth = 2.0;

  Landmark actual_lmk;
  camera.backProject(kpt, depth, &actual_lmk);
  const double Z = actual_lmk(2);

  gtsam::Pose3 pose(gtsam::Rot3::Rodrigues(0, 2, 3), gtsam::Point3(1, 2, 0));

  auto numerical_deriv_func =
      [&camera](const gtsam::Vector3& uvz,
                const gtsam::Pose3& X_world) -> gtsam::Vector3 {
    Landmark lmk;
    camera.backProjectFromZ(gtsam::Point2(uvz(0), uvz(1)), uvz(2), &lmk,
                            X_world);
    return lmk;
  };

  // construct 3x1 vector of input to satisfy the matrix structure of the
  // problem we want to sove
  gtsam::Vector3 input(kpt(0), kpt(1), Z);
  // numericalDerivative21 -> 2 function arguments, derivative w.r.t uvz
  gtsam::Matrix33 numerical_J =
      gtsam::numericalDerivative21<gtsam::Vector3, gtsam::Vector3,
                                   const gtsam::Pose3&>(numerical_deriv_func,
                                                        input, pose);

  Landmark lmk;  // unused
  gtsam::Matrix33 analytical_J;
  camera.backProjectFromZ(kpt, Z, &lmk, pose, analytical_J);

  EXPECT_TRUE(gtsam::assert_equal(numerical_J, analytical_J));
}
