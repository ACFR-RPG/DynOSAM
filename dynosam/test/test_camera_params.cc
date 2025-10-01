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

#include <config_utilities/parsing/yaml.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <gtsam/geometry/Cal3DS2.h>

#include <thread>

#include "dynosam_vision_common/CameraParams.hpp"
#include "internal/helpers.hpp"

DECLARE_string(test_data_path);

using namespace dyno;

TEST(testCameraParamss, basicConstructionCal3DS2) {
  // Intrinsics.
  const std::vector<double> intrinsics_expected = {458.654, 457.296, 367.215,
                                                   248.375};
  //   // Distortion coefficients.
  const std::vector<double> distortion_expected = {-0.28340811, 0.07395907,
                                                   0.00019359, 1.76187114e-05};

  const cv::Size size_expected(752, 480);

  const std::string expected_distortion_model = "radtan";

  // Sensor extrinsics wrt. the body-frame.
  gtsam::Rot3 R_expected(0.0148655429818, -0.999880929698, 0.00414029679422,
                         0.999557249008, 0.0149672133247, 0.025715529948,
                         -0.0257744366974, 0.00375618835797, 0.999660727178);
  gtsam::Point3 T_expected(-0.0216401454975, -0.064676986768, 0.00981073058949);
  gtsam::Pose3 pose_expected(R_expected, T_expected);

  CameraParams params(intrinsics_expected, distortion_expected, size_expected,
                      expected_distortion_model, pose_expected);

  EXPECT_EQ(size_expected.width, params.ImageWidth());
  EXPECT_EQ(size_expected.height, params.ImageHeight());

  // for (int c = 0u; c < 4u; c++)
  // {
  //   EXPECT_DOUBLE_EQ(intrinsics_expected[c], params.intrinsics_[c]);
  // }
  EXPECT_DOUBLE_EQ(intrinsics_expected[0],
                   params.getCameraMatrix().at<double>(0, 0));
  EXPECT_DOUBLE_EQ(intrinsics_expected[1],
                   params.getCameraMatrix().at<double>(1, 1));
  EXPECT_DOUBLE_EQ(intrinsics_expected[2],
                   params.getCameraMatrix().at<double>(0, 2));
  EXPECT_DOUBLE_EQ(intrinsics_expected[3],
                   params.getCameraMatrix().at<double>(1, 2));

  EXPECT_DOUBLE_EQ(intrinsics_expected[0], params.fx());
  EXPECT_DOUBLE_EQ(intrinsics_expected[1], params.fy());
  EXPECT_DOUBLE_EQ(intrinsics_expected[2], params.cu());
  EXPECT_DOUBLE_EQ(intrinsics_expected[3], params.cv());
  // //   EXPECT_EQ(cam_params.intrinsics_.size(), 4u);
  gtsam::Cal3DS2 gtsam_calib =
      params.constructGtsamCalibration<gtsam::Cal3DS2>();

  EXPECT_DOUBLE_EQ(intrinsics_expected[0], gtsam_calib.fx());
  EXPECT_DOUBLE_EQ(intrinsics_expected[1], gtsam_calib.fy());
  EXPECT_DOUBLE_EQ(0u, gtsam_calib.skew());
  EXPECT_DOUBLE_EQ(intrinsics_expected[2], gtsam_calib.px());
  EXPECT_DOUBLE_EQ(intrinsics_expected[3], gtsam_calib.py());

  EXPECT_TRUE(assert_equal(pose_expected, params.getExtrinsics()));

  for (int c = 0u; c < 4u; c++) {
    EXPECT_DOUBLE_EQ(distortion_expected[c],
                     params.getDistortionCoeffs().at<double>(c));
  }
  EXPECT_EQ(params.getDistortionCoeffs().rows, 1u);
  EXPECT_EQ(params.getDistortionCoeffs().cols, 4u);
  EXPECT_DOUBLE_EQ(distortion_expected[0], gtsam_calib.k1());
  EXPECT_DOUBLE_EQ(distortion_expected[1], gtsam_calib.k2());
  EXPECT_DOUBLE_EQ(distortion_expected[2], gtsam_calib.p1());
  EXPECT_DOUBLE_EQ(distortion_expected[3], gtsam_calib.p2());
}

TEST(testCameraParams, parseYAML) {
  // CameraParams cam_params = CameraParams::fromYamlFile(getTestDataPath() +
  // "/sensor.yaml");
  auto cam_params =
      config::fromYamlFile<CameraParams>(getTestDataPath() + "/sensor.yaml");

  // Frame rate.
  const double frame_rate_expected = 1.0 / 20.0;
  // EXPECT_DOUBLE_EQ(frame_rate_expected, cam_params.frame_rate_);

  // Image size.
  const cv::Size size_expected(752, 480);
  EXPECT_EQ(size_expected.width, cam_params.ImageWidth());
  EXPECT_EQ(size_expected.height, cam_params.ImageHeight());

  // Intrinsics.
  const std::vector<double> intrinsics_expected = {458.654, 457.296, 367.215,
                                                   248.375};
  EXPECT_DOUBLE_EQ(intrinsics_expected[0], cam_params.fx());
  EXPECT_DOUBLE_EQ(intrinsics_expected[1], cam_params.fy());
  EXPECT_DOUBLE_EQ(intrinsics_expected[2], cam_params.cu());
  EXPECT_DOUBLE_EQ(intrinsics_expected[3], cam_params.cv());

  EXPECT_DOUBLE_EQ(intrinsics_expected[0],
                   cam_params.getCameraMatrix().at<double>(0, 0));
  EXPECT_DOUBLE_EQ(intrinsics_expected[1],
                   cam_params.getCameraMatrix().at<double>(1, 1));
  EXPECT_DOUBLE_EQ(intrinsics_expected[2],
                   cam_params.getCameraMatrix().at<double>(0, 2));
  EXPECT_DOUBLE_EQ(intrinsics_expected[3],
                   cam_params.getCameraMatrix().at<double>(1, 2));
  gtsam::Cal3DS2 gtsam_calib =
      cam_params.constructGtsamCalibration<gtsam::Cal3DS2>();

  EXPECT_DOUBLE_EQ(intrinsics_expected[0], gtsam_calib.fx());
  EXPECT_DOUBLE_EQ(intrinsics_expected[1], gtsam_calib.fy());
  EXPECT_DOUBLE_EQ(0u, gtsam_calib.skew());
  EXPECT_DOUBLE_EQ(intrinsics_expected[2], gtsam_calib.px());
  EXPECT_DOUBLE_EQ(intrinsics_expected[3], gtsam_calib.py());

  // Sensor extrinsics wrt. the body-frame.
  gtsam::Rot3 R_expected(0.0148655429818, -0.999880929698, 0.00414029679422,
                         0.999557249008, 0.0149672133247, 0.025715529948,
                         -0.0257744366974, 0.00375618835797, 0.999660727178);
  gtsam::Point3 T_expected(-0.0216401454975, -0.064676986768, 0.00981073058949);
  gtsam::Pose3 pose_expected(R_expected, T_expected);
  EXPECT_TRUE(assert_equal(pose_expected, cam_params.getExtrinsics()));

  // Distortion coefficients.
  const std::vector<double> distortion_expected = {-0.28340811, 0.07395907,
                                                   0.00019359, 1.76187114e-05};
  for (size_t c = 0u; c < 4u; c++) {
    EXPECT_DOUBLE_EQ(distortion_expected[c],
                     cam_params.getDistortionCoeffs().at<double>(c));
  }
  EXPECT_EQ(cam_params.getDistortionCoeffs().rows, 1u);
  EXPECT_EQ(cam_params.getDistortionCoeffs().cols, 4u);
  EXPECT_DOUBLE_EQ(distortion_expected[0], gtsam_calib.k1());
  EXPECT_DOUBLE_EQ(distortion_expected[1], gtsam_calib.k2());
  EXPECT_DOUBLE_EQ(distortion_expected[2], gtsam_calib.p1());
  EXPECT_DOUBLE_EQ(distortion_expected[3], gtsam_calib.p2());

  LOG(INFO) << "Dyno printing " << cam_params.toString();
  LOG(INFO) << "CU printing " << config::toString(cam_params);
}

TEST(testCameraParamss, convertDistortionVectorToMatrix) {
  std::vector<double> distortion_coeffs;

  // 4 distortion params
  distortion_coeffs = {1.0, -2.0, 1.3, 10};
  cv::Mat distortion_coeffs_mat;
  CameraParams::convertDistortionVectorToMatrix(distortion_coeffs,
                                                &distortion_coeffs_mat);
  EXPECT_EQ(distortion_coeffs_mat.cols, distortion_coeffs.size());
  EXPECT_EQ(distortion_coeffs_mat.rows, 1u);
  for (size_t i = 0u; i < distortion_coeffs.size(); i++) {
    EXPECT_EQ(distortion_coeffs_mat.at<double>(0, i), distortion_coeffs.at(i));
  }

  // 5 distortion params
  distortion_coeffs = {1, 1.2f, 3u, 4l, 5.34};  //! randomize types as well
  CameraParams::convertDistortionVectorToMatrix(distortion_coeffs,
                                                &distortion_coeffs_mat);
  EXPECT_EQ(distortion_coeffs_mat.cols, distortion_coeffs.size());
  EXPECT_EQ(distortion_coeffs_mat.rows, 1u);
  for (size_t i = 0u; i < distortion_coeffs.size(); i++) {
    EXPECT_EQ(distortion_coeffs_mat.at<double>(0u, i), distortion_coeffs.at(i));
  }

  // n distortion params
  distortion_coeffs = {1.0, 1.2, 3.2, 4.3, 5.34, 10203, 1818.9, 1.9};
  CameraParams::convertDistortionVectorToMatrix(distortion_coeffs,
                                                &distortion_coeffs_mat);
  EXPECT_EQ(distortion_coeffs_mat.cols, distortion_coeffs.size());
  EXPECT_EQ(distortion_coeffs_mat.rows, 1u);
  for (size_t i = 0u; i < distortion_coeffs.size(); i++) {
    EXPECT_EQ(distortion_coeffs_mat.at<double>(0u, i), distortion_coeffs.at(i));
  }
}
