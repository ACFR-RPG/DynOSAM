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

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "dynosam_ros_common/adaptors/CameraParamsAdaptor.hpp"
#include "sensor_msgs/msg/camera_info.hpp"

using namespace dyno;

static dyno::CameraParams makeDefaultCameraParams() {
  dyno::CameraParams::IntrinsicsCoeffs intrinsics(4);
  dyno::CameraParams::DistortionCoeffs distortion(4);
  intrinsics.at(0) = 554.256;  // fx
  intrinsics.at(1) = 554.256;  // fy
  intrinsics.at(2) = 640 / 2;  // u0
  intrinsics.at(3) = 480 / 2;  // v0
  return dyno::CameraParams(intrinsics, distortion, cv::Size(640, 480),
                            "radtan");
}

TEST(CameraParamsAdaptor, testToROS) {
  CameraParams dyno_params = makeDefaultCameraParams();

  sensor_msgs::msg::CameraInfo camera_info;

  using Adaptor =
      rclcpp::TypeAdapter<dyno::CameraParams, sensor_msgs::msg::CameraInfo>;
  Adaptor::convert_to_ros_message(dyno_params, camera_info);

  EXPECT_DOUBLE_EQ(camera_info.k[0], dyno_params.fx());
  EXPECT_DOUBLE_EQ(camera_info.k[4], dyno_params.fy());
  EXPECT_DOUBLE_EQ(camera_info.k[2], dyno_params.cu());
  EXPECT_DOUBLE_EQ(camera_info.k[5], dyno_params.cv());

  // Assert: Check the image dimensions
  EXPECT_EQ(camera_info.width, dyno_params.ImageWidth());
  EXPECT_EQ(camera_info.height, dyno_params.ImageHeight());

  // Assert: Check distortion coefficients
  const auto& expected_distortion = dyno_params.getDistortionCoeffs();
  ASSERT_EQ(camera_info.d.size(), expected_distortion.total());
  for (size_t i = 0; i < camera_info.d.size(); ++i) {
    EXPECT_DOUBLE_EQ(camera_info.d[i], expected_distortion.at<double>(i));
  }
}

TEST(CameraParamsAdaptor, testFromROS) {
  sensor_msgs::msg::CameraInfo camera_info;
  camera_info.width = 640;
  camera_info.height = 480;
  camera_info.k = {500.0, 0.0, 320.0, 0.0, 500.0,
                   240.0, 0.0, 0.0,   1.0};      // Intrinsic matrix
  camera_info.d = {0.1, -0.1, 0.01, 0.01, 0.0};  // Distortion coefficients
  camera_info.distortion_model = "plumb_bob";

  using Adaptor =
      rclcpp::TypeAdapter<dyno::CameraParams, sensor_msgs::msg::CameraInfo>;

  CameraParams dyno_params;
  Adaptor::convert_to_custom(camera_info, dyno_params);

  // Assert: Check intrinsic parameters
  EXPECT_DOUBLE_EQ(dyno_params.fx(), 500.0);
  EXPECT_DOUBLE_EQ(dyno_params.fy(), 500.0);
  EXPECT_DOUBLE_EQ(dyno_params.cu(), 320.0);
  EXPECT_DOUBLE_EQ(dyno_params.cv(), 240.0);

  // Assert: Check image size
  EXPECT_EQ(dyno_params.ImageWidth(), 640);
  EXPECT_EQ(dyno_params.ImageHeight(), 480);

  // Assert: Check distortion coefficients
  const auto& expected_distortion = dyno_params.getDistortionCoeffs();
  ASSERT_EQ(expected_distortion.total(), camera_info.d.size());
  for (size_t i = 0; i < expected_distortion.total(); ++i) {
    EXPECT_DOUBLE_EQ(expected_distortion.at<double>(i), camera_info.d[i]);
  }

  // Assert: Check distortion model
  EXPECT_EQ(dyno_params.getDistortionModel(), DistortionModel::RADTAN);
}
