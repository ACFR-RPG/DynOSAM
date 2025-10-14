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

#pragma once

#include <glog/logging.h>

#include "dynosam_cv/CameraParams.hpp"
#include "rclcpp/type_adapter.hpp"
#include "sensor_msgs/msg/camera_info.hpp"

template <>
struct rclcpp::TypeAdapter<dyno::CameraParams, sensor_msgs::msg::CameraInfo> {
  using is_specialized = std::true_type;
  using custom_type = dyno::CameraParams;
  using ros_message_type = sensor_msgs::msg::CameraInfo;

  static void convert_to_ros_message(const custom_type& source,
                                     ros_message_type& destination) {
    // intrisincs
    const cv::Mat& K = source.getCameraMatrix();
    std::copy(K.begin<double>(), K.end<double>(), destination.k.begin());

    // distortion
    const cv::Mat& D = source.getDistortionCoeffs();
    destination.d.resize(D.total());
    destination.d = std::vector<double>(D.begin<double>(), D.end<double>());

    destination.height = source.ImageHeight();
    destination.width = source.ImageWidth();

    std::string camera_model;
    CHECK(dyno::CameraParams::distortionModelToString(
        source.getDistortionModel(), destination.distortion_model,
        camera_model));
    (void)camera_model;
  }

  static void convert_to_custom(const ros_message_type& source,
                                custom_type& destination) {
    // bit gross but cv::Mat does not accept a const void* for data (which
    // source.k.data()) is so to ensure the data is safe (ie. rather than doing
    // a const conversion), I decide to copy the data into std::vector and then
    // copy into cv::Mat. this is inefficient but this conversion happens so
    // rarely and the data pakcet is tiny...
    std::vector<double> k_vector(source.k.begin(), source.k.end());
    cv::Mat K(3, 3, CV_64F, std::data(k_vector));
    dyno::CameraParams::IntrinsicsCoeffs intrinsics;
    // TODO: should make this a constructor...
    dyno::CameraParams::convertKMatrixToIntrinsicsCoeffs(K, intrinsics);
    dyno::CameraParams::DistortionCoeffs distortion(source.d.begin(),
                                                    source.d.end());
    cv::Size size(source.width, source.height);
    const std::string& distortion_model = source.distortion_model;

    destination =
        dyno::CameraParams(intrinsics, distortion, size, distortion_model);
  }
};

RCLCPP_USING_CUSTOM_TYPE_AS_ROS_MESSAGE_TYPE(dyno::CameraParams,
                                             sensor_msgs::msg::CameraInfo);
