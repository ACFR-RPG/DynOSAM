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

#include "dynosam_ros/OnlineDataProviderRos.hpp"

#include "dynosam_ros/RosUtils.hpp"

namespace dyno {

OnlineDataProviderRos::OnlineDataProviderRos(
    rclcpp::Node::SharedPtr node, const OnlineDataProviderRosParams &params)
    : DataProviderRos(node), frame_id_(0u) {
  if (params.wait_for_camera_params) {
    waitAndSetCameraParams(
        std::chrono::milliseconds(params.camera_params_timeout));
  }

  connect();
  CHECK_EQ(shutdown_, false);
}

bool OnlineDataProviderRos::spin() { return !shutdown_; }

void OnlineDataProviderRos::shutdown() {
  shutdown_ = true;
  // shutdown synchronizer
  RCLCPP_INFO_STREAM(node_->get_logger(),
                     "Shutting down OnlineDataProviderRos");
  image_subscriber_->shutdown();
}

void OnlineDataProviderRos::connect() {
  connectImages();
  connectImu();
  shutdown_ = false;
}

void OnlineDataProviderRos::connectImages() {
  rclcpp::Node &node_ref = *node_;

  static const std::array<std::string, 4> &topics = {
      "image/rgb", "image/depth", "image/flow", "image/mask"};

  std::shared_ptr<MultiImageSync4> multi_image_sync =
      std::make_shared<MultiImageSync4>(node_ref, topics, 20);
  multi_image_sync->registerCallback(
      [this](const sensor_msgs::msg::Image::ConstSharedPtr &rgb_msg,
             const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg,
             const sensor_msgs::msg::Image::ConstSharedPtr &flow_msg,
             const sensor_msgs::msg::Image::ConstSharedPtr &mask_msg) {
        if (!image_container_callback_) {
          RCLCPP_ERROR_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                                "Image Sync callback triggered but "
                                "image_container_callback_ is not registered!");
          return;
        }

        const cv::Mat rgb = readRgbRosImage(rgb_msg);
        const cv::Mat depth = readDepthRosImage(depth_msg);
        const cv::Mat flow = readFlowRosImage(flow_msg);
        const cv::Mat mask = readMaskRosImage(mask_msg);

        const Timestamp timestamp = utils::fromRosTime(rgb_msg->header.stamp);
        const FrameId frame_id = frame_id_;
        frame_id_++;

        ImageContainer image_container(frame_id, timestamp);
        image_container.rgb(rgb)
            .depth(depth)
            .opticalFlow(flow)
            .objectMotionMask(mask);

        image_container_callback_(
            std::make_shared<ImageContainer>(image_container));
      });
  CHECK(multi_image_sync->connect());
  image_subscriber_ = multi_image_sync;
}

void OnlineDataProviderRos::connectImu() {
  if (imu_sub_) imu_sub_.reset();

  imu_callback_group_ = node_->create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive);

  rclcpp::SubscriptionOptions imu_sub_options;
  imu_sub_options.callback_group = imu_callback_group_;

  imu_sub_ = node_->create_subscription<ImuAdaptedType>(
      "imu", rclcpp::SensorDataQoS(),
      [&](const dyno::ImuMeasurement &imu) -> void {
        if (!imu_single_input_callback_) {
          RCLCPP_ERROR_THROTTLE(
              node_->get_logger(), *node_->get_clock(), 1000,
              "Imu callback triggered but "
              "imu_single_input_callback_ is not registered!");
          return;
        }
        imu_single_input_callback_(imu);
      },
      imu_sub_options);
}

}  // namespace dyno
