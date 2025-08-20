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
  if (sync_) sync_.reset();

  rgb_image_sub_.unsubscribe();
  depth_image_sub_.unsubscribe();
  flow_image_sub_.unsubscribe();
  mask_image_sub_.unsubscribe();
}

void OnlineDataProviderRos::connect() {
  rclcpp::Node *node_ptr = node_.get();
  CHECK_NOTNULL(node_ptr);
  rgb_image_sub_.subscribe(node_ptr, "image/rgb");
  depth_image_sub_.subscribe(node_ptr, "image/depth");
  flow_image_sub_.subscribe(node_ptr, "image/flow");
  mask_image_sub_.subscribe(node_ptr, "image/mask");

  if (sync_) sync_.reset();

  // static constexpr size_t kQueueSize = 20u;
  static constexpr size_t kQueueSize = 1000u;
  sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
      SyncPolicy(kQueueSize), rgb_image_sub_, depth_image_sub_, flow_image_sub_,
      mask_image_sub_);

  sync_->registerCallback(std::bind(
      &OnlineDataProviderRos::imageSyncCallback, this, std::placeholders::_1,
      std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));

  RCLCPP_INFO_STREAM(
      node_->get_logger(),
      "OnlineDataProviderRos has been connected. Subscribed to image topics: "
          << rgb_image_sub_.getSubscriber()->get_topic_name() << " "
          << depth_image_sub_.getSubscriber()->get_topic_name() << " "
          << flow_image_sub_.getSubscriber()->get_topic_name() << " "
          << mask_image_sub_.getSubscriber()->get_topic_name() << ".");

  shutdown_ = false;
}

void OnlineDataProviderRos::imageSyncCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr &rgb_msg,
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
  image_container.rgb(rgb).depth(depth).opticalFlow(flow).objectMotionMask(
      mask);

  cv::Mat of_viz, motion_viz, depth_viz;
  of_viz = ImageType::OpticalFlow::toRGB(flow);
  motion_viz = ImageType::MotionMask::toRGB(mask);
  depth_viz = ImageType::Depth::toRGB(depth);

  cv::imshow("Optical Flow", of_viz);
  cv::imshow("Motion mask", motion_viz);
  cv::imshow("Depth", depth_viz);
  cv::waitKey(1);

  // trigger callback to send data to the DataInterface!
  // image_container_callback_(std::make_shared<ImageContainer>(image_container));
}

}  // namespace dyno
