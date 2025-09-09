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

#include <dynosam/backend/BackendDefinitions.hpp>  //TODO: just for the gps stuff for dyno_mpc... for now

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
  // flow_image_sub_.unsubscribe();
  mask_image_sub_.unsubscribe();
}

void OnlineDataProviderRos::connect() {
  rclcpp::Node *node_ptr = node_.get();
  CHECK_NOTNULL(node_ptr);
  rgb_image_sub_.subscribe(node_ptr, "image/rgb");
  depth_image_sub_.subscribe(node_ptr, "image/depth");
  // flow_image_sub_.subscribe(node_ptr, "image/flow");
  mask_image_sub_.subscribe(node_ptr, "image/mask");

  if (sync_) sync_.reset();

  // static constexpr size_t kQueueSize = 20u;
  static constexpr size_t kQueueSize = 1000u;
  // sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
  //     SyncPolicy(kQueueSize), rgb_image_sub_, depth_image_sub_,
  //     flow_image_sub_, mask_image_sub_);
  sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
      SyncPolicy(kQueueSize), rgb_image_sub_, depth_image_sub_,
      mask_image_sub_);

  // sync_->registerCallback(std::bind(
  //     &OnlineDataProviderRos::imageSyncCallback, this, std::placeholders::_1,
  //     std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
  sync_->registerCallback(std::bind(
      &OnlineDataProviderRos::imageSyncCallback, this, std::placeholders::_1,
      std::placeholders::_2, std::placeholders::_3));

  RCLCPP_INFO_STREAM(
      node_->get_logger(),
      "OnlineDataProviderRos has been connected. Subscribed to image topics: "
          << rgb_image_sub_.getSubscriber()->get_topic_name() << " "
          << depth_image_sub_.getSubscriber()->get_topic_name()
          << " "
          // << flow_image_sub_.getSubscriber()->get_topic_name() << " "
          << mask_image_sub_.getSubscriber()->get_topic_name() << ".");

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

  external_measurements_callback_group_ = node_->create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive);

  rclcpp::SubscriptionOptions external_measurement_sub_options;
  external_measurement_sub_options.callback_group =
      external_measurements_callback_group_;

  gps_like_sub_ = node_->create_subscription<PoseWithCovarianceStampted>(
      "gps", rclcpp::SensorDataQoS(),
      [&](const PoseWithCovarianceStampted &pose_msg) -> void {
        if (!external_measurement_callback_) {
          RCLCPP_ERROR_THROTTLE(
              node_->get_logger(), *node_->get_clock(), 1000,
              "GPS like callback triggered but "
              "external_measurement_callback_ is not registered!");
          return;
        }

        dyno::Timestamp timestamp;
        dyno::convert(pose_msg.header.stamp, timestamp);

        gtsam::Pose3 pose;
        dyno::convert(pose_msg.pose.pose, pose);

        // convert covariance matrix
        Eigen::Matrix<double, 6, 6> cov_ros;
        for (int i = 0; i < 6; ++i) {
          for (int j = 0; j < 6; ++j) {
            cov_ros(i, j) = pose_msg.pose.covariance[i * 6 + j];
          }
        }

        // Step 2. Permutation matrix (xyzrpy → rpyxyz)
        Eigen::Matrix<double, 6, 6> P;
        P << 0, 0, 0, 1, 0, 0,  // roll
            0, 0, 0, 0, 1, 0,   // pitch
            0, 0, 0, 0, 0, 1,   // yaw
            1, 0, 0, 0, 0, 0,   // x
            0, 1, 0, 0, 0, 0,   // y
            0, 0, 1, 0, 0, 0;   // z

        // Step 3. Convert covariance
        Eigen::Matrix<double, 6, 6> cov_gtsam = P * cov_ros * P.transpose();

        external_measurement_callback_(
            "gps", std::make_shared<dyno::UnaryPoseMeasurement<gtsam::Pose3>>(
                       timestamp, Pose3Measurement(pose, cov_gtsam)));
      },
      external_measurement_sub_options);

  shutdown_ = false;
}

void OnlineDataProviderRos::imageSyncCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr &rgb_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg,
    // const sensor_msgs::msg::Image::ConstSharedPtr &flow_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr &mask_msg) {
  if (!image_container_callback_) {
    RCLCPP_ERROR_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                          "Image Sync callback triggered but "
                          "image_container_callback_ is not registered!");
    return;
  }

  const cv::Mat rgb = readRgbRosImage(rgb_msg);
  const cv::Mat depth = readDepthRosImage(depth_msg);
  // const cv::Mat flow = readFlowRosImage(flow_msg);
  const cv::Mat mask = readMaskRosImage(mask_msg);

  const Timestamp timestamp = utils::fromRosTime(rgb_msg->header.stamp);
  const FrameId frame_id = frame_id_;
  frame_id_++;

  ImageContainer image_container(frame_id, timestamp);
  image_container.rgb(rgb).depth(depth).objectMotionMask(mask);

  // try and send associated ground truth to the data-interface if we can!
  if (ground_truth_handler_) {
    ground_truth_handler_->emit(timestamp, frame_id);

    // skip first frame so that we always have a previous gt
    if (frame_id == 0u) {
      return;
    }
  }

  // cv::Mat of_viz, motion_viz, depth_viz;
  // of_viz = ImageType::OpticalFlow::toRGB(flow);
  // motion_viz = ImageType::MotionMask::toRGB(mask);
  // depth_viz = ImageType::Depth::toRGB(depth);

  // cv::imshow("Optical Flow", of_viz);
  // cv::imshow("Motion mask", motion_viz);
  // cv::imshow("Depth", depth_viz);
  // cv::waitKey(1);

  // trigger callback to send data to the DataInterface!
  image_container_callback_(std::make_shared<ImageContainer>(image_container));
}

}  // namespace dyno
