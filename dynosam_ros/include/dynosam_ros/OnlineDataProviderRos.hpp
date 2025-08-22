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

#include "dynosam_ros/DataProviderRos.hpp"
#include "dynosam_ros/adaptors/ImuMeasurementAdaptor.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/exact_time.h"
#include "message_filters/synchronizer.h"
#include "rclcpp/node.hpp"
#include "rclcpp/node_options.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/imu.hpp"

namespace dyno {

struct OnlineDataProviderRosParams {
  bool wait_for_camera_params{true};
  int32_t camera_params_timeout{-1};
};

/**
 * @brief Online data-provider for DynoSAM.
 *
 * Subscribes to four synchronized image topics (rgb, depth, mask and optical
 * flow) and a camera_info topic (to define the camera intrinsics of the
 * system).
 *
 *
 *
 */
class OnlineDataProviderRos : public DataProviderRos {
 public:
  /**
   * @brief Construct a new OnlineDataProviderRos.
   *
   * Constructor will block until camera info has been received (if
   * OnlineDataProviderRosParams::wait_for_camera_params is true).
   *
   * @param node rclcpp::Node::SharedPtr
   * @param params const OnlineDataProviderRosParams&
   */
  OnlineDataProviderRos(rclcpp::Node::SharedPtr node,
                        const OnlineDataProviderRosParams &params);

  /**
   * @brief Indicates that there is no known end to the dataset
   *
   * @return int
   */
  int datasetSize() const override { return -1; }

  /**
   * @brief Returns true while not shutdown
   *
   * @return true
   * @return false
   */
  bool spin() override;

  /**
   * @brief Disconnects all subscribers
   *
   */
  void shutdown() override;

  /**
   * @brief Connects all subscribers.
   *
   */
  void connect();

 private:
  using SyncPolicy = message_filters::sync_policies::ExactTime<
      sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image,
      sensor_msgs::msg::Image>;

  void imageSyncCallback(
      const sensor_msgs::msg::Image::ConstSharedPtr &rgb_msg,
      const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg,
      const sensor_msgs::msg::Image::ConstSharedPtr &flow_msg,
      const sensor_msgs::msg::Image::ConstSharedPtr &mask_msg);

 private:
  FrameId frame_id_;

  message_filters::Subscriber<sensor_msgs::msg::Image> rgb_image_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> depth_image_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> flow_image_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> mask_image_sub_;

  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

  rclcpp::CallbackGroup::SharedPtr imu_callback_group_;

  using ImuAdaptedType =
      rclcpp::adapt_type<dyno::ImuMeasurement>::as<sensor_msgs::msg::Imu>;
  rclcpp::Subscription<ImuAdaptedType>::SharedPtr imu_sub_;

  // for dyno_mpc
  using PoseWithCovarianceStampted =
      geometry_msgs::msg::PoseWithCovarianceStamped;
  rclcpp::Subscription<PoseWithCovarianceStampted>::SharedPtr gps_like_sub_;
};

}  // namespace dyno
