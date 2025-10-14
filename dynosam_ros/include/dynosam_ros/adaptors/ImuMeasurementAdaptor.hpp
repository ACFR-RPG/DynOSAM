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

#include <dynosam/frontend/imu/ImuMeasurements.hpp>

#include "dynosam_ros/RosUtils.hpp"
#include "rclcpp/type_adapter.hpp"
#include "sensor_msgs/msg/imu.hpp"

template <>
struct rclcpp::TypeAdapter<dyno::ImuMeasurement, sensor_msgs::msg::Imu> {
  using is_specialized = std::true_type;
  using custom_type = dyno::ImuMeasurement;
  using ros_message_type = sensor_msgs::msg::Imu;

  static void convert_to_ros_message(const custom_type& source,
                                     ros_message_type& destination) {
    // first timestamp
    dyno::convert(source.timestamp_, destination.header.stamp);

    // our imu does not contain a quaternion
    destination.linear_acceleration.x = source.acc_gyr_(0);
    destination.linear_acceleration.y = source.acc_gyr_(1);
    destination.linear_acceleration.z = source.acc_gyr_(2);

    destination.angular_velocity.x = source.acc_gyr_(3);
    destination.angular_velocity.y = source.acc_gyr_(4);
    destination.angular_velocity.z = source.acc_gyr_(5);
  }

  static void convert_to_custom(const ros_message_type& source,
                                custom_type& destination) {
    // first timestamp
    dyno::convert(source.header.stamp, destination.timestamp_);

    destination.acc_gyr_ << source.linear_acceleration.x,
        source.linear_acceleration.y, source.linear_acceleration.z,
        source.angular_velocity.x, source.angular_velocity.y,
        source.angular_velocity.z;
  }
};

RCLCPP_USING_CUSTOM_TYPE_AS_ROS_MESSAGE_TYPE(dyno::ImuMeasurement,
                                             sensor_msgs::msg::Imu);
