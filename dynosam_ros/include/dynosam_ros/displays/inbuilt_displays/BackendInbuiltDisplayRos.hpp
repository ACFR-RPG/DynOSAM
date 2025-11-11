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

#include <dynosam/visualizer/Display.hpp>

#include "dynosam_ros/Display-Definitions.hpp"
#include "dynosam_ros/displays/inbuilt_displays/InbuiltDisplayCommon.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

namespace dyno {

class BackendInbuiltDisplayRos : public BackendDisplay, InbuiltDisplayCommon {
 public:
  BackendInbuiltDisplayRos(const DisplayParams params,
                           rclcpp::Node::SharedPtr node);

  void spinOnceImpl(
      const BackendOutputPacket::ConstPtr& backend_output) override;

 private:
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      static_tracked_points_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      dynamic_tracked_points_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      dynamic_initial_points_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      new_scaled_dynamic_points_pub_;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_pub_;

  // optimzied
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr odometry_path_pub_;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      object_pose_path_pub_;  //! Path of propogated object poses using the
                              //! motion estimate
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      object_pose_pub_;  //! Propogated object poses using the motion estimate

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      object_aabb_pub_;  //! Draw object bounding boxes as cubes
};

}  // namespace dyno
