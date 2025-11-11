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

#include <pcl_conversions/pcl_conversions.h>

#include <sensor_msgs/msg/point_cloud2.hpp>

#include "dynosam_common/PointCloudProcess.hpp"  //for CloudPerObject
#include "dynosam_common/Types.hpp"
#include "dynosam_ros/Display-Definitions.hpp"
#include "image_transport/image_transport.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

namespace dyno {

using PointCloud2 = sensor_msgs::msg::PointCloud2;
using MarkerArray =
    visualization_msgs::msg::MarkerArray;        //! Typedef for MarkerArray msg
using Marker = visualization_msgs::msg::Marker;  //! Typedef for Marker msg

using PointCloud2Pub = rclcpp::Publisher<sensor_msgs::msg::PointCloud2>;
using OdometryPub = rclcpp::Publisher<nav_msgs::msg::Odometry>;
using PathPub = rclcpp::Publisher<nav_msgs::msg::Path>;
using MarkerArrayPub = rclcpp::Publisher<MarkerArray>;

/**
 * @brief Common stateless (free) functions for all ROS displays.
 *
 */
struct DisplayCommon {
  static CloudPerObject publishPointCloud(PointCloud2Pub::SharedPtr pub,
                                          const StatusLandmarkVector& landmarks,
                                          const gtsam::Pose3& T_world_camera,
                                          const std::string& frame_id);
  static void publishOdometry(OdometryPub::SharedPtr pub,
                              const gtsam::Pose3& T_world_camera,
                              Timestamp timestamp, const std::string& frame_id,
                              const std::string& child_frame_id);
  static void publishOdometryPath(PathPub::SharedPtr pub,
                                  const gtsam::Pose3Vector& poses,
                                  Timestamp latest_timestamp,
                                  const std::string& frame_id);

  static std::vector<Marker> objectBBXToRvizMarker(
      const ObjectBBX& bounding_box, const ObjectId object_id,
      const Timestamp latest_timestamp, const std::string& frame_id);
};

}  // namespace dyno
