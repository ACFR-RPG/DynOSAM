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

#include "dynosam_ros/displays/inbuilt_displays/BackendInbuiltDisplayRos.hpp"

#include <glog/logging.h>
#include <pcl_conversions/pcl_conversions.h>

#include <dynosam/visualizer/ColourMap.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "dynosam_ros/RosUtils.hpp"
#include "rclcpp/qos.hpp"

namespace dyno {

BackendInbuiltDisplayRos::BackendInbuiltDisplayRos(const DisplayParams params,
                                                   rclcpp::Node::SharedPtr node)
    : InbuiltDisplayCommon(params, node) {
  // const rclcpp::QoS& sensor_data_qos = rclcpp::SensorDataQoS();
  static_tracked_points_pub_ =
      node->create_publisher<sensor_msgs::msg::PointCloud2>("static_cloud", 1);
  dynamic_tracked_points_pub_ =
      node->create_publisher<sensor_msgs::msg::PointCloud2>("dynamic_cloud", 1);

  odometry_pub_ = node->create_publisher<nav_msgs::msg::Odometry>("odom", 1);
  object_pose_pub_ =
      node->create_publisher<visualization_msgs::msg::MarkerArray>(
          "composed_object_poses", 1);
  object_pose_path_pub_ =
      node->create_publisher<visualization_msgs::msg::MarkerArray>(
          "composed_object_paths", 1);

  object_aabb_pub_ =
      node->create_publisher<visualization_msgs::msg::MarkerArray>(
          "object_aabb", 1);

  odometry_path_pub_ =
      node->create_publisher<nav_msgs::msg::Path>("odom_path", 2);
}

void BackendInbuiltDisplayRos::spinOnce(
    const BackendOutputPacket::ConstPtr& backend_output) {
  publishPointCloud(static_tracked_points_pub_,
                    backend_output->static_landmarks, backend_output->pose());
  CloudPerObject clouds_per_obj = publishPointCloud(
      dynamic_tracked_points_pub_, backend_output->dynamic_landmarks,
      backend_output->pose());

  publishObjectBoundingBox(object_aabb_pub_, nullptr, /* no Obb publisher */
                           clouds_per_obj, utils::fromRosTime(node_->now()),
                           "backend");
  publishObjectPositions(
      object_pose_pub_, backend_output->optimized_object_poses,
      backend_output->getFrameId(), backend_output->getTimestamp(), "backend");

  publishObjectPaths(object_pose_path_pub_,
                     backend_output->optimized_object_poses,
                     backend_output->getFrameId(),
                     backend_output->getTimestamp(), "backend", 60);

  {
    nav_msgs::msg::Odometry odom_msg;
    utils::convertWithHeader(backend_output->pose(), odom_msg,
                             backend_output->getTimestamp(), "world", "camera");
    odometry_pub_->publish(odom_msg);
  }

  {
    nav_msgs::msg::Path odom_path_msg;

    for (const gtsam::Pose3& T_world_camera :
         backend_output->optimized_camera_poses) {
      // optimized camera traj
      geometry_msgs::msg::PoseStamped pose_stamped;
      utils::convertWithHeader(T_world_camera, pose_stamped,
                               backend_output->getTimestamp(), "world");

      static std_msgs::msg::Header header;
      header.stamp = utils::toRosTime(backend_output->getTimestamp());
      header.frame_id = "world";
      odom_path_msg.header = header;
      odom_path_msg.poses.push_back(pose_stamped);
    }

    odometry_path_pub_->publish(odom_path_msg);
  }
}

}  // namespace dyno
