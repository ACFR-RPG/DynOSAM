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

#include "dynamic_slam_interfaces/msg/multi_object_odometry_path.hpp"
#include "dynamic_slam_interfaces/msg/object_odometry.hpp"
#include "dynamic_slam_interfaces/msg/object_odometry_path.hpp"
#include "dynosam_common/DynoState.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/Macros.hpp"
#include "dynosam_ros/Display-Definitions.hpp"
#include "dynosam_ros/RosUtils.hpp"
#include "dynosam_ros/displays/DisplaysCommon.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/header.hpp"
#include "tf2_ros/transform_broadcaster.h"

namespace dyno {

using ObjectOdometry = dynamic_slam_interfaces::msg::ObjectOdometry;
using ObjectOdometryPath = dynamic_slam_interfaces::msg::ObjectOdometryPath;
using MultiObjectOdometryPath =
    dynamic_slam_interfaces::msg::MultiObjectOdometryPath;

using ObjectOdometryPub = rclcpp::Publisher<ObjectOdometry>;
using MultiObjectOdometryPathPub = rclcpp::Publisher<MultiObjectOdometryPath>;

//! Map of object id link (child frame id) to ObjectOdometry (for a single
//! frame, no frame ids)
using ObjectOdometryMap = gtsam::FastMap<std::string, ObjectOdometry>;

class DynoStatePublisher {
 public:
  DynoStatePublisher(const DisplayParams& params, rclcpp::Node::SharedPtr node);

  DYNO_POINTER_TYPEDEFS(DynoStatePublisher)

  void publish(const DynoState& state);

  DynoStatePublisher& publishVisualOdomTF(bool flag);
  DynoStatePublisher& publishObjectOdomTF(bool flag);

 private:
  void publishObjects(FrameId frame_id,
                      const MultiObjectTrajectories& object_trajectories);

  ObjectOdometry constructObjectOdometry(
      ObjectId object_id, FrameId frame_id,
      const PoseWithMotionTrajectory& trajectory) const;

  void sendObjectOdometryTransform(const ObjectOdometry& object_odom);

  /**
   * @brief Publishes a pose like object as a transform.
   * T must be convertable to a geometry_msgs::msg::Transform via the
   * dyno::convert function.
   * @tparam T
   * @param pose const T&
   * @param header const std_msgs::msg::Header&
   * @param child_frame_link const std::string&
   */
  template <typename T>
  void sendTransform(const T& pose, const std_msgs::msg::Header& header,
                     const std::string& child_frame_link) {
    geometry_msgs::msg::TransformStamped t;

    dyno::convert(pose, t.transform);
    t.header = header;
    t.child_frame_id = child_frame_link;

    tf_broadcaster_->sendTransform(t);
  }

  /**
   * @brief Publishes a pose like object as a transform.
   * See sendTransform(const T&, const std_msgs::msg::Header&, const
   * std::string& ).
   * @tparam T
   * @param pose const T&
   * @param timestamp const Timestamp&
   * @param frame_link const std::string&
   * @param child_frame_link const std::string&
   */
  template <typename T>
  void sendTransform(const T& pose, const Timestamp& timestamp,
                     const std::string& frame_link,
                     const std::string& child_frame_link) {
    std_msgs::msg::Header header;
    header.stamp = utils::toRosTime(timestamp);
    header.frame_id = frame_link;

    sendTransform(pose, header, child_frame_link);
  }

 protected:
  const DisplayParams params_;
  rclcpp::Node::SharedPtr node_;
  //! TF broadcaster for the odometry.
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  OdometryPub::SharedPtr vo_publisher_;
  PathPub::SharedPtr vo_path_publisher_;

  ObjectOdometryPub::SharedPtr object_odom_publisher_;
  MultiObjectOdometryPathPub::SharedPtr multi_object_odom_path_publisher_;

  PointCloud2Pub::SharedPtr static_points_pub_;
  PointCloud2Pub::SharedPtr dynamic_points_pub_;

  //! Settings

  //! Publish TF for visual odom
  bool publish_vo_tf_{false};
  //! Publish TF for object odometry
  bool publish_oo_tf_{false};
};

}  // namespace dyno
