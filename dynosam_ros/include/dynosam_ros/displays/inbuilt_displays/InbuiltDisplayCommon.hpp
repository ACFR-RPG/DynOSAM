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

#include <dynosam/common/Camera.hpp>
#include <dynosam/common/Exceptions.hpp>
#include <dynosam/common/PointCloudProcess.hpp>
#include <dynosam/common/Types.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "dynosam_ros/Display-Definitions.hpp"
#include "dynosam_ros/displays/DisplaysCommon.hpp"
#include "image_transport/image_transport.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

namespace dyno {

class InbuiltDisplayCommon {
 public:
  InbuiltDisplayCommon(const ReferenceFrames& params,
                       rclcpp::Node::SharedPtr node);
  virtual ~InbuiltDisplayCommon() = default;

  virtual CloudPerObject publishPointCloud(
      PointCloud2Pub::SharedPtr pub, const StatusLandmarkVector& landmarks,
      const gtsam::Pose3& T_world_camera);
  virtual void publishOdometry(OdometryPub::SharedPtr pub,
                               const gtsam::Pose3& T_world_camera,
                               Timestamp timestamp);
  virtual void publishOdometryPath(PathPub::SharedPtr pub,
                                   const gtsam::Pose3Vector& poses,
                                   Timestamp latest_timestamp);

  virtual void publishObjectPositions(
      MarkerArrayPub::SharedPtr pub, const ObjectPoseMap& object_positions,
      FrameId frame_id, Timestamp latest_timestamp,
      const std::string& prefix_marker_namespace, bool draw_labels = false,
      double scale = 1.0);

  // if -1 min_poses, draw all
  virtual void publishObjectPaths(MarkerArrayPub::SharedPtr pub,
                                  const ObjectPoseMap& object_positions,
                                  FrameId frame_id, Timestamp latest_timestamp,
                                  const std::string& prefix_marker_namespace,
                                  const int min_poses = 60);

  virtual void publishObjectBoundingBox(
      MarkerArrayPub::SharedPtr aabb_pub, MarkerArrayPub::SharedPtr obb_pub,
      const CloudPerObject& cloud_per_object, Timestamp timestamp,
      const std::string& prefix_marker_namespace);

  /**
   * @brief Constructs axis-aligned BB (AABB) and oriented BB (OBB) marker
   * arrays from a set of point clouds which are coloured per object.
   *
   * NOTE:(jesse) right now OBB are not gravity aligned
   *
   * @param cloud_per_object
   * @param aabb_markers
   * @param obb_markers
   * @param latest_timestamp
   * @param prefix_marker_namespace
   */
  static void constructBoundingBoxeMarkers(
      const CloudPerObject& cloud_per_object, MarkerArray& aabb_markers,
      MarkerArray& obb_markers, Timestamp timestamp,
      const std::string& prefix_marker_namespace);

  static void createAxisMarkers(const gtsam::Pose3& pose,
                                MarkerArray& axis_markers, Timestamp timestamp,
                                const cv::Scalar& colour,
                                const std::string& frame, const std::string& ns,
                                double length = 0.8, double radius = 0.1);

  // T_world_x should put something in the local frame to the world frame
  // taken from
  // https://github.com/uzh-rpg/rpg_svo_pro_open/blob/master/vikit/vikit_ros/src/output_helper.cpp
  MarkerArray createCameraMarker(const gtsam::Pose3& T_world_x,
                                 Timestamp timestamp, const std::string& ns,
                                 const cv::Scalar& colour,
                                 double marker_scale = 1.0);

  void publishCameraMarker(const gtsam::Pose3& T_world_x, Timestamp timestamp,
                           const std::string& ns, const cv::Scalar& colour,
                           double marker_scale = 1.0) {
    camera_frustrum_pub_->publish(
        createCameraMarker(T_world_x, timestamp, ns, colour, marker_scale));
  }

  inline visualization_msgs::msg::Marker getDeletionMarker() const {
    static visualization_msgs::msg::Marker delete_marker;
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    return delete_marker;
  }

 protected:
  const ReferenceFrames params_;
  rclcpp::Node::SharedPtr node_;

  MarkerArrayPub::SharedPtr
      camera_frustrum_pub_;  //! publishes camera pose as a marker array,
                             //! representing the frustrum, in the world frame
};

}  // namespace dyno
