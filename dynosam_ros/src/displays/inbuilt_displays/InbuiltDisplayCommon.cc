/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam_ros/displays/inbuilt_displays/InbuiltDisplayCommon.hpp"

#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/memory.h>

#include <dynosam/common/PointCloudProcess.hpp>
#include <dynosam/visualizer/ColourMap.hpp>

#include "dynosam_ros/RosUtils.hpp"
#include "dynosam_ros/displays/DisplaysCommon.hpp"

namespace dyno {

InbuiltDisplayCommon::InbuiltDisplayCommon(const DisplayParams& params,
                                           rclcpp::Node::SharedPtr node)
    : params_(params), node_(node) {}

CloudPerObject InbuiltDisplayCommon::publishPointCloud(
    PointCloud2Pub::SharedPtr pub, const StatusLandmarkVector& landmarks,
    const gtsam::Pose3& T_world_camera) {
  return DisplayCommon::publishPointCloud(pub, landmarks, T_world_camera,
                                          params_.world_frame_id);
}

void InbuiltDisplayCommon::publishOdometry(OdometryPub::SharedPtr pub,
                                           const gtsam::Pose3& T_world_camera,
                                           Timestamp timestamp) {
  DisplayCommon::publishOdometry(pub, T_world_camera, timestamp,
                                 params_.world_frame_id,
                                 params_.camera_frame_id);
}

void InbuiltDisplayCommon::publishOdometryPath(PathPub::SharedPtr pub,
                                               const gtsam::Pose3Vector& poses,
                                               Timestamp latest_timestamp) {
  DisplayCommon::publishOdometryPath(pub, poses, latest_timestamp,
                                     params_.world_frame_id);
}

void InbuiltDisplayCommon::publishObjectPositions(
    MarkerArrayPub::SharedPtr pub, const ObjectPoseMap& object_positions,
    FrameId frame_id, Timestamp latest_timestamp,
    const std::string& prefix_marker_namespace, bool draw_labels,
    double scale) {
  visualization_msgs::msg::MarkerArray object_pose_marker_array;
  static visualization_msgs::msg::Marker delete_marker;
  delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;

  auto ros_time = utils::toRosTime(latest_timestamp);

  object_pose_marker_array.markers.push_back(delete_marker);

  for (const auto& [object_id, poses_map] : object_positions) {
    // do not draw if in current frame
    if (!poses_map.exists(frame_id)) {
      continue;
    }

    const gtsam::Pose3& pose = poses_map.at(frame_id);

    // assume
    // object centroid per frame
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = params_.world_frame_id;
    marker.ns = prefix_marker_namespace + "_object_positions";
    marker.id = object_id;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.header.stamp = ros_time;
    marker.pose.position.x = pose.x();
    marker.pose.position.y = pose.y();
    marker.pose.position.z = pose.z();
    marker.scale.x = scale;
    marker.scale.y = scale;
    marker.scale.z = scale;
    marker.color.a = 1.0;  // Don't forget to set the alpha!

    const cv::Scalar colour = Color::uniqueId(object_id);
    marker.color.r = colour(0) / 255.0;
    marker.color.g = colour(1) / 255.0;
    marker.color.b = colour(2) / 255.0;

    object_pose_marker_array.markers.push_back(marker);

    if (draw_labels) {
      visualization_msgs::msg::Marker text_marker = marker;
      text_marker.ns = prefix_marker_namespace + "_object_labels";
      text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text_marker.text = std::to_string(object_id);
      text_marker.pose.position.z += 1.0;  // make it higher than the pose
                                           // marker
      text_marker.scale.z = 0.7;
      object_pose_marker_array.markers.push_back(text_marker);
    }
  }

  pub->publish(object_pose_marker_array);
}

void InbuiltDisplayCommon::publishObjectPaths(
    MarkerArrayPub::SharedPtr pub, const ObjectPoseMap& object_positions,
    FrameId frame_id, Timestamp latest_timestamp,
    const std::string& prefix_marker_namespace, const int min_poses) {
  visualization_msgs::msg::MarkerArray object_path_marker_array;
  static visualization_msgs::msg::Marker delete_marker;
  delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;

  auto ros_time = utils::toRosTime(latest_timestamp);
  object_path_marker_array.markers.push_back(delete_marker);

  for (const auto& [object_id, poses_map] : object_positions) {
    if (poses_map.size() < 2u) {
      continue;
    }

    // draw a line list for viz
    visualization_msgs::msg::Marker line_list_marker;
    line_list_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    line_list_marker.header.frame_id = "world";
    line_list_marker.ns = prefix_marker_namespace + "_object_path";
    line_list_marker.id = object_id;
    line_list_marker.header.stamp = ros_time;
    line_list_marker.scale.x = 0.05;

    line_list_marker.pose.orientation.x = 0;
    line_list_marker.pose.orientation.y = 0;
    line_list_marker.pose.orientation.z = 0;
    line_list_marker.pose.orientation.w = 1;

    const cv::Scalar colour = Color::uniqueId(object_id);
    line_list_marker.color.r = colour(0) / 255.0;
    line_list_marker.color.g = colour(1) / 255.0;
    line_list_marker.color.b = colour(2) / 255.0;
    line_list_marker.color.a = 1;

    size_t traj_size;
    // draw all the poses
    if (min_poses == -1) {
      traj_size = poses_map.size();
    } else {
      traj_size = std::min(min_poses, static_cast<int>(poses_map.size()));
    }

    // totally assume in order
    auto map_iter = poses_map.end();
    // equivalant to map_iter-traj_size + 1
    // we want to go backwards to the starting point specified by trajectory
    // size
    std::advance(map_iter, (-traj_size + 1));
    for (; map_iter != poses_map.end(); map_iter++) {
      auto prev_iter = std::prev(map_iter, 1);
      const gtsam::Pose3& prev_pose = prev_iter->second;
      ;
      const gtsam::Pose3& curr_pose = map_iter->second;

      // check frames are consequative
      //  CHECK_EQ(prev_iter->first + 1, map_iter->first) << " For object " <<
      //  object_id;
      LOG_IF(INFO, prev_iter->first + 1 != map_iter->first)
          << " Frames not consequative for object " << object_id;

      {
        geometry_msgs::msg::Point p;
        p.x = prev_pose.x();
        p.y = prev_pose.y();
        p.z = prev_pose.z();

        line_list_marker.points.push_back(p);
      }

      {
        geometry_msgs::msg::Point p;
        p.x = curr_pose.x();
        p.y = curr_pose.y();
        p.z = curr_pose.z();

        line_list_marker.points.push_back(p);
      }
    }

    object_path_marker_array.markers.push_back(line_list_marker);
  }

  pub->publish(object_path_marker_array);
}

void InbuiltDisplayCommon::publishObjectBoundingBox(
    MarkerArrayPub::SharedPtr aabb_pub, MarkerArrayPub::SharedPtr obb_pub,
    const CloudPerObject& cloud_per_object, Timestamp timestamp,
    const std::string& prefix_marker_namespace) {
  if (!aabb_pub && !obb_pub) {
    return;
  }

  visualization_msgs::msg::MarkerArray aabb_markers;  // using linelist
  visualization_msgs::msg::MarkerArray obb_markers;   // using cube
  aabb_markers.markers.push_back(getDeletionMarker());
  obb_markers.markers.push_back(getDeletionMarker());

  constructBoundingBoxeMarkers(cloud_per_object, aabb_markers, obb_markers,
                               timestamp, prefix_marker_namespace);

  if (aabb_pub) aabb_pub->publish(aabb_markers);
  if (obb_pub) aabb_pub->publish(obb_markers);
}

void InbuiltDisplayCommon::createAxisMarkers(
    const gtsam::Pose3& pose, MarkerArray& axis_markers, Timestamp timestamp,
    const cv::Scalar& colour, const std::string& frame, const std::string& ns,
    double length, double radius) {
  const auto pose_matrix = pose.matrix();
  Eigen::Isometry3d x = Eigen::Translation3d(length / 2.0, 0, 0) *
                        Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitY());
  gtsam::Pose3 x_pose(pose_matrix * x.matrix());

  // Publish y axis
  Eigen::Isometry3d y = Eigen::Translation3d(0, length / 2.0, 0) *
                        Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitX());
  gtsam::Pose3 y_pose(pose_matrix * y.matrix());

  Eigen::Isometry3d z = Eigen::Translation3d(0, 0, length / 2.0) *
                        Eigen::AngleAxisd(0, Eigen::Vector3d::UnitZ());
  gtsam::Pose3 z_pose(pose_matrix * z.matrix());

  const auto ros_time = utils::toRosTime(timestamp);

  auto make_cylinder =
      [&length, &radius, &ros_time, &ns, &frame,
       &colour](const gtsam::Pose3& pose) -> visualization_msgs::msg::Marker {
    visualization_msgs::msg::Marker marker;
    marker.header.stamp = ros_time;
    marker.ns = ns;
    marker.header.frame_id = frame;

    geometry_msgs::msg::Pose pose_msg;
    dyno::convert(pose, pose_msg);

    marker.pose = pose_msg;
    marker.scale.x = radius;
    marker.scale.y = radius;
    marker.scale.z = length;

    marker.color.r = colour(0) / 255.0;
    marker.color.g = colour(1) / 255.0;
    marker.color.b = colour(2) / 255.0;

    return marker;
  };

  axis_markers.markers.push_back(make_cylinder(x_pose));
  axis_markers.markers.push_back(make_cylinder(y_pose));
  axis_markers.markers.push_back(make_cylinder(z_pose));
}

void InbuiltDisplayCommon::constructBoundingBoxeMarkers(
    const CloudPerObject& cloud_per_object, MarkerArray& aabb_markers,
    MarkerArray& obb_markers, Timestamp timestamp,
    const std::string& prefix_marker_namespace) {
  auto ros_time = utils::toRosTime(timestamp);

  for (const auto& [object_id, obj_cloud] : cloud_per_object) {
    pcl::PointXYZ centroid;
    pcl::computeCentroid(obj_cloud, centroid);

    const cv::Scalar colour = Color::uniqueId(object_id);

    visualization_msgs::msg::Marker txt_marker;
    txt_marker.header.frame_id = "world";
    txt_marker.ns = prefix_marker_namespace + "_object_id_txt";
    txt_marker.id = object_id;
    txt_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    txt_marker.action = visualization_msgs::msg::Marker::ADD;
    txt_marker.header.stamp = ros_time;

    txt_marker.scale.z = 0.5;
    // txt_marker.scale.z = 2.0;
    txt_marker.color.r = colour(0) / 255.0;
    txt_marker.color.g = colour(1) / 255.0;
    txt_marker.color.b = colour(2) / 255.0;
    txt_marker.color.a = 1;
    txt_marker.text = "Obj " + std::to_string(object_id);
    txt_marker.pose.position.x = centroid.x;
    // txt_marker.pose.position.y = centroid.y - 2.0;
    // txt_marker.pose.position.z = centroid.z - 1.0;
    txt_marker.pose.position.y = centroid.y - 0.6;
    txt_marker.pose.position.z = centroid.z - 0.5;
    aabb_markers.markers.push_back(txt_marker);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr obj_cloud_ptr =
        pcl::make_shared<pcl::PointCloud<pcl::PointXYZRGB> >(obj_cloud);
    ObjectBBX aabb = findAABBFromCloud<pcl::PointXYZRGB>(obj_cloud_ptr);
    ObjectBBX obb = findOBBFromCloud<pcl::PointXYZRGB>(obj_cloud_ptr);

    visualization_msgs::msg::Marker aabb_marker;
    aabb_marker.header.frame_id = "world";
    aabb_marker.ns = prefix_marker_namespace + "object_aabb";
    aabb_marker.id = object_id;
    aabb_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
    aabb_marker.action = visualization_msgs::msg::Marker::ADD;
    aabb_marker.header.stamp = ros_time;
    aabb_marker.scale.x = 0.03;

    aabb_marker.pose.orientation.x = 0;
    aabb_marker.pose.orientation.y = 0;
    aabb_marker.pose.orientation.z = 0;
    aabb_marker.pose.orientation.w = 1;

    aabb_marker.color.r = colour(0) / 255.0;
    aabb_marker.color.g = colour(1) / 255.0;
    aabb_marker.color.b = colour(2) / 255.0;
    aabb_marker.color.a = 1;

    for (const pcl::PointXYZ& this_line_list_point :
         findLineListPointsFromAABBMinMax(aabb.min_bbx_point_,
                                          aabb.max_bbx_point_)) {
      geometry_msgs::msg::Point p;
      p.x = this_line_list_point.x;
      p.y = this_line_list_point.y;
      p.z = this_line_list_point.z;
      aabb_marker.points.push_back(p);
    }
    aabb_markers.markers.push_back(aabb_marker);

    visualization_msgs::msg::Marker obbx_marker;
    obbx_marker.header.frame_id = "world";
    obbx_marker.ns = prefix_marker_namespace + "object_obb";
    obbx_marker.id = object_id;
    obbx_marker.type = visualization_msgs::msg::Marker::CUBE;
    obbx_marker.action = visualization_msgs::msg::Marker::ADD;
    obbx_marker.header.stamp = ros_time;
    obbx_marker.scale.x = obb.max_bbx_point_.x() - obb.min_bbx_point_.x();
    obbx_marker.scale.y = obb.max_bbx_point_.y() - obb.min_bbx_point_.y();
    obbx_marker.scale.z = obb.max_bbx_point_.z() - obb.min_bbx_point_.z();

    obbx_marker.pose.position.x = obb.bbx_position_.x();
    obbx_marker.pose.position.y = obb.bbx_position_.y();
    obbx_marker.pose.position.z = obb.bbx_position_.z();

    const gtsam::Quaternion& q = obb.orientation_.toQuaternion();
    obbx_marker.pose.orientation.x = q.x();
    obbx_marker.pose.orientation.y = q.y();
    obbx_marker.pose.orientation.z = q.z();
    obbx_marker.pose.orientation.w = q.w();

    obbx_marker.color.r = colour(0) / 255.0;
    obbx_marker.color.g = colour(1) / 255.0;
    obbx_marker.color.b = colour(2) / 255.0;
    obbx_marker.color.a = 0.2;

    obb_markers.markers.push_back(obbx_marker);
  }
}

MarkerArray InbuiltDisplayCommon::createCameraMarker(
    const gtsam::Pose3& T_world_x, Timestamp timestamp, const std::string& ns,
    const cv::Scalar& colour, double marker_scale) {
  auto transform_odom_to =
      [&T_world_x](
          const geometry_msgs::msg::Pose& msg) -> geometry_msgs::msg::Pose {
    // convert msg to pose
    gtsam::Pose3 msg_as_pose;
    dyno::convert(msg, msg_as_pose);
    // transform pose into world frame
    gtsam::Pose3 transformed_pose = T_world_x * msg_as_pose;
    // transform pose back to ROS msg and return
    geometry_msgs::msg::Pose transformed_msg;
    dyno::convert(transformed_pose, transformed_msg);
    return transformed_msg;
  };

  auto ros_time = utils::toRosTime(timestamp);

  // make rectangles as frame
  const double r_w = 1.0;
  const double z_plane = (r_w / 2.0) * marker_scale;

  static constexpr double sqrt2_2 = sqrt(2) / 2;

  MarkerArray marker_array;
  visualization_msgs::msg::Marker marker;

  // the marker will be displayed in frame_id
  marker.header.frame_id = params_.world_frame_id;
  marker.header.stamp = ros_time;
  marker.ns = ns;
  marker.action = 0;
  marker.id = 8;  // 8 of 8 markers

  marker.type = visualization_msgs::msg::Marker::CUBE;
  marker.scale.x = r_w * marker_scale;
  marker.scale.y = 0.04 * marker_scale;
  marker.scale.z = 0.04 * marker_scale;
  marker.color.r = colour(0) / 255.0;
  marker.color.g = colour(1) / 255.0;
  marker.color.b = colour(2) / 255.0;
  marker.color.a = 1.0;

  marker.pose.orientation.x = 0;
  marker.pose.orientation.y = 0;
  marker.pose.orientation.z = 0;
  marker.pose.orientation.w = 1;
  marker.id--;
  marker_array.markers.push_back(marker);
  marker.pose.position.y = -(r_w / 4.0) * marker_scale;
  marker.id--;
  marker_array.markers.push_back(marker);

  marker.scale.x = (r_w / 2.0) * marker_scale;
  marker.pose.position.x = (r_w / 2.0) * marker_scale;
  marker.pose.position.y = 0;
  marker.pose.orientation.w = sqrt2_2;
  marker.pose.orientation.z = sqrt2_2;
  marker.id--;
  marker_array.markers.push_back(marker);
  marker.pose.position.x = -(r_w / 2.0) * marker_scale;
  marker.id--;
  marker_array.markers.push_back(marker);

  // make pyramid edges
  marker.scale.x = (3.0 * r_w / 4.0) * marker_scale;
  marker.pose.position.z = 0.5 * z_plane;

  marker.pose.position.x = (r_w / 4.0) * marker_scale;
  marker.pose.position.y = (r_w / 8.0) * marker_scale;
  //  0.08198092, -0.34727674,  0.21462883,  0.9091823
  marker.pose.orientation.x = 0.08198092;
  marker.pose.orientation.y = -0.34727674;
  marker.pose.orientation.z = 0.21462883;
  marker.pose.orientation.w = 0.9091823;
  marker.id--;
  marker_array.markers.push_back(marker);

  marker.pose.position.x = -(r_w / 4.0) * marker_scale;
  marker.pose.position.y = (r_w / 8.0) * marker_scale;
  // -0.27395078, -0.22863284,  0.9091823 ,  0.21462883
  marker.pose.orientation.x = 0.08198092;
  marker.pose.orientation.y = 0.34727674;
  marker.pose.orientation.z = -0.21462883;
  marker.pose.orientation.w = 0.9091823;
  marker.id--;
  marker_array.markers.push_back(marker);

  marker.pose.position.x = -(r_w / 4.0) * marker_scale;
  marker.pose.position.y = -(r_w / 8.0) * marker_scale;
  //  -0.08198092,  0.34727674,  0.21462883,  0.9091823
  marker.pose.orientation.x = -0.08198092;
  marker.pose.orientation.y = 0.34727674;
  marker.pose.orientation.z = 0.21462883;
  marker.pose.orientation.w = 0.9091823;
  marker.id--;
  marker_array.markers.push_back(marker);

  marker.pose.position.x = (r_w / 4.0) * marker_scale;
  marker.pose.position.y = -(r_w / 8.0) * marker_scale;
  // -0.08198092, -0.34727674, -0.21462883,  0.9091823
  marker.pose.orientation.x = -0.08198092;
  marker.pose.orientation.y = -0.34727674;
  marker.pose.orientation.z = -0.21462883;
  marker.pose.orientation.w = 0.9091823;
  marker.id--;
  marker_array.markers.push_back(marker);

  for (auto& marker : marker_array.markers) {
    // put in world frame
    marker.pose = transform_odom_to(marker.pose);
  }
  return marker_array;
}

}  // namespace dyno
