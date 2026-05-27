#include "dynosam_ros/displays/DisplaysCommon.hpp"

#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/memory.h>

#include <dynosam_common/Transforms.hpp>

#include "dynosam_common/PointCloudProcess.hpp"
#include "dynosam_common/viz/Colour.hpp"
#include "dynosam_ros/RosUtils.hpp"

namespace dyno {

CloudPerObject DisplayCommon::publishPointCloud(
    PointCloud2Pub::SharedPtr pub, const StatusLandmarkVector& landmarks,
    const std::string& frame_id, const gtsam::Pose3& T) {
  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  CloudPerObject clouds_per_obj = groupObjectCloud(
      landmarks, [&cloud](const pcl::PointXYZRGB& point, ObjectId) {
        cloud.points.push_back(point);
      });

  pcl::PointCloud<pcl::PointXYZRGB> filtered_and_merged_cloud;
  for (auto& [_, obj_cloud] : clouds_per_obj) {
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>(obj_cloud);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(
        new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(50);             // Number of neighbors to analyze
    sor.setStddevMulThresh(1.0);  // Threshold based on std dev
    sor.filter(*cloud_filtered);

    // update object cloud with the one that is actually published
    obj_cloud = *cloud_filtered;
    filtered_and_merged_cloud += (*cloud_filtered);
  }

  // apply final transform to cloud if needed
  Eigen::Matrix<float, 4, 4> transform = T.matrix().cast<float>();
  pcl::transformPointCloud(filtered_and_merged_cloud, filtered_and_merged_cloud,
                           transform);

  sensor_msgs::msg::PointCloud2 pc2_msg;
  pcl::toROSMsg(filtered_and_merged_cloud, pc2_msg);
  pc2_msg.header.frame_id = frame_id;
  pub->publish(pc2_msg);

  return clouds_per_obj;
}

void DisplayCommon::publishOdometry(OdometryPub::SharedPtr pub,
                                    const gtsam::Pose3& T_world_camera,
                                    Timestamp timestamp,
                                    const std::string& frame_id,
                                    const std::string& child_frame_id) {
  nav_msgs::msg::Odometry odom_msg;
  ros::convertWithHeader(T_world_camera, odom_msg, timestamp, frame_id,
                         child_frame_id);
  pub->publish(odom_msg);
}

void DisplayCommon::publishOdometryPath(PathPub::SharedPtr pub,
                                        const gtsam::Pose3Vector& poses,
                                        Timestamp latest_timestamp,
                                        const std::string& frame_id) {
  nav_msgs::msg::Path path;
  for (const gtsam::Pose3& odom : poses) {
    geometry_msgs::msg::PoseStamped pose_stamped;
    ros::convertWithHeader(odom, pose_stamped, latest_timestamp, frame_id);
    path.poses.push_back(pose_stamped);
  }

  path.header.stamp = ros::toRosTime(latest_timestamp);
  path.header.frame_id = frame_id;
  pub->publish(path);
}

std::vector<Marker> DisplayCommon::objectBBXToRvizMarker(
    const ObjectBBX& bounding_box, const ObjectId object_id,
    const Timestamp latest_timestamp, const std::string& frame_id) {
  std::vector<Marker> markers;

  Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = ros::toRosTime(latest_timestamp);
  marker.ns = "object_bbx";
  marker.id = object_id * 2;  // even ids for boxes
  marker.type = visualization_msgs::msg::Marker::CUBE;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.lifetime = rclcpp::Duration::from_seconds(3.0);

  std_msgs::msg::ColorRGBA colour_msg;
  convert(Color::uniqueId(object_id), colour_msg);
  marker.color = colour_msg;
  marker.color.a = 0.25f;  // translucent so points/objects remain visible

  // Compute scale from min/max points
  gtsam::Point3 size =
      bounding_box.max_bbx_point_ - bounding_box.min_bbx_point_;
  marker.scale.x = std::abs(size.x());
  marker.scale.y = std::abs(size.y());
  marker.scale.z = std::abs(size.z());

  geometry_msgs::msg::Pose pose_msg;
  // convert(bounding_box.pose(), pose);
  gtsam::Pose3 bbx_pose(gtsam::Rot3::Identity(), bounding_box.bbx_position_);
  convert(bbx_pose, pose_msg);

  marker.pose = pose_msg;
  markers.push_back(marker);

  // SOLID EDGES (wireframe outline)
  Marker edges;
  edges.header = marker.header;
  edges.ns = "object_bbx_edges";
  edges.id = object_id * 2 + 1;  // odd ids for edges
  edges.type = visualization_msgs::msg::Marker::LINE_LIST;
  edges.action = visualization_msgs::msg::Marker::ADD;
  edges.lifetime = rclcpp::Duration::from_seconds(3.0);

  edges.scale.x = 0.015;  // line thickness
  edges.color = marker.color;
  edges.color.a = 1.0f;  // fully opaque edges

  // Compute 8 corners
  std::vector<Eigen::Vector3d> corners;
  corners.reserve(8);
  Eigen::Vector3d min(bounding_box.min_bbx_point_.x(),
                      bounding_box.min_bbx_point_.y(),
                      bounding_box.min_bbx_point_.z());
  Eigen::Vector3d max(bounding_box.max_bbx_point_.x(),
                      bounding_box.max_bbx_point_.y(),
                      bounding_box.max_bbx_point_.z());

  for (int i = 0; i < 8; ++i)
    corners.emplace_back((i & 1) ? max.x() : min.x(),
                         (i & 2) ? max.y() : min.y(),
                         (i & 4) ? max.z() : min.z());

  // Transform for OBB
  const Eigen::Matrix3d R = bbx_pose.rotation().matrix();
  const Eigen::Vector3d t = bbx_pose.translation();
  for (auto& c : corners) c = R * (c - Eigen::Vector3d::Zero()) + t;

  auto add_edge = [&](int i, int j) {
    geometry_msgs::msg::Point p1, p2;
    p1.x = corners[i].x();
    p1.y = corners[i].y();
    p1.z = corners[i].z();
    p2.x = corners[j].x();
    p2.y = corners[j].y();
    p2.z = corners[j].z();
    edges.points.push_back(p1);
    edges.points.push_back(p2);
  };

  // 12 edges of a cube
  add_edge(0, 1);
  add_edge(1, 3);
  add_edge(3, 2);
  add_edge(2, 0);
  add_edge(4, 5);
  add_edge(5, 7);
  add_edge(7, 6);
  add_edge(6, 4);
  add_edge(0, 4);
  add_edge(1, 5);
  add_edge(2, 6);
  add_edge(3, 7);

  markers.push_back(edges);

  return markers;
}

Marker DisplayCommon::poseToCameraFrustrum(
    const gtsam::Pose3& T_WR, const Timestamp timestamp,
    const std::string& frame_id, const Color& colour, const int id,
    const double scale, const double line_width, const std::string& ns) {
  Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = ros::toRosTime(timestamp);

  marker.ns = ns;
  marker.id = id;

  marker.type = Marker::LINE_LIST;
  marker.action = Marker::ADD;

  marker.pose.orientation.w = 1.0;

  marker.scale.x = line_width;

  std_msgs::msg::ColorRGBA colour_msg;
  convert(colour, colour_msg);
  marker.color = colour_msg;
  marker.color.a = 1.0f;

  marker.lifetime = rclcpp::Duration::from_seconds(0.0);

  const double w = scale;
  const double h = w * 0.75;
  const double z = w * 0.6;

  // Frustum corners in camera optical frame
  const gtsam::Point3 O(0.0, 0.0, 0.0);

  const gtsam::Point3 P1(w, h, z);
  const gtsam::Point3 P2(w, -h, z);
  const gtsam::Point3 P3(-w, -h, z);
  const gtsam::Point3 P4(-w, h, z);

  auto transformPoint = [&](const gtsam::Point3& p_cam) {
    static const auto T_robot_opencv = openCVToRobotTransform();
    auto p_cam_robot = T_robot_opencv * p_cam;
    return T_WR.transformFrom(p_cam_robot);
  };

  auto toRosPoint = [](const gtsam::Point3& p) {
    geometry_msgs::msg::Point msg;
    dyno::convert(p, msg);
    return msg;
  };

  auto addLine = [&](const gtsam::Point3& a, const gtsam::Point3& b) {
    marker.points.push_back(toRosPoint(transformPoint(a)));
    marker.points.push_back(toRosPoint(transformPoint(b)));
  };

  // Pyramid edges
  addLine(O, P1);
  addLine(O, P2);
  addLine(O, P3);
  addLine(O, P4);

  // Front rectangle
  addLine(P1, P2);
  addLine(P4, P3);
  addLine(P4, P1);
  addLine(P3, P2);

  return marker;
}

}  // namespace dyno
