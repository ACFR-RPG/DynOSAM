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

#include "dynosam_ros/displays/dynamic_slam_displays/FrontendDSDRos.hpp"

#include "cv_bridge/cv_bridge.hpp"
#include "dynosam_common/utils/SafeCast.hpp"
#include "dynosam_ros/RosUtils.hpp"
#include "rclcpp/qos.hpp"

namespace dyno {

FrontendDSDRos::FrontendDSDRos(const DisplayParams params,
                               rclcpp::Node::SharedPtr node,
                               rclcpp::Node::SharedPtr ground_truth_node)
    : FrontendDisplay(), DSDRos(params, node) {
  tracking_image_pub_ =
      image_transport::create_publisher(node.get(), "tracking_image");

  // const rclcpp::SensorDataQoS sensor_qos;
  dense_dynamic_cloud_pub_ =
      node->create_publisher<sensor_msgs::msg::PointCloud2>(
          "dense_labelled_cloud", rclcpp::SensorDataQoS());

  if (ground_truth_node) {
    RCLCPP_INFO_STREAM(node->get_logger(), "Creating ground truth publishers");
    ground_truth_publishers_ = GroundTruthPublishers(ground_truth_node);
  }
}

FrontendDSDRos::GroundTruthPublishers::GroundTruthPublishers(
    rclcpp::Node::SharedPtr ground_truth_node)
    : dsd_transport_(CHECK_NOTNULL(ground_truth_node)) {
  vo_publisher_ = ground_truth_node->create_publisher<nav_msgs::msg::Odometry>(
      "odometry", rclcpp::SensorDataQoS());
  vo_path_publisher_ = ground_truth_node->create_publisher<nav_msgs::msg::Path>(
      "odometry_path", rclcpp::SensorDataQoS());
}

void FrontendDSDRos::spinOnceImpl(
    const VisionImuPacket::ConstPtr& frontend_output) {
  updateAccumulatedDataStructured(frontend_output);

  // publish debug imagery
  tryPublishDebugImagery(frontend_output);

  // // publish ground truth
  tryPublishGroundTruth(frontend_output);

  // publish odometry
  tryPublishVisualOdometry(frontend_output);

  tryPublishPointClouds(frontend_output);

  tryPublishObjects(frontend_output);
}

void FrontendDSDRos::tryPublishDebugImagery(
    const VisionImuPacket::ConstPtr& frontend_output) {
  if (!frontend_output->debugImagery()) return;

  const DebugImagery& debug_imagery = *frontend_output->debugImagery();
  if (debug_imagery.tracking_image.empty()) return;

  std_msgs::msg::Header hdr;
  sensor_msgs::msg::Image::SharedPtr msg =
      cv_bridge::CvImage(hdr, "bgr8", debug_imagery.tracking_image)
          .toImageMsg();
  tracking_image_pub_.publish(msg);
}

void FrontendDSDRos::tryPublishGroundTruth(
    const VisionImuPacket::ConstPtr& frontend_output) {
  if (!ground_truth_publishers_ || !frontend_output->groundTruthPacket() ||
      !frontend_output->debugImagery())
    return;

  const DebugImagery& debug_imagery = *frontend_output->debugImagery();
  const cv::Mat& rgb_image = debug_imagery.rgb_viz;
  const auto timestamp = frontend_output->timestamp();
  const auto frame_id = frontend_output->frameId();

  if (rgb_image.empty()) return;

  // collect gt poses and motions
  static ObjectPoseMap poses;
  static ObjectMotionMap motions;
  static FrameIdTimestampMap timestamp_map;
  // hack! recreate the timestamp since it is not in the frontend base packet!!
  timestamp_map.insert2(frame_id, timestamp);

  const GroundTruthInputPacket& gt_packet =
      frontend_output->groundTruthPacket().value();

  for (const auto& object_pose_gt : gt_packet.object_poses_) {
    // check we have a gt motion here
    // in the case that we dont, this might be the first time the object
    // appears...
    if (!object_pose_gt.prev_H_current_world_) {
      continue;
    }

    poses.insert22(object_pose_gt.object_id_, gt_packet.frame_id_,
                   object_pose_gt.L_world_);

    // motion in world k-1 to k
    Motion3ReferenceFrame gt_motion(
        *object_pose_gt.prev_H_current_world_, MotionRepresentationStyle::F2F,
        ReferenceFrame::GLOBAL, gt_packet.frame_id_ - 1u, gt_packet.frame_id_);
    motions.insert22(object_pose_gt.object_id_, gt_packet.frame_id_, gt_motion);
  }

  GroundTruthPublishers& ground_truth_pubs = ground_truth_publishers_.value();
  // will this result in confusing tf's since the gt object and estimated
  // objects use the same link?
  DSDTransport::Publisher publisher =
      ground_truth_pubs.dsd_transport_.addObjectInfo(
          motions, poses, params_.world_frame_id, timestamp_map, frame_id,
          timestamp);
  publisher.publishObjectOdometry();

  // publish ground truth odom
  const gtsam::Pose3& T_world_camera = gt_packet.X_world_;
  nav_msgs::msg::Odometry odom_msg;
  utils::convertWithHeader(T_world_camera, odom_msg, timestamp,
                           params_.world_frame_id, params_.camera_frame_id);
  ground_truth_pubs.vo_publisher_->publish(odom_msg);

  // odom path gt
  // make static variable since we dont build up the path anywhere else
  // and just append the last gt camera pose to the path msg
  static nav_msgs::msg::Path gt_odom_path_msg;
  static std_msgs::msg::Header header;

  geometry_msgs::msg::PoseStamped pose_stamped;
  utils::convertWithHeader(T_world_camera, pose_stamped, timestamp,
                           params_.world_frame_id);

  header.stamp = utils::toRosTime(timestamp);
  header.frame_id = params_.world_frame_id;
  gt_odom_path_msg.header = header;
  gt_odom_path_msg.poses.push_back(pose_stamped);

  ground_truth_pubs.vo_path_publisher_->publish(gt_odom_path_msg);
}
void FrontendDSDRos::tryPublishVisualOdometry(
    const VisionImuPacket::ConstPtr& frontend_output) {
  // publish vo
  constexpr static bool kPublishOdomAsTf = true; 

  this->publishVisualOdometry(frontend_output->cameraPose(),
                              frontend_output->timestamp(), 
                              kPublishOdomAsTf,
                              frontend_output->getBodyVelocity());

  // relies on correct accumulation of internal objects
  this->publishVisualOdometryPath(camera_poses_, frontend_output->timestamp());
}

void FrontendDSDRos::tryPublishPointClouds(
    const VisionImuPacket::ConstPtr& frontend_output) {
  StatusLandmarkVector static_landmarks =
      frontend_output->staticLandmarkMeasurements();
  if (!static_landmarks.empty()) {
    this->publishStaticPointCloud(static_landmarks,
                                  frontend_output->cameraPose());
  }

  StatusLandmarkVector dynamic_landmarks =
      frontend_output->dynamicLandmarkMeasurements();
  if (!dynamic_landmarks.empty()) {
    this->publishDynamicPointCloud(dynamic_landmarks,
                                   frontend_output->cameraPose());
  }

  if (auto labelled_point_cloud = frontend_output->denseLabelledCloud();
      labelled_point_cloud) {
    sensor_msgs::msg::PointCloud2 pc2_msg;
    pcl::toROSMsg(*labelled_point_cloud, pc2_msg);
    pc2_msg.header.frame_id = params_.camera_frame_id;
    pc2_msg.header.stamp = utils::toRosTime(frontend_output->timestamp());
    dense_dynamic_cloud_pub_->publish(pc2_msg);
  }
}
void FrontendDSDRos::tryPublishObjects(
    const VisionImuPacket::ConstPtr& frontend_output) {
  // relies on correct accumulation of internal objects AND that the shared
  // module data is updated with timestamp/frame data (as in Display.hpp)
  const auto& object_motions = object_motions_;
  const auto& object_poses = object_poses_;
  const auto& timestamp_map = this->shared_module_info.getTimestampMap();
  const auto& label_map = this->shared_module_info.getClassLabelMap(); 

  DSDTransport::Publisher object_poses_publisher = dsd_transport_.addObjectInfo(
      object_motions, object_poses, params_.world_frame_id, timestamp_map,
      frontend_output->frameId(), frontend_output->timestamp(), label_map);
  object_poses_publisher.publishObjectOdometry();
  object_poses_publisher.publishObjectTransforms();
  object_poses_publisher.publishObjectPaths();
}

void FrontendDSDRos::updateAccumulatedDataStructured(
    const VisionImuPacket::ConstPtr& frontend_output) {
  camera_poses_.push_back(frontend_output->cameraPose());
  object_motions_.insert2(frontend_output->frameId(),
                          frontend_output->objectMotions());
  object_poses_.insert2(frontend_output->frameId(),
                        frontend_output->objectPoses());
}

}  // namespace dyno
