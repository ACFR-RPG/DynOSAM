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

#include "dynosam_ros/displays/dynamic_slam_displays/MPCEstimationVizRos.hpp"

#include <glog/logging.h>

#include "dynosam_ros/RosUtils.hpp"

namespace dyno {

MPCEstimationVizRos::MPCEstimationVizRos(const DisplayParams params,
                                         rclcpp::Node::SharedPtr node,
                                         MPCFormulation* formulation)
    : params_(params),
      node_(node),
      prediction_transport_(params, node->create_sub_node("prediction")),
      formulation_(CHECK_NOTNULL(formulation)) {
  cmd_vel_pub_ =
      node->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);
  local_goal_marker_pub_ =
      node->create_publisher<visualization_msgs::msg::MarkerArray>("local_goal",
                                                                   10);

  global_path_subscriber_ = node_->create_subscription<nav_msgs::msg::Path>(
      "global_plan", rclcpp::SensorDataQoS(),
      [&](const nav_msgs::msg::Path::ConstSharedPtr& global_path) -> void {
        Timestamp timestamp = utils::fromRosTime(global_path->header.stamp);

        gtsam::Pose3Vector path;
        for (const auto& pose_stamped_ros : global_path->poses) {
          gtsam::Pose3 pose_gtsam;
          convert(pose_stamped_ros.pose, pose_gtsam);

          // https://github.com/ros-navigation/navigation2/issues/2186
          // planner ignores frame_id request from action and assumes it matches
          // the global/map frame since our map frame is in robotic convention
          // and dynosam operates in opencv coonvention we need to manually
          // apply a transform we ignore the orientation so just use identity!
          auto translation_robotic = pose_gtsam.translation();
          // gtsam::Point3 translation_opencv(
          //   translation_robotic.y(),
          //   -translation_robotic.z(),
          //   translation_robotic.x()
          // );
          // path.push_back(gtsam::Pose3(gtsam::Rot3::Identity(),
          // translation_opencv));
        }
        LOG(INFO) << "Recieved global plan at time " << timestamp;
        CHECK(formulation_);
        formulation_->updateGlobalPath(timestamp, path);
      });
}

void MPCEstimationVizRos::spin(Timestamp timestamp, FrameId frame_k,
                               const MPCFormulation* formulation) {
  auto predicted_camera_poses = formulation->getPredictedCameraPoses(frame_k);
  auto [predicted_object_motions, predicted_object_poses] =
      formulation->getObjectPredictions(frame_k);

  LOG(INFO) << "Predicted poses of size " << predicted_camera_poses.size();

  prediction_transport_.publishVisualOdometryPath(predicted_camera_poses,
                                                  timestamp);

  // we dont have future timestamps so make ones up
  FrameIdTimestampMap fake_future_timestamps;
  FrameId frame_N = frame_k + formulation->horizon();
  // values init camera pose, 2dvelocity, 2d acceletation
  Timestamp future_t = timestamp;
  for (FrameId frame_id = frame_k; frame_id < frame_N; frame_id++) {
    fake_future_timestamps.insert2(frame_id, future_t);
    future_t += 1;
  }

  auto local_goal = formulation->local_goal_;
  if (local_goal) {
    LOG(INFO) << "Publishing local goal";
    publishLocalGoalMarker(*local_goal, "Local Goal");
  }

  auto command_query = formulation->getControlCommand(frame_k + 1);
  if (!command_query) {
    LOG(WARNING) << "Cannot emit control command for " << frame_k + 1
                 << ": Invalid query!!";
  } else {
    auto lim_lin_vel = formulation->lin_vel_;
    auto lim_ang_vel = formulation->ang_vel_;

    gtsam::Vector2 command = *command_query;
    double lin_vel = command.x();
    if (lin_vel < lim_lin_vel.min) {
      lin_vel = lim_lin_vel.min;
    } else if (lin_vel > lim_lin_vel.max) {
      lin_vel = lim_lin_vel.max;
    }

    double ang_vel = command.y();
    if (ang_vel < lim_ang_vel.min) {
      ang_vel = lim_ang_vel.min;
    } else if (ang_vel > lim_ang_vel.max) {
      ang_vel = lim_ang_vel.max;
    }

    LOG(INFO) << "Emitting command linear: " << lin_vel << " angular "
              << ang_vel << " t=" << timestamp << " k=" << frame_k;
    geometry_msgs::msg::Twist pub_msg;
    pub_msg.linear.x = lin_vel;
    pub_msg.angular.z = ang_vel;
    cmd_vel_pub_->publish(pub_msg);
  }

  DSDTransport::Publisher object_poses_publisher =
      prediction_transport_.getDSDTransport().addObjectInfo(
          predicted_object_motions, predicted_object_poses,
          params_.world_frame_id, fake_future_timestamps, frame_k, timestamp);
  object_poses_publisher.publishObjectPaths();

  LOG(INFO) << "In MPCEstimationVizRos spin";
}

void MPCEstimationVizRos::publishLocalGoalMarker(const gtsam::Pose3& pose,
                                                 const std::string& name) {
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id =
      params_.world_frame_id;  // change to your fixed frame
  marker.header.stamp = node_->get_clock()->now();
  marker.ns = "local_goal";
  marker.id = 0;
  marker.type =
      visualization_msgs::msg::Marker::SPHERE;  // can be SPHERE, CUBE, etc.
  marker.action = visualization_msgs::msg::Marker::ADD;

  // Set position
  marker.pose.position.x = pose.translation().x();
  marker.pose.position.y = pose.translation().y();
  marker.pose.position.z = pose.translation().z();

  // Set orientation
  auto q = pose.rotation().toQuaternion();  // gtsam::Quaternion
  marker.pose.orientation.x = 0;
  marker.pose.orientation.y = 0;
  marker.pose.orientation.z = 0;
  marker.pose.orientation.w = 1;

  // Marker scale
  marker.scale.x = 0.5;  // shaft length for arrow
  marker.scale.y = 0.5;
  marker.scale.z = 0.5;

  // Color (RGBA)
  marker.color.r = 0.0f;
  marker.color.g = 1.0f;
  marker.color.b = 0.0f;
  marker.color.a = 1.0f;

  // Optional: text marker for title
  visualization_msgs::msg::Marker text_marker;
  text_marker = marker;
  text_marker.id = 1;
  text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  text_marker.text = name;
  text_marker.pose.position.z += 0.3;  // offset above the arrow
  text_marker.scale.z = 0.1;
  text_marker.color.r = 1.0f;
  text_marker.color.g = 1.0f;
  text_marker.color.b = 1.0f;
  text_marker.color.a = 1.0f;

  // Publish as a MarkerArray
  visualization_msgs::msg::MarkerArray marker_array;
  marker_array.markers.push_back(marker);
  marker_array.markers.push_back(text_marker);

  local_goal_marker_pub_->publish(marker_array);
}

}  // namespace dyno
