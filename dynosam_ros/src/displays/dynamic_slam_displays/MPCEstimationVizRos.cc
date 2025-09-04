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
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace dyno {

MPCEstimationVizRos::MPCEstimationVizRos(const DisplayParams params,
                                         rclcpp::Node::SharedPtr node,
                                         MPCFormulation* formulation)
    : params_(params),
      node_(node),
      prediction_transport_(params, node->create_sub_node("prediction")),
      formulation_(CHECK_NOTNULL(formulation)),
      tf_buffer_(node->get_clock()),
      tf_listener_(tf_buffer_) {
  cmd_vel_pub_ =
      node->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);
  local_goal_marker_pub_ =
      node->create_publisher<visualization_msgs::msg::MarkerArray>("local_goal",
                                                                   10);

  global_path_subscriber_ = node_->create_subscription<nav_msgs::msg::Path>(
      "global_plan", rclcpp::SensorDataQoS(),
      [&](const nav_msgs::msg::Path::ConstSharedPtr& global_path) -> void {
        // need it in the camera odom frame as this is what dynosam estimates
        // w.r.t however the goal is generated with respect to the world frame!!
        const auto target_frame = params_.world_frame_id;
        const auto fixed_frame = global_path->header.frame_id;
        // time in seconds
        Timestamp timestamp = utils::fromRosTime(global_path->header.stamp);
        tf2::TimePoint target_time = tf2::timeFromSec(timestamp);

        LOG(INFO) << "Looking up tf transfor " << fixed_frame << " -> "
                  << target_frame;

        try {
          // TODO: could (maybe should) do transform pose in one go and then
          // loop through and convert to Pose3Vector
          gtsam::Pose3Vector path;
          // for (const geometry_msgs::msg::PoseStamped& pose_stamped :
          // global_path->poses) {
          //   //for some reason the path comes with a default header (time 0,0
          //   and frame_id='')
          //   //this means that the tf_buffer_.transform does not work!
          //   geometry_msgs::msg::PoseStamped pose_stamped_with_proper_header =
          //   pose_stamped; pose_stamped_with_proper_header.header =
          //   global_path->header;

          //   geometry_msgs::msg::PoseStamped transformed_pose_ros =
          //     tf_buffer_.transform(
          //       pose_stamped_with_proper_header,
          //       target_frame,
          //       tf2::durationFromSec(0.1));

          //   gtsam::Pose3 pose_gtsam;
          //   convert(transformed_pose_ros.pose, pose_gtsam);
          //   path.push_back(pose_gtsam);
          // }

          // LOG(INFO) << "Recieved global plan at time " << timestamp;
          // CHECK(formulation_);
          // formulation_->updateGlobalPath(timestamp, path);

          geometry_msgs::msg::TransformStamped tf_stamped =
              tf_buffer_.lookupTransform(target_frame, fixed_frame,
                                         tf2::TimePointZero  // latest available
              );

          for (const geometry_msgs::msg::PoseStamped& pose_stamped :
               global_path->poses) {
            geometry_msgs::msg::PoseStamped pose_stamped_transform;
            tf2::doTransform(pose_stamped, pose_stamped_transform, tf_stamped);

            gtsam::Pose3 pose_gtsam;
            convert(pose_stamped_transform.pose, pose_gtsam);
            path.push_back(pose_gtsam);
          }

          LOG(INFO) << "Recieved global plan at time " << timestamp;
          CHECK(formulation_);
          formulation_->updateGlobalPath(timestamp, path);

        } catch (const tf2::TransformException& ex) {
          RCLCPP_WARN(node_->get_logger(),
                      "Transform failed: %s. Cannot set global path!",
                      ex.what());
        }
      });

  world_control_client_ =
      node_->create_client<ControlWorld>("/world/empty/control");
}

void MPCEstimationVizRos::inPreUpdate() { pauseWorld(); }
void MPCEstimationVizRos::inPostUpdate() { startWorld(); }

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
    publishLocalGoalMarker(*local_goal, timestamp, "Local Goal");
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

bool MPCEstimationVizRos::queryGlobalOffset(gtsam::Pose3& T_world_camera) {
  const std::string map_frame = "odom";  /// hardcoded!!

  try {
    geometry_msgs::msg::TransformStamped tf_stamped =
        tf_buffer_.lookupTransform(map_frame, params_.world_frame_id,
                                   tf2::TimePointZero);

    convert(tf_stamped, T_world_camera);
    return true;

  } catch (const tf2::TransformException& ex) {
    RCLCPP_WARN(node_->get_logger(),
                "Transform failed: %s. Cannot set get transform between map "
                "and dynosam odom!",
                ex.what());
    return false;
  }
}

void MPCEstimationVizRos::publishLocalGoalMarker(const gtsam::Pose3& pose,
                                                 Timestamp timestamp,
                                                 const std::string& name) {
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id =
      params_.world_frame_id;  // change to your fixed frame
  marker.header.stamp = utils::toRosTime(timestamp);
  marker.ns = name;
  marker.id = 0;
  marker.type =
      visualization_msgs::msg::Marker::ARROW;  // can be SPHERE, CUBE, etc.
  marker.action = visualization_msgs::msg::Marker::ADD;

  // Set position
  marker.pose.position.x = pose.translation().x();
  marker.pose.position.y = pose.translation().y();
  marker.pose.position.z = pose.translation().z();

  // Set orientation
  auto q = pose.rotation().toQuaternion();  // gtsam::Quaternion
  marker.pose.orientation.x = q.x();
  marker.pose.orientation.y = q.y();
  marker.pose.orientation.z = q.z();
  marker.pose.orientation.w = q.w();

  // Marker scale
  marker.scale.x = 0.2;  // shaft length for arrow
  marker.scale.y = 0.2;
  marker.scale.z = 0.2;

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

void MPCEstimationVizRos::pauseWorld() {
  // RCLCPP_INFO(node_->get_logger(), "Pausing simulation");
  if (!checkWorldControlServiceAvailable()) {
    RCLCPP_INFO(node_->get_logger(),
                "Unable to pause world: control service unavailable!");
    return;
  }

  auto request = std::make_shared<ControlWorld::Request>();
  request->world_control.pause = true;

  using ServiceResponseFuture = rclcpp::Client<ControlWorld>::SharedFuture;
  auto response_received_callback = [&](ServiceResponseFuture future) {
    (void)future;  // service response is empty
    RCLCPP_INFO(node_->get_logger(), "World paused successfully.");
  };

  world_control_client_->async_send_request(request,
                                            response_received_callback);

  // if (rclcpp::spin_until_future_complete(node_->get_node_base_interface(),
  // result) ==
  //     rclcpp::FutureReturnCode::SUCCESS)
  // {
  //   RCLCPP_INFO(node_->get_logger(), "World paused successfully.");
  // } else {
  //   RCLCPP_ERROR(node_->get_logger(), "Failed to pause world.");
  // }

  // NOTE: hack to get blocking servic call
  // if we do node_->get_node_base_interface() it tries to spin the same node
  // that is already managed by the top level executor. In reality we should do
  // an asynchronous call but here the point is to wait until completing before
  // continuing.
  // It actually might be fine to do the asynch call as this should not take
  // very long...
  //  rclcpp::executors::SingleThreadedExecutor exec;
  //  exec.add_node(node_->get_node_base_interface());

  // if (exec.spin_until_future_complete(result) ==
  // rclcpp::FutureReturnCode::SUCCESS)
  // {
  //   RCLCPP_INFO(node_->get_logger(), "World paused successfully.");
  // } else {
  //   RCLCPP_ERROR(node_->get_logger(), "Failed to paused world.");
  // }

  // exec.remove_node(node_->get_node_base_interface());
}

void MPCEstimationVizRos::startWorld() {
  RCLCPP_INFO(node_->get_logger(), "Starting simulation");

  if (!checkWorldControlServiceAvailable()) {
    RCLCPP_INFO(node_->get_logger(),
                "Unable to start world: control service unavailable!");
    return;
  }

  auto request = std::make_shared<ControlWorld::Request>();
  request->world_control.pause = false;

  using ServiceResponseFuture = rclcpp::Client<ControlWorld>::SharedFuture;
  auto response_received_callback = [&](ServiceResponseFuture future) {
    (void)future;  // service response is empty
    RCLCPP_INFO(node_->get_logger(), "World unpaused successfully.");
  };

  world_control_client_->async_send_request(request,
                                            response_received_callback);

  // auto result = world_control_client_->async_send_request(request);
  // if (rclcpp::spin_until_future_complete(node_->get_node_base_interface(),
  // result) ==
  //     rclcpp::FutureReturnCode::SUCCESS)
  // {
  //   RCLCPP_INFO(node_->get_logger(), "World unpaused successfully.");
  // } else {
  //   RCLCPP_ERROR(node_->get_logger(), "Failed to unpause world.");
  // }

  // rclcpp::executors::SingleThreadedExecutor exec;
  // exec.add_node(node_->get_node_base_interface());

  // if (exec.spin_until_future_complete(result) ==
  // rclcpp::FutureReturnCode::SUCCESS)
  // {
  //   RCLCPP_INFO(node_->get_logger(), "World unpaused successfully.");
  // } else {
  //   RCLCPP_ERROR(node_->get_logger(), "Failed to unpaused world.");
  // }

  // exec.remove_node(node_->get_node_base_interface());
}

bool MPCEstimationVizRos::checkWorldControlServiceAvailable(int timeout_s) {
  // return
  // world_control_client_->wait_for_service(std::chrono::seconds(timeout_s));
  return world_control_client_->service_is_ready();
}

}  // namespace dyno
