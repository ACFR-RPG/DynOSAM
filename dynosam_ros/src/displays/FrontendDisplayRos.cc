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

#include "dynosam_ros/displays/FrontendDisplayRos.hpp"

#include "cv_bridge/cv_bridge.hpp"
#include "dynosam_ros/RosUtils.hpp"
#include "rclcpp/qos.hpp"

namespace dyno {

FrontendDisplayRos::FrontendDisplayRos(
    const CanonicalSensorRig::ConstPtr& sensor_rig,
    rclcpp::Node::SharedPtr node, rclcpp::Node::SharedPtr ground_truth_node)
    : FrontendDisplay(),
      dyno_state_publisher_(sensor_rig, node,
                            // pulish vo, oo transforms and wireframe cameras
                            // and convert to optical
                            {true, true, true, true}) {
  tracking_image_pub_ =
      image_transport::create_publisher(node.get(), "tracking_image");

  // const rclcpp::SensorDataQoS sensor_qos;
  dense_dynamic_cloud_pub_ =
      node->create_publisher<sensor_msgs::msg::PointCloud2>(
          "dense_labelled_cloud", rclcpp::SensorDataQoS());

  if (ground_truth_node) {
    RCLCPP_INFO_STREAM(node->get_logger(), "Creating ground truth publishers");
    // dont convert to optical for now (hack as we precompute gt poses in robot
    // frame) this is just for visualisation during testing of dataset

    ground_truth_publishers_.emplace(
        sensor_rig, ground_truth_node,
        DynoStatePublisherOptions{true, true, true, false});
  }
}

FrontendDisplayRos::GroundTruthPublishers::GroundTruthPublishers(
    const CanonicalSensorRig::ConstPtr& sensor_rig,
    rclcpp::Node::SharedPtr ground_truth_node,
    const DynoStatePublisherOptions& options)
    : dyno_state_publisher_(sensor_rig, CHECK_NOTNULL(ground_truth_node),
                            options) {}

void FrontendDisplayRos::spinOnce(
    const RealtimeOutput::ConstPtr& frontend_output) {
  VLOG(20) << "Spinning FrontendDisplayRos k="
           << frontend_output->state.frame_id;
  dyno_state_publisher_.publish(frontend_output->state);

  // publish debug imagery
  tryPublishDebugImagery(frontend_output);

  // // // publish ground truth
  tryPublishGroundTruth(frontend_output);
}

void FrontendDisplayRos::tryPublishDebugImagery(
    const RealtimeOutput::ConstPtr& frontend_output) {
  const DebugImagery& debug_imagery = frontend_output->debug_imagery;
  if (debug_imagery.tracking_image.empty()) return;

  std_msgs::msg::Header hdr;
  sensor_msgs::msg::Image::SharedPtr msg =
      cv_bridge::CvImage(hdr, "bgr8", debug_imagery.tracking_image)
          .toImageMsg();
  tracking_image_pub_.publish(msg);
}

void FrontendDisplayRos::tryPublishGroundTruth(
    const RealtimeOutput::ConstPtr& frontend_output) {
  // for historical and structural reasons we expect the ground truth packet
  // to be provided in the frontend output for cisualisation
  if (!ground_truth_publishers_ || !frontend_output->ground_truth) return;

  DynoState& ground_truth_dyno_state =
      ground_truth_publishers_->ground_truth_state_;

  const GroundTruthInputPacket& gt_packet =
      frontend_output->ground_truth.value();

  const auto frame_id = gt_packet.frame_id_;
  const auto timestamp = gt_packet.timestamp_;

  // must update frame id and timestamp on state otherwise publishing wont work
  // as it uses the frame id to build the object odometries for this frame!
  ground_truth_dyno_state.frame_id = frame_id;
  ground_truth_dyno_state.timestamp = timestamp;

  for (const auto& object_pose_gt : gt_packet.object_poses_) {
    // check we have a gt motion here
    // in the case that we dont, this might be the first time the object
    // appears...
    if (!object_pose_gt.prev_H_current_world_) {
      continue;
    }

    const auto object_id = object_pose_gt.object_id_;

    gtsam::Pose3 L_W_k_gt = object_pose_gt.L_world_;
    // motion in world k-1 to k
    Motion3ReferenceFrame H_W_km1_k_gt(
        *object_pose_gt.prev_H_current_world_, MotionRepresentationStyle::F2F,
        ReferenceFrame::GLOBAL, frame_id - 1u, frame_id);

    LOG(INFO) << "Here " << object_id << " k= " << frame_id;
    ground_truth_dyno_state.object_trajectories.insert(
        object_id, frame_id, timestamp, {L_W_k_gt, H_W_km1_k_gt});
  }

  ground_truth_dyno_state.camera_trajectory.insert(frame_id, timestamp,
                                                   gt_packet.X_world_);

  ground_truth_publishers_->dyno_state_publisher_.publish(
      ground_truth_dyno_state);
}

}  // namespace dyno
