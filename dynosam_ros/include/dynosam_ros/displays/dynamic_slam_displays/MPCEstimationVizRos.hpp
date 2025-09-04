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

#include "dynosam/backend/rgbd/MPCEstimator.hpp"
#include "dynosam_ros/Display-Definitions.hpp"
#include "dynosam_ros/displays/dynamic_slam_displays/DSDCommonRos.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/node.hpp"
#include "ros_gz_interfaces/srv/control_world.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

namespace dyno {

class MPCEstimationVizRos : public MPCEstimationViz {
 public:
  MPCEstimationVizRos(const DisplayParams params, rclcpp::Node::SharedPtr node,
                      MPCFormulation* formulation);

  void spin(Timestamp timestamp, FrameId frame_id,
            const MPCFormulation* formulation) override;

  bool queryGlobalOffset(gtsam::Pose3& T_world_camera) override;

 private:
  void publishLocalGoalMarker(const gtsam::Pose3& pose, Timestamp timestamp,
                              const std::string& name);
  void inPreUpdate() override;
  void inPostUpdate() override;

  // for simulating testing ICRA 2026
  void pauseWorld();
  void startWorld();
  bool checkWorldControlServiceAvailable(int timeout_s = 1);

 private:
  const DisplayParams params_;
  rclcpp::Node::SharedPtr node_;
  DSDRos prediction_transport_;

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr global_path_subscriber_;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      local_goal_marker_pub_;

  MPCFormulation* formulation_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  using ControlWorld = ros_gz_interfaces::srv::ControlWorld;
  rclcpp::Client<ControlWorld>::SharedPtr world_control_client_;
};

}  // namespace dyno
