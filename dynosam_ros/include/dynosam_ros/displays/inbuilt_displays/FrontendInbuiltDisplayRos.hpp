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

#include <dynosam/common/GroundTruthPacket.hpp>
#include <dynosam/frontend/RGBDInstance-Definitions.hpp>
#include <dynosam/visualizer/Display.hpp>
#include <opencv4/opencv2/videoio.hpp>

#include "dynosam_ros/Display-Definitions.hpp"
#include "dynosam_ros/displays/inbuilt_displays/InbuiltDisplayCommon.hpp"
#include "image_transport/image_transport.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "visualization_msgs/msg/marker_array.hpp"

namespace dyno {

class FrontendInbuiltDisplayRos : public FrontendDisplay, InbuiltDisplayCommon {
 public:
  FrontendInbuiltDisplayRos(const DisplayParams params,
                            rclcpp::Node::SharedPtr node);
  ~FrontendInbuiltDisplayRos();

  void spinOnceImpl(
      const FrontendOutputPacketBase::ConstPtr& frontend_output) override;

 private:
  void processRGBDOutputpacket(
      const RGBDInstanceOutputPacket::ConstPtr& rgbd_frontend_output);

  void publishOdometry(const gtsam::Pose3& T_world_camera, Timestamp timestamp);
  // void publishOdometryPath(const gtsam::Pose3& T_world_camera, Timestamp
  // timestamp);
  void publishDebugImage(const DebugImagery& debug_imagery);

  void publishGroundTruthInfo(Timestamp timestamp,
                              const GroundTruthInputPacket& gt_packet,
                              const cv::Mat& rgb);

 private:
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      static_tracked_points_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      dynamic_tracked_points_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_pub_;

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr odometry_path_pub_;
  nav_msgs::msg::Path odom_path_msg_;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      object_pose_path_pub_;  //! Path of propogated object poses using the
                              //! motion estimate
  std::map<ObjectId, gtsam::Pose3Vector> object_trajectories_;
  std::map<ObjectId, FrameId>
      object_trajectories_update_;  //! The last frame id that the object was
                                    //! seen in

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr
      object_motion_pub_;  //! Publish object motions per frame as an array of
                           //! SE(3) transformations (a Path) where frame_id per
                           //! pose is object id

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      object_pose_pub_;  //! Propogated object poses using the motion estimate
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      object_bbx_line_pub_;  //! Draw object bounding boxes as line lists
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      object_bbx_pub_;  //! Draw object bounding boxes as cubes
  image_transport::Publisher tracking_image_pub_;

  // ground truth publishers
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      gt_object_pose_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      gt_object_path_pub_;  //! Path of objects with gt
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr gt_odometry_pub_;

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr gt_odom_path_pub_;
  nav_msgs::msg::Path gt_odom_path_msg_;

  image_transport::Publisher gt_bounding_box_pub_;

  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  std::unique_ptr<cv::VideoWriter> video_writer_;  // for now
};

}  // namespace dyno
