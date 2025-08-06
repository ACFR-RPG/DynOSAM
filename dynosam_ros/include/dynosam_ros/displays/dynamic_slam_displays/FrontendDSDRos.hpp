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
#include <dynosam/frontend/FrontendOutputPacket.hpp>
#include <dynosam/visualizer/Display.hpp>

#include "dynamic_slam_interfaces/msg/object_odometry.hpp"
#include "dynosam_ros/Display-Definitions.hpp"
#include "dynosam_ros/displays/DisplaysCommon.hpp"
#include "dynosam_ros/displays/dynamic_slam_displays/DSDCommonRos.hpp"
#include "image_transport/image_transport.hpp"
#include "rclcpp/node.hpp"

namespace dyno {

class FrontendDSDRos : public FrontendDisplay, DSDRos {
 public:
  FrontendDSDRos(const DisplayParams params, rclcpp::Node::SharedPtr node);
  ~FrontendDSDRos() = default;

  void spinOnce(const VisionImuPacket::ConstPtr& frontend_output) override;

 private:
  void tryPublishDebugImagery(const VisionImuPacket::ConstPtr& frontend_output);
  void tryPublishGroundTruth(const VisionImuPacket::ConstPtr& frontend_output);
  void tryPublishVisualOdometry(
      const VisionImuPacket::ConstPtr& frontend_output);

  // TODO: not publishing object stuff and point cloud!!!
  //   void processRGBDOutputpacket(
  //       const RGBDInstanceOutputPacket::ConstPtr& rgbd_packet);

 private:
  //! Transport for ground truth publishing
  DSDTransport::UniquePtr dsd_ground_truth_transport_;
  //! Image Transport for tracking image
  image_transport::Publisher tracking_image_pub_;

  OdometryPub::SharedPtr vo_ground_truth_publisher_;
  PathPub::SharedPtr vo_path_ground_truth_publisher_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      dense_dynamic_cloud_pub_;
};

}  // namespace dyno
