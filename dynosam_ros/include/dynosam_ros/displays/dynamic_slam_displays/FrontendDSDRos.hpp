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

#include <dynosam/visualizer/VisualizerPipelines.hpp>

#include "dynosam_common/GroundTruthPacket.hpp"
#include "dynosam_ros/displays/DisplaysCommon.hpp"
#include "dynosam_ros/displays/dynamic_slam_displays/DSDCommonRos.hpp"
#include "image_transport/image_transport.hpp"
#include "rclcpp/node.hpp"

namespace dyno {

class FrontendDSDRos : public FrontendDisplay {
 public:
  FrontendDSDRos(const CanonicalSensorRig::ConstPtr& sensor_rig,
                 rclcpp::Node::SharedPtr node,
                 rclcpp::Node::SharedPtr ground_truth_node = nullptr);
  ~FrontendDSDRos() = default;

  void spinOnce(const RealtimeOutput::ConstPtr& frontend_output) override;

 private:
  void tryPublishDebugImagery(const RealtimeOutput::ConstPtr& frontend_output);
  void tryPublishGroundTruth(const RealtimeOutput::ConstPtr& frontend_output);

 private:
  struct GroundTruthPublishers {
    //! Transport for ground truth publishing
    DynoStatePublisher dyno_state_publisher_;

    //! Accumulated ground truth state
    DynoState ground_truth_state_;

    GroundTruthPublishers(const CanonicalSensorRig::ConstPtr& sensor_rig,
                          rclcpp::Node::SharedPtr ground_truth_node);
  };
  DynoStatePublisher dyno_state_publisher_;
  //! Image Transport for tracking image
  image_transport::Publisher tracking_image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      dense_dynamic_cloud_pub_;

  std::optional<GroundTruthPublishers> ground_truth_publishers_;
};

}  // namespace dyno
