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

#include <dynosam/backend/BackendOutputPacket.hpp>
#include <dynosam/common/GroundTruthPacket.hpp>
#include <dynosam/common/SharedModuleInfo.hpp>
#include <dynosam/visualizer/Display.hpp>

#include "dynamic_slam_interfaces/msg/object_odometry.hpp"
#include "dynosam_ros/Display-Definitions.hpp"
#include "dynosam_ros/displays/DisplaysCommon.hpp"
#include "dynosam_ros/displays/dynamic_slam_displays/DSDCommonRos.hpp"
#include "rclcpp/node.hpp"

namespace dyno {

class BackendDSDRos : public BackendDisplay, DSDRos {
 public:
  BackendDSDRos(const DisplayParams params, rclcpp::Node::SharedPtr node);
  ~BackendDSDRos() = default;

  void spinOnceImpl(
      const BackendOutputPacket::ConstPtr& backend_output) override;

 private:
  void publishTemporalDynamicMaps(
      const BackendOutputPacket::ConstPtr& latest_backend_output);

 private:
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      temporal_dynamic_points_pub_;
  gtsam::FastMap<ObjectId, std::deque<pcl::PointCloud<pcl::PointXYZRGB>>>
      temporal_clouds_;
};

}  // namespace dyno
