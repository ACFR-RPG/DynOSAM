/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam_ros/PipelineRos.hpp"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <dynosam/dataprovider/DataProviderFactory.hpp>
#include <dynosam/dataprovider/DataProviderUtils.hpp>
#include <dynosam/pipeline/PipelineHooks.hpp>
#include <dynosam/pipeline/PipelineParams.hpp>
#include <dynosam/visualizer/OpenCVFrontendDisplay.hpp>

#include "dynosam_ros/Display-Definitions.hpp"
#include "dynosam_ros/OnlineDataProviderRos.hpp"
#include "dynosam_ros/RosUtils.hpp"
#include "dynosam_ros/displays/DisplaysImpl.hpp"
#include "rcl_interfaces/msg/parameter.hpp"
#include "rclcpp/parameter.hpp"
#include "rosgraph_msgs/msg/clock.hpp"

// TODO: just for dyno mppc
#include <dynosam/backend/RegularBackendModule.hpp>
#include <dynosam/backend/rgbd/MPCEstimator.hpp>

#include "dynosam_ros/displays/dynamic_slam_displays/MPCEstimationVizRos.hpp"

namespace dyno {

DynoNode::DynoNode(const std::string& node_name,
                   const rclcpp::NodeOptions& options)
    : Node(node_name, "dynosam", options) {
  RCLCPP_INFO_STREAM(this->get_logger(), "Starting DynoNode");
  auto params_path = getParamsPath();
  RCLCPP_INFO_STREAM(this->get_logger(),
                     "Loading Dyno VO params from: " << params_path);

  is_online_ = ParameterConstructor(this, "online", false)
                   .description("If the online DataProvider should be used")
                   .finish()
                   .get<bool>();

  dyno_params_ = std::make_unique<DynoParams>(params_path);
}

dyno::DataProvider::Ptr DynoNode::createDataProvider() {
  if (is_online_) {
    RCLCPP_INFO_STREAM(
        this->get_logger(),
        "Online DataProvider selected. Waiting for ROS topics...");
    OnlineDataProviderRosParams online_params;
    online_params.wait_for_camera_params =
        ParameterConstructor(this, "wait_for_camera_params",
                             online_params.wait_for_camera_params)
            .description(
                "If the online DataProvider should wait for the camera params "
                "on a ROS topic!")
            .finish()
            .get<bool>();
    online_params.camera_params_timeout =
        ParameterConstructor(this, "camera_params_timeout",
                             online_params.camera_params_timeout)
            .description(
                "When waiting for camera params, how long the online "
                "DataProvider should wait before time out (ms)")
            .finish()
            .get<int>();
    return std::make_shared<OnlineDataProviderRos>(
        this->create_sub_node("dataprovider"), online_params);
  }

  auto params_path = getParamsPath();
  auto dataset_path = getDatasetPath();
  auto dyno_params = getDynoParams();

  RCLCPP_INFO_STREAM(this->get_logger(),
                     "Loading dataset from: " << dataset_path);

  dyno::DataProvider::Ptr data_loader = dyno::DataProviderFactory::Create(
      dataset_path, params_path,
      static_cast<dyno::DatasetType>(dyno_params.dataProviderType()));
  RCLCPP_INFO_STREAM(this->get_logger(), "Constructed data loader");
  return data_loader;
}

std::string DynoNode::searchForPathWithParams(const std::string& param_name,
                                              const std::string& default_path,
                                              const std::string& description) {
  // check if we've alrady declared this param
  // use non-default version so that ParameterConstructor throws exception if no
  // parameter is provided on the param server
  const std::string path = ParameterConstructor(this, param_name)
                               .description(description)
                               .finish()
                               .get<std::string>();
  throwExceptionIfPathInvalid(path);
  return path;
}

DynoPipelineManagerRos::DynoPipelineManagerRos(
    const rclcpp::NodeOptions& options)
    : DynoNode("dynosam", options) {}

void DynoPipelineManagerRos::initalisePipeline() {
  RCLCPP_INFO_STREAM(this->get_logger(), "Starting DynoPipelineManagerRos");

  auto params = getDynoParams();

  // setup display params
  DisplayParams display_params;
  display_params.camera_frame_id =
      ParameterConstructor(this, "camera_frame_id",
                           display_params.camera_frame_id)
          .description(
              "ROS frame id for the camera (ie. the measured odometry)")
          .finish()
          .get<std::string>();
  display_params.world_frame_id =
      ParameterConstructor(this, "world_frame_id",
                           display_params.world_frame_id)
          .description("ROS frame id for the static workd frame (ie. odometry)")
          .finish()
          .get<std::string>();

  LOG(INFO) << "Camera frame id " << display_params.camera_frame_id;
  LOG(INFO) << "World frame id " << display_params.world_frame_id;

  auto frontend_display = std::make_shared<dyno::FrontendDisplayRos>(
      display_params, this->create_sub_node("frontend"));
  auto backend_display = std::make_shared<dyno::BackendDisplayRos>(
      display_params, this->create_sub_node("backend"));

  ExternalHooks::Ptr hooks = std::make_shared<ExternalHooks>();
  // if online then we are using OnlineDataProviderRos, which should collect the
  // timestamp from ROS anyway. Otherwise, the timestamp comes from the dynosam
  // DataLoaders and so we need to artifially tell the ROS network what the time
  // is
  if (!is_online_) {
    RCLCPP_INFO_STREAM(this->get_logger(),
                       "Update time external hook created. This will publish "
                       "internal dynosam timestamp's to /clock!");
    rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr clock_pub =
        this->create_publisher<rosgraph_msgs::msg::Clock>("/clock", 10);
    hooks->update_time = [clock_pub](Timestamp timestamp) -> void {
      auto msg = rosgraph_msgs::msg::Clock();
      msg.clock = utils::toRosTime(timestamp);
      CHECK_NOTNULL(clock_pub)->publish(msg);
    };
  }

  auto data_loader = createDataProvider();
  pipeline_ = std::make_unique<DynoPipelineManager>(
      params, data_loader, frontend_display, backend_display, hooks);

  // JUST FOR DYNO MPC
  auto backend_module = pipeline_->getBackendModule();
  if (backend_module) {
    auto regular_backend_module =
        std::dynamic_pointer_cast<RegularBackendModule>(backend_module);
    if (regular_backend_module) {
      LOG(INFO) << "Is regular backend module";

      auto formulation = regular_backend_module->formulation();
      MPCFormulation* mpc_formulation =
          static_cast<MPCFormulation*>(formulation);
      if (mpc_formulation) {
        LOG(INFO) << "Is mpc formulation!";
        mpc_formulation->viz_ = std::make_unique<MPCEstimationVizRos>(
            display_params, this->create_sub_node("control"), mpc_formulation);
      }
    }
  }
}

}  // namespace dyno
