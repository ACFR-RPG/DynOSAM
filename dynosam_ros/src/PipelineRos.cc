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

#include "dynosam/backend/BackendFactory.hpp"
#include "dynosam/dataprovider/DataProviderFactory.hpp"
#include "dynosam/dataprovider/DataProviderUtils.hpp"
#include "dynosam/pipeline/PipelineHooks.hpp"
#include "dynosam/pipeline/PipelineParams.hpp"
#include "dynosam_ros/BackendDisplayPolicyRos.hpp"
#include "dynosam_ros/CameraSystem.hpp"
#include "dynosam_ros/RosUtils.hpp"
#include "dynosam_ros/Subscriber.hpp"
#include "dynosam_ros/displays/BackendDisplayRos.hpp"
#include "dynosam_ros/displays/FrontendDisplayRos.hpp"
#include "rcl_interfaces/msg/parameter.hpp"
#include "rclcpp/parameter.hpp"
#include "rosgraph_msgs/msg/clock.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace dyno {

DynoNode::DynoNode(const std::string& node_name,
                   const rclcpp::NodeOptions& options)
    : Node(node_name, "dynosam", options) {
  RCLCPP_INFO_STREAM(this->get_logger(), "Starting DynoNode");
  auto params_path = getParamsPath();
  RCLCPP_INFO_STREAM(this->get_logger(),
                     "Loading Dyno VO params from: " << params_path);

  auto dyno_params = std::make_unique<DynoParams>(params_path);

  is_online_ = ros::Parameter::Builder(this, "online", false)
                   .description("If the online DataProvider should be used")
                   .finish()
                   .get<bool>();

  // dyno params may additionally get modified with the online data provider
  dyno::DataProvider::Ptr data_provider =
      createDataProvider(*dyno_params, is_online_);
  dyno_params_ = std::move(dyno_params);
  data_provider_ = data_provider;
}

dyno::DataProvider::Ptr DynoNode::createDataProvider(DynoParams& dyno_params,
                                                     bool is_online) {
  if (is_online) {
    RCLCPP_INFO_STREAM(this->get_logger(), "Creating online data-provider");
    return createOnlineDataProvider(dyno_params);
  } else {
    return createDatasetDataProvider(dyno_params);
  }
}

dyno::DataProvider::Ptr DynoNode::createOnlineDataProvider(
    DynoParams& dyno_params) {
  auto mode =
      ros::Parameter::Builder(this, "input_image_mode", "rgb+aligned_depth")
          .description(
              "Specify sensor and camera stream modes (ie. rgb+aligned_depth, "
              "+stereo, +imu etc.)")
          .finish()
          .get<std::string>();

  SensorMode sensor_mode(mode);
  sensor_mode.reconfigure(dyno_params);

  SensorSystem::Ptr sensor_system = std::make_shared<SensorSystem>(
      this->create_sub_node("dataprovider"), sensor_mode.depthRigType(),
      getParamsPath(), true);
  for (const auto& configs : sensor_mode.configs()) {
    sensor_system->addCamera(configs);
  }

  sensor_system->enableImu(sensor_mode.useImu());
  sensor_system->finalise();

  // do better with the subnodes
  auto subscriber = std::make_shared<Subscriber>(
      sensor_system, this->create_sub_node("images"));
  return subscriber;
}

dyno::DataProvider::Ptr DynoNode::createDatasetDataProvider(
    const DynoParams& dyno_params) {
  auto params_path = getParamsPath();
  auto dataset_path = getDatasetPath();

  RCLCPP_INFO_STREAM(this->get_logger(),
                     "Loading dataset from: " << dataset_path);

  dyno::DataProvider::Ptr data_loader = dyno::DataProviderFactory::Create(
      dataset_path, params_path,
      static_cast<dyno::DatasetType>(dyno_params.dataProviderType()));
  RCLCPP_INFO_STREAM(this->get_logger(), "Constructed data loader");
  return data_loader;
}

std::string DynoNode::searchForPathWithParams(
    const std::string& param_name, const std::string& /*default_path*/,
    const std::string& description) {
  // check if we've alrady declared this param
  // use non-default version so that Builder throws exception if no
  // parameter is provided on the param server
  const std::string path = ros::Parameter::Builder(this, param_name)
                               .description(description)
                               .finish()
                               .get<std::string>();
  utils::throwExceptionIfPathInvalid(path);
  return path;
}

DynoPipelineManagerRos::DynoPipelineManagerRos(
    const rclcpp::NodeOptions& options)
    : DynoNode("dynosam", options), diagnostics_updater_(nullptr) {
  bool publish_diagnostics =
      ros::Parameter::Builder(this, "publish_diagnostics", true)
          .description(
              "If the diagnostics publisher should run, reporting stats for all"
              " pipelines and modules.")
          .finish()
          .get<bool>();

  if (publish_diagnostics) {
    diagnostics_updater_ = std::make_unique<DynoDiagnosticsTaskManager>(this);
  }
}

DynoPipelineManagerRos::~DynoPipelineManagerRos() {
  // stop diagnostic updater before pipeline as all the pipeline owns
  //  the memory for all diagnostics tasks which are only stored in the
  //  updater with references/raw pointers
  if (diagnostics_updater_) diagnostics_updater_.reset(nullptr);
  if (pipeline_) pipeline_.reset(nullptr);
}

void DynoPipelineManagerRos::initalisePipeline() {
  RCLCPP_INFO_STREAM(this->get_logger(), "Starting DynoPipelineManagerRos");

  // load data provider first as this could change some params to ensure
  // they match with the data-provider selected!
  auto data_loader = getDataProvider();
  auto params = getDynoParams();

  const auto sensor_rig = data_loader->sensorRig();
  auto frontend_display = std::make_shared<dyno::FrontendDisplayRos>(
      sensor_rig, this->create_sub_node("frontend"),
      this->create_sub_node("ground_truth"));
  auto backend_display = std::make_shared<dyno::BackendDisplayRos>(
      sensor_rig, this->create_sub_node("backend"));

  ExternalHooks::Ptr hooks = std::make_shared<ExternalHooks>();
  // if online then we are using OnlineDataProviderRos, which should collect the
  // timestamp from ROS anyway. Otherwise, the timestamp comes from the dynosam
  // DataLoaders and so we need to artifially tell the ROS network what the time
  // is
  if (!isOnline()) {
    RCLCPP_INFO_STREAM(this->get_logger(),
                       "Update time external hook created. This will publish "
                       "internal dynosam timestamp's to /clock!");
    rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr clock_pub =
        this->create_publisher<rosgraph_msgs::msg::Clock>("/clock", 10);
    hooks->update_time = [clock_pub](Timestamp timestamp) -> void {
      auto msg = rosgraph_msgs::msg::Clock();
      msg.clock = ros::toRosTime(timestamp);
      CHECK_NOTNULL(clock_pub)->publish(msg);
    };
  }

  // proxy for checking if diagnostics should be published
  // if true set up task register for the pipeline
  if (diagnostics_updater_) {
    hooks->register_diagnostics_task = [&](const std::string& name,
                                           DiagnosticTaskRunner* task) {
      CHECK_NOTNULL(task);
      RCLCPP_INFO_STREAM(this->get_logger(),
                         "Registering diagnostics task: " << name);
      diagnostics_updater_->registerTask(
          std::make_shared<DynoDiagnosticsTask>(name, task));
    };
  }

  // Define a backend factory with a ROS specific policy
  // This allows custom displays to be loaded at runtime depending
  // on the formulation/module requested
  // TODO: RF for now!
  using RosBackendFactory = BackendFactory<BackendModulePolicyRos>;
  auto factory = RosBackendFactory::Create(
      params.backend_type, sensor_rig->getReferenceFrames(), this);

  pipeline_ = std::make_unique<DynoPipelineManager>(
      params, data_loader, frontend_display, backend_display, factory, hooks);
}

}  // namespace dyno
