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
#include "dynosam_ros/PipelineRos-impl.hpp"
#include "dynosam_ros/RosUtils.hpp"
#include "dynosam_ros/Subscriber.hpp"
#include "dynosam_ros/displays/BackendDisplayRos.hpp"
#include "dynosam_ros/displays/FrontendDisplayRos.hpp"
#include "rcl_interfaces/msg/parameter.hpp"
#include "rclcpp/parameter.hpp"
#include "rosgraph_msgs/msg/clock.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace dyno {

DynosamNodeImpl::DynosamNodeImpl(rclcpp::Node& node) : dynosam_node(node) {}

void DynosamNodeImpl::init() {
  rclcpp::Node* node_ptr = &dynosam_node;

  RCLCPP_INFO_STREAM(node_ptr->get_logger(), "Starting DynoNode");
  auto params_path = getParamsPath();
  RCLCPP_INFO_STREAM(node_ptr->get_logger(),
                     "Loading Dyno VO params from: " << params_path);

  DynoParams dyno_params(params_path);

  const bool is_online =
      ros::Parameter::Builder(node_ptr, "online", false)
          .description("If the online DataProvider should be used")
          .finish()
          .get<bool>();

  // dyno params may additionally get modified with the online data provider
  dyno::DataProvider::Ptr data_provider =
      createDataProvider(dyno_params, is_online);

  bool publish_diagnostics =
      ros::Parameter::Builder(node_ptr, "publish_diagnostics", true)
          .description(
              "If the diagnostics publisher should run, reporting stats for all"
              " pipelines and modules.")
          .finish()
          .get<bool>();

  if (publish_diagnostics) {
    diagnostics_updater =
        std::make_unique<DynoDiagnosticsTaskManager>(&dynosam_node);
  }

  const auto sensor_rig = data_provider->sensorRig();
  auto frontend_display = std::make_shared<dyno::FrontendDisplayRos>(
      sensor_rig, node_ptr->create_sub_node("frontend"),
      node_ptr->create_sub_node("ground_truth"));
  auto backend_display = std::make_shared<dyno::BackendDisplayRos>(
      sensor_rig, node_ptr->create_sub_node("backend"));

  ExternalHooks::Ptr hooks = std::make_shared<ExternalHooks>();
  // if online then we are using OnlineDataProviderRos, which should collect the
  // timestamp from ROS anyway. Otherwise, the timestamp comes from the dynosam
  // DataLoaders and so we need to artifially tell the ROS network what the time
  // is
  if (!is_online) {
    RCLCPP_INFO_STREAM(node_ptr->get_logger(),
                       "Update time external hook created. This will publish "
                       "internal dynosam timestamp's to /clock!");
    rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr clock_pub =
        node_ptr->create_publisher<rosgraph_msgs::msg::Clock>("/clock", 10);
    hooks->update_time = [clock_pub](Timestamp timestamp) -> void {
      auto msg = rosgraph_msgs::msg::Clock();
      msg.clock = ros::toRosTime(timestamp);
      CHECK_NOTNULL(clock_pub)->publish(msg);
    };
  }

  // proxy for checking if diagnostics should be published
  // if true set up task register for the pipeline
  if (diagnostics_updater) {
    hooks->register_diagnostics_task = [&](const std::string& name,
                                           DiagnosticTaskRunner* task) {
      CHECK_NOTNULL(task);
      RCLCPP_INFO_STREAM(node_ptr->get_logger(),
                         "Registering diagnostics task: " << name);
      diagnostics_updater->registerTask(
          std::make_shared<DynoDiagnosticsTask>(name, task));
    };
  }

  // Define a backend factory with a ROS specific policy
  // This allows custom displays to be loaded at runtime depending
  // on the formulation/module requested
  // TODO: RF for now!
  using RosBackendFactory = BackendFactory<BackendModulePolicyRos>;
  auto factory = RosBackendFactory::Create(
      dyno_params.backend_type, sensor_rig->getReferenceFrames(), node_ptr);

  pipeline = std::make_unique<DynoPipelineManager>(
      dyno_params, data_provider, frontend_display, backend_display, factory,
      hooks);

  auto getStats = []() -> std::string { return utils::Statistics::Print(); };

  auto spinOnce = [&]() -> void {
    RCLCPP_INFO_STREAM_THROTTLE(dynosam_node.get_logger(),
                                *dynosam_node.get_clock(), 2000, getStats());

    if (!pipeline->spin()) {
      auto context = dynosam_node.get_node_options().context();
      context->shutdown("Dynosam pipeline has finished processing all data");
    }
  };

  // so that all process is encapsulated within this class we use a timer
  // to trigger the pipelines spin function
  // this hides a bit of complex behaviour because depending on the setup
  // (ie. parallel_run, is_online etc) this may or may not limit the processing
  // rate of the pipeline
  // in most cases this should not matter (ie. when we are processing dataset
  // the processing speed is not an issue) in either case we expect the limit of
  // processing to be about 15-20Hz (which is realtime anyway) so we attempt to
  // spin at this rate regardless
  spin_timer_group =
      node_ptr->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  spin_timer = node_ptr->create_wall_timer(std::chrono::milliseconds(50),
                                           spinOnce, spin_timer_group);
}

DynosamNodeImpl::~DynosamNodeImpl() {
  // stop diagnostic updater before pipeline as all the pipeline owns
  //  the memory for all diagnostics tasks which are only stored in the
  //  updater with references/raw pointers
  if (diagnostics_updater) diagnostics_updater.reset(nullptr);
  if (pipeline) pipeline.reset(nullptr);
}

dyno::DataProvider::Ptr DynosamNodeImpl::createDataProvider(
    DynoParams& dyno_params, bool is_online) {
  if (is_online) {
    RCLCPP_INFO_STREAM(dynosam_node.get_logger(),
                       "Creating online data-provider");
    return createOnlineDataProvider(dyno_params);
  } else {
    return createDatasetDataProvider(dyno_params);
  }
}

dyno::DataProvider::Ptr DynosamNodeImpl::createOnlineDataProvider(
    DynoParams& dyno_params) {
  auto mode =
      ros::Parameter::Builder(&dynosam_node, "input_image_mode",
                              "rgb+aligned_depth")
          .description(
              "Specify sensor and camera stream modes (ie. rgb+aligned_depth, "
              "+stereo, +imu etc.)")
          .finish()
          .get<std::string>();

  SensorMode sensor_mode(mode);
  sensor_mode.reconfigure(dyno_params);

  SensorSystem::Ptr sensor_system = std::make_shared<SensorSystem>(
      dynosam_node.create_sub_node("dataprovider"), sensor_mode.depthRigType(),
      getParamsPath(), true);
  for (const auto& configs : sensor_mode.configs()) {
    sensor_system->addCamera(configs);
  }

  sensor_system->enableImu(sensor_mode.useImu());
  sensor_system->finalise();

  // do better with the subnodes
  auto subscriber = std::make_shared<Subscriber>(
      sensor_system, dynosam_node.create_sub_node("images"));
  return subscriber;
}

dyno::DataProvider::Ptr DynosamNodeImpl::createDatasetDataProvider(
    const DynoParams& dyno_params) {
  auto params_path = getParamsPath();
  auto dataset_path = getDatasetPath();

  RCLCPP_INFO_STREAM(dynosam_node.get_logger(),
                     "Loading dataset from: " << dataset_path);

  dyno::DataProvider::Ptr data_loader = dyno::DataProviderFactory::Create(
      dataset_path, params_path,
      static_cast<dyno::DatasetType>(dyno_params.dataProviderType()));
  RCLCPP_INFO_STREAM(dynosam_node.get_logger(), "Constructed data loader");
  return data_loader;
}

std::string DynosamNodeImpl::safeGetFilePath(const std::string& param_name,
                                             const std::string& default_path,
                                             const std::string& description) {
  // check if we've alrady declared this param
  // use non-default version so that Builder throws exception if no
  // parameter is provided on the param server
  const std::string path = ros::Parameter::Builder(&dynosam_node, param_name)
                               .description(description)
                               .finish()
                               .get<std::string>();
  utils::throwExceptionIfPathInvalid(path);
  return path;
}

std::string DynosamNodeImpl::getDatasetPath() {
  return safeGetFilePath("dataset_path", "dataset", "Path to the dataset.");
}

std::string DynosamNodeImpl::getParamsPath() {
  return safeGetFilePath("params_path", "dynosam/params/",
                         "Path to the folder containing the yaml "
                         "files with the Dynosam parameters.");
}

DynosamNode::DynosamNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("dynosam_node", "dynosam", options) {
  impl_ = std::make_unique<DynosamNodeImpl>(*this);
  impl_->init();
}
DynosamNode::~DynosamNode() { impl_.reset(); }

DynosamComposableNode::DynosamComposableNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("dynosam_node", "dynosam", options) {
  // init glogging and
  impl_ = std::make_unique<DynosamNodeImpl>(*this);
  // load all gflags via flag files auto-discovered in the params folder
  ros::initGlog(impl_->getParamsPath(), "dynosam_node");
  impl_->init();
}
DynosamComposableNode::~DynosamComposableNode() { impl_.reset(); }

}  // namespace dyno

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(dyno::DynosamComposableNode)
