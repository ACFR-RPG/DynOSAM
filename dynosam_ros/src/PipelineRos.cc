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
#include "dynosam_ros/displays/DisplaysImpl.hpp"
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

  is_online_ = ParameterConstructor(this, "online", false)
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
  std::vector<std::string> cameras_needed_for_depth;
  std::vector<std::string> cameras_params;
  std::vector<std::string> cameras;

  auto mode =
      ParameterConstructor(this, "input_image_mode", "rgb+aligned_depth")
          .description(
              "Which input image mode to run the pipeline in (e.g "
              "ALL, RGBD, STEREO)...")
          .finish()
          .get<std::string>();

  SensorMode sensor_mode(mode);
  sensor_mode.reconfigure(dyno_params);

  SensorSystem::Ptr sensor_system = std::make_shared<SensorSystem>(
      this->create_sub_node("dataprovider"), sensor_mode.depthRigType(),
      getParamsPath(), true);
  for (const auto& configs : sensor_mode.configs()) {
    LOG(INFO) << configs;
    sensor_system->addCamera(configs);
  }

  sensor_system->finalise();

  // do better with the subnodes
  auto subscriber = std::make_shared<Subscriber>(
      sensor_system, this->create_sub_node("images"));
  return subscriber;

  // wait for all camera params as necessary

  // LOG(FATAL) << "BLAH!";
  // OnlineDataProviderRosParams online_params;
  // online_params.wait_for_camera_params =
  //     ParameterConstructor(this, "wait_for_camera_params",
  //                          online_params.wait_for_camera_params)
  //         .description(
  //             "If the online DataProvider should wait for the camera params "
  //             "on a ROS topic!")
  //         .finish()
  //         .get<bool>();
  // online_params.camera_params_timeout =
  //     ParameterConstructor(this, "camera_params_timeout",
  //                          online_params.camera_params_timeout)
  //         .description(
  //             "When waiting for camera params, how long the online "
  //             "DataProvider should wait before time out (ms)")
  //         .finish()
  //         .get<int>();
  // InputImageMode image_mode = static_cast<InputImageMode>(
  //     ParameterConstructor(this, "input_image_mode",
  //                          static_cast<int>(InputImageMode::ALL))
  //         .description("Which input image mode to run the pipeline in (e.g "
  //                      "ALL, RGBD, STEREO)...")
  //         .finish()
  //         .get<int>());

  // // TODO: make image input mode like OKVIS (ie. all+imu)
  // OnlineDataProviderRos::Ptr online_data_provider = nullptr;
  // switch (image_mode) {
  //   case InputImageMode::ALL:
  //     online_data_provider = std::make_shared<AllImagesOnlineProviderRos>(
  //         this->create_sub_node("dataprovider"), online_params);
  //     break;
  //   case InputImageMode::RGBD:
  //     online_data_provider = std::make_shared<RGBDOnlineProviderRos>(
  //         this->create_sub_node("dataprovider"), online_params);
  //     break;
  //   case InputImageMode::RGBDM:
  //     online_data_provider = std::make_shared<RGBDMOnlineProviderRos>(
  //         this->create_sub_node("dataprovider"), online_params);
  //     break;
  //   case InputImageMode::STEREO:
  //     online_data_provider = std::make_shared<StereoOnlineProviderRos>(
  //         this->create_sub_node("dataprovider"), online_params);
  //     break;
  //   default:
  //     LOG(FATAL) << "Unknown image_mode";
  //     return nullptr;
  // }

  // CHECK(online_data_provider);
  // // update any params in case they do not conflixt with the expected input
  // online_data_provider->updateAndCheckParams(dyno_params);
  // online_data_provider->setupSubscribers();
  // return online_data_provider;
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
  // use non-default version so that ParameterConstructor throws exception if no
  // parameter is provided on the param server
  const std::string path = ParameterConstructor(this, param_name)
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
      ParameterConstructor(this, "publish_diagnostics", true)
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

  const auto reference_frames = data_loader->sensorRig()->getReferenceFrames();
  // TODO: shoudl take the sensor system to also get the xtrinsics...
  auto frontend_display = std::make_shared<dyno::FrontendDisplayRos>(
      reference_frames, this->create_sub_node("frontend"),
      this->create_sub_node("ground_truth"));
  auto backend_display = std::make_shared<dyno::BackendDisplayRos>(
      reference_frames, this->create_sub_node("backend"));

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
      msg.clock = utils::toRosTime(timestamp);
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
  using RosBackendFactory = BackendFactory<BackendModulePolicyRos>;
  auto factory =
      RosBackendFactory::Create(params.backend_type, reference_frames, this);

  pipeline_ = std::make_unique<DynoPipelineManager>(
      params, data_loader, frontend_display, backend_display, factory, hooks);
}

}  // namespace dyno
