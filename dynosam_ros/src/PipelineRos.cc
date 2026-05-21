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
#include "dynosam_ros/Display-Definitions.hpp"
#include "dynosam_ros/OnlineDataProviderRos.hpp"
#include "dynosam_ros/RosUtils.hpp"
#include "dynosam_ros/displays/DisplaysImpl.hpp"
#include "rcl_interfaces/msg/parameter.hpp"
#include "rclcpp/parameter.hpp"
#include "rosgraph_msgs/msg/clock.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace dyno {

DynoNode::DynoNode(const std::string& node_name,
                   const rclcpp::NodeOptions& options)
    : Node(node_name, "dynosam", options),
      broadcaster_(this),
      tf_buffer_(this->get_clock()),
      tf_listener_(tf_buffer_) {
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
  CHECK_NOTNULL(data_provider);
  // optionally update imu and camera params with config from the data-provider
  // if required
  updateSensorParams(*dyno_params, data_provider);
  // once the sensor params have finally been updated we know which camera
  // params we will use. Use this to set the camera_frame value in
  // rf_definitions which indicates the reference frame of the camera we are
  // using and will form the basis of the odom_frame -> camera_frame tf
  // published by the Displays
  loadReferenceFrameDefinitions(rf_definitions_, *dyno_params);

  // set up tf tree which contains camera (usually optical) to base frame
  // (usually camera link) setupTFTree(rf_definitions_);
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

void DynoNode::loadReferenceFrameDefinitions(
    ReferenceFrameDefinitions& rf_definitions, const DynoParams& dyno_params) {
  rf_definitions.base_frame =
      ParameterConstructor(this, "base_frame", rf_definitions.base_frame)
          .description("ROS frame id for base link of the robot")
          .finish()
          .get<std::string>();
  rf_definitions.odom_frame =
      ParameterConstructor(this, "odom_frame", rf_definitions.odom_frame)
          .description("ROS frame id for the static workd frame (ie. odometry)")
          .finish()
          .get<std::string>();

  rf_definitions.imu_frame =
      ParameterConstructor(this, "imu_frame", rf_definitions.imu_frame)
          .description("ROS frame id imu frame")
          .finish()
          .get<std::string>();

  // now we have a full set of camera params, set the camera_frame param used
  // for publishes the VO
  rf_definitions.camera_frame = dyno_params.camera_params_.referenceFrame();
  RCLCPP_INFO_STREAM(this->get_logger(),
                     "Camera frame: " << rf_definitions.camera_frame);
  RCLCPP_INFO_STREAM(this->get_logger(),
                     "Odom frame: " << rf_definitions.odom_frame);
}

void DynoNode::updateSensorParams(
    DynoParams& dyno_params, const dyno::DataProvider::Ptr& data_provider) {
  // update dyno params with parameters from data-provider
  if (dyno_params.preferDataProviderCameraParams() &&
      data_provider->getCameraParams().has_value()) {
    RCLCPP_INFO_STREAM(
        this->get_logger(),
        "Using camera params from DataProvider, not the config in the "
        "CameraParams.yaml!");
    dyno_params.camera_params_ = *data_provider->getCameraParams();
  } else {
    RCLCPP_INFO_STREAM(this->get_logger(),
                       "Using camera params specified in CameraParams.yaml");
  }

  ImuParams imu_params;
  if (dyno_params.preferDataProviderImuParams() &&
      data_provider->getImuParams().has_value()) {
    RCLCPP_INFO_STREAM(
        this->get_logger(),
        "Using imu params from DataProvider, not the config in the "
        "ImuParams.yaml!");
    imu_params = *data_provider->getImuParams();
  } else {
    RCLCPP_INFO_STREAM(this->get_logger(),
                       "Using imu params specified in ImuParams.yaml!");
    imu_params = dyno_params.imu_params_;
  }
  // update the imu params that will actually get sent to the frontend
  dyno_params.frontend_params_.imu_params = imu_params;
  dyno_params.imu_params_ = imu_params;
}

void DynoNode::setupTFTree(const ReferenceFrameDefinitions& rf_definitions) {
  const std::string camera_frame = rf_definitions.camera_frame;
  const std::string base_frame = rf_definitions.base_frame;

  RCLCPP_INFO_STREAM(this->get_logger(), "Publishing " << base_frame << " to "
                                                       << camera_frame
                                                       << " transform");

  geometry_msgs::msg::TransformStamped T_B_C;

  T_B_C.header.stamp = this->now();
  T_B_C.header.frame_id = base_frame;
  T_B_C.child_frame_id = camera_frame;

  if (isOnline()) {
    // look up link between camera (usually optical frame) and robot base frame
    // (usually camera_link) and republish
    const tf2::Transform base_link_pose_camera_optical =
        getLatestTransform(base_frame, camera_frame);

    T_B_C.transform = tf2::toMsg(base_link_pose_camera_optical);
  } else {
    // if offline, just use the cv->robot rotation as a transform
    // No translation
    T_B_C.transform.translation.x = 0.0;
    T_B_C.transform.translation.y = 0.0;
    T_B_C.transform.translation.z = 0.0;

    // Rotation: ROS base_link → OpenCV optical frame
    tf2::Quaternion q;
    q.setRPY(-M_PI_2, 0.0, -M_PI_2);  // roll, pitch, yaw

    T_B_C.transform.rotation.x = q.x();
    T_B_C.transform.rotation.y = q.y();
    T_B_C.transform.rotation.z = q.z();
    T_B_C.transform.rotation.w = q.w();
  }
  broadcaster_.sendTransform(T_B_C);
}

dyno::DataProvider::Ptr DynoNode::createOnlineDataProvider(
    DynoParams& dyno_params) {
  std::vector<std::string> cameras_needed_for_depth;
  std::vector<std::string> cameras_params;
  std::vector<std::string> cameras;
  SensorMode sensor_mode("rgb+depth+imu");
  for (const auto& configs : sensor_mode.configs()) {
    LOG(INFO) << configs;
  }

  // wait for all camera params as necessary

  LOG(FATAL) << "BLAH!";
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

tf2::Transform DynoNode::getLatestTransform(const std::string& target,
                                            const std::string& source) const {
  geometry_msgs::msg::TransformStamped transform_stamped;
  tf2::Transform pose;

  // Time out duration for TF tree lookup before throwing an exception.
  constexpr int32_t kTimeOutSeconds = 10;

  try {
    if (!tf_buffer_.canTransform(target, source, tf2::TimePointZero,
                                 tf2::durationFromSec(kTimeOutSeconds))) {
      RCLCPP_ERROR(
          this->get_logger(),
          "Transform is impossible. canTransform(%s->%s) returns false",
          target.c_str(), source.c_str());
    }
    transform_stamped =
        tf_buffer_.lookupTransform(target, source, tf2::TimePointZero,
                                   tf2::durationFromSec(kTimeOutSeconds));
    tf2::fromMsg(transform_stamped.transform, pose);
  } catch (tf2::TransformException& ex) {
    RCLCPP_INFO(this->get_logger(), "Could not transform %s to %s: %s",
                source.c_str(), target.c_str(), ex.what());
    throw std::runtime_error("Could not find the requested transform!");
  }
  return pose;
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
  auto rf_definitions = getReferenceFrameDefinitions();

  auto frontend_display = std::make_shared<dyno::FrontendDisplayRos>(
      rf_definitions, this->create_sub_node("frontend"),
      this->create_sub_node("ground_truth"));
  auto backend_display = std::make_shared<dyno::BackendDisplayRos>(
      rf_definitions, this->create_sub_node("backend"));

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
      RosBackendFactory::Create(params.backend_type, rf_definitions, this);

  pipeline_ = std::make_unique<DynoPipelineManager>(
      params, data_loader, frontend_display, backend_display, factory, hooks);
}

}  // namespace dyno
