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

#include "dynosam_ros/OnlineDataProviderRos.hpp"

#include "dynosam_common/Types.hpp"
#include "dynosam_ros/RosUtils.hpp"

namespace dyno {

template <>
std::string to_string(const InputImageMode& input_image_mode) {
  std::string status_str = "";
  switch (input_image_mode) {
    case InputImageMode::ALL: {
      status_str = "ALL";
      break;
    }
    case InputImageMode::RGBD: {
      status_str = "RGBD";
      break;
    }
    case InputImageMode::STEREO: {
      status_str = "STEREO";
      break;
    }
  }
  return status_str;
}

OnlineDataProviderRos::OnlineDataProviderRos(
    rclcpp::Node::SharedPtr node, const OnlineDataProviderRosParams& params)
    : DataProviderRos(node), params_(params), frame_id_(0u) {
  CHECK_EQ(shutdown_, false);
}

bool OnlineDataProviderRos::spin() {
  if (!is_connected) {
    RCLCPP_ERROR_THROTTLE(
        node_->get_logger(), *node_->get_clock(), 1000,
        "OnlineDataProviderRos spinning but subscribers are not connected. "
        "Did you forget to call OnlineDataProviderRos::setupSubscribers()?");
  }
  return !shutdown_;
}

void OnlineDataProviderRos::shutdown() {
  shutdown_ = true;
  // shutdown synchronizer
  RCLCPP_INFO_STREAM(node_->get_logger(),
                     "Shutting down OnlineDataProviderRos");
  if (imu_sub_) imu_sub_.reset();
  unsubscribeImages();
  is_connected = false;
}

void OnlineDataProviderRos::setupSubscribers() {
  subscribeImages();
  subscribeImu();
  shutdown_ = false;
  is_connected = true;
}

void OnlineDataProviderRos::subscribeImu() {
  if (imu_sub_) imu_sub_.reset();

  imu_callback_group_ =
      node_->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

  rclcpp::SubscriptionOptions imu_sub_options;
  imu_sub_options.callback_group = imu_callback_group_;

  imu_sub_ = node_->create_subscription<ImuAdaptedType>(
      "imu", rclcpp::SensorDataQoS(),
      [&](const dyno::ImuMeasurement& imu) -> void {
        if (!imu_single_input_callback_) {
          RCLCPP_ERROR_THROTTLE(
              node_->get_logger(), *node_->get_clock(), 1000,
              "Imu callback triggered but "
              "imu_single_input_callback_ is not registered!");
          return;
        }
        imu_single_input_callback_(imu);
      },
      imu_sub_options);
}

RGBDTypeCalibrationHelper::RGBDTypeCalibrationHelper(
    rclcpp::Node::SharedPtr node, const OnlineDataProviderRosParams& params)
    : node_(node) {
  if (params.wait_for_camera_params) {
    const CameraParams original_camera_params = waitAndSetCameraParams(
        node, "camera/camera_info",
        std::chrono::milliseconds(params.camera_params_timeout));

    int rescale_width, rescale_height;
    getParamsFromRos(original_camera_params, rescale_width, rescale_height);

    CameraParams camera_params;
    setupNewCameraParams(original_camera_params, camera_params, rescale_width,
                         rescale_height);

    original_camera_params_ = original_camera_params;
    camera_params_ = camera_params;
  }
}

void RGBDTypeCalibrationHelper::undistort(const cv::Mat& src, cv::Mat& dst) {
  cv::remap(src, dst, mapx_, mapy_, cv::INTER_LINEAR);

  // output will have the same type as mapx/y so covnert back to required type
  dst.convertTo(dst, src.type());
}

const CameraParams::Optional&
RGBDTypeCalibrationHelper::getOriginalCameraParams() const {
  return original_camera_params_;
}
const CameraParams::Optional& RGBDTypeCalibrationHelper::getCameraParams()
    const {
  return camera_params_;
}

void RGBDTypeCalibrationHelper::setupNewCameraParams(
    const CameraParams& original_camera_params, CameraParams& new_camera_params,
    const int& rescale_width, const int& rescale_height) {
  const auto original_size = original_camera_params.imageSize();
  const cv::Size rescale_size = cv::Size(rescale_width, rescale_height);
  cv::Mat original_K = original_camera_params.getCameraMatrix();
  const cv::Mat distortion = original_camera_params.getDistortionCoeffs();

  cv::Mat new_K = cv::getOptimalNewCameraMatrix(
      original_K, distortion, original_size, 1.0, rescale_size);

  cv::initUndistortRectifyMap(original_K, distortion, cv::Mat(), new_K,
                              rescale_size, CV_32FC1, mapx_, mapy_);

  dyno::CameraParams::IntrinsicsCoeffs intrinsics;
  cv::Mat K_double;
  new_K.convertTo(K_double, CV_64F);
  dyno::CameraParams::convertKMatrixToIntrinsicsCoeffs(K_double, intrinsics);
  dyno::CameraParams::DistortionCoeffs zero_distortion(4, 0);

  new_camera_params = CameraParams(intrinsics, zero_distortion, rescale_size,
                                   original_camera_params.getDistortionModel(),
                                   original_camera_params.getExtrinsics());
}

void RGBDTypeCalibrationHelper::getParamsFromRos(
    const CameraParams& original_camera_params, int& rescale_width,
    int& rescale_height) {
  rescale_width = ParameterConstructor(node_.get(), "rescale_width",
                                       original_camera_params.ImageWidth())
                      .description(
                          "Image width to rescale to. If not provided or -1 "
                          "image will be inchanged")
                      .finish()
                      .get<int>();
  if (rescale_width == -1) {
    rescale_width = original_camera_params.ImageWidth();
  }

  rescale_height = ParameterConstructor(node_.get(), "rescale_height",
                                        original_camera_params.ImageHeight())
                       .description(
                           "Image height to rescale to. If not provided or -1 "
                           "image will be inchanged")
                       .finish()
                       .get<int>();
  if (rescale_height == -1) {
    rescale_height = original_camera_params.ImageHeight();
  }
}

void updateAndCheckDynoParamsForRawImageInput(DynoParams& dyno_params) {
  auto& tracker_params = dyno_params.frontend_params_.tracker_params;
  if (tracker_params.prefer_provided_optical_flow) {
    LOG(WARNING)
        << "InputImageMode not set to ALL but prefer_provided_optical_flow is "
           "true - param will be updated!";
    tracker_params.prefer_provided_optical_flow = false;
  }
  if (tracker_params.prefer_provided_object_detection) {
    LOG(WARNING)
        << "InputImageMode not set to ALL but prefer_provided_object_detection "
           "is true - param will be updated!";
    tracker_params.prefer_provided_object_detection = false;
    // TODO: should also warn in this case that gt tracking will not match!!
  }
}

AllImagesOnlineProviderRos::AllImagesOnlineProviderRos(
    rclcpp::Node::SharedPtr node, const OnlineDataProviderRosParams& params)
    : OnlineDataProviderRos(node, params) {
  LOG(INFO) << "Creating AllImagesOnlineProviderRos";
  calibration_helper_ =
      std::make_unique<RGBDTypeCalibrationHelper>(node, params);

  // All Images only works with undisroted images as the pre-processing must be
  // done on undistorted
  //  images particularly for optical flow
  auto original_camera_params = calibration_helper_->getOriginalCameraParams();
  if (original_camera_params) {
    const cv::Mat distortion = original_camera_params->getDistortionCoeffs();
    if (cv::countNonZero(distortion.reshape(1)) != 0) {
      // not all zeros
      DYNO_THROW_MSG(DynosamException)
          << "In AllImagesOnlineProviderRos the original camera params has "
             "distortion coeffs which means the images "
             " propvided have not been undisroted!";
    }
  } else {
    DYNO_THROW_MSG(DynosamException)
        << "No original camera params found for AllImagesOnlineProviderRos";
  }
}

void AllImagesOnlineProviderRos::subscribeImages() {
  rclcpp::Node& node_ref = *node_;
  static const std::array<std::string, 4>& topics = {
      "image/rgb", "image/depth", "image/flow", "image/mask"};

  MultiSyncConfig config;
  config.queue_size = 20;
  // config.subscriber_options.callback_group =
  //     node_ref.create_callback_group(rclcpp::CallbackGroupType::Reentrant);

  std::shared_ptr<MultiImageSync4> multi_image_sync =
      std::make_shared<MultiImageSync4>(node_ref, topics, config);
  multi_image_sync->registerCallback(
      [this](const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg,
             const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg,
             const sensor_msgs::msg::Image::ConstSharedPtr& flow_msg,
             const sensor_msgs::msg::Image::ConstSharedPtr& mask_msg) {
        if (!image_container_callback_) {
          RCLCPP_ERROR_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                                "Image Sync callback triggered but "
                                "image_container_callback_ is not registered!");
          return;
        }

        const cv::Mat rgb = readRgbRosImage(rgb_msg);
        const cv::Mat depth = readDepthRosImage(depth_msg);
        const cv::Mat flow = readFlowRosImage(flow_msg);
        const cv::Mat mask = readMaskRosImage(mask_msg);

        const Timestamp timestamp = utils::fromRosTime(rgb_msg->header.stamp);
        const FrameId frame_id = frame_id_;
        frame_id_++;

        ImageContainer image_container(frame_id, timestamp);
        image_container.rgb(rgb)
            .depth(depth)
            .opticalFlow(flow)
            .objectMotionMask(mask);

        image_container_callback_(
            std::make_shared<ImageContainer>(image_container));
      });
  CHECK(multi_image_sync->connect());
  image_subscriber_ = multi_image_sync;
}

void AllImagesOnlineProviderRos::unsubscribeImages() {
  if (image_subscriber_) image_subscriber_->shutdown();
}

CameraParams::Optional AllImagesOnlineProviderRos::getCameraParams() const {
  return calibration_helper_->getCameraParams();
}

RGBDOnlineProviderRos::RGBDOnlineProviderRos(
    rclcpp::Node::SharedPtr node, const OnlineDataProviderRosParams& params)
    : OnlineDataProviderRos(node, params) {
  LOG(INFO) << "Creating RGBDOnlineProviderRos";
  calibration_helper_ =
      std::make_unique<RGBDTypeCalibrationHelper>(node, params);
}

void RGBDOnlineProviderRos::subscribeImages() {
  rclcpp::Node& node_ref = *node_;
  static const std::array<std::string, 2>& topics = {"image/rgb",
                                                     "image/depth"};

  MultiSyncConfig config;
  config.queue_size = 20;
  // config.subscriber_options.callback_group =
  //     node_ref.create_callback_group(rclcpp::CallbackGroupType::Reentrant);

  std::shared_ptr<MultiImageSync2> multi_image_sync =
      std::make_shared<MultiImageSync2>(node_ref, topics, config);
  multi_image_sync->registerCallback(
      [this](const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg,
             const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg) {
        if (!image_container_callback_) {
          RCLCPP_ERROR_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                                "Image Sync callback triggered but "
                                "image_container_callback_ is not registered!");
          return;
        }

        cv::Mat rgb = readRgbRosImage(rgb_msg);
        cv::Mat depth = readDepthRosImage(depth_msg);

        calibration_helper_->undistort(rgb, rgb);
        calibration_helper_->undistort(depth, depth);

        const Timestamp timestamp = utils::fromRosTime(rgb_msg->header.stamp);
        const FrameId frame_id = frame_id_;
        frame_id_++;

        ImageContainer image_container(frame_id, timestamp);
        image_container.rgb(rgb).depth(depth);

        image_container_callback_(
            std::make_shared<ImageContainer>(image_container));
      });
  CHECK(multi_image_sync->connect());
  image_subscriber_ = multi_image_sync;
}

void RGBDOnlineProviderRos::unsubscribeImages() {
  if (image_subscriber_) image_subscriber_->shutdown();
}

CameraParams::Optional RGBDOnlineProviderRos::getCameraParams() const {
  return calibration_helper_->getCameraParams();
}

void RGBDOnlineProviderRos::updateAndCheckParams(DynoParams& dyno_params) {
  updateAndCheckDynoParamsForRawImageInput(dyno_params);
}

}  // namespace dyno
