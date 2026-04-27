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
#include "geometry_msgs/msg/transform_stamped.hpp"

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
    case InputImageMode::RGBDM: {
      status_str = "RGBDM";
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

StereoCalibrationHelper::StereoCalibrationHelper(
    rclcpp::Node::SharedPtr node, const OnlineDataProviderRosParams& params)
    : buffer_(node->get_clock()), listener_(buffer_) {
  const CameraParams original_cam0_params = waitAndSetCameraParams(
      node, "cam0/camera_info",
      std::chrono::milliseconds(params.camera_params_timeout));
  original_camera_params_ = original_cam0_params;

  CameraParams original_cam1_params = waitAndSetCameraParams(
      node, "cam1/camera_info",
      std::chrono::milliseconds(params.camera_params_timeout));
  // hardcoded sort of ish for the real-sense camera for now!
  // stereo-camera uses extrinsics to work out baseline... even though this is
  // provided as part of the projection matrix P our not-very-ideal handling of
  // camera params data-structures means we dont even have access to this data
  // out of ROS. Instead rely on the tf frames to get good extrinsics
  // in this case we get the transform of T_c1_c2 (ie transform of camera 2 into
  // camera) and set the inrinsics of cam1 such that the reference frame is cam!
  geometry_msgs::msg::TransformStamped tf = buffer_.lookupTransform(
      "camera_infra2_frame", "camera_infra1_frame", tf2::TimePointZero);

  gtsam::Pose3 T_C1_C2;
  convert(tf, T_C1_C2);
  original_cam1_params.setExtrinsics(T_C1_C2);

  stereo_camera_ = std::make_shared<StereoCamera>(original_cam0_params,
                                                  original_cam1_params);

  const gtsam::Cal3_S2Stereo& new_calibration =
      stereo_camera_->getStereoCameraCalibration();

  dyno::CameraParams::IntrinsicsCoeffs intrinsics{
      new_calibration.fx(),
      new_calibration.fy(),
      new_calibration.px(),
      new_calibration.py(),
  };
  dyno::CameraParams::DistortionCoeffs zero_distortion(4, 0);

  // not sure if image size is correct here!
  camera_params_ = CameraParams(intrinsics, zero_distortion,
                                original_cam0_params.imageSize(),
                                original_cam0_params.getDistortionModel());

  camera_params_->setDepthParams(stereo_camera_->getBaseline());
}

const CameraParams::Optional& StereoCalibrationHelper::getOriginalCameraParams()
    const {
  return original_camera_params_;
}
const CameraParams::Optional& StereoCalibrationHelper::getCameraParams() const {
  return camera_params_;
}

void StereoCalibrationHelper::processPair(cv::Mat& rectify_left,
                                          cv::Mat& rectify_right,
                                          const cv::Mat& left,
                                          const cv::Mat& right) const {
  stereo_camera_->undistortRectifyImages(rectify_left, rectify_right, left,
                                         right);
}

RGBDTypeCalibrationHelper::RGBDTypeCalibrationHelper(
    rclcpp::Node::SharedPtr node, const OnlineDataProviderRosParams& params)
    : node_(node) {
  if (params.wait_for_camera_params) {
    const CameraParams original_camera_params = waitAndSetCameraParams(
        node, "camera/camera_info",
        std::chrono::milliseconds(params.camera_params_timeout));

    int rescale_width, rescale_height;
    getParamsFromRos(original_camera_params, rescale_width, rescale_height,
                     depth_scale_, baseline_);

    CameraParams camera_params;
    setupNewCameraParams(original_camera_params, camera_params, rescale_width,
                         rescale_height);

    original_camera_params_ = original_camera_params;
    camera_params_ = camera_params;
  }
}

void RGBDTypeCalibrationHelper::processRGB(const cv::Mat& src,
                                           cv::Mat& dst) const {
  undistortWithMaps(src, dst);
}
void RGBDTypeCalibrationHelper::processDepth(const cv::Mat& src,
                                             cv::Mat& dst) const {
  CHECK(src.type() == ImageType::Depth::OpenCVType);
  undistortWithMaps(src, dst);
  // convert the depth map to metirc scale
  // data-type shoule match
  dst *= depth_scale_;
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

  double alpha = 0.0;  // crop to valid region
  cv::Mat new_K = cv::getOptimalNewCameraMatrix(
      original_K, distortion, original_size, alpha, rescale_size);

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

  new_camera_params.setDepthParams(baseline_);
}

void RGBDTypeCalibrationHelper::getParamsFromRos(
    const CameraParams& original_camera_params, int& rescale_width,
    int& rescale_height, double& depth_scale, double& baseline) {
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

  depth_scale = ParameterConstructor(node_.get(), "depth_scale", 0.001)
                    .description(
                        "Value to scale the depth image from a disparity map "
                        "to metric depth")
                    .finish()
                    .get<double>();

  baseline = ParameterConstructor(node_.get(), "baseline", 0.1)
                 .description(
                     "Stereo camera baseline needed for virtual-stereo system")
                 .finish()
                 .get<double>();
}

void RGBDTypeCalibrationHelper::undistortWithMaps(const cv::Mat& src,
                                                  cv::Mat& dst) const {
  cv::remap(src, dst, mapx_, mapy_, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
  // output will have the same type as mapx/y so covnert back to required type
  dst.convertTo(dst, src.type());
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

RGBDMTypeCalibrationHelper::RGBDMTypeCalibrationHelper(
    rclcpp::Node::SharedPtr node, const OnlineDataProviderRosParams& params)
    : RGBDTypeCalibrationHelper(node, params) {}

void RGBDMTypeCalibrationHelper::processMask(const cv::Mat& src,
                                             cv::Mat& dst) const {
  CHECK(src.type() == ImageType::MotionMask::OpenCVType);
  undistortWithMaps(src, dst);
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

  // make multiimage sync and and queue have similar depth
  static constexpr size_t queue_size = 1000;
  auto image_qos = rclcpp::SensorDataQoS()
                       .keep_last(queue_size)  // large queue
                       .best_effort();

  MultiSyncConfig config;
  config.queue_size = queue_size;
  config.subscriber_qos = image_qos;
  config.subscriber_options.callback_group =
      node_ref.create_callback_group(rclcpp::CallbackGroupType::Reentrant);

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

  // make multiimage sync and and queue have similar depth
  // reliable important so we dont drop frames we're quite reliant on frame
  // to frame tracking!
  static constexpr size_t queue_size = 1000;
  auto image_qos = rclcpp::SensorDataQoS().keep_last(queue_size).reliable();

  MultiSyncConfig config;
  config.queue_size = queue_size;
  config.subscriber_qos = image_qos;
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

        cv::Mat rgb = readRgbRosImage(rgb_msg).clone();
        cv::Mat depth = readDepthRosImage(depth_msg).clone();

        calibration_helper_->processRGB(rgb, rgb);
        calibration_helper_->processDepth(depth, depth);

        const Timestamp timestamp = utils::fromRosTime(rgb_msg->header.stamp);
        const FrameId frame_id = frame_id_;
        frame_id_++;

        auto image_container =
            std::make_shared<ImageContainer>(frame_id, timestamp);
        (*image_container).rgb(rgb).depth(depth);

        image_container_callback_(image_container);
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

RGBDMOnlineProviderRos::RGBDMOnlineProviderRos(
    rclcpp::Node::SharedPtr node, const OnlineDataProviderRosParams& params)
    : OnlineDataProviderRos(node, params) {
  LOG(INFO) << "Creating RGBDMOnlineProviderRos";
  calibration_helper_ =
      std::make_unique<RGBDMTypeCalibrationHelper>(node, params);
}

void RGBDMOnlineProviderRos::subscribeImages() {
  rclcpp::Node& node_ref = *node_;
  static const std::array<std::string, 3>& topics = {"image/rgb", "image/depth",
                                                     "image/mask"};

  // make multiimage sync and and queue have similar depth
  // reliable important so we dont drop frames we're quite reliant on frame
  // to frame tracking!
  static constexpr size_t queue_size = 1000;
  auto image_qos = rclcpp::SensorDataQoS().keep_last(queue_size).reliable();

  MultiSyncConfig config;
  config.queue_size = queue_size;
  config.subscriber_qos = image_qos;
  // config.subscriber_options.callback_group =
  //     node_ref.create_callback_group(rclcpp::CallbackGroupType::Reentrant);

  std::shared_ptr<MultiImageSync3> multi_image_sync =
      std::make_shared<MultiImageSync3>(node_ref, topics, config);
  multi_image_sync->registerCallback(
      [this](const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg,
             const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg,
             const sensor_msgs::msg::Image::ConstSharedPtr& mask_msg) {
        if (!image_container_callback_) {
          RCLCPP_ERROR_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                                "Image Sync callback triggered but "
                                "image_container_callback_ is not registered!");
          return;
        }

        cv::Mat rgb = readRgbRosImage(rgb_msg).clone();
        cv::Mat depth = readDepthRosImage(depth_msg).clone();
        cv::Mat mask = readMaskRosImage(mask_msg).clone();

        calibration_helper_->processRGB(rgb, rgb);
        calibration_helper_->processDepth(depth, depth);
        // for now do not process and assume all images are undistorted!
        // calibration_helper_->processMask(mask, mask);

        const Timestamp timestamp = utils::fromRosTime(rgb_msg->header.stamp);
        const FrameId frame_id = frame_id_;
        frame_id_++;

        auto image_container =
            std::make_shared<ImageContainer>(frame_id, timestamp);
        (*image_container).rgb(rgb).depth(depth).objectMotionMask(mask);

        image_container_callback_(image_container);
      });
  CHECK(multi_image_sync->connect());
  image_subscriber_ = multi_image_sync;
}

void RGBDMOnlineProviderRos::unsubscribeImages() {
  if (image_subscriber_) image_subscriber_->shutdown();
}

CameraParams::Optional RGBDMOnlineProviderRos::getCameraParams() const {
  return calibration_helper_->getCameraParams();
}

void RGBDMOnlineProviderRos::updateAndCheckParams(DynoParams& dyno_params) {
  auto& tracker_params = dyno_params.frontend_params_.tracker_params;
  if (tracker_params.prefer_provided_optical_flow) {
    LOG(WARNING)
        << "InputImageMode not set to ALL but prefer_provided_optical_flow is "
           "true - param will be updated!";
    tracker_params.prefer_provided_optical_flow = false;
  }
  if (!tracker_params.prefer_provided_object_detection) {
    LOG(WARNING) << "InputImageMode not set to RGBD but "
                    "prefer_provided_object_detection "
                    "is false - param will be updated to true!!!";
    tracker_params.prefer_provided_object_detection = true;
  }
}

StereoOnlineProviderRos::StereoOnlineProviderRos(
    rclcpp::Node::SharedPtr node, const OnlineDataProviderRosParams& params)
    : OnlineDataProviderRos(node, params) {
  LOG(INFO) << "Creating StereoOnlineProviderRos";
  calibration_helper_ = std::make_unique<StereoCalibrationHelper>(node, params);
}

void StereoOnlineProviderRos::subscribeImages() {
  rclcpp::Node& node_ref = *node_;
  static const std::array<std::string, 2>& topics = {"cam0/image_raw",
                                                     "cam1/image_raw"};

  // make multiimage sync and and queue have similar depth
  // reliable important so we dont drop frames we're quite reliant on frame
  // to frame tracking!
  static constexpr size_t queue_size = 1000;
  auto image_qos = rclcpp::SensorDataQoS().keep_last(queue_size).reliable();

  MultiSyncConfig config;
  config.queue_size = queue_size;
  config.subscriber_qos = image_qos;
  // config.subscriber_options.callback_group =
  //     node_ref.create_callback_group(rclcpp::CallbackGroupType::Reentrant);

  std::shared_ptr<MultiImageSync2> multi_image_sync =
      std::make_shared<MultiImageSync2>(node_ref, topics, config);
  multi_image_sync->registerCallback(
      [this](const sensor_msgs::msg::Image::ConstSharedPtr& left_msg,
             const sensor_msgs::msg::Image::ConstSharedPtr& right_msg) {
        if (!image_container_callback_) {
          RCLCPP_ERROR_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                                "Image Sync callback triggered but "
                                "image_container_callback_ is not registered!");
          return;
        }

        cv::Mat left = readRgbRosImage(left_msg).clone();
        cv::Mat right = readRgbRosImage(right_msg).clone();
        calibration_helper_->processPair(left, right, left, right);

        const Timestamp timestamp = utils::fromRosTime(left_msg->header.stamp);
        const FrameId frame_id = frame_id_;
        frame_id_++;

        auto image_container =
            std::make_shared<ImageContainer>(frame_id, timestamp);
        (*image_container).rgb(left).rightRgb(right);

        image_container_callback_(image_container);
      });
  CHECK(multi_image_sync->connect());
  image_subscriber_ = multi_image_sync;
}

void StereoOnlineProviderRos::unsubscribeImages() {
  if (image_subscriber_) image_subscriber_->shutdown();
}

CameraParams::Optional StereoOnlineProviderRos::getCameraParams() const {
  return calibration_helper_->getCameraParams();
}

void StereoOnlineProviderRos::updateAndCheckParams(DynoParams& dyno_params) {
  // TODO: warning!
  // TODO: not right logging as the hardcoded "ALL" is in this function!
  updateAndCheckDynoParamsForRawImageInput(dyno_params);
}

}  // namespace dyno
