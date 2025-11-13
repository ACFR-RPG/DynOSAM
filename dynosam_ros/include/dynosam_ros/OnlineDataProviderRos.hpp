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

#include "dynosam/pipeline/PipelineParams.hpp"
#include "dynosam_ros/DataProviderRos.hpp"
#include "dynosam_ros/MultiSync.hpp"
#include "dynosam_ros/adaptors/ImuMeasurementAdaptor.hpp"
#include "rclcpp/node.hpp"
#include "rclcpp/node_options.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/imu.hpp"

namespace dyno {

enum InputImageMode : int {
  //! Expects rgb, depth, flow and semantics to be provided
  ALL = 0,
  //! Expects only rgb and depth images to be provided
  RGBD = 1,
  STEREO = 2
};

struct OnlineDataProviderRosParams {
  bool subscribe_imu{true};
  bool wait_for_camera_params{true};
  int32_t camera_params_timeout{-1};
};

/**
 * @brief Online data-provider for DynoSAM which handles subcription to the IMU
 * and the interface for subscribing to images but does not implement this.
 *
 * Instead input image mode specific classes derive from this class to implement
 * the image subscription
 *
 *
 *
 *
 */
class OnlineDataProviderRos : public DataProviderRos {
 public:
  DYNO_POINTER_TYPEDEFS(OnlineDataProviderRos)

  /**
   * @brief Construct a new OnlineDataProviderRos.
   *
   * Constructor will block until camera info has been received (if
   * OnlineDataProviderRosParams::wait_for_camera_params is true).
   *
   * @param node rclcpp::Node::SharedPtr
   * @param params const OnlineDataProviderRosParams&
   */
  OnlineDataProviderRos(rclcpp::Node::SharedPtr node,
                        const OnlineDataProviderRosParams& params);

  virtual ~OnlineDataProviderRos() = default;

  /**
   * @brief Indicates that there is no known end to the dataset
   *
   * @return int
   */
  int datasetSize() const override { return -1; }

  /**
   * @brief Returns true while not shutdown
   *
   * @return true
   * @return false
   */
  bool spin() override;

  /**
   * @brief Disconnects all subscribers
   *
   */
  void shutdown() override;

  /**
   * @brief Connects all subscribers.
   *
   */
  void setupSubscribers();

  /**
   * @brief Checks all params are configured correctly for this (derived)
   * data-loader and updates them if necessary
   *
   * @param dyno_params
   */
  virtual void updateAndCheckParams(DynoParams&){};

 protected:
  virtual void subscribeImages() = 0;
  virtual void unsubscribeImages() = 0;

  OnlineDataProviderRosParams params_;
  //! Driving frame id for the enture dynosam pipeline
  FrameId frame_id_;

 private:
  void subscribeImu();

  rclcpp::CallbackGroup::SharedPtr imu_callback_group_;
  using ImuAdaptedType =
      rclcpp::adapt_type<dyno::ImuMeasurement>::as<sensor_msgs::msg::Imu>;
  rclcpp::Subscription<ImuAdaptedType>::SharedPtr imu_sub_;

  std::atomic_bool is_connected{false};
};

/**
 * @brief Class to help setup calibration for undistortion/resizing etc
 * for an RGBD style input where only a single CameraParams is needed to
 * represent a pinhole camera, rather than a stereo setup which needs different
 * management
 *
 * TODO: why not use the UndistortRectifier?!
 *
 */
class RGBDTypeCalibrationHelper {
 public:
  RGBDTypeCalibrationHelper(rclcpp::Node::SharedPtr node,
                            const OnlineDataProviderRosParams& params);

  void undistort(const cv::Mat& src, cv::Mat& dst);

  const CameraParams::Optional& getOriginalCameraParams() const;
  const CameraParams::Optional& getCameraParams() const;

 private:
  void setupNewCameraParams(const CameraParams& original_camera_params,
                            CameraParams& new_camera_params,
                            const int& rescale_width,
                            const int& rescale_height);

  void getParamsFromRos(const CameraParams& original_camera_params,
                        int& rescale_width, int& rescale_height);

 private:
  rclcpp::Node::SharedPtr node_;
  CameraParams::Optional original_camera_params_;
  CameraParams::Optional camera_params_;

  //! Undistort maps
  cv::Mat mapx_;
  cv::Mat mapy_;
};

/**
 * @brief Updates and checks that all dyno params are set correctly
 * for when only raw image data is providd (ie. RBGD or Stereo) and no
 * object/flow pre-processing has been done
 *
 * @param dyno_params
 */
void updateAndCheckDynoParamsForRawImageInput(DynoParams& dyno_params);

/**
 * @brief Class that subscribes to rgb, depth, motion mask and dense optical
 * flow topics.
 *
 */
class AllImagesOnlineProviderRos : public OnlineDataProviderRos {
 public:
  AllImagesOnlineProviderRos(rclcpp::Node::SharedPtr node,
                             const OnlineDataProviderRosParams& params);

  void subscribeImages() override;
  void unsubscribeImages() override;
  CameraParams::Optional getCameraParams() const override;

 private:
  std::unique_ptr<RGBDTypeCalibrationHelper> calibration_helper_;
  MultiSyncBase::Ptr image_subscriber_;
};

/**
 * @brief Class that subscribes to rgb, depth, motion mask and dense optical
 * flow topics.
 *
 */
class RGBDOnlineProviderRos : public OnlineDataProviderRos {
 public:
  RGBDOnlineProviderRos(rclcpp::Node::SharedPtr node,
                        const OnlineDataProviderRosParams& params);

  void subscribeImages() override;
  void unsubscribeImages() override;
  CameraParams::Optional getCameraParams() const override;

  void updateAndCheckParams(DynoParams& dyno_params) override;

 private:
  std::unique_ptr<RGBDTypeCalibrationHelper> calibration_helper_;
  MultiSyncBase::Ptr image_subscriber_;
};

}  // namespace dyno
