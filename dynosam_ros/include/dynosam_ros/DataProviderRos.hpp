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

#include <dynosam/common/Exceptions.hpp>
#include <dynosam/common/GroundTruthPacket.hpp>
#include <dynosam/common/ImageTypes.hpp>
#include <dynosam/dataprovider/DataProvider.hpp>
#include <dynosam/pipeline/ThreadSafeTemporalBuffer.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "cv_bridge/cv_bridge.h"
#include "dynamic_slam_interfaces/msg/multi_object_odometry_path.hpp"
#include "dynosam_ros/RosUtils.hpp"
#include "dynosam_ros/adaptors/CameraParamsAdaptor.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/exact_time.h"
#include "message_filters/synchronizer.h"
#include "rclcpp/node.hpp"
#include "rclcpp/node_options.hpp"
#include "rclcpp/wait_for_message.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace dyno {

/**
 * @brief Intermediate conversion struct to convert ROS types into dynosam
 * ground truth packets.
 *
 *
 */
struct GroundTruthRosWrapper {
  rclcpp::Time timestamp;
  //! Frame id does not come with the ROS messages and must be found by the
  //! DynosamRos pipeline (ie by synchronization or other methdos)
  nav_msgs::msg::Odometry X_W;
  dynamic_slam_interfaces::msg::MultiObjectOdometryPath object_paths;

  //! For IO
  GroundTruthRosWrapper() {}

  GroundTruthRosWrapper(
      rclcpp::Time timestamp_, const nav_msgs::msg::Odometry& X_W_,
      const dynamic_slam_interfaces::msg::MultiObjectOdometryPath&
          object_paths_)
      : timestamp(timestamp_), X_W(X_W_), object_paths(object_paths_) {}
};

class GroundTruthInputHandler {
 public:
  GroundTruthInputHandler() = default;
  virtual ~GroundTruthInputHandler() = default;

  void add(const GroundTruthRosWrapper& input, FrameId frame_id);
  bool query(GroundTruthInputPacket* ground_truth, FrameId query_frame) const;

 private:
  GroundTruthPacketMap ground_truths_;
};

class GroundTruthInputHandlerROS : public GroundTruthInputHandler {
 public:
  GroundTruthInputHandlerROS(
      rclcpp::Node::SharedPtr node,
      DataProvider::GroundTruthPacketCallback* ground_truth_packet_callback);

  /**
   * @brief Call the ground_truth_packet_callback (to send a packet to the data
   * interface) now that we have a timestamp-frameid association.
   *
   * Looks inside the cache_ for the closest timestamp and then constructs a
   * GroundTruthInputPacket given frame_id.
   *
   * @param timestamp
   * @param frame_id
   */
  void emit(Timestamp timestamp, FrameId frame_id);

 private:
  void callback(const nav_msgs::msg::Odometry::ConstSharedPtr& X_W,
                const dynamic_slam_interfaces::msg::MultiObjectOdometryPath::
                    ConstSharedPtr& object_paths);

 private:
  rclcpp::Node::SharedPtr node_;
  //! Callback that sends a packet to the data interface via the provider
  //! Note we pass a pointer to the callback becuase the callback only gets set
  //! later So it will not be valid by the time of the constructor!
  DataProvider::GroundTruthPacketCallback* ground_truth_packet_callback_;

  message_filters::Subscriber<nav_msgs::msg::Odometry> robot_odometry_sub_;
  message_filters::Subscriber<
      dynamic_slam_interfaces::msg::MultiObjectOdometryPath>
      object_odometry_sub_;

  using SyncPolicy = message_filters::sync_policies::ExactTime<
      nav_msgs::msg::Odometry,
      dynamic_slam_interfaces::msg::MultiObjectOdometryPath>;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

  using GroundTruthTemporalBuffer =
      ThreadsafeTemporalBuffer<GroundTruthRosWrapper>;
  // temporal cache that holds GroundTruthRosWrapper until we can associate a
  // timestamp with a frameid this will happen when we get an image callback
  GroundTruthTemporalBuffer cache_;
};

/**
 * @brief Base Dataprovider for ROS that implements common image processing
 * functionalities.
 *
 */
class DataProviderRos : public DataProvider {
 public:
  DataProviderRos(rclcpp::Node::SharedPtr node);
  virtual ~DataProviderRos() = default;

  /**
   * @brief Convers a sensor_msgs::msg::Image to a cv::Mat while testing that
   * the input has the correct datatype for an RGB image (as defined by
   * ImageType::RGBMono).
   *
   * @param img_msg const sensor_msgs::msg::Image::ConstSharedPtr&
   * @return const cv::Mat
   */
  const cv::Mat readRgbRosImage(
      const sensor_msgs::msg::Image::ConstSharedPtr& img_msg) const;

  /**
   * @brief Convers a sensor_msgs::msg::Image to a cv::Mat while testing that
   * the input has the correct datatype for an Depth image (as defined by
   * ImageType::Depth).
   *
   * @param img_msg const sensor_msgs::msg::Image::ConstSharedPtr&
   * @return const cv::Mat
   */
  const cv::Mat readDepthRosImage(
      const sensor_msgs::msg::Image::ConstSharedPtr& img_msg) const;

  /**
   * @brief Convers a sensor_msgs::msg::Image to a cv::Mat while testing that
   * the input has the correct datatype for an Optical Flow image (as defined by
   * ImageType::OpticalFlow).
   *
   * @param img_msg const sensor_msgs::msg::Image::ConstSharedPtr&
   * @return const cv::Mat
   */
  const cv::Mat readFlowRosImage(
      const sensor_msgs::msg::Image::ConstSharedPtr& img_msg) const;

  /**
   * @brief Convers a sensor_msgs::msg::Image to a cv::Mat while testing that
   * the input has the correct datatype for an Motion Mask image (as defined by
   * ImageType::MotionMask).
   *
   * @param img_msg const sensor_msgs::msg::Image::ConstSharedPtr&
   * @return const cv::Mat
   */
  const cv::Mat readMaskRosImage(
      const sensor_msgs::msg::Image::ConstSharedPtr& img_msg) const;

  /**
   * @brief Gets CameraParams from a sensor_msgs::msg::CameraInfo recieved on
   * the specified topic. This function is blocking until a message is recieved
   * (or until the time_to_wait) elapses.
   *
   * While this function returns a const ref to the CameraParams it also sets
   * the internal camera_params_. The camera params are then returned by the
   * overwritten getCameraParams, allowing the PipelineManager to access the
   * correct camera paramters.
   *
   * @tparam Rep int64_t,
   * @tparam Period std::milli
   * @param time_to_wait const std::chrono::duration<Rep, Period>&
   * @param topic const std::string&. Defaults to "image/camera_info"
   * @return const CameraParams&
   */
  template <class Rep = int64_t, class Period = std::milli>
  const CameraParams& waitAndSetCameraParams(
      const std::chrono::duration<Rep, Period>& time_to_wait =
          std::chrono::duration<Rep, Period>(-1),
      const std::string& topic = "image/camera_info") {
    RCLCPP_INFO_STREAM(node_->get_logger(),
                       "Waiting for camera params on topic: " << topic);
    // it seems rclcpp::Adaptors do not work yet with wait for message
    sensor_msgs::msg::CameraInfo camera_info;
    if (rclcpp::wait_for_message<sensor_msgs::msg::CameraInfo, Rep, Period>(
            camera_info, node_, topic, time_to_wait)) {
      using Adaptor =
          rclcpp::TypeAdapter<dyno::CameraParams, sensor_msgs::msg::CameraInfo>;
      CameraParams camera_params;
      Adaptor::convert_to_custom(camera_info, camera_params);
      RCLCPP_INFO_STREAM(node_->get_logger(), "Received camera params: "
                                                  << camera_params.toString());
      camera_params_ = camera_params;
      return *camera_params_;
    } else {
      const auto milliseconds =
          std::chrono::duration_cast<std::chrono::milliseconds>(time_to_wait);
      throw DynosamException("Failed to receive camera params on topic " +
                             topic + " (waited with timeout " +
                             std::to_string(milliseconds.count()) + " ms).");
    }
  }

  CameraParams::Optional getCameraParams() const override {
    return camera_params_;
  }

 protected:
  /**
   * @brief Helper function to convert a ROS Image message to a CvImageConstPtr
   * via the cv bridge.
   *
   * @param img_msg const sensor_msgs::msg::Image::ConstSharedPtr&
   * @return const cv_bridge::CvImageConstPtr
   */
  const cv_bridge::CvImageConstPtr readRosImage(
      const sensor_msgs::msg::Image::ConstSharedPtr& img_msg) const;

  /**
   * @brief Helper function to convert a
   * sensor_msgs::msg::Image::ConstSharedPtr& to a cv::Mat with the right
   * datatype.
   *
   * The datatype is specified from the template IMAGETYPE::OpenCVType and
   * ensures the passed in image has the correct datatype for the desired
   * IMAGETYPE.
   *
   * ROS will be shutdown if the incoming image has an incorrect type.
   *
   * @tparam IMAGETYPE
   * @param img_msg  const sensor_msgs::msg::Image::ConstSharedPtr&
   * @return const cv::Mat
   */
  template <typename IMAGETYPE>
  const cv::Mat convertRosImage(
      const sensor_msgs::msg::Image::ConstSharedPtr& img_msg) const {
    const cv_bridge::CvImageConstPtr cvb_image = readRosImage(img_msg);

    try {
      const cv::Mat img = cvb_image->image;
      image_traits<IMAGETYPE>::validate(img);
      return img;

    } catch (const InvalidImageTypeException& exception) {
      RCLCPP_FATAL_STREAM(node_->get_logger(),
                          image_traits<IMAGETYPE>::name()
                              << " Image msg was of the wrong type (validate "
                                 "failed with exception "
                              << exception.what() << "). "
                              << "ROS encoding type used was "
                              << cvb_image->encoding);
      rclcpp::shutdown();
      return cv::Mat();
    }
  }

 protected:
  rclcpp::Node::SharedPtr node_;
  std::unique_ptr<GroundTruthInputHandlerROS> ground_truth_handler_;

  CameraParams::Optional camera_params_;
};

}  // namespace dyno
