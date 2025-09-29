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
#include <dynosam/common/ImageTypes.hpp>
#include <dynosam/dataprovider/DataProvider.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "cv_bridge/cv_bridge.hpp"
#include "dynosam_ros/adaptors/CameraParamsAdaptor.hpp"
#include "rclcpp/node.hpp"
#include "rclcpp/node_options.hpp"
#include "rclcpp/wait_for_message.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace dyno {

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
  CameraParams::Optional camera_params_;
};

}  // namespace dyno
