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

#include "dynosam_ros/DataProviderRos.hpp"

#include <dynosam/common/ImageTypes.hpp>

#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace dyno {

DataProviderRos::DataProviderRos(rclcpp::Node::SharedPtr node)
    : DataProvider(), node_(node) {}

// specalise conversion function for depth image to handle the case we are given
// float32 bit images
template <>
const cv::Mat DataProviderRos::convertRosImage<ImageType::Depth>(
    const sensor_msgs::msg::Image::ConstSharedPtr& img_msg) const {
  const cv_bridge::CvImageConstPtr cvb_image = readRosImage(img_msg);
  const cv::Mat img = cvb_image->image;

  try {
    image_traits<ImageType::Depth>::validate(img);
    return img;

  } catch (const InvalidImageTypeException& exception) {
    // handle the case its a float32
    if (img.type() == CV_32FC1) {
      cv::Mat depth64;
      img.convertTo(depth64, CV_64FC1);
      image_traits<ImageType::Depth>::validate(depth64);
      return depth64;
    }

    RCLCPP_FATAL_STREAM(node_->get_logger(),
                        image_traits<ImageType::Depth>::name()
                            << " Image msg was of the wrong type (validate "
                               "failed with exception "
                            << exception.what() << "). "
                            << "ROS encoding type used was "
                            << cvb_image->encoding);

    rclcpp::shutdown();
    return cv::Mat();
  }
}

const cv::Mat DataProviderRos::readRgbRosImage(
    const sensor_msgs::msg::Image::ConstSharedPtr& img_msg) const {
  cv::Mat rgb_mono_image = convertRosImage<ImageType::RGBMono>(img_msg);
  if (img_msg->encoding == "rgb8") {
    cv::cvtColor(rgb_mono_image, rgb_mono_image, cv::COLOR_RGB2BGR);
  }
  return rgb_mono_image;
}

const cv::Mat DataProviderRos::readDepthRosImage(
    const sensor_msgs::msg::Image::ConstSharedPtr& img_msg) const {
  return convertRosImage<ImageType::Depth>(img_msg);
}

const cv::Mat DataProviderRos::readFlowRosImage(
    const sensor_msgs::msg::Image::ConstSharedPtr& img_msg) const {
  return convertRosImage<ImageType::OpticalFlow>(img_msg);
}

const cv::Mat DataProviderRos::readMaskRosImage(
    const sensor_msgs::msg::Image::ConstSharedPtr& img_msg) const {
  return convertRosImage<ImageType::MotionMask>(img_msg);
}

const cv_bridge::CvImageConstPtr DataProviderRos::readRosImage(
    const sensor_msgs::msg::Image::ConstSharedPtr& img_msg) const {
  CHECK(img_msg);
  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    // important to copy to ensure that memory does not go out of scope (which
    // it seems to !!!)
    cv_ptr = cv_bridge::toCvCopy(img_msg);
  } catch (cv_bridge::Exception& exception) {
    RCLCPP_FATAL(node_->get_logger(), "cv_bridge exception: %s",
                 exception.what());
    rclcpp::shutdown();
  }
  return cv_ptr;
}

}  // namespace dyno
