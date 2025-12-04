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

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <dynosam/test/helpers.hpp>

#include "dynosam_ros/MultiSync.hpp"
#include "dynosam_ros/OnlineDataProviderRos.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/imu.hpp"

typedef sensor_msgs::msg::Imu Msg;
typedef std::shared_ptr<sensor_msgs::msg::Imu const> MsgConstPtr;
typedef std::shared_ptr<sensor_msgs::msg::Imu> MsgPtr;

class Helper {
 public:
  Helper() : count(0) {}

  void cb(const MsgConstPtr) { ++count; }

  int32_t count;
};

using namespace dyno;
using namespace std::chrono_literals;

TEST(OnlineDataProviderRos, testwaitCameraInfoSubscribe) {
  auto node = std::make_shared<rclcpp::Node>("test_wait_for_camera_info_sub");

  auto publisher = node->create_publisher<sensor_msgs::msg::CameraInfo>(
      "image/camera_info", 10);
  auto camera_info_msg = sensor_msgs::msg::CameraInfo();
  camera_info_msg.header.stamp = node->now();
  camera_info_msg.header.frame_id = "camera_frame";
  camera_info_msg.width = 640;
  camera_info_msg.height = 480;
  camera_info_msg.k = {500.0, 0.0, 320.0, 0.0, 500.0,
                       240.0, 0.0, 0.0,   1.0};      // Intrinsic matrix
  camera_info_msg.d = {0.1, -0.1, 0.01, 0.01, 0.0};  // Distortion coefficients
  camera_info_msg.distortion_model = "plumb_bob";

  auto received = false;
  std::shared_ptr<OnlineDataProviderRos> odpr = nullptr;
  std::shared_future<bool> wait = std::async(std::launch::async, [&]() {
    OnlineDataProviderRosParams params;
    params.wait_for_camera_params = true;
    params.camera_params_timeout = -1;
    odpr = std::make_shared<OnlineDataProviderRos>(node, params);
    received = true;
    return true;
  });

  for (auto i = 0u; i < 10 && received == false; ++i) {
    publisher->publish(camera_info_msg);
    std::this_thread::sleep_for(1s);
  }

  ASSERT_NO_THROW(wait.get());
  ASSERT_TRUE(received);
  EXPECT_TRUE(odpr->getCameraParams());
}

TEST(OnlineDataProviderRos, testNowaitCameraInfoSubscribe) {
  auto node =
      std::make_shared<rclcpp::Node>("test_no_wait_for_camera_info_sub");

  OnlineDataProviderRosParams params;
  params.wait_for_camera_params = false;
  auto odpr = std::make_shared<OnlineDataProviderRos>(node, params);
  EXPECT_FALSE(odpr->getCameraParams());
}

TEST(MultiSync, printVersionMessageFilters) {
  std::cout << "--- Detection Check ---" << std::endl;
  std::cout << "Message filters use NodeParametersInterface: " << std::boolalpha
            << message_filters_uses_node_interface << std::endl;
  // This will print the version number defined by the preprocessor logic
  std::cout << "Defined MESSAGE_FILTERS_USES_NODE_INTERFACE: "
            << MESSAGE_FILTERS_USES_NODE_INTERFACE << std::endl;
}

TEST(MultiSync, basicInvalidConnect) {
  auto node = std::make_shared<rclcpp::Node>("test");

  using MIS = MultiImageSync<2>;
  MIS image_sync(*node, {"image_raw1", "image_raw2"}, 10);
  EXPECT_FALSE(image_sync.connect());
}

TEST(MultiSync, basicConnect) {
  auto node = std::make_shared<rclcpp::Node>("test");

  using MIS = MultiImageSync<2>;
  MIS image_sync(*node, {"image_raw1", "image_raw2"}, 10);
  image_sync.registerCallback(
      [](const sensor_msgs::msg::Image::ConstSharedPtr&,
         const sensor_msgs::msg::Image::ConstSharedPtr&) {});

  EXPECT_TRUE(image_sync.connect());
}

TEST(MultiSync, basicSubscribe) {
  auto node = std::make_shared<rclcpp::Node>("test_node");
  Helper h;

  using MIS = MultiSync<Msg, 2>;
  MIS image_sync(*node, {"test_topic1", "test_topic2"}, 10);
  image_sync.registerCallback(
      [&h](const MsgConstPtr& msg1, const MsgConstPtr&) { h.cb(msg1); });

  auto pub1 = node->create_publisher<Msg>("test_topic1", 10);
  auto pub2 = node->create_publisher<Msg>("test_topic2", 10);

  EXPECT_TRUE(image_sync.connect());

  rclcpp::Clock ros_clock;
  auto start = ros_clock.now();
  while (h.count == 0 && (ros_clock.now() - start) < rclcpp::Duration(1, 0)) {
    pub1->publish(Msg());
    pub2->publish(Msg());
    rclcpp::Rate(50).sleep();
    rclcpp::spin_some(node);
  }

  ASSERT_GT(h.count, 0);
}
