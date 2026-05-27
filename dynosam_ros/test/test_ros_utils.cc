/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam_ros/RosUtils.hpp"

using namespace dyno;
using namespace dyno::ros;

#include "geometry_msgs/msg/point.hpp"
#include "nav_msgs/msg/odometry.hpp"

TEST(TestConcepts, SubNodeNamespacing) {
  auto node = std::make_shared<rclcpp::Node>("my_node", "my_ns");
  LOG(INFO) << node->get_effective_namespace();  // -> "/my_ns"
  auto sub_node1 = node->create_sub_node("a");
  LOG(INFO) << sub_node1->get_effective_namespace();  // -> "/my_ns/a"

  auto pub = sub_node1->create_publisher<nav_msgs::msg::Odometry>("test", 1);
  LOG(INFO) << pub->get_topic_name();
}

TEST(RosUtils, HasMsgHeader) {
  EXPECT_FALSE(ros::internal::HasMsgHeader<geometry_msgs::msg::Point>::value);
  EXPECT_TRUE(ros::internal::HasMsgHeader<nav_msgs::msg::Odometry>::value);
}

TEST(RosTraits, testParamTypes) {
  EXPECT_TRUE(
      (std::is_same_v<
          dyno::ros::param_traits<
              rcl_interfaces::msg::ParameterType::PARAMETER_DOUBLE>::cpp_type,
          double>));
  EXPECT_EQ(dyno::traits<double>::ros_parameter_type,
            rcl_interfaces::msg::ParameterType::PARAMETER_DOUBLE);

  EXPECT_TRUE(
      (std::is_same_v<
          dyno::ros::param_traits<
              rcl_interfaces::msg::ParameterType::PARAMETER_STRING>::cpp_type,
          const char *>));
  EXPECT_EQ(dyno::traits<const char *>::ros_parameter_type,
            rcl_interfaces::msg::ParameterType::PARAMETER_STRING);

  EXPECT_TRUE((std::is_same_v<
               dyno::ros::param_traits<rcl_interfaces::msg::ParameterType::
                                           PARAMETER_DOUBLE_ARRAY>::cpp_type,
               std::vector<double>>));
  EXPECT_EQ(dyno::traits<std::vector<double>>::ros_parameter_type,
            rcl_interfaces::msg::ParameterType::PARAMETER_DOUBLE_ARRAY);
}

TEST(ParameterBuilder, defaultConstructionShared) {
  auto node = std::make_shared<rclcpp::Node>("test_node");
  ros::Parameter::Builder pc(node, "test_param");
  const rclcpp::ParameterValue &pc_as_value = pc;
  EXPECT_EQ(pc_as_value.get_type(), rclcpp::PARAMETER_NOT_SET);
  EXPECT_EQ(pc.name(), "test_param");
}

TEST(ParameterBuilder, defaultConstructionRawP) {
  auto node = std::make_shared<rclcpp::Node>("test_node");
  ros::Parameter::Builder pc(node.get(), "test_param");
  const rclcpp::ParameterValue &pc_as_value = pc;
  EXPECT_EQ(pc_as_value.get_type(), rclcpp::PARAMETER_NOT_SET);
  EXPECT_EQ(pc.name(), "test_param");
}

TEST(ParameterBuilder, valueConstruction) {
  auto node = std::make_shared<rclcpp::Node>("test_node");
  ros::Parameter::Builder pc(node, "test_param", 10);
  const rclcpp::ParameterValue &pc_as_value = pc;
  EXPECT_EQ(pc_as_value.get_type(), rclcpp::PARAMETER_INTEGER);
  EXPECT_EQ(pc_as_value.get<int>(), 10);
  EXPECT_EQ(pc.name(), "test_param");
}

TEST(ParameterBuilder, testOptionUpdates) {
  auto node = std::make_shared<rclcpp::Node>("test_node");
  ros::Parameter::Builder pc = ros::Parameter::Builder(node, "test_param", 10)
                                   .description("a test")
                                   .read_only(true);
  const rcl_interfaces::msg::ParameterDescriptor &p_as_descriptor = pc;
  EXPECT_EQ(p_as_descriptor.description, "a test");
  EXPECT_EQ(p_as_descriptor.read_only, true);
}

TEST(ParameterBuilder, testFinishWithDefault) {
  auto node = std::make_shared<rclcpp::Node>("test_node");
  Parameter detail = ros::Parameter::Builder(node, "test_param", 10)
                         .description("a test")
                         .read_only(true)
                         .finish();
  const rclcpp::ParameterValue &detail_as_value = detail;
  EXPECT_EQ(detail_as_value.get_type(), rclcpp::PARAMETER_INTEGER);
  EXPECT_EQ(detail_as_value.get<int>(), 10);
}

TEST(ParameterBuilder, testFinishWithoutDefault) {
  auto node = std::make_shared<rclcpp::Node>("test_node");
  Parameter detail_shared = ros::Parameter::Builder(node, "test_param")
                                .description("a test")
                                .read_only(true)
                                .finish();
  EXPECT_THROW({ detail_shared.get(); }, InvalidDefaultParameter);

  Parameter detail_raw = ros::Parameter::Builder(node.get(), "test_param")
                             .description("a test")
                             .read_only(true)
                             .finish();
  EXPECT_THROW({ detail_raw.get(); }, InvalidDefaultParameter);
}

TEST(ParameterBuilder, testFinishWithoutDefaultButOverwrite) {
  rclcpp::NodeOptions no;
  no.parameter_overrides({
      {"test_param", 10},
  });

  auto node = std::make_shared<rclcpp::Node>("test_node", no);
  Parameter detail = ros::Parameter::Builder(node, "test_param")
                         .description("a test")
                         .read_only(true)
                         .finish();
  const rclcpp::ParameterValue &detail_as_value = detail;
  EXPECT_EQ(detail_as_value.get_type(), rclcpp::PARAMETER_INTEGER);
  EXPECT_EQ(detail_as_value.get<int>(), 10);
  EXPECT_EQ(detail.get<int>(), 10);
}

TEST(Parameter, testDeclareWithDefault) {
  auto node = std::make_shared<rclcpp::Node>("test_node");
  ros::Parameter::Builder pc(node, "test", "some_value");
  EXPECT_FALSE(node->has_parameter("test"));

  Parameter detail = pc.finish();

  EXPECT_TRUE(node->has_parameter("test"));
  EXPECT_EQ(rclcpp::PARAMETER_STRING,
            node->get_parameter("test").get_parameter_value().get_type());

  EXPECT_EQ(node->get_parameter("test").get_value<std::string>(), "some_value");

  std::string value = detail.get<std::string>();
  EXPECT_EQ(value, "some_value");

  // check we can use another detail to get the same value even if no default
  // has been set
  Parameter detail1 = pc.finish();
  value = detail1.get<std::string>(detail1);
  EXPECT_EQ(value, "some_value");

  // update param and check we can still get the correct one!!
  node->set_parameter(rclcpp::Parameter("test", "a_new_value"));
  value = detail1.get<std::string>();
  EXPECT_EQ(value, "a_new_value");
}

TEST(Parameter, testGetParamWithOverrides) {
  rclcpp::NodeOptions no;
  no.parameter_overrides({
      {"parameter_no_default", 42},
      {"parameter_wrong_type", "value"},
  });

  auto node = std::make_shared<rclcpp::Node>("test_node", no);

  // okay even with overrides, we MUST declare it for it to appear
  // this will not be the case if we set allow_uncleared_overrides on the
  // NodeOptions
  EXPECT_FALSE(node->has_parameter("parameter_no_default"));
  Parameter detail =
      ros::Parameter::Builder(node, "parameter_no_default").finish();
  EXPECT_TRUE(node->has_parameter("parameter_no_default"));
  EXPECT_EQ(node->get_parameter("parameter_no_default").get_value<int>(), 42);

  // this has a different type than is in the override
  EXPECT_THROW(
      { ros::Parameter::Builder(node, "parameter_wrong_type", 10).finish(); },
      rclcpp::exceptions::InvalidParameterTypeException);
}

TEST(Parameter, testGetParamWithMutableString) {
  rclcpp::NodeOptions no;
  no.parameter_overrides({
      {"param", "value"},
  });

  auto node = std::make_shared<rclcpp::Node>("test_node", no);

  struct Params {
    std::string value = "a";
  };

  Params params;
  // this tests the case where the ValueT type is not a const& and we can still
  // use it!!
  params.value = ros::Parameter::Builder(node, "param", params.value)
                     .finish()
                     .get<std::string>();
  // test the value is in the override not the default one
  EXPECT_EQ(params.value, "value");
}

TEST(Parameter, testGetWithNoOverridesWithDefault) {
  auto node = std::make_shared<rclcpp::Node>("test_node");
  Parameter detail =
      ros::Parameter::Builder(node, "parameter_no_default").finish();

  // attempt to get a param that has been declared but not set without providing
  // a default value
  int value = detail.get(5);
  EXPECT_EQ(value, 5);
  // the node should not be updated
  EXPECT_EQ(rclcpp::PARAMETER_NOT_SET,
            node->get_parameter("parameter_no_default")
                .get_parameter_value()
                .get_type());
}

TEST(Parameter, testGetWithOverridesWithDefault) {
  auto node = std::make_shared<rclcpp::Node>("test_node");

  Parameter detail =
      ros::Parameter::Builder(node, "parameter_default", 10).finish();
  // attempt to get a param that has been declared and set
  int value = detail.get(5);
  EXPECT_EQ(value, 10);
  // the node should not be updated
  EXPECT_EQ(
      node->get_parameter("parameter_default").get_parameter_value().get<int>(),
      10);
}

TEST(Parameter, testGetWithOverridesWithString) {
  auto node = std::make_shared<rclcpp::Node>("test_node");

  Parameter detail =
      ros::Parameter::Builder(node, "string_param", "a string").finish();
  // attempt to get a param that has been declared but not set without providing
  // a default value
  std::string value = detail.get<std::string>();
  EXPECT_EQ(value, "a string");
  // the node should not be updated
  EXPECT_EQ(
      rclcpp::PARAMETER_STRING,
      node->get_parameter("string_param").get_parameter_value().get_type());
}

TEST(Parameter, testGetWithDefaultString) {
  auto node = std::make_shared<rclcpp::Node>("test_node");
  Parameter detail = ros::Parameter::Builder(node, "string_param").finish();

  // attempt to get a param that has been declared but not set without providing
  // a default value
  std::string value = detail.get<std::string>("default string");
  EXPECT_EQ(value, "default string");
  // the node should not be updated
  EXPECT_EQ(
      rclcpp::PARAMETER_NOT_SET,
      node->get_parameter("string_param").get_parameter_value().get_type());
}

// TEST(Parameter, testUpdateOnParamChange) {

//   auto node = std::make_shared<rclcpp::Node>("test_node");

//   rclcpp::executors::SingleThreadedExecutor executor;
//   executor.add_node(node);

//   rclcpp::Publisher<rcl_interfaces::msg::ParameterEvent>::SharedPtr
//     event_publisher =
//     node->create_publisher<rcl_interfaces::msg::ParameterEvent>("/parameter_events",
//     1);

//   auto param_event = std::make_shared<rcl_interfaces::msg::ParameterEvent>();

//   rcl_interfaces::msg::Parameter p;
//   p.name = "parameter_default";
//   p.value.type = rcl_interfaces::msg::ParameterType::PARAMETER_INTEGER;
//   p.value.integer_value = 42;

//   param_event->node = node->get_fully_qualified_name();
//   param_event->changed_parameters.push_back(p);

//   Parameter detail = ros::Parameter::Builder(node, "parameter_default",
//   10).finish(); const int original_value = detail.get<int>();
//   EXPECT_EQ(original_value, 10);

//   event_publisher->publish(*param_event);

//   executor.spin_some();

//   const int updated_value = detail.get<int>();
//   EXPECT_EQ(updated_value, 42);

// }

// TEST(Parameter, testUpdateOnParamChangeCallback) {

//   auto node = std::make_shared<rclcpp::Node>("test_node");

//   rclcpp::executors::SingleThreadedExecutor executor;
//   executor.add_node(node);

//   rclcpp::Publisher<rcl_interfaces::msg::ParameterEvent>::SharedPtr
//     event_publisher =
//     node->create_publisher<rcl_interfaces::msg::ParameterEvent>("/parameter_events",
//     1);

//   auto param_event = std::make_shared<rcl_interfaces::msg::ParameterEvent>();

//   rcl_interfaces::msg::Parameter p;
//   p.name = "parameter_default";
//   p.value.type = rcl_interfaces::msg::ParameterType::PARAMETER_INTEGER;
//   p.value.integer_value = 42;

//   param_event->node = node->get_fully_qualified_name();
//   param_event->changed_parameters.push_back(p);

//   Parameter detail = ros::Parameter::Builder(node, "parameter_default",
//   10).finish();

//   int updated_value = -1;
//   detail.registerParamCallback<int>([&updated_value](int value) -> void {
//     updated_value = value;
//   });

//   const int original_value = detail.get<int>();
//   EXPECT_EQ(original_value, 10);

//   // param_handler->test_event(param_event);
//   event_publisher->publish(*param_event);

//   executor.spin_some();
//   EXPECT_EQ(updated_value, 42);

// }
