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
#include "dynosam_ros/RosUtils.hpp"

#include <gtsam/geometry/Pose3.h>

#include "dynosam_common/Types.hpp"
#include "dynosam_common/viz/Colour.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/time.hpp"
#include "std_msgs/msg/color_rgba.hpp"

template <>
bool dyno::convert(const dyno::Timestamp& time_seconds, rclcpp::Time& time) {
  uint64_t nanoseconds = static_cast<uint64_t>(time_seconds * 1e9);
  time = rclcpp::Time(nanoseconds);
  return true;
}

template <>
bool dyno::convert(const rclcpp::Time& time, dyno::Timestamp& time_seconds) {
  uint64_t nanoseconds = time.nanoseconds();
  time_seconds = static_cast<double>(nanoseconds) / 1e9;
  return true;
}

template <>
bool dyno::convert(const dyno::Timestamp& time_seconds,
                   builtin_interfaces::msg::Time& time) {
  rclcpp::Time ros_time;
  convert(time_seconds, ros_time);
  time = ros_time;
  return true;
}

template <>
bool dyno::convert(const RGBA<float>& colour, std_msgs::msg::ColorRGBA& msg) {
  msg.r = colour.r;
  msg.g = colour.g;
  msg.b = colour.b;
  msg.a = colour.a;
  return true;
}

template <>
bool dyno::convert(const Color& colour, std_msgs::msg::ColorRGBA& msg) {
  return convert(RGBA<float>(colour), msg);
}

template <>
bool dyno::convert(const gtsam::Pose3& pose, geometry_msgs::msg::Pose& msg) {
  const gtsam::Rot3& rotation = pose.rotation();
  const gtsam::Quaternion& quaternion = rotation.toQuaternion();

  // Position
  msg.position.x = pose.x();
  msg.position.y = pose.y();
  msg.position.z = pose.z();

  // Orientation
  msg.orientation.w = quaternion.w();
  msg.orientation.x = quaternion.x();
  msg.orientation.y = quaternion.y();
  msg.orientation.z = quaternion.z();
  return true;
}

template <>
bool dyno::convert(const geometry_msgs::msg::Pose& msg, gtsam::Pose3& pose) {
  gtsam::Point3 translation(msg.position.x, msg.position.y, msg.position.z);

  gtsam::Rot3 rotation(msg.orientation.w, msg.orientation.x, msg.orientation.y,
                       msg.orientation.z);

  pose = gtsam::Pose3(rotation, translation);
  return true;
}

template <>
bool dyno::convert(const gtsam::Pose3& pose,
                   geometry_msgs::msg::PoseStamped& msg) {
  return convert<gtsam::Pose3, geometry_msgs::msg::Pose>(pose, msg.pose);
}

// will not do time or tf links or covariance....
template <>
bool dyno::convert(const gtsam::Pose3& pose, nav_msgs::msg::Odometry& odom) {
  return convert<gtsam::Pose3, geometry_msgs::msg::Pose>(pose, odom.pose.pose);
}

template <>
bool dyno::convert(const geometry_msgs::msg::Pose& pose,
                   geometry_msgs::msg::Transform& transform) {
  transform.translation.x = pose.position.x;
  transform.translation.y = pose.position.y;
  transform.translation.z = pose.position.z;

  transform.rotation.x = pose.orientation.x;
  transform.rotation.y = pose.orientation.y;
  transform.rotation.z = pose.orientation.z;
  transform.rotation.w = pose.orientation.w;
  return true;
}

template <>
bool dyno::convert(const gtsam::Pose3& pose,
                   geometry_msgs::msg::Transform& transform) {
  transform.translation.x = pose.x();
  transform.translation.y = pose.y();
  transform.translation.z = pose.z();

  const gtsam::Rot3& rotation = pose.rotation();
  const gtsam::Quaternion& quaternion = rotation.toQuaternion();
  transform.rotation.x = quaternion.x();
  transform.rotation.y = quaternion.y();
  transform.rotation.z = quaternion.z();
  transform.rotation.w = quaternion.w();
  return true;
}

template <>
bool dyno::convert(const gtsam::Pose3& pose,
                   geometry_msgs::msg::TransformStamped& transform) {
  return convert<gtsam::Pose3, geometry_msgs::msg::Transform>(
      pose, transform.transform);
}

namespace dyno {

std::ostream& operator<<(std::ostream& stream, const ParameterDetails& param) {
  stream << (std::string)param;
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const rclcpp::Parameter& param) {
  stream << param.get_name() << ": " << param.value_to_string() << " ("
         << param.get_type_name() << ")";
  return stream;
}

const std::string& ParameterDetails::name() const {
  return default_parameter_.get_name();
}

std::string ParameterDetails::node_name() const { return node_->get_name(); }

rclcpp::Parameter ParameterDetails::get() const {
  return this->get_param(this->default_parameter_);
}

std::string ParameterDetails::get(const char* default_value) const {
  return this->get_param<std::string>(
      rclcpp::Parameter(this->name(), std::string(default_value)));
}

std::string ParameterDetails::get(const std::string& default_value) const {
  return this->get_param<std::string>(
      rclcpp::Parameter(this->name(), default_value));
}

ParameterDetails::operator std::string() const {
  std::stringstream ss;
  ss << "[ name: " << this->name();
  ss << " value: " << this->get().value_to_string();
  ss << " description: " << description_.description << "]";
  return ss.str();
}

ParameterDetails::ParameterDetails(
    rclcpp::Node* node, const rclcpp::Parameter& parameter,
    const rcl_interfaces::msg::ParameterDescriptor& description)
    : node_(node), default_parameter_(parameter), description_(description) {
  declare();
}

rclcpp::Parameter ParameterDetails::get_param(
    const rclcpp::Parameter& default_param) const {
  const bool is_set = isSet();
  bool has_default = default_param.get_type() != rclcpp::PARAMETER_NOT_SET;

  if (!is_set) {
    // if no default value we treat a not set parameter as a error as the use
    // MUST override it via runtime configuration
    if (!has_default) {
      throw InvalidDefaultParameter(this->name());
    } else {
      return default_param;
    }
  }
  return node_->get_parameter(this->name());
}

void ParameterDetails::declare() {
  // only declare if needed
  if (!node_->has_parameter(this->name())) {
    const rclcpp::ParameterValue default_value =
        default_parameter_.get_parameter_value();
    const rclcpp::ParameterValue effective_value =
        node_->declare_parameter(this->name(), default_value, description_);
    (void)effective_value;
  }

  // add the param subscriber if we dont have one!!
  // if(!param_subscriber_) {
  //   param_subscriber_ =
  //   std::make_shared<rclcpp::ParameterEventHandler>(node_);

  //   auto cb = [&](const rclcpp::Parameter& new_parameter) {
  //       node_->set_parameter(new_parameter);
  //       //should check same type?
  //       CHECK_EQ(new_parameter.get_name(), this->name());

  //       property_handler_.update(this->name(),new_parameter);
  //   };
  //   //callback handler must be set and remain in scope for the cb's to
  //   trigger cb_handle_ =
  //   param_subscriber_->add_parameter_callback(this->name(), cb);
  // }
}

ParameterConstructor::ParameterConstructor(rclcpp::Node::SharedPtr node,
                                           const std::string& name)
    : ParameterConstructor(node.get(), name) {}

ParameterConstructor::ParameterConstructor(rclcpp::Node* node,
                                           const std::string& name)
    : node_(node), parameter_(name) {
  CHECK_NOTNULL(node_);
  parameter_descriptor_.name = name;
  parameter_descriptor_.dynamic_typing = true;
}

ParameterDetails ParameterConstructor::finish() const {
  return ParameterDetails(node_, parameter_, parameter_descriptor_);
}

ParameterConstructor& ParameterConstructor::description(
    const std::string& description) {
  parameter_descriptor_.description = description;
  return *this;
}

ParameterConstructor& ParameterConstructor::read_only(bool read_only) {
  parameter_descriptor_.read_only = read_only;
  return *this;
}

ParameterConstructor& ParameterConstructor::parameter_description(
    const rcl_interfaces::msg::ParameterDescriptor& parameter_description) {
  parameter_descriptor_ = parameter_description;
  return *this;
}

namespace utils {

Timestamp fromRosTime(const rclcpp::Time& time) {
  Timestamp timestamp;
  convert(time, timestamp);
  return timestamp;
}

rclcpp::Time toRosTime(Timestamp timestamp) {
  rclcpp::Time time;
  convert(timestamp, time);
  return time;
}

}  // namespace utils
}  // namespace dyno
