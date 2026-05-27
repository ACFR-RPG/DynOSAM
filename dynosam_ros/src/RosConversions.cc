#include "dynosam_common/Types.hpp"
#include "dynosam_common/viz/Colour.hpp"
#include "dynosam_ros/RosUtils.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
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
bool dyno::convert(const builtin_interfaces::msg::Time& time,
                   dyno::Timestamp& time_seconds) {
  rclcpp::Time ros_time = time;
  convert(ros_time, time_seconds);
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
bool dyno::convert(const geometry_msgs::msg::Vector3& vec3,
                   gtsam::Point3& point) {
  point = gtsam::Point3(vec3.x, vec3.y, vec3.z);
  return true;
}

template <>
bool dyno::convert(const geometry_msgs::msg::Point& vec3,
                   gtsam::Point3& point) {
  point = gtsam::Point3(vec3.x, vec3.y, vec3.z);
  return true;
}

template <>
bool dyno::convert(const gtsam::Point3& point, geometry_msgs::msg::Point& msg) {
  msg.x = point(0);
  msg.y = point(1);
  msg.z = point(2);
  return true;
}

template <>
bool dyno::convert(const geometry_msgs::msg::Quaternion& orientation,
                   gtsam::Rot3& rot) {
  rot = gtsam::Rot3(orientation.w, orientation.x, orientation.y, orientation.z);
  return true;
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
  // gtsam::Point3 translation(msg.position.x, msg.position.y, msg.position.z);

  // gtsam::Rot3 rotation(msg.orientation.w, msg.orientation.x,
  // msg.orientation.y,
  //                      msg.orientation.z);

  gtsam::Point3 translation;
  convert(msg.position, translation);

  gtsam::Rot3 rotation;
  convert(msg.orientation, rotation);

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
bool dyno::convert(const gtsam::Vector6& vel,
                   geometry_msgs::msg::Twist& twist) {
  // linear velocity components
  twist.linear.x = vel(3);
  twist.linear.y = vel(4);
  twist.linear.z = vel(5);

  // angular velocity components
  twist.angular.x = vel(0);
  twist.angular.y = vel(1);
  twist.angular.z = vel(2);

  return true;
}

template <>
bool dyno::convert(const geometry_msgs::msg::Transform& transform,
                   gtsam::Pose3& pose) {
  gtsam::Point3 translation;
  convert(transform.translation, translation);

  gtsam::Rot3 rotation;
  convert(transform.rotation, rotation);

  pose = gtsam::Pose3(rotation, translation);
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

template <>
bool dyno::convert(const geometry_msgs::msg::TransformStamped& transform,
                   gtsam::Pose3& pose) {
  return convert<geometry_msgs::msg::Transform, gtsam::Pose3>(
      transform.transform, pose);
}
