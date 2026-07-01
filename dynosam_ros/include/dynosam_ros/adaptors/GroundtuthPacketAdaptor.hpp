#include <dynosam_common/GroundTruthPacket.hpp>

#include "dynamic_slam_interfaces/msg/groundtruth_packet.hpp"
#include "dynosam_ros/RosUtils.hpp"
#include "rclcpp/type_adapter.hpp"

template <>
struct rclcpp::TypeAdapter<dyno::GroundTruthInputPacket,
                           dynamic_slam_interfaces::msg::GroundtruthPacket> {
  using is_specialized = std::true_type;
  using custom_type = dyno::GroundTruthInputPacket;
  using ros_message_type = dynamic_slam_interfaces::msg::GroundtruthPacket;

  static void convert_to_ros_message(const custom_type& source,
                                     ros_message_type& destination) {
    // first timestamp
    dyno::convert(source.timestamp_, destination.header.stamp);
    destination.sequence = source.frame_id_;

    // no header conversion for odom
    dyno::convert(source.X_world_, destination.odom.pose);
  }

  static void convert_to_custom(const ros_message_type& source,
                                custom_type& destination) {
    // first timestamp
    dyno::convert(source.header.stamp, destination.timestamp_);
    destination.frame_id_ = source.sequence;

    // no header conversion for odom
    dyno::convert(source.odom.pose, destination.X_world_);
  }
};

RCLCPP_USING_CUSTOM_TYPE_AS_ROS_MESSAGE_TYPE(
    dyno::GroundTruthInputPacket,
    dynamic_slam_interfaces::msg::GroundtruthPacket);
