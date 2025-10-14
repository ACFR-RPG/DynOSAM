#include "dynosam_ros/MultiImageSync.hpp"

namespace dyno {

MultiImageSync2::MultiImageSync2(rclcpp::Node& node,
                                 const std::array<std::string, 2>& topics,
                                 uint32_t queue_size,
                                 const ImageContainerCallback& cb)
    : SyncBase(node, topics, queue_size), ImageContainerCallbackWrapper(cb) {
  RCLCPP_INFO(node_.get_logger(), "Creating MultiImageSync2");
}

MultiImageSync3::MultiImageSync3(rclcpp::Node& node,
                                 const std::array<std::string, 3>& topics,
                                 uint32_t queue_size,
                                 const ImageContainerCallback& cb)
    : SyncBase(node, topics, queue_size), ImageContainerCallbackWrapper(cb) {
  RCLCPP_INFO(node_.get_logger(), "Creating MultiImageSync3");
}

MultiImageSync4::MultiImageSync4(rclcpp::Node& node,
                                 const std::array<std::string, 4>& topics,
                                 uint32_t queue_size,
                                 const ImageContainerCallback& cb)
    : SyncBase(node, topics, queue_size), ImageContainerCallbackWrapper(cb) {
  RCLCPP_INFO(node_.get_logger(), "Creating MultiImageSync4");
}

void MultiImageSync2::imageSyncCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg) {}

void MultiImageSync3::imageSyncCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr& flow_msg) {}

void MultiImageSync4::imageSyncCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr& flow_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr& mask_msg) {}

}  // namespace dyno
