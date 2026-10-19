#pragma once

#include <pcl_conversions/pcl_conversions.h>

#include <sensor_msgs/msg/point_cloud2.hpp>

#include "dynamic_slam_interfaces/msg/multi_object_odometry_path.hpp"
#include "dynamic_slam_interfaces/msg/object_odometry.hpp"
#include "dynamic_slam_interfaces/msg/object_odometry_path.hpp"
#include "dynosam2/DynamicSlamVisualizer.hpp"
#include "dynosam2/common/DynamicSlamInterface.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_processing/PipelineBase.hpp"
#include "dynosam_ros_common/RosUtils.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/header.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "visualization_msgs/msg/marker_array.hpp"

namespace dyno2 {

typedef dyno::SIMOPipelineModule<PublicationData, dyno::EmptyPayload>
    PublicationDataPipeline;
typedef PublicationDataPipeline::InputQueue PublicationDataQueue;

/// @brief Both the visualzier itself AND the Pipeline to recieve the data
class DynamicSlamVisualizerRosBase : public DynamicSlamVisualizer,
                                     public PublicationDataPipeline {
 public:
  DYNO_POINTER_TYPEDEFS(DynamicSlamVisualizerRosBase)
  DynamicSlamVisualizerRosBase(dyno::CanonicalSensorRig::Ptr sensor_rig,
                               const SlamBackend* estimator,
                               InputQueue* input_queue)
      : DynamicSlamVisualizer(sensor_rig, estimator),
        PublicationDataPipeline("dymosam2-viz", input_queue) {}

 private:
  inline dyno::EmptyPayload::ConstPtr process(
      const PublicationData::ConstPtr& input) override {
    // perform publishing
    this->publishEstimatorUpdate(*input);

    static const auto empty_payload = std::make_shared<dyno::EmptyPayload>();
    return empty_payload;
  }
};

using MarkerArray = visualization_msgs::msg::MarkerArray;
using Marker = visualization_msgs::msg::Marker;

class DynamicSlamVisualizerRos : public DynamicSlamVisualizerRosBase {
 public:
  DYNO_POINTER_TYPEDEFS(DynamicSlamVisualizerRos)
  DynamicSlamVisualizerRos(rclcpp::Node::SharedPtr node,
                           dyno::CanonicalSensorRig::Ptr sensor_rig,
                           const SlamBackend* estimator,
                           InputQueue* input_queue);

 private:
  void publishEstimatorUpdate(const PublicationData& publication_data) override;

  /**
   * @brief Publishes a pose like object as a transform.
   * T must be convertable to a geometry_msgs::msg::Transform via the
   * dyno::convert function.
   * @tparam T
   * @param pose const T&
   * @param header const std_msgs::msg::Header&
   * @param child_frame_link const std::string&
   */
  template <typename T>
  void sendTransform(const T& pose, const std_msgs::msg::Header& header,
                     const std::string& child_frame_link) {
    geometry_msgs::msg::TransformStamped t;

    dyno::convert(pose, t.transform);
    t.header = header;
    t.child_frame_id = child_frame_link;

    tf_broadcaster_->sendTransform(t);
  }

  /**
   * @brief Publishes a pose like object as a transform.
   * See sendTransform(const T&, const std_msgs::msg::Header&, const
   * std::string& ).
   * @tparam T
   * @param pose const T&
   * @param timestamp const Timestamp&
   * @param frame_link const std::string&
   * @param child_frame_link const std::string&
   */
  template <typename T>
  void sendTransform(const T& pose, const dyno::Timestamp& timestamp,
                     const std::string& frame_link,
                     const std::string& child_frame_link) {
    std_msgs::msg::Header header;
    header.stamp = dyno::ros::toRosTime(timestamp);
    header.frame_id = frame_link;

    sendTransform(pose, header, child_frame_link);
  }

 private:
  rclcpp::Node::SharedPtr node_;

  using OdometryPub = rclcpp::Publisher<nav_msgs::msg::Odometry>;
  using PathPub = rclcpp::Publisher<nav_msgs::msg::Path>;
  using MarkerArrayPub = rclcpp::Publisher<MarkerArray>;

  OdometryPub::SharedPtr vo_publisher_;
  PathPub::SharedPtr vo_path_publisher_;
  MarkerArrayPub::SharedPtr vo_keyframes_publisher_;

  using ObjectOdometry = dynamic_slam_interfaces::msg::ObjectOdometry;
  using ObjectOdometryPath = dynamic_slam_interfaces::msg::ObjectOdometryPath;
  using MultiObjectOdometryPath =
      dynamic_slam_interfaces::msg::MultiObjectOdometryPath;

  using ObjectOdometryPub = rclcpp::Publisher<ObjectOdometry>;
  using MultiObjectOdometryPathPub = rclcpp::Publisher<MultiObjectOdometryPath>;

  ObjectOdometryPub::SharedPtr object_odom_publisher_;
  MultiObjectOdometryPathPub::SharedPtr multi_object_odom_path_publisher_;

  //! TF broadcaster for the odometry.
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

}  // namespace dyno2
