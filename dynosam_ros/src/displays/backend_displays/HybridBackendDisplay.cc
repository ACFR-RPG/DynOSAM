#include "dynosam_ros/displays/backend_displays/HybridBackendDisplay.hpp"

#include "dynosam_common/PointCloudProcess.hpp"
#include "dynosam_ros/BackendDisplayPolicyRos.hpp"
#include "dynosam_ros/RosUtils.hpp"
#include "dynosam_ros/displays/BackendDisplayRos.hpp"

namespace dyno {

HybridModuleDisplayCommon::HybridModuleDisplayCommon(
    const DisplayParams& params, rclcpp::Node::SharedPtr node,
    HybridAccessorCommon::Ptr hybrid_accessor)
    : BackendModuleDisplayRos(params, node),
      hybrid_accessor_(CHECK_NOTNULL(hybrid_accessor)) {
  object_bounding_box_pub_ =
      node_->create_publisher<MarkerArray>("object_bounding_boxes", 1);
  object_key_frame_pub_ =
      node_->create_publisher<MarkerArray>("object_keyframes", 1);
}

void HybridModuleDisplayCommon::publishObjectBoundingBoxes(
    const BackendOutputPacket::ConstPtr& output) {
  CloudPerObject clouds_per_obj =
      groupObjectCloud(output->dynamic_landmarks, output->pose());

  visualization_msgs::msg::MarkerArray array;

  for (const auto& [object_id, object_cloud] : clouds_per_obj) {
    const ObjectBBX bbx = findOBBFromCloud<pcl::PointXYZRGB>(
        pcl::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>(object_cloud));

    std::vector<Marker> markers = DisplayCommon::objectBBXToRvizMarker(
        bbx, object_id, output->getTimestamp(), params_.world_frame_id);
    array.markers.insert(array.markers.end(), markers.begin(), markers.end());
  }

  object_bounding_box_pub_->publish(array);
}

void HybridModuleDisplayCommon::publishObjectKeyFrames(
    const FrameId frame_id, const Timestamp timestamp) {
  // bit weird, use the object motion map to find which objects exist at the
  // current frame
  const MotionEstimateMap motions =
      hybrid_accessor_->getObjectMotions(frame_id);
  auto ros_time = utils::toRosTime(timestamp);

  visualization_msgs::msg::MarkerArray array;
  for (const auto& [object_id, _] : motions) {
    const KeyFrameRanges* object_range = nullptr;
    if (hybrid_accessor_->getObjectKeyFrameHistory(object_id, object_range)) {
      CHECK_NOTNULL(object_range);
      std_msgs::msg::ColorRGBA colour_msg;
      convert(Color::uniqueId(object_id), colour_msg);

      // iterate over all keyframes and draw
      size_t count = 0;
      for (const FrameRange<gtsam::Pose3>::Ptr& frame_range : *object_range) {
        const auto [keyframe_id, L_e] = frame_range->dataPair();

        visualization_msgs::msg::Marker marker;
        // Header and Metadata
        marker.header.frame_id = params_.world_frame_id;
        marker.header.stamp = ros_time;
        marker.ns = "obj_" + std::to_string(object_id) + "_keyframe";
        marker.id = count;
        marker.action = visualization_msgs::msg::Marker::ADD;

        // Marker Type: LINE_LIST allows us to draw multiple lines (the three
        // axes)
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        // marker.lifetime =
        // Translation
        marker.pose.position.x = L_e.x();
        marker.pose.position.y = L_e.y();
        marker.pose.position.z = L_e.z();

        // --- Line Properties ---
        marker.scale.x = 0.04;                      // Line width in meters
        constexpr static double axis_length = 0.4;  // Length of the axes

        // Orientation (Convert GTSAM Rotation to ROS Quaternion)
        const gtsam::Quaternion gtsam_q = L_e.rotation().toQuaternion();
        marker.pose.orientation.w = gtsam_q.w();
        marker.pose.orientation.x = gtsam_q.x();
        marker.pose.orientation.y = gtsam_q.y();
        marker.pose.orientation.z = gtsam_q.z();

        geometry_msgs::msg::Point origin;  // (0, 0, 0)
        origin.x = 0.0;
        origin.y = 0.0;
        origin.z = 0.0;

        // X-Axis (start at origin, end at +X)
        geometry_msgs::msg::Point x_end = origin;
        x_end.x = axis_length;
        marker.points.push_back(origin);
        marker.points.push_back(x_end);
        marker.colors.push_back(colour_msg);
        marker.colors.push_back(colour_msg);

        // Y-Axis (start at origin, end at +Y)
        geometry_msgs::msg::Point y_end = origin;
        y_end.y = axis_length;
        marker.points.push_back(origin);
        marker.points.push_back(y_end);
        marker.colors.push_back(colour_msg);
        marker.colors.push_back(colour_msg);

        // Z-Axis (start at origin, end at +Z)
        geometry_msgs::msg::Point z_end = origin;
        z_end.z = axis_length;
        marker.points.push_back(origin);
        marker.points.push_back(z_end);
        marker.colors.push_back(colour_msg);
        marker.colors.push_back(colour_msg);

        array.markers.push_back(marker);

        count++;
      }
    }
  }
  object_key_frame_pub_->publish(array);
}

void ParalleHybridModuleDisplay::spin(
    const BackendOutputPacket::ConstPtr& output) {
  this->publishObjectBoundingBoxes(output);
  this->publishObjectKeyFrames(output->getFrameId(), output->getTimestamp());
}

void RegularHybridFormulationDisplay::spin(
    const BackendOutputPacket::ConstPtr& output) {
  this->publishObjectBoundingBoxes(output);
  this->publishObjectKeyFrames(output->getFrameId(), output->getTimestamp());
}

}  // namespace dyno
