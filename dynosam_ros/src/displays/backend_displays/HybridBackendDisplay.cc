#include "dynosam_ros/displays/backend_displays/HybridBackendDisplay.hpp"

#include "dynosam_common/PointCloudProcess.hpp"
#include "dynosam_ros/BackendDisplayPolicyRos.hpp"
#include "dynosam_ros/displays/BackendDisplayRos.hpp"

namespace dyno {

HybridModuleDisplayCommon::HybridModuleDisplayCommon(
    const DisplayParams& params, rclcpp::Node* node)
    : BackendModuleDisplayRos(params, node) {
  object_bounding_box_pub_ =
      node_->create_publisher<MarkerArray>("object_bounding_boxes", 1);
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

void ParalleHybridModuleDisplay::spin(
    const BackendOutputPacket::ConstPtr& output) {
  this->publishObjectBoundingBoxes(output);
}

void RegularHybridFormulationDisplay::spin(
    const BackendOutputPacket::ConstPtr& output) {
  this->publishObjectBoundingBoxes(output);
}

}  // namespace dyno
