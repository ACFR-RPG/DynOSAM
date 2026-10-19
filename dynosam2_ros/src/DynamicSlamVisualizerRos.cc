#include "dynosam2_ros/DynamicSlamVisualizerRos.hpp"

#include "dynosam_common/viz/Colour.hpp"
#include "dynosam_ros_common/RosUtils.hpp"

namespace dyno2 {

DynamicSlamVisualizerRos::DynamicSlamVisualizerRos(
    rclcpp::Node::SharedPtr node, dyno::CanonicalSensorRig::Ptr sensor_rig,
    const SlamBackend* estimator, InputQueue* input_queue)
    : DynamicSlamVisualizerRosBase(sensor_rig, estimator, input_queue),
      node_(node) {
  auto odom_qos =
      dyno::ros::addQosParameter(*node, "SYSTEM_DEFAULT", "odometry_qos");
  vo_publisher_ =
      node_->create_publisher<nav_msgs::msg::Odometry>("odometry", odom_qos);
  vo_path_publisher_ =
      node_->create_publisher<nav_msgs::msg::Path>("odometry_path", odom_qos);

  vo_keyframes_publisher_ =
      node_->create_publisher<MarkerArray>("keyframes", odom_qos);

  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*node_);

  object_odom_publisher_ =
      node->create_publisher<ObjectOdometry>("object_odometry", odom_qos);
  multi_object_odom_path_publisher_ =
      node->create_publisher<MultiObjectOdometryPath>("object_odometry_path",
                                                      odom_qos);
}

void DynamicSlamVisualizerRos::publishEstimatorUpdate(
    const PublicationData& publication_data) {
  CHECK_NOTNULL(estimator_);

  // add new vo state estimate
  const dyno::FrameId frame_id = publication_data.frame_id;
  const dyno::Timestamp timestamp = publication_data.timestamp;
  const auto timestamp_ros = dyno::ros::toRosTime(timestamp);

  gtsam::Pose3 X_WR_k = cameraToRobotFrame(estimator_->getVioPose(frame_id));
  camera_trajectory_.insert(frame_id, timestamp, X_WR_k);

  // first upate internal trajectory
  for (dyno::FrameId updated_state : publication_data.updated_states.vo) {
    gtsam::Pose3 X_W_C = estimator_->getVioPose(updated_state);
    gtsam::Pose3 X_W_R = cameraToRobotFrame(X_W_C);
    camera_trajectory_.update(updated_state, X_W_R);
  }

  const auto reference_frames = sensor_rig_->getReferenceFrames();
  // publish VO transform
  std_msgs::msg::Header header;
  header.stamp = timestamp_ros;
  header.frame_id = reference_frames.odom_frame;
  sendTransform(X_WR_k, header, reference_frames.base_frame);

  // publish VO
  nav_msgs::msg::Odometry odom_msg;
  dyno::ros::convertWithHeader(X_WR_k, odom_msg, timestamp,
                               reference_frames.odom_frame,
                               reference_frames.base_frame);
  vo_publisher_->publish(odom_msg);

  // publish VO path
  nav_msgs::msg::Path path;
  for (const gtsam::Pose3& odom : camera_trajectory_.toDataVector()) {
    geometry_msgs::msg::PoseStamped pose_stamped;
    // Timestamp wrong?
    dyno::ros::convertWithHeader(odom, pose_stamped, timestamp,
                                 reference_frames.odom_frame);
    path.poses.push_back(pose_stamped);
  }

  path.header.stamp = timestamp_ros;
  path.header.frame_id = reference_frames.odom_frame;
  vo_path_publisher_->publish(path);

  // now for objects
  // TODO: estimator should hold the velocities
  // and we should make a set of object trajectories that holds them
  // instead of calculating the velocity every time like we do in the old
  // publishObjects function!
  for (dyno::TemporalObjectIdentifier updated_state :
       publication_data.updated_states.objects) {
    dyno::PoseWithMotion L_W_R =
        cameraToRobotFrame(estimator_->getDooEntry(updated_state));

    // // bit of a hack but we can get the timestamps from the camera
    // trajectory?
    CHECK(camera_trajectory_.exists(updated_state.frame_id))
        << updated_state.frame_id;
    const dyno::Timestamp timestamp =
        camera_trajectory_.timeAt(updated_state.frame_id);
    object_trajectories_[updated_state.object_id].insertOrUpdate(
        updated_state.frame_id, timestamp, L_W_R);
  }

  if (!publication_data.observed_objects.empty()) {
    MultiObjectOdometryPath multi_object_odom_paths;
    multi_object_odom_paths.header.stamp = timestamp_ros;
    multi_object_odom_paths.header.frame_id = reference_frames.odom_frame;

    // now get trajectories only for seen objects
    for (dyno::ObjectId object_id : publication_data.observed_objects) {
      // TODO: estimator should give us velocity!
      const auto& object_trajectory = object_trajectories_[object_id];

      // if traj.size() > 2

      // Full trajectory path.
      std_msgs::msg::ColorRGBA colour_msg;
      dyno::convert(dyno::Color::uniqueId(object_id), colour_msg);

      multi_object_odom_paths.paths.emplace_back();
      ObjectOdometryPath& path_per_segment =
          multi_object_odom_paths.paths.back();
      path_per_segment.colour = colour_msg;
      path_per_segment.object_id = object_id;
      path_per_segment.path_segment = 0;
      path_per_segment.header = multi_object_odom_paths.header;

      // dummy way of calculating object odom now -> just fill pose!
      const auto frame_link = reference_frames.odom_frame;
      const auto child_link = "object_" + std::to_string(object_id) + "_link";

      // pose with motion entry (for now?)
      for (const auto& entry : object_trajectory.toVector()) {
        path_per_segment.object_odometries.emplace_back();

        const gtsam::Pose3& L_W_k = entry.data.pose;
        const dyno::Timestamp timestamp = entry.timestamp;

        ObjectOdometry& object_odom = path_per_segment.object_odometries.back();
        dyno::ros::convertWithHeader(L_W_k, object_odom.odom, timestamp,
                                     frame_link, child_link);

        object_odom.object_id = object_id;
        object_odom.sequence = entry.frame_id;
      }
    }

    multi_object_odom_path_publisher_->publish(multi_object_odom_paths);
  }
}

}  // namespace dyno2
