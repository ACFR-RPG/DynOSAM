#include "dynosam_ros/displays/dynamic_slam_displays/DSDCommonRos.hpp"

#include <glog/logging.h>

#include "dynosam_common/DynamicObjects.hpp"
#include "dynosam_common/viz/Colour.hpp"
#include "dynosam_ros/RosUtils.hpp"
#include "dynosam_ros/displays/DisplaysCommon.hpp"

namespace dyno {

DynoStatePublisher::DynoStatePublisher(
    const CanonicalSensorRig::ConstPtr& sensor_rig,
    rclcpp::Node::SharedPtr node)
    : sensor_rig_(sensor_rig), node_(node) {
  vo_publisher_ =
      node_->create_publisher<nav_msgs::msg::Odometry>("odometry", 1);
  vo_path_publisher_ =
      node_->create_publisher<nav_msgs::msg::Path>("odometry_path", 1);
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*node_);

  object_odom_publisher_ =
      node->create_publisher<ObjectOdometry>("object_odometry", 1);
  multi_object_odom_path_publisher_ =
      node->create_publisher<MultiObjectOdometryPath>("object_odometry_path",
                                                      1);

  static_points_pub_ =
      node->create_publisher<sensor_msgs::msg::PointCloud2>("static_cloud", 1);
  dynamic_points_pub_ =
      node->create_publisher<sensor_msgs::msg::PointCloud2>("dynamic_cloud", 1);
}

DynoStatePublisher& DynoStatePublisher::publishVisualOdomTF(bool flag) {
  publish_vo_tf_ = flag;
  return *this;
}

DynoStatePublisher& DynoStatePublisher::publishObjectOdomTF(bool flag) {
  publish_oo_tf_ = flag;
  return *this;
}

void DynoStatePublisher::publish(const DynoState& state) {
  const FrameId frame_id = state.frame_id;
  const Timestamp timestamp = state.timestamp;
  const auto reference_frames = sensor_rig_->getReferenceFrames();
  const gtsam::Pose3 T_RC = sensor_rig_->getCanonicalExtrinsics();

  auto camera_to_base_frame = [&T_RC](gtsam::Pose3& X_WC) {
    X_WC = T_RC.compose(X_WC);
  };

  std::vector<gtsam::Pose3> camera_trajectory =
      state.camera_trajectory.toDataVector();
  // transform camera from estimated (usually optical frame) to base frame
  std::for_each(camera_trajectory.begin(), camera_trajectory.end(),
                camera_to_base_frame);

  const gtsam::Pose3 X_W_k = camera_trajectory.back();
  DisplayCommon::publishOdometry(vo_publisher_, X_W_k, timestamp,
                                 reference_frames.odom_frame,
                                 reference_frames.base_frame);
  if (publish_vo_tf_) {
    std_msgs::msg::Header header;
    header.stamp = utils::toRosTime(timestamp);
    header.frame_id = reference_frames.odom_frame;
    sendTransform(X_W_k, header, reference_frames.base_frame);
  }

  // publish trajectory
  DisplayCommon::publishOdometryPath(vo_path_publisher_, camera_trajectory,
                                     timestamp, reference_frames.odom_frame);

  // publish local(?) static points
  // TODO: now if in the original X_W (which at least some of them are!)
  DisplayCommon::publishPointCloud(static_points_pub_, state.local_static_map,
                                   X_W_k, reference_frames.odom_frame);

  DisplayCommon::publishPointCloud(dynamic_points_pub_, state.dynamic_map,
                                   X_W_k, reference_frames.odom_frame);

  publishObjects(frame_id, state.object_trajectories);
}

void DynoStatePublisher::publishObjects(
    FrameId frame_id, const MultiObjectTrajectories& object_trajectories) {
  // get subset of trajectories that has an object observed at k
  // TODO: for now!
  // auto object_trajectories_k =
  //     object_trajectories.trajectoriesAtFrame(frame_id);

  // if (object_trajectories_k.empty()) {
  //   return;
  // }
  MultiObjectTrajectories object_trajectories_k = object_trajectories;

  if (object_trajectories_k.empty()) {
    return;
  }

  ObjectOdometryMap object_odometries;

  MultiObjectOdometryPath multi_object_odom_paths;
  multi_object_odom_paths.header.stamp =
      utils::toRosTime(object_trajectories_k.lastTimestamp());

  const auto reference_frames = sensor_rig_->getReferenceFrames();
  multi_object_odom_paths.header.frame_id = reference_frames.odom_frame;

  for (const auto& [object_id, object_trajectory] : object_trajectories_k) {
    // latest object odometry
    ObjectOdometry object_odometry = constructObjectOdometry(
        object_id, object_trajectory.maxFrame(), object_trajectory);
    object_odom_publisher_->publish(object_odometry);

    if (publish_oo_tf_) {
      sendObjectOdometryTransform(object_odometry);
    }

    // full path for object j
    ObjectOdometryPath object_path;

    std_msgs::msg::ColorRGBA colour_msg;
    convert(Color::uniqueId(object_id), colour_msg);

    // TODO: for now ignore segmenebts becuase the backend will always be
    // segmented due to non-consequative frames
    ObjectOdometryPath path_per_segment;
    path_per_segment.colour = colour_msg;
    path_per_segment.object_id = object_id;
    path_per_segment.path_segment = 0;
    path_per_segment.header = multi_object_odom_paths.header;
    for (FrameId frame_i : object_trajectory.toFrameIds()) {
      path_per_segment.object_odometries.push_back(
          constructObjectOdometry(object_id, frame_i, object_trajectory));
    }

    multi_object_odom_paths.paths.push_back(path_per_segment);
    // // construct full paths
    // const auto trajectory_segments = object_trajectory.segments();
    // for (size_t i = 0; i < trajectory_segments.size(); i++) {
    //   size_t segment_id = i + 1;

    //   const auto& segment = trajectory_segments.at(i);

    //   ObjectOdometryPath path_per_segment;
    //   path_per_segment.colour = colour_msg;
    //   path_per_segment.object_id = object_id;
    //   path_per_segment.path_segment = segment_id;
    //   path_per_segment.header = multi_object_odom_paths.header;

    //   for (const auto& entry : segment.trajectory) {
    //     path_per_segment.object_odometries.push_back(
    //         constructObjectOdometry(object_id, entry));
    //   }

    //   multi_object_odom_paths.paths.push_back(path_per_segment);
    // }
  }

  multi_object_odom_path_publisher_->publish(multi_object_odom_paths);
}

ObjectOdometry DynoStatePublisher::constructObjectOdometry(
    ObjectId object_id, FrameId frame_id,
    const PoseWithMotionTrajectory& trajectory) const {
  CHECK(trajectory.exists(frame_id));

  const PoseWithMotionEntry& pose_with_motion = trajectory.get(frame_id);
  const auto& entry = pose_with_motion.data;
  const auto L_W_k = entry.pose;
  const auto H_W_km1_k = entry.motion;
  const auto timestamp_k = pose_with_motion.timestamp;
  const auto frame_id_k = pose_with_motion.frame_id;

  CHECK_EQ(H_W_km1_k.to(), frame_id);
  CHECK_EQ(frame_id_k, frame_id);
  // backend may not send F2F
  // CHECK_EQ(H_W_km1_k.style(), MotionRepresentationStyle::F2F);

  const auto reference_frames = sensor_rig_->getReferenceFrames();
  const auto frame_link = reference_frames.odom_frame;
  const auto child_link = "object_" + std::to_string(object_id) + "_link";

  ObjectOdometry object_odom;
  utils::convertWithHeader(L_W_k, object_odom.odom, timestamp_k, frame_link,
                           child_link);

  dyno::convert(H_W_km1_k.estimate(), object_odom.h_w_km1_k.pose);

  const std::optional<PoseWithMotionEntry> previous_entry =
      trajectory.getPrevious(pose_with_motion);
  // may not be velocity in the case of the backend currentlly due to keyframing
  if (previous_entry && H_W_km1_k.from() == previous_entry->frame_id) {
    // this also implicitly checks that the previous entry is the immediately
    // // previous entry
    // CHECK_EQ(H_W_km1_k.from(), previous_entry->frame_id);

    const gtsam::Pose3 L_W_km1 = previous_entry->data.pose;
    const Timestamp timestamp_km1 = previous_entry->timestamp;

    gtsam::Vector6 body_velocity =
        calculateBodyMotion(H_W_km1_k, L_W_km1, timestamp_k, timestamp_km1);

    dyno::convert(body_velocity, object_odom.odom.twist.twist);
  }

  object_odom.object_id = object_id;
  object_odom.sequence = frame_id_k;

  return object_odom;
}

void DynoStatePublisher::sendObjectOdometryTransform(
    const ObjectOdometry& object_odom) {
  sendTransform(object_odom.odom.pose.pose, object_odom.odom.header,
                object_odom.odom.child_frame_id);
}

}  // namespace dyno
