#include "dynosam_ros/displays/DSDCommonRos.hpp"

#include <glog/logging.h>

#include "dynosam_common/DynamicObjects.hpp"
#include "dynosam_common/Transforms.hpp"
#include "dynosam_common/viz/Colour.hpp"
#include "dynosam_ros/RosUtils.hpp"
#include "dynosam_ros/displays/DisplaysCommon.hpp"

namespace dyno {

DynoStatePublisher::DynoStatePublisher(
    const CanonicalSensorRig::ConstPtr& sensor_rig,
    rclcpp::Node::SharedPtr node, const DynoStatePublisherOptions& options)
    : sensor_rig_(sensor_rig), node_(node), options_(options) {
  auto odom_qos = ros::addQosParameter(*node, "SYSTEM_DEFAULT", "odometry_qos");
  vo_publisher_ =
      node_->create_publisher<nav_msgs::msg::Odometry>("odometry", odom_qos);
  vo_path_publisher_ =
      node_->create_publisher<nav_msgs::msg::Path>("odometry_path", odom_qos);

  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*node_);

  object_odom_publisher_ =
      node->create_publisher<ObjectOdometry>("object_odometry", odom_qos);
  multi_object_odom_path_publisher_ =
      node->create_publisher<MultiObjectOdometryPath>("object_odometry_path",
                                                      odom_qos);

  auto pc_qos =
      ros::addQosParameter(*node, "SYSTEM_DEFAULT", "point_cloud_qos");
  static_points_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "static_cloud", pc_qos);
  dynamic_points_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "dynamic_cloud", pc_qos);

  auto markers_qos =
      ros::addQosParameter(*node, "SYSTEM_DEFAULT", "markers_qos");

  if (options_.publish_wireframe_cameras) {
    camera_wireframe_pub_ =
        node->create_publisher<MarkerArray>("camera_frustrum", markers_qos);
  }
}

void DynoStatePublisher::publish(const DynoState& state) {
  const FrameId frame_id = state.frame_id;
  const Timestamp timestamp = state.timestamp;
  const auto reference_frames = sensor_rig_->getReferenceFrames();
  const gtsam::Pose3 T_RC = sensor_rig_->getCanonicalExtrinsics();

  // camera (optical) frame to robot (base) frame
  auto camera_to_base_frame = [&T_RC](PoseTrajectoryEntry& X_WC_entry) {
    X_WC_entry.data = changeBasis(T_RC, X_WC_entry.data);
  };

  PoseTrajectory camera_trajectory = state.camera_trajectory;
  // transform camera from estimated (usually optical frame) to base frame
  if (options_.convert_to_optical) {
    std::for_each(camera_trajectory.begin(), camera_trajectory.end(),
                  camera_to_base_frame);
  }

  // pose of the robot (base link)
  const gtsam::Pose3 X_WR = camera_trajectory.last().data;
  DisplayCommon::publishOdometry(vo_publisher_, X_WR, timestamp,
                                 reference_frames.odom_frame,
                                 reference_frames.base_frame);
  if (options_.publish_vo_tf) {
    std_msgs::msg::Header header;
    header.stamp = ros::toRosTime(timestamp);
    header.frame_id = reference_frames.odom_frame;
    sendTransform(X_WR, header, reference_frames.base_frame);
  }

  if (options_.publish_wireframe_cameras) {
    MarkerArray marker_array;

    static constexpr double kLineWidth = 0.02;
    static constexpr double kFrustrumScale = 0.2;
    static constexpr int kShowLastNKeyframes = 10;

    visualization_msgs::msg::Marker delete_marker;
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;

    marker_array.markers.push_back(delete_marker);

    marker_array.markers.push_back(DisplayCommon::poseToCameraFrustrum(
        X_WR, timestamp, reference_frames.odom_frame, NiceColors::bluishgreen(),
        static_cast<int>(frame_id), 2.0 * kFrustrumScale, 2.0 * kLineWidth));

    if (camera_trajectory.size() >= 2) {
      // find last N keyframes
      std::set<FrameId> ckf_ids;
      for (const auto& [frame_id, kf_info] : state.keyframe_infos) {
        if (kf_info.camera_keyframe) {
          ckf_ids.insert(frame_id);
        }
      }
      // erase current frame (if exists) as we have already displayed
      ckf_ids.erase(frame_id);

      const auto start_it = ckf_ids.size() > kShowLastNKeyframes
                                ? std::prev(ckf_ids.end(), kShowLastNKeyframes)
                                : ckf_ids.begin();

      for (auto i = start_it; i != ckf_ids.end(); ++i) {
        const auto frame_id_i = *i;

        CHECK(camera_trajectory.exists(frame_id_i));
        const auto entry_i = camera_trajectory.get(frame_id_i);
        const auto T_WR_i = entry_i.data;
        const auto timestamp_i = entry_i.timestamp;
        marker_array.markers.push_back(DisplayCommon::poseToCameraFrustrum(
            T_WR_i, timestamp_i, reference_frames.odom_frame,
            NiceColors::vermillion(), static_cast<int>(frame_id_i),
            kFrustrumScale, kLineWidth));
      }
    }

    camera_wireframe_pub_->publish(marker_array);
  }

  // publish trajectory
  DisplayCommon::publishOdometryPath(vo_path_publisher_,
                                     camera_trajectory.toDataVector(),
                                     timestamp, reference_frames.odom_frame);

  // publish local(?) static points
  DisplayCommon::publishPointCloud(static_points_pub_, state.static_map,
                                   reference_frames.odom_frame, T_RC);

  DisplayCommon::publishPointCloud(dynamic_points_pub_, state.dynamic_map,
                                   reference_frames.odom_frame, T_RC);

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

  const auto reference_frames = sensor_rig_->getReferenceFrames();
  const gtsam::Pose3 T_RC = sensor_rig_->getCanonicalExtrinsics();

  // camera (optical) frame to robot (base) frame
  auto camera_to_base_frame_object = [&T_RC](PoseWithMotionEntry& entry_WS) {
    // change basis for both pose and motion to go from sensor-world to
    // robot-world
    entry_WS.data.pose = changeBasis(T_RC, entry_WS.data.pose);

    // loose notation here to indicate that motion is in W as defined by sensor
    // (S)
    gtsam::Pose3 H_WS = entry_WS.data.motion;
    gtsam::Pose3 H_WR = changeBasis(T_RC, H_WS);
    entry_WS.data.motion.estimate() = H_WR;
  };

  ObjectOdometryMap object_odometries;

  MultiObjectOdometryPath multi_object_odom_paths;
  multi_object_odom_paths.header.stamp =
      ros::toRosTime(object_trajectories_k.lastTimestamp());

  multi_object_odom_paths.header.frame_id = reference_frames.odom_frame;

  for (const auto& [object_id, object_trajectory_S] : object_trajectories_k) {
    // object trajectory in the world frame as defined by the robot frame
    PoseWithMotionTrajectory object_trajectory_R = object_trajectory_S;

    if (options_.convert_to_optical) {
      std::for_each(object_trajectory_R.begin(), object_trajectory_R.end(),
                    camera_to_base_frame_object);
    }

    // latest object odometry
    ObjectOdometry object_odometry = constructObjectOdometry(
        object_id, object_trajectory_R.maxFrame(), object_trajectory_R);
    object_odom_publisher_->publish(object_odometry);

    if (options_.publish_oo_tf) {
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
    for (FrameId frame_i : object_trajectory_R.toFrameIds()) {
      path_per_segment.object_odometries.push_back(
          constructObjectOdometry(object_id, frame_i, object_trajectory_R));
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
  ros::convertWithHeader(L_W_k, object_odom.odom, timestamp_k, frame_link,
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
