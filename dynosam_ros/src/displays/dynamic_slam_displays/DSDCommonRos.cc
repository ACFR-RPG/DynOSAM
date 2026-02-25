#include "dynosam_ros/displays/dynamic_slam_displays/DSDCommonRos.hpp"

#include <glog/logging.h>

#include "dynosam_common/DynamicObjects.hpp"
#include "dynosam_common/viz/Colour.hpp"
#include "dynosam_ros/RosUtils.hpp"
#include "dynosam_ros/displays/DisplaysCommon.hpp"

namespace dyno {

DSDTransport::DSDTransport(rclcpp::Node::SharedPtr node) : node_(node) {
  object_odom_publisher_ =
      node->create_publisher<ObjectOdometry>("object_odometry", 1);
  multi_object_odom_path_publisher_ =
      node->create_publisher<MultiObjectOdometryPath>("object_odometry_path",
                                                      1);
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*node_);

  VLOG(20) << "Constructed DSDTransport with effective namespace "
           << node_->get_effective_namespace();
}

std::string DSDTransport::constructObjectFrameLink(ObjectId object_id) {
  return "object_" + std::to_string(object_id) + "_link";
}

// ================================================================================================================
// TODO: Fill in the velocity and check calculateBodyMotion is correct (only linear, add angular component too)
// ================================================================================================================
ObjectOdometry DSDTransport::constructObjectOdometry(
    const gtsam::Pose3& e_H_k_world, const gtsam::Pose3& pose_km1,
    ObjectId object_id, FrameId frame_id_k, Timestamp timestamp_k, Timestamp timestamp_km1,
    const std::string& frame_id_link, const std::string& child_frame_id_link,
    const ObjectIdClassMap& object_class_map) {
  ObjectOdometry object_odom;

  // technically this should be k-1
  gtsam::Vector6 body_velocity = calculateBodyMotion(e_H_k_world, pose_km1, timestamp_km1, timestamp_k);

  nav_msgs::msg::Odometry odom_msg;
  utils::convertWithHeader(pose_km1, odom_msg, timestamp_k, frame_id_link,
                           child_frame_id_link);

  object_odom.odom = odom_msg;
  // TODO: can check if correct representation?

  dyno::convert(e_H_k_world, object_odom.h_w_km1_k.pose);

  dyno::convert(body_velocity, object_odom.odom.twist.twist); // Add velocity to message
 
  object_odom.object_id = object_id;
  object_odom.sequence = frame_id_k;

  // Assigning the class label
  auto observation = object_class_map.find(object_id);
  if (observation != object_class_map.end()) {
    object_odom.class_label=observation->second;
  } else {
    object_odom.class_label="unknown";
  }


  return object_odom;
}

ObjectOdometryMap DSDTransport::constructObjectOdometries(
    const ObjectMotionMap& motions, const ObjectPoseMap& poses,
    FrameId frame_id_k, Timestamp timestamp_k, Timestamp timestamp_km1,
    const std::string& frame_id_link, 
    const ObjectIdClassMap& object_class_map) {
  // need to get poses for k-1
  // TODO: no way to ensure that the motions are for frame k
  // this is a weird data-structure to use and motions are per frame and
  // ObjectPoseMap is over all k to K
  //  const FrameId frame_id_k_1 = frame_id_k - 1u;
  ObjectOdometryMap object_odom_map;
  for (const auto& [object_id, per_frame_motions] : motions) {
    if (!per_frame_motions.exists(frame_id_k)) {
      // object does not exist at this frame.
      continue;
    }
    const gtsam::Pose3& e_H_k_world = per_frame_motions.at(frame_id_k);

    if (!poses.exists(object_id, frame_id_k)) {
      VLOG(30) << "Cannot construct ObjectOdometry for object " << object_id
               << ", at frame " << frame_id_k
               << " Missing entry in ObjectPoseMap (but object motion "
                  "entry found!!)";
      continue;
    }
    const gtsam::Pose3& pose_k = poses.at(object_id, frame_id_k);

    const std::string child_frame_id_link = constructObjectFrameLink(object_id);

    object_odom_map.insert2(
        child_frame_id_link,
        constructObjectOdometry(e_H_k_world, pose_k, object_id, frame_id_k,
                                timestamp_k, timestamp_km1, 
                                frame_id_link, child_frame_id_link, object_class_map));
  }

  return object_odom_map;
}

MultiObjectOdometryPath DSDTransport::constructMultiObjectOdometryPaths(
    const ObjectMotionMap& motions, const ObjectPoseMap& poses,
    Timestamp timestamp_k, const FrameIdTimestampMap& frame_timestamp_map,
    const std::string& frame_id_link, bool interpolate_missing_segments, const ObjectIdClassMap& object_class_map) {
  MultiObjectOdometryPath multi_path;
  multi_path.header.stamp = utils::toRosTime(timestamp_k);

  multi_path.header.frame_id = frame_id_link;

  // TODO: right now dont have the motion for every timestep so... just leave
  // blank?
  for (const auto& [object_id, frame_pose_map] : poses) {
    const std::string child_frame_id_link = constructObjectFrameLink(object_id);
    // NOTE: cannot use FrameId for type (size_t)
    int previous_frame_id = -1;
    bool first = true;
    int path_segment = 0;

    std_msgs::msg::ColorRGBA colour_msg;
    convert(Color::uniqueId(object_id), colour_msg);
    // paths for this object, broken into segments
    gtsam::FastMap<int, ObjectOdometryPath> segmented_paths;

    for (const auto& [frame_id, object_pose] : frame_pose_map) {
      if (!motions.exists(object_id, frame_id)) {
        // LOG(WARNING)
        //     << "Cannot construct ObjectOdometry for object " << object_id
        //     << ", at frame " << frame_id
        //     << " for MultiObjectOdometryPath. Missing entry in
        //     ObjectMotionMap";
        continue;
      }

      const gtsam::Pose3& object_motion = motions.at(object_id, frame_id);

      if (!frame_timestamp_map.exists(frame_id)) {
        LOG(WARNING) << "Cannot construct ObjectOdometry for object "
                     << object_id << ", at frame " << frame_id
                     << " for MultiObjectOdometryPath. Missing entry in "
                        "FrameIdTimestampMap";
        continue;
      }

      const Timestamp& timestamp = frame_timestamp_map.at(frame_id);

      // if (!first && static_cast<int>(frame_id) != previous_frame_id + 1) {
      //   path_segment++;
      // }
      first = false;
      previous_frame_id = static_cast<int>(frame_id);

      // RIGHT NOW MOTION IDENTITY
      // timestamp is wrong

      // Adding the (k-1)th timestamp
      Timestamp timestamp_km1; 
      if (previous_frame_id >= 0) {
        timestamp_km1 = frame_timestamp_map.at(previous_frame_id);
      }
      else {
        timestamp_km1 = timestamp;
      }

      gtsam::Pose3 motion;
      const ObjectOdometry object_odometry = constructObjectOdometry(
          object_motion, object_pose, object_id, frame_id, timestamp, timestamp_km1,
          frame_id_link, child_frame_id_link, object_class_map);

      if (!segmented_paths.exists(path_segment)) {
        ObjectOdometryPath path;
        path.colour = colour_msg;
        path.object_id = object_id;
        path.path_segment = path_segment;

        path.header = multi_path.header;

        segmented_paths.insert2(path_segment, path);
      }

      ObjectOdometryPath& path = segmented_paths.at(path_segment);
      path.object_odometries.push_back(object_odometry);
    }

    // TODO: make this a flag and the frame diff!
    if (interpolate_missing_segments) {
      // assume segmented paths are in order
      for (auto it = segmented_paths.begin();
           std::next(it) != segmented_paths.end(); ++it) {
        auto current_segment_id = it->first;
        auto next_segment_id = std::next(it)->first;
        CHECK_EQ(current_segment_id + 1, next_segment_id);

        const ObjectOdometryPath& current_path_segment = it->second;
        const ObjectOdometryPath& next_path_segment = std::next(it)->second;

        const auto begin_object_odom =
            current_path_segment.object_odometries.back();
        const auto end_object_odom =
            next_path_segment.object_odometries.front();
        // interpolate between end of current and start of next
        // less than 5 frames appart

        if (end_object_odom.sequence - begin_object_odom.sequence < 5) {
          ObjectOdometryPath interpolated_segment;
          interpolated_segment.colour = colour_msg;
          interpolated_segment.object_id = object_id;
          //-1 for path segment
          interpolated_segment.path_segment = -1;
          interpolated_segment.header = multi_path.header;
          // NOTE: this is not actually interpolating its just going to make
          // RVIZ draw a line between them!!
          interpolated_segment.object_odometries.push_back(begin_object_odom);
          interpolated_segment.object_odometries.push_back(end_object_odom);
          multi_path.paths.push_back(interpolated_segment);
        }
      }
    }

    for (const auto& [_, path] : segmented_paths) {
      multi_path.paths.push_back(path);
    }
  }

  return multi_path;
}

void DSDTransport::Publisher::publishObjectOdometry() {
  for (const auto& [_, object_odom] : object_odometries_)
    object_odom_publisher_->publish(object_odom);
}

void DSDTransport::Publisher::publishObjectTransforms() {
  for (const auto& [object_child_frame, object_odom] : object_odometries_) {
    geometry_msgs::msg::TransformStamped t;
    dyno::convert(object_odom.odom.pose.pose, t.transform);

    t.header.stamp = utils::toRosTime(timestamp_);
    t.header.frame_id = frame_id_link_;
    t.child_frame_id = object_child_frame;

    tf_broadcaster_->sendTransform(t);
  }
}

void DSDTransport::Publisher::publishObjectPaths() {
  multi_object_odom_path_publisher_->publish(object_paths_);
}

// ============================== ADDED HELPER FUNCTION TO GET PREVIOUS TIMESTAMP ==============================
Timestamp getPrevTimestamp(const FrameIdTimestampMap& map, FrameId id, Timestamp current_timestamp) {

  if (id < 0) return current_timestamp;

  FrameId prev_id = id - 1;

  auto it = map.find(prev_id);
  if (it != map.end()) {
    return it->second;
  } 
  else {
    return current_timestamp;
  }
}
// ============================================================================================================

DSDTransport::Publisher::Publisher(
    rclcpp::Node::SharedPtr node,
    ObjectOdometryPub::SharedPtr object_odom_publisher,
    MultiObjectOdometryPathPub::SharedPtr multi_object_odom_path_publisher,
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster,
    const ObjectMotionMap& motions, const ObjectPoseMap& poses,
    const std::string& frame_id_link,
    const FrameIdTimestampMap& frame_timestamp_map, FrameId frame_id,
    Timestamp timestamp,
    const ObjectIdClassMap& object_class_map)
    : node_(node),
      object_odom_publisher_(object_odom_publisher),
      multi_object_odom_path_publisher_(multi_object_odom_path_publisher),
      tf_broadcaster_(tf_broadcaster),
      frame_id_link_(frame_id_link),
      frame_id_(frame_id),
      timestamp_(timestamp),
      timestamp_km1_(getPrevTimestamp(frame_timestamp_map, frame_id, timestamp)), // added to get (k-1)th timestamp
      object_odometries_(DSDTransport::constructObjectOdometries(
          motions, poses, frame_id, timestamp, timestamp_km1_, frame_id_link, object_class_map)),
      object_paths_(DSDTransport::constructMultiObjectOdometryPaths(
          motions, poses, timestamp, frame_timestamp_map, frame_id_link, true, object_class_map)),
      object_class_map_(object_class_map){}


DSDTransport::Publisher DSDTransport::addObjectInfo(
    const ObjectMotionMap& motions, const ObjectPoseMap& poses,
    const std::string& frame_id_link,
    const FrameIdTimestampMap& frame_timestamp_map,
    FrameId frame_id,
    Timestamp timestamp,
    const ObjectIdClassMap& object_class_map) {
  return Publisher(node_, object_odom_publisher_,
                   multi_object_odom_path_publisher_, tf_broadcaster_, motions,
                   poses, frame_id_link, frame_timestamp_map, frame_id,
                   timestamp, object_class_map);
}

DSDRos::DSDRos(const DisplayParams& params, rclcpp::Node::SharedPtr node)
    : params_(params), node_(node), dsd_transport_(node) {
  vo_publisher_ =
      node_->create_publisher<nav_msgs::msg::Odometry>("odometry", 1);
  vo_path_publisher_ =
      node_->create_publisher<nav_msgs::msg::Path>("odometry_path", 1);
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*node_);

  static_points_pub_ =
      node->create_publisher<sensor_msgs::msg::PointCloud2>("static_cloud", 1);
  dynamic_points_pub_ =
      node->create_publisher<sensor_msgs::msg::PointCloud2>("dynamic_cloud", 1);
}

void DSDRos::publishVisualOdometry(const gtsam::Pose3& T_world_camera,
                                   Timestamp timestamp, const bool publish_tf,
                                   const gtsam::Vector6& velocity) {
  DisplayCommon::publishOdometry(vo_publisher_, T_world_camera, timestamp,
                                 params_.world_frame_id,
                                 params_.camera_frame_id,
                                 velocity);

  if (publish_tf) {
    geometry_msgs::msg::TransformStamped t;
    dyno::convert<gtsam::Pose3, geometry_msgs::msg::TransformStamped>(
        T_world_camera, t);

    // t.header.stamp = node_->now();
    t.header.stamp = utils::toRosTime(timestamp);
    t.header.frame_id = params_.world_frame_id;
    t.child_frame_id = params_.camera_frame_id;
    tf_broadcaster_->sendTransform(t);
  }
}
void DSDRos::publishVisualOdometryPath(const gtsam::Pose3Vector& poses,
                                       Timestamp latest_timestamp) {
  DisplayCommon::publishOdometryPath(vo_path_publisher_, poses,
                                     latest_timestamp, params_.world_frame_id);
}

CloudPerObject DSDRos::publishStaticPointCloud(
    const StatusLandmarkVector& landmarks, const gtsam::Pose3& T_world_camera) {
  return DisplayCommon::publishPointCloud(
      static_points_pub_, landmarks, T_world_camera, params_.world_frame_id);
}
CloudPerObject DSDRos::publishDynamicPointCloud(
    const StatusLandmarkVector& landmarks, const gtsam::Pose3& T_world_camera) {
  return DisplayCommon::publishPointCloud(
      dynamic_points_pub_, landmarks, T_world_camera, params_.world_frame_id);
}

}  // namespace dyno
