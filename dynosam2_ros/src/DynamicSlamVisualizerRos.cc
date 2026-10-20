#include "dynosam2_ros/DynamicSlamVisualizerRos.hpp"

#include <pcl/common/transforms.h>

#include "dynosam_common/viz/Colour.hpp"
#include "dynosam_ros_common/RosUtils.hpp"

namespace dyno2 {

// helper functions
static Marker poseToCameraFrustrum(const gtsam::Pose3& T_WR,
                                   const dyno::Timestamp timestamp,
                                   const std::string& frame_id,
                                   const dyno::Color& colour, const int id,
                                   const double scale = 0.15,
                                   const double line_width = 0.1,
                                   const std::string& ns = "camera_frustum");

static Marker poseToObjectArrow(const gtsam::Pose3& T_WR,
                                const dyno::Timestamp timestamp,
                                const std::string& frame_id,
                                const dyno::Color& colour, int id,
                                const double scale = 0.15,
                                const double line_width = 0.1,
                                const std::string& ns = "object_frustum");

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

  object_keyframes_publisher_ =
      node_->create_publisher<MarkerArray>("object_keyframes", odom_qos);

  map_publisher_ = node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "map", dyno::ros::addQosParameter(*node, "SENSOR_DATA", "map_qos"));
}

void DynamicSlamVisualizerRos::publishEstimatorUpdate(
    const PublicationData& publication_data) {
  CHECK_NOTNULL(estimator_);

  publishVisualOdometry(publication_data);
  publishObjects(publication_data);
  publishMap(publication_data);
}

void DynamicSlamVisualizerRos::publishVisualOdometry(
    const PublicationData& publication_data) {
  const dyno::FrameId frame_id = publication_data.frame_id;
  const dyno::Timestamp timestamp = publication_data.timestamp;
  const auto reference_frames = sensor_rig_->getReferenceFrames();
  const auto timestamp_ros = dyno::ros::toRosTime(timestamp);

  static constexpr double kLineWidth = 0.02;
  static constexpr double kFrustrumScale = 0.2;
  static constexpr int kShowLastNKeyframes = 10;

  MarkerArray camera_frustrum_marker_array;
  visualization_msgs::msg::Marker delete_marker;
  delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
  camera_frustrum_marker_array.markers.push_back(delete_marker);

  const gtsam::Pose3 X_WR_k =
      cameraToRobotFrame(estimator_->getVioPose(frame_id));

  camera_trajectory_.insert(frame_id, timestamp, X_WR_k);
  // first upate internal trajectory
  for (dyno::FrameId updated_state : publication_data.updated_states.vo) {
    gtsam::Pose3 X_W_C = estimator_->getVioPose(updated_state);
    gtsam::Pose3 X_W_R = cameraToRobotFrame(X_W_C);
    camera_trajectory_.update(updated_state, X_W_R);
  }

  // update camera frustrum markers
  camera_frustrum_marker_array.markers.push_back(poseToCameraFrustrum(
      X_WR_k, timestamp, reference_frames.odom_frame,
      dyno::NiceColors::bluishgreen(), static_cast<int>(frame_id),
      2.0 * kFrustrumScale, 2.0 * kLineWidth));

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

  // publish VO path and fill keyframe markers
  nav_msgs::msg::Path path;
  std::set<dyno::FrameId> camera_keyframes;
  for (const auto& entry_k : camera_trajectory_.toVector()) {
    geometry_msgs::msg::PoseStamped pose_stamped;
    const dyno::Timestamp timestamp_k = entry_k.timestamp;
    const dyno::FrameId frame_id_k = entry_k.frame_id;
    const gtsam::Pose3& odom = entry_k.data;
    dyno::ros::convertWithHeader(odom, pose_stamped, timestamp_k,
                                 reference_frames.odom_frame);
    path.poses.push_back(pose_stamped);

    if (estimator_->isCameraKeyframe(frame_id_k)) {
      camera_keyframes.insert(frame_id_k);
    }
  }
  path.header.stamp = timestamp_ros;
  path.header.frame_id = reference_frames.odom_frame;
  vo_path_publisher_->publish(path);

  // build keyframe markers
  // erase current frame (if exists) as we have already displayed
  camera_keyframes.erase(frame_id);

  if (camera_trajectory_.size() >= 2) {
    const auto start_it =
        camera_keyframes.size() > kShowLastNKeyframes
            ? std::prev(camera_keyframes.end(), kShowLastNKeyframes)
            : camera_keyframes.begin();

    for (auto it = start_it; it != camera_keyframes.end(); ++it) {
      const dyno::FrameId k0 = *it;
      const auto entry = camera_trajectory_.get(k0);
      const auto T_WR = entry.data;
      const auto timestamp_i = entry.timestamp;

      camera_frustrum_marker_array.markers.push_back(poseToCameraFrustrum(
          T_WR, timestamp_i, reference_frames.odom_frame,
          dyno::NiceColors::vermillion(), static_cast<int>(k0), kFrustrumScale,
          kLineWidth));
    }
  }
  vo_keyframes_publisher_->publish(camera_frustrum_marker_array);
}

void DynamicSlamVisualizerRos::publishObjects(
    const PublicationData& publication_data) {
  const dyno::Timestamp timestamp = publication_data.timestamp;
  const auto timestamp_ros = dyno::ros::toRosTime(timestamp);

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

  if (!publication_data.do_tracking_states.empty()) {
    MultiObjectOdometryPath multi_object_odom_paths;
    multi_object_odom_paths.header.stamp = timestamp_ros;
    multi_object_odom_paths.header.frame_id = reference_frames_.odom_frame;

    MarkerArray frustrum_markers;
    visualization_msgs::msg::Marker delete_marker;
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    frustrum_markers.markers.push_back(delete_marker);

    // Marker IDs must be unique within the marker namespace.
    int marker_id{0};

    // now get trajectories only for seen objects
    for (const dyno::DoTrackingState& tracking_state :
         publication_data.do_tracking_states) {
      const dyno::ObjectId object_id = tracking_state.object_id;
      // TODO: estimator should give us velocity!
      const auto& object_trajectory = object_trajectories_[object_id];

      if (object_trajectory.size() < 2) {
        continue;
      }

      // Full trajectory path.
      std_msgs::msg::ColorRGBA colour_msg;
      const dyno::Color object_colour = dyno::Color::uniqueId(object_id);
      dyno::convert(object_colour, colour_msg);

      multi_object_odom_paths.paths.emplace_back();
      ObjectOdometryPath& path_per_segment =
          multi_object_odom_paths.paths.back();
      path_per_segment.colour = colour_msg;
      path_per_segment.object_id = object_id;
      path_per_segment.path_segment = 0;
      path_per_segment.header = multi_object_odom_paths.header;

      // dummy way of calculating object odom now -> just fill pose!
      const auto frame_link = reference_frames_.odom_frame;
      const auto child_link = "object_" + std::to_string(object_id) + "_link";

      static constexpr int kShowLastNKeyframes = 10;
      static constexpr double kArrowScale = 0.4;
      static constexpr double kLineWidth = 0.02;

      const dyno::Color anchor_color = object_colour.brightness(0.2f);
      const dyno::Color regular_color = object_colour.brightness(0.5f);

      struct ObjectKeyframe {
        dyno::FrameId frame_id;
        dyno::DoKeyframeType keyframe_type;
        gtsam::Pose3 L_W_k;
        bool operator<(const ObjectKeyframe& other) const {
          return frame_id < other.frame_id;
        }
      };

      std::set<ObjectKeyframe> object_keyframes;
      for (const auto& entry_k : object_trajectory.toVector()) {
        path_per_segment.object_odometries.emplace_back();

        const gtsam::Pose3& L_W_k = entry_k.data.pose;
        const dyno::Timestamp timestamp_k = entry_k.timestamp;
        const dyno::FrameId frame_id_k = entry_k.frame_id;

        ObjectOdometry& object_odom = path_per_segment.object_odometries.back();
        dyno::ros::convertWithHeader(L_W_k, object_odom.odom, timestamp_k,
                                     frame_link, child_link);

        object_odom.object_id = object_id;
        object_odom.sequence = frame_id_k;

        // fill object keyframes;
        dyno::DoKeyframeType keyframe_status;
        if (estimator_->isObjectKeyframe(
                dyno::TemporalObjectIdentifier{frame_id_k, object_id},
                keyframe_status)) {
          object_keyframes.insert(
              ObjectKeyframe{frame_id_k, keyframe_status, L_W_k});
        }
      }

      if (object_keyframes.size() > 2) {
        const auto start_it =
            object_keyframes.size() > kShowLastNKeyframes
                ? std::prev(object_keyframes.end(), kShowLastNKeyframes)
                : object_keyframes.begin();

        for (auto it = start_it; it != object_keyframes.end(); ++it) {
          const ObjectKeyframe& object_keyframe = *it;

          double arrorw_scale = kArrowScale;
          double line_width = kLineWidth;
          dyno::Color marker_colour = regular_color;

          if (object_keyframe.keyframe_type ==
              dyno::DoKeyframeType::AnchorKeyFrame) {
            arrorw_scale *= 2.0;
            line_width *= 2.0;
            marker_colour = anchor_color;
          }

          frustrum_markers.markers.push_back(poseToObjectArrow(
              object_keyframe.L_W_k, timestamp, reference_frames_.odom_frame,
              marker_colour, marker_id++, kArrowScale, kLineWidth));
        }
      }
    }

    object_keyframes_publisher_->publish(frustrum_markers);
    multi_object_odom_path_publisher_->publish(multi_object_odom_paths);
  }
}

void DynamicSlamVisualizerRos::publishMap(
    const PublicationData& publication_data) {
  dyno::ObjectIds object_tracked;
  std::unordered_map<dyno::ObjectId, dyno::Color> colours;
  for (const auto& tracking_state : publication_data.do_tracking_states) {
    const dyno::ObjectId object_id = tracking_state.object_id;

    object_tracked.push_back(object_id);
    colours[object_id] = dyno::Color::uniqueId(object_id);
  }
  colours[dyno::background_label] = dyno::Color(0, 0, 0);

  // get static and dynamic map points
  MapPointVector map_points;
  estimator_->getLandmarks(map_points, object_tracked);
  if (map_points.size() > 0) {
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    cloud.reserve(map_points.size());
    for (const auto& map_point : map_points) {
      const dyno::Landmark& p_W = map_point.p_W;

      const dyno::ObjectId object_id = map_point.object_id;
      const dyno::Color& color = colours[object_id];

      pcl::PointXYZRGB point(
          static_cast<float>(p_W(0)), static_cast<float>(p_W(1)),
          static_cast<float>(p_W(2)), static_cast<std::uint8_t>(color.r),
          static_cast<std::uint8_t>(color.g),
          static_cast<std::uint8_t>(color.b));
      cloud.push_back(point);
    }

    // apply transform to point cloud from estimation (camera) to robot frame
    Eigen::Matrix<float, 4, 4> T_RC_eigen = T_RC_.matrix().cast<float>();
    pcl::transformPointCloud(cloud, cloud, T_RC_eigen);

    sensor_msgs::msg::PointCloud2 pc2_msg;
    pcl::toROSMsg(cloud, pc2_msg);
    pc2_msg.header.frame_id = reference_frames_.odom_frame;
    map_publisher_->publish(pc2_msg);
  }
}

Marker poseToCameraFrustrum(const gtsam::Pose3& T_WR,
                            const dyno::Timestamp timestamp,
                            const std::string& frame_id,
                            const dyno::Color& colour, const int id,
                            const double scale, const double line_width,
                            const std::string& ns) {
  Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = dyno::ros::toRosTime(timestamp);

  marker.ns = ns;
  marker.id = id;

  marker.type = Marker::LINE_LIST;
  marker.action = Marker::ADD;

  marker.pose.orientation.w = 1.0;

  marker.scale.x = line_width;

  std_msgs::msg::ColorRGBA colour_msg;
  convert(colour, colour_msg);
  marker.color = colour_msg;
  marker.color.a = 1.0f;

  marker.lifetime = rclcpp::Duration::from_seconds(0.0);

  const double w = scale;
  const double h = w * 0.75;
  const double z = w * 0.6;

  // Frustum corners in camera optical frame
  const gtsam::Point3 O(0.0, 0.0, 0.0);

  const gtsam::Point3 P1(w, h, z);
  const gtsam::Point3 P2(w, -h, z);
  const gtsam::Point3 P3(-w, -h, z);
  const gtsam::Point3 P4(-w, h, z);

  auto transformPoint = [&](const gtsam::Point3& p_cam) {
    static const auto T_robot_opencv = dyno::openCVToRobotTransform();
    auto p_cam_robot = T_robot_opencv * p_cam;
    return T_WR.transformFrom(p_cam_robot);
  };

  auto toRosPoint = [](const gtsam::Point3& p) {
    geometry_msgs::msg::Point msg;
    dyno::convert(p, msg);
    return msg;
  };

  auto addLine = [&](const gtsam::Point3& a, const gtsam::Point3& b) {
    marker.points.push_back(toRosPoint(transformPoint(a)));
    marker.points.push_back(toRosPoint(transformPoint(b)));
  };

  // Pyramid edges
  addLine(O, P1);
  addLine(O, P2);
  addLine(O, P3);
  addLine(O, P4);

  // Front rectangle
  addLine(P1, P2);
  addLine(P4, P3);
  addLine(P4, P1);
  addLine(P3, P2);

  return marker;
}

Marker poseToObjectArrow(const gtsam::Pose3& T_WR,
                         const dyno::Timestamp timestamp,
                         const std::string& frame_id, const dyno::Color& colour,
                         int id, const double scale, const double line_width,
                         const std::string& ns) {
  Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = dyno::ros::toRosTime(timestamp);
  marker.ns = ns;
  marker.id = id;
  marker.type = Marker::LINE_LIST;
  marker.action = Marker::ADD;

  marker.pose.orientation.w = 1.0;
  marker.scale.x = line_width;

  std_msgs::msg::ColorRGBA colour_msg;
  dyno::convert(colour, colour_msg);
  marker.color = colour_msg;
  marker.color.a = 1.0f;

  // Square-based pyramid, pointing along the object's local +X axis.
  constexpr double kBackHalfWidth = 0.12;
  //   constexpr double kBackHalfWidth = 0.09;
  constexpr double kTipX = 0.48;

  const double half_width = kBackHalfWidth * scale;
  const double tip_x = kTipX * scale;

  using Point3 = gtsam::Point3;

  // Four corners of the flat back face, perpendicular to local +X.
  const std::array<Point3, 4> back = {Point3(0.0, -half_width, -half_width),
                                      Point3(0.0, half_width, -half_width),
                                      Point3(0.0, half_width, half_width),
                                      Point3(0.0, -half_width, half_width)};

  // Single tip extending forward along local +X.
  const Point3 tip(tip_x, 0.0, 0.0);

  marker.points.reserve(16);

  auto addLine = [&](const Point3& p_local, const Point3& q_local) {
    const Point3 p_world = T_WR.transformFrom(p_local);
    const Point3 q_world = T_WR.transformFrom(q_local);

    geometry_msgs::msg::Point p;
    p.x = p_world.x();
    p.y = p_world.y();
    p.z = p_world.z();

    geometry_msgs::msg::Point q;
    q.x = q_world.x();
    q.y = q_world.y();
    q.z = q_world.z();

    marker.points.push_back(p);
    marker.points.push_back(q);
  };

  // Four edges of the flat back face.
  for (std::size_t i = 0; i < back.size(); ++i) {
    addLine(back[i], back[(i + 1) % back.size()]);
  }

  // Four edges converging at the tip.
  for (const Point3& corner : back) {
    addLine(corner, tip);
  }

  return marker;
}

}  // namespace dyno2
