/*
 *   Copyright (c) 2025 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a
 copy
 *   of this software and associated documentation files (the "Software"), to
 deal
 *   in the Software without restriction, including without limitation the
 rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in
 all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE
 *   SOFTWARE.
 */

#pragma once

#include "dynamic_slam_interfaces/msg/multi_object_odometry_path.hpp"
#include "dynamic_slam_interfaces/msg/object_odometry.hpp"
#include "dynamic_slam_interfaces/msg/object_odometry_path.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/Macros.hpp"
#include "dynosam_core/DynoState.hpp"
#include "dynosam_ros/displays/DisplaysCommon.hpp"
#include "dynosam_ros_common/RosUtils.hpp"
#include "dynosam_sensors/SensorRig.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/header.hpp"
#include "tf2_ros/transform_broadcaster.h"

namespace dyno {

using ObjectOdometry = dynamic_slam_interfaces::msg::ObjectOdometry;
using ObjectOdometryPath = dynamic_slam_interfaces::msg::ObjectOdometryPath;
using MultiObjectOdometryPath =
    dynamic_slam_interfaces::msg::MultiObjectOdometryPath;

using ObjectOdometryPub = rclcpp::Publisher<ObjectOdometry>;
using MultiObjectOdometryPathPub = rclcpp::Publisher<MultiObjectOdometryPath>;

//! Map of object id link (child frame id) to ObjectOdometry (for a single
//! frame, no frame ids)
using ObjectOdometryMap = gtsam::FastMap<std::string, ObjectOdometry>;

struct DynoStatePublisherOptions {
  //! Publish TF for visual odom
  bool publish_vo_tf{false};
  //! Publish TF for object odometry
  bool publish_oo_tf{false};
  //! Publish camera pose as additional wireframe orb-slam style
  bool publish_wireframe_cameras{false};

  bool convert_to_optical{true};
};

class DynoStatePublisher {
 public:
  DynoStatePublisher(
      const CanonicalSensorRig::ConstPtr& sensor_rig,
      rclcpp::Node::SharedPtr node,
      const DynoStatePublisherOptions& options = DynoStatePublisherOptions());

  DYNO_POINTER_TYPEDEFS(DynoStatePublisher)

  void publish(const DynoState& state);

 private:
  void publishObjects(FrameId frame_id,
                      const MultiObjectTrajectories& object_trajectories,
                      const std::vector<DoTrackingState>& do_tracking_states);

  ObjectOdometry constructObjectOdometry(
      ObjectId object_id, FrameId frame_id,
      const PoseWithMotionTrajectory& trajectory) const;

  void sendObjectOdometryTransform(const ObjectOdometry& object_odom);

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
  void sendTransform(const T& pose, const Timestamp& timestamp,
                     const std::string& frame_link,
                     const std::string& child_frame_link) {
    std_msgs::msg::Header header;
    header.stamp = ros::toRosTime(timestamp);
    header.frame_id = frame_link;

    sendTransform(pose, header, child_frame_link);
  }

 protected:
  CanonicalSensorRig::ConstPtr sensor_rig_;
  rclcpp::Node::SharedPtr node_;
  DynoStatePublisherOptions options_;
  //! TF broadcaster for the odometry.
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  OdometryPub::SharedPtr vo_publisher_;
  PathPub::SharedPtr vo_path_publisher_;

  ObjectOdometryPub::SharedPtr object_odom_publisher_;
  MultiObjectOdometryPathPub::SharedPtr multi_object_odom_path_publisher_;

  PointCloud2Pub::SharedPtr static_points_pub_;
  PointCloud2Pub::SharedPtr dynamic_points_pub_;

  MarkerArrayPub::SharedPtr camera_wireframe_pub_;
  MarkerArrayPub::SharedPtr object_wireframe_pub_;

  struct ViTrackingStatesRecord {
    // All key frames to display
    // must be ordered so we can show only the last N states
    std::set<FrameId> keyframes;
    // All vi states
    dyno::FastUnorderedMap<FrameId, ViTrackingState> temporal_tracking_states;

    void update(const ViTrackingState& update) {
      temporal_tracking_states[update.frame_id] = update;
      if (update.is_keyframe) {
        keyframes.insert(update.frame_id);
      }
    }

    std::optional<ViTrackingState> maybeTrackingState(FrameId frame_id) const {
      auto it = temporal_tracking_states.find(frame_id);
      if (it == temporal_tracking_states.end()) {
        return std::optional<ViTrackingState>{};
      } else {
        return it->second;
      }
    }
  };

  // record per object
  struct DoTrackingStatesRecord {
    /// @brief Update for a  the record of temporal tracking states for a frame
    /// k and insert to the list of keyframes if the update is a keyframe
    /// @param update const DoTrackingState& object update for a frame k
    void update(const DoTrackingState& update) {
      temporal_tracking_states[update.frame_id] = update;
      if (update.isKeyframe()) {
        keyframes.insert(update.frame_id);
      }
    }

    DoKeyframeType getKeyframeType(FrameId frame_id) const {
      return temporal_tracking_states.at(frame_id).keyframe_type;
    }

    // All key frames to display
    std::set<FrameId> keyframes;
    // All do states per object
    dyno::FastUnorderedMap<FrameId, DoTrackingState> temporal_tracking_states;
  };

  // All object tracking states
  struct AllDoTrackingStatesRecord {
    dyno::FastUnorderedMap<ObjectId, DoTrackingStatesRecord> objects;

    void update(const std::vector<DoTrackingState>& updates) {
      for (const DoTrackingState& do_tracking_state : updates) {
        objects[do_tracking_state.object_id].update(do_tracking_state);
      }
    }
  };

  ViTrackingStatesRecord vi_tracking_states_record_;
  AllDoTrackingStatesRecord do_tracking_states_records_;
};

}  // namespace dyno
