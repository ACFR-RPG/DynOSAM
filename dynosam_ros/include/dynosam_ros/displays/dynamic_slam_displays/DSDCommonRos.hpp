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

#include <dynosam/common/Types.hpp>
#include <dynosam/utils/Macros.hpp>

#include "dynamic_slam_interfaces/msg/object_odometry.hpp"
#include "dynosam_ros/Display-Definitions.hpp"
#include "dynosam_ros/displays/DisplaysCommon.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/transform_broadcaster.h"

namespace dyno {

using ObjectOdometry = dynamic_slam_interfaces::msg::ObjectOdometry;
using ObjectOdometryPub = rclcpp::Publisher<ObjectOdometry>;

//! Map of object id link (child frame id) to ObjectOdometry (for a single
//! frame, no frame ids)
using ObjectOdometryMap = gtsam::FastMap<std::string, ObjectOdometry>;

/**
 * @brief Class for managing the publishing and conversion of ObjectOdometry
 * messages.
 * DSD is shorthand for Dynamic SLAM Display.
 *
 */
class DSDTransport {
 public:
  DYNO_POINTER_TYPEDEFS(DSDTransport)

  DSDTransport(rclcpp::Node::SharedPtr node);

  /**
   * @brief Construct object id tf link from an object id.
   * This will be used as the link in the ROS tf tree and as the child frame id
   * for the object odometry.
   *
   * @param object_id ObjectId
   * @return std::string
   */
  static std::string constructObjectFrameLink(ObjectId object_id);

  // this is technically wrong as we should have a motion at k and a pose at k-1
  // to get velocity...
  static ObjectOdometry constructObjectOdometry(
      const gtsam::Pose3& motion_k, const gtsam::Pose3& pose_k,
      ObjectId object_id, Timestamp timestamp_k,
      const std::string& frame_id_link, const std::string& child_frame_id_link);

  static ObjectOdometryMap constructObjectOdometries(
      const MotionEstimateMap& motions_k, const ObjectPoseMap& poses,
      FrameId frame_id_k, Timestamp timestamp_k,
      const std::string& frame_id_link);

  /**
   * @brief Nested Publisher that publishes all the object odometries for a
   * single frame/timestamp. Object odometries are set on construction.
   * Functionality for publishing the object odom's themselves and broadcasting
   * their position on the tf tree using the object current pose is addionally
   * included.
   *
   *
   */
  class Publisher {
   public:
    /**
     * @brief Publish the contained object odometries.
     *
     */
    void publishObjectOdometry();

    /**
     * @brief Broadcast the transform of each object using their pose.
     * Transforms will be constructed between the Publisher::frame_id_link_
     * and object frame link (as the child_frame_id).
     *
     */
    void publishObjectTransforms();

    /**
     * @brief Get the frame id
     *
     * @return FrameId
     */
    inline FrameId getFrameId() const { return frame_id_; }
    inline Timestamp getTimestamp() const { return timestamp_; }

    /**
     * @brief Get the (tf) frame id used as the parent of the tree.
     *
     * @return const std::string&
     */
    inline const std::string& getFrameIdLink() const { return frame_id_link_; }

   private:
    rclcpp::Node::SharedPtr node_;
    ObjectOdometryPub::SharedPtr object_odom_publisher_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::string frame_id_link_;
    FrameId frame_id_;
    Timestamp timestamp_;

    ObjectOdometryMap object_odometries_;

    friend class DSDTransport;

    /**
     * @brief Construct a new Publisher object.
     * Upon construction the publisher will broadcast to the 'object_odometry'
     * topic under the effective namespace of the provided node.
     *
     * @param node rclcpp::Node::SharedPtr
     * @param object_odom_publisher ObjectOdometryPub::SharedPtr Object odom
     * publisher to share between all Publishers.
     * @param tf_broadcaster std::shared_ptr<tf2_ros::TransformBroadcaster> TF
     * broadcaster to share between all Publishers.
     * @param motions const MotionEstimateMap& estimated motions at time (k)
     * @param poses const ObjectPoseMap& poses at time (... to k)
     * @param frame_id_link const std::string& parent tf tree link (e.g.
     * odom/world.)
     * @param frame_id FrameId current frame (k)
     * @param timestamp Timestamp
     */
    Publisher(rclcpp::Node::SharedPtr node,
              ObjectOdometryPub::SharedPtr object_odom_publisher,
              std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster,
              const MotionEstimateMap& motions, const ObjectPoseMap& poses,
              const std::string& frame_id_link, FrameId frame_id,
              Timestamp timestamp);
  };

  /**
   * @brief Create a object odometry Publisher given object state (motion and
   * pose) information for a given frame (k). The resulting publisher can then
   * be used to published ObjectOdometry messages and update the tf tree with
   * the object poses w.r.t to the parent frame id link.
   *
   * @param motions_k const MotionEstimateMap& estimated motions at time (k)
   * @param poses const ObjectPoseMap& poses at time (... to k)
   * @param frame_id_link const std::string& parent tf tree link (e.g.
   * odom/world.)
   * @param frame_id FrameId current frame (k)
   * @param timestamp Timestamp
   * @return Publisher
   */
  Publisher addObjectInfo(const MotionEstimateMap& motions_k,
                          const ObjectPoseMap& poses,
                          const std::string& frame_id_link, FrameId frame_id,
                          Timestamp timestamp);

 private:
  rclcpp::Node::SharedPtr node_;
  ObjectOdometryPub::SharedPtr object_odom_publisher_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

/**
 * @brief Shorthand for Dynamic Slam Display ROS
 *
 */
class DSDRos {
 public:
  DSDRos(const DisplayParams& params, rclcpp::Node::SharedPtr node);

  void publishVisualOdometry(const gtsam::Pose3& T_world_camera,
                             Timestamp timestamp, const bool publish_tf);
  void publishVisualOdometryPath(const gtsam::Pose3Vector& poses,
                                 Timestamp latest_timestamp);

  CloudPerObject publishStaticPointCloud(const StatusLandmarkVector& landmarks,
                                         const gtsam::Pose3& T_world_camera);

  // struct PubDynamicCloudOptions {
  //   //TODO: unused
  //   bool publish_object_bounding_box{true};

  //   // PubDynamicCloudOptions() = default;
  //   ~PubDynamicCloudOptions() = default;
  // };

  CloudPerObject publishDynamicPointCloud(const StatusLandmarkVector& landmarks,
                                          const gtsam::Pose3& T_world_camera);

 private:
 protected:
  const DisplayParams params_;
  rclcpp::Node::SharedPtr node_;
  //! Dynamic SLAM display transport for estimated object odometry
  DSDTransport dsd_transport_;
  //! TF broadcaster for the odometry.
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  OdometryPub::SharedPtr vo_publisher_;
  PathPub::SharedPtr vo_path_publisher_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      static_points_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      dynamic_points_pub_;
};

}  // namespace dyno
