/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
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

#include <gtsam/geometry/Pose3.h>  //for Pose3

#include <nlohmann/json.hpp>
#include <opencv4/opencv2/opencv.hpp>  //for cv::Rect

#include "dynosam/common/Types.hpp"
#include "dynosam/pipeline/PipelinePayload.hpp"

using json = nlohmann::json;

namespace dyno {

// should only be included if visible??
struct ObjectPoseGT {
  DYNO_POINTER_TYPEDEFS(ObjectPoseGT)

  struct MotionInfo {
    bool is_moving_;    // moving between k-1 and k
    bool has_stopped_;  // has stopped since the previous frame. If true, then
                        // is_moving_ should be false
  };

  FrameId frame_id_;  // k
  ObjectId object_id_;
  gtsam::Pose3 L_camera_;  //! object pose in camera frame (using camera ie.
                           //! OpenCV frame convention)
  gtsam::Pose3 L_world_;   //! object pose in world frame (using camera ie.
                           //! OpenCV frame convention)
  cv::Rect bounding_box_;  //! box of detection on image plane

  /// @brief 3D object 'dimensions' in meters. Not all datasets will contain.
  /// Used to represent a 3D bounding box. Expected order is {width, height,
  /// length}
  std::optional<gtsam::Vector3> object_dimensions_;

  // TODO: update these to use Motion3ReferenceFrame!!
  /// @brief Motion in the world frame that takes us from k-1 (frame_id -1) to k
  /// (frame_id)
  std::optional<gtsam::Pose3> prev_H_current_world_;

  /// @brief Motion in the (ground truth object) frame (^WL_{k-1}) that takes us
  /// from k-1 (frame_id -1) to k (frame_id)
  std::optional<gtsam::Pose3> prev_H_current_L_;

  /// @brief Motion in the (ground truth camera) frame (^WX_{k-1}) that takes us
  /// from k-1 (frame_id -1) to k (frame_id)
  std::optional<gtsam::Pose3> prev_H_current_X_;

  std::optional<MotionInfo> motion_info_;

  /**
   * @brief Draws the object label and bounding box on the provided image
   *
   * @param img
   */
  void drawBoundingBox(cv::Mat& img) const;

  /**
   * @brief Calculates and sets prev_*_current_world_ using ground truth object
   * in the previous frame. Shoudl be used when constructed the gt packets in
   * the various DataProviders
   *
   * This object is epxected to be at frame k, and the previous motion should be
   * at frame k-1. The previous_object_gt should have the same object ID and an
   * exception will be thrown if either of these checks fail.
   *
   * Both ObjectPoseGT are expected to have the L_world_ and L_camera_ variable
   * set correctly.
   *
   * @param previous_motion const ObjectPoseGT&
   * @param prev_X_world const gtsam::Pose3& Camera pose at the previous frame
   * (k-1).
   * @param curr_X_world const gtsam::Pose3& Camera pose at the current frame
   * (k).
   */
  void setMotions(const ObjectPoseGT& previous_object_gt,
                  const gtsam::Pose3& prev_X_world,
                  const gtsam::Pose3& curr_X_world);

  operator std::string() const;

  // TODO: dont really have full tests over this
  bool operator==(const ObjectPoseGT& other) const;
  friend std::ostream& operator<<(std::ostream& os,
                                  const ObjectPoseGT& object_pose);
};

class GroundTruthInputPacket : public PipelinePayload {
 public:
  DYNO_POINTER_TYPEDEFS(GroundTruthInputPacket)

  // must have a default constructor for dataset loading and IO
  GroundTruthInputPacket() {}

  GroundTruthInputPacket(Timestamp timestamp, FrameId id, const gtsam::Pose3 X,
                         const std::vector<ObjectPoseGT>& poses)
      : PipelinePayload(timestamp),
        frame_id_(id),
        X_world_(X),
        object_poses_(poses) {}

  FrameId frame_id_;
  gtsam::Pose3 X_world_;  // camera pose in world frame (using camera ie. OpenCV
                          // frame convention)
  //! Vector of GT objects. This should contain all the objects in the sequence
  std::vector<ObjectPoseGT> object_poses_;

  /**
   * @brief Attempts to get the ObjectPoseGT with the requested id from the
   * packet. If the object id does not exist, false is returned.
   *
   * Log severity indicates what should happen if the object id does NOT exist
   * 0: nothing, and the function returns false
   * 1: a GLOG(WARNING) message is thrown, and the function returns false
   * 2: An exception is thrown, and the function returns false
   *
   * @param object_id
   * @param object_pose_gt
   * @param log_severity
   * @return true
   * @return false
   */
  bool getObject(ObjectId object_id, ObjectPoseGT& object_pose_gt,
                 int log_severity = 0) const;

  ObjectIds getObjectIds() const;

  /**
   * @brief Query an ObjectPoseGT in this packet and anOTHER packet using a
   * object label
   *
   * If the query object is in both this and the other packet, true is returned
   * and obj and other_obj are set
   *
   * We pass in a pointer to a pointer so we can modify the value of pointer
   * itself
   *
   * @param label ObjectId
   * @param other GroundTruthInputPacket&
   * @param obj ObjectPoseGT** Pointer to the object in THIS packet. Pointer is
   * passed in as address so the pointer can be changed. This points to the
   * obejct contained in this vector.
   * @param other_obj ObjectPoseGT** Pointer to the object in the OTHER packet.
   * Pointer is passed in as address so the pointer can be changed. This points
   * to the obejct contained in the OTHER vector.
   * @return true
   * @return false
   */
  bool findAssociatedObject(ObjectId label, GroundTruthInputPacket& other,
                            ObjectPoseGT** obj, ObjectPoseGT** other_obj);
  bool findAssociatedObject(ObjectId label, const GroundTruthInputPacket& other,
                            ObjectPoseGT** obj, const ObjectPoseGT** other_obj);

  /**
   * @brief Query an ObjectPoseGT in this packet and anOTHER packet using a
   * object label
   *
   * If the query object is in both this and the other packet, true is returned
   * and the index location obj and other_obj are set. The index is the position
   * in the object_poses_ vector (respectively) where the ObjectPoseGT can be
   * found
   *
   * e.g.
   *
   * GroundTruthPacket packet, other_packet;
   *
   * size_t packet_idx, other_packet_idx;
   * packet.findAssociatedObject(1u, other_packet, packet_idx,
   * other_packet_idx);
   *
   * assert(packet.object_poses_.at(packet_idx).object_id ==
   * other_packet.object_poses_.at(other_packet_idx).object_id == 1u)
   *
   * @param label
   * @param other
   * @param obj_idx
   * @param other_obj_idx
   * @return true
   * @return false
   */
  bool findAssociatedObject(ObjectId label, const GroundTruthInputPacket& other,
                            size_t& obj_idx, size_t& other_obj_idx) const;
  bool findAssociatedObject(ObjectId label, const GroundTruthInputPacket& other,
                            const ObjectPoseGT** obj,
                            const ObjectPoseGT** other_obj) const;

  /**
   * @brief Calcualtes and sets the object motion ground truth variables of this
   * GroundTruthInputPacket using the previous object motions. This packet is
   * considered to be time k and the previous object packet is k-1. This should
   * only be used in some derived DataProvider/Loader when constructing the
   * GroundTruthPacket.
   *
   * If the previous obejct packet is not at the right frame (k-1), false will
   * be returned. Each object in the current packet will be queried in the
   * previous packet, and, if exists, the motion will be calcuted at set.
   *
   * The vector of ObjectId's indicate which objects motions were calculated
   *
   * @param previous_object_packet
   * @param motions_set
   * @return size_t number of motions set
   */
  size_t calculateAndSetMotions(
      const GroundTruthInputPacket& previous_object_packet,
      ObjectIds& motions_set);
  size_t calculateAndSetMotions(
      const GroundTruthInputPacket& previous_object_packet);

  operator std::string() const;

  bool operator==(const GroundTruthInputPacket& other) const;
  friend std::ostream& operator<<(std::ostream& os,
                                  const GroundTruthInputPacket& gt_packet);

 private:
};

/**
 * @brief FastMap from frame id to GroundTruthInputPacket
 *
 */
class GroundTruthPacketMap
    : public gtsam::FastMap<FrameId, GroundTruthInputPacket> {
 public:
  using Base = gtsam::FastMap<FrameId, GroundTruthInputPacket>;
  using Base::Base;

  GroundTruthPacketMap() {}
};

/**
 * @brief Below is the JSON seralize/deseralize functions for all the
 * GroundTruthPacket types. We use the nlohmann::json library
 * (https://json.nlohmann.me/) - see the Arbitrary types conversions section in
 * the readme (https://github.com/nlohmann/json?tab=readme-ov-file#integration)
 * for details on the implemententation.
 */

/**
 * @brief Implements the required to_json (seralize) function for the
 * nlohmann::json to automatically convert ObjectPoseGT::MotionInfo to a json
 * type.
 *
 * Used internally when seralizing the ObjectPoseGT.
 *
 * @param j json&
 * @param motion_info const ObjectPoseGT::MotionInfo&
 */
void to_json(json& j, const ObjectPoseGT::MotionInfo& motion_info);

/**
 * @brief Implements the required from_json (deseralizes) function for the
 * nlohmann::json to automatically load ObjectPoseGT::MotionInfo from a json
 * object.
 *
 * Used internally when deseralizing the ObjectPoseGT.
 *
 * @param j const json&
 * @param motion_info  ObjectPoseGT::MotionInfo&
 */
void from_json(const json& j, ObjectPoseGT::MotionInfo& motion_info);

/**
 * @brief Implements the required to_json (seralize) function for the
 * nlohmann::json to automatically convert ObjectPoseGT to a json type.
 *
 * Used internally when seralizing the GroundTruthInputPacket.
 *
 * @param j json&
 * @param object_pose_gt  const ObjectPoseGT&
 */
void to_json(json& j, const ObjectPoseGT& object_pose_gt);

/**
 * @brief Implements the required from_json (deseralizes) function for the
 * nlohmann::json to automatically load ObjectPoseGT from a json object.
 *
 * Used internally when deseralizing the GroundTruthInputPacket.
 *
 * @param j const json&
 * @param object_pose_gt ObjectPoseGT&
 */
void from_json(const json& j, ObjectPoseGT& object_pose_gt);

/**
 * @brief Implements the required to_json (seralize) function for the
 * nlohmann::json to automatically convert GroundTruthInputPacket to a json
 * type.
 *
 * This allows code like:
 *
 * GroundTruthInputPacket gt_packet;
 * using json = nlohmann::json;
 * json j = gt_packet;
 * auto gt_packet_2 = j.template get<GroundTruthInputPacket>();
 *
 * to work automagically.
 *
 * Note: we don't use adl (argument dependant lookup) because we control the
 * namespace here
 *
 * @param j json&
 * @param gt_packet const GroundTruthInputPacket&
 */
void to_json(json& j, const GroundTruthInputPacket& gt_packet);

/**
 * @brief Implements the required from_json (deseralizes) function for the
 * nlohmann::json to automatically load GroundTruthInputPacket from a json
 * object.
 *
 * @param j const json&
 * @param gt_packet GroundTruthInputPacket&
 */
void from_json(const json& j, GroundTruthInputPacket& gt_packet);

/**
 * @brief Implements the required to_json (seralize) function for the
 * nlohmann::json to automatically convert GroundTruthPacketMap to a json type.
 *
 * @param j json&
 * @param gt_packet_map GroundTruthPacketMap&
 */
void to_json(json& j, const GroundTruthPacketMap& gt_packet_map);

/**
 * @brief Implements the required from_json (deseralizes) function for the
 * nlohmann::json to automatically load GroundTruthPacketMap from a json object.
 *
 * @param j const json&
 * @param gt_packet_map GroundTruthPacketMap&
 */
void from_json(const json& j, GroundTruthPacketMap& gt_packet_map);

}  // namespace dyno
