#pragma once

#include "dynosam_common/DynoState.hpp"
#include "dynosam_common/Types.hpp"

/**
 * Common definitions for the PoseChange/MotionKeyFrame modules
 *
 */

namespace dyno {

enum class ObjectKeyFrameStatus {
  NonKeyFrame = 0,
  RegularKeyFrame = 1,
  AnchorKeyFrame = 2
};

struct KeyframeInfo {
  bool camera_keyframe{false};

  struct MotionPair {
    ObjectId object_id;
    FrameId from_motion;
    FrameId to_motion;
  };
  std::vector<MotionPair> object_keyframes;
};

/// @brief Map representing keyframe meta-data for each
using KeyFrameInfoMap = gtsam::FastMap<FrameId, KeyframeInfo>;

/** Data parsed from the backend to the frontend when an update is complete */
struct PoseChangeUpdateComplete {
  // for batch data
  FrameId ending_frame_id;
  FrameId starting_frame_id;
  // Records the state of the map for each frame in the batch input
  // need to record separately as during the optimisation the map state will
  // change (ie. new keyframes are made)
  KeyFrameInfoMap keyframe_infos;
};

using PoseChangeUpdateCompleteCallback =
    std::function<void(const PoseChangeUpdateComplete&)>;

// TODO: so inconsistent with names!!! Hybrid/Keyframe/PoseChange!?
struct HybridKeyFrameUpdate {
  FrameId frame_id;
  Timestamp timestamp;

  struct Object {
    ObjectId object_id;
    PoseWithMotionTrajectory trajectory;
    //! Tracked object points in L (ie. ^Lm)
    TrackedPointsPerObject::mapped_type object_points;
  };

  PoseTrajectory camera_trajectory;
  std::vector<Object> object_infos;

  const Object* getObject(ObjectId object_id) const;
};

struct ObjectPoseChangeInfo {
  FrameId frame_id;

  StatusLandmarkVector initial_object_points;
  //! Associated keyframe
  //! if keyframe then this value is NEW (ie changed from the previous one)
  //! and the initial motion should be identity
  gtsam::Pose3 L_W_KF;
  //! This is the preintegrated motion immediately before the current
  //! keyframe at k
  Motion3ReferenceFrame H_W_KF_k;
  gtsam::Pose3 L_W_k;

  ObjectKeyFrameStatus keyframe_status{ObjectKeyFrameStatus::NonKeyFrame};

  bool isKeyFrame() const {
    return keyframe_status != ObjectKeyFrameStatus::NonKeyFrame;
  }
};

using ObjectPoseChangeInfoMap = gtsam::FastMap<ObjectId, ObjectPoseChangeInfo>;

std::ostream& operator<<(std::ostream& os, const ObjectKeyFrameStatus& status);

}  // namespace dyno
