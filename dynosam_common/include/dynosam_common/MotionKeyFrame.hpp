#pragma once

#include "dynosam_common/DynamicObjects.hpp"
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

/**
 * @brief KeyframeInfo for a single frame k
 *
 */
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

/// @brief Per object mapping of frame id to keyframe index (ie frame k (for
/// object j) is the Nth keyframe)
using FrameKeyframeIndexMapping = gtsam::FastMap<FrameObjectPair, int>;

/** Data parsed from the backend to the frontend when an update is complete */
struct PoseChangeUpdateComplete {
  // for batch data
  FrameId ending_frame_id;
  FrameId starting_frame_id;
  // Records the state of the map for each frame in the batch input
  // need to record separately as during the optimisation the map state will
  // change (ie. new keyframes are made)
  KeyFrameInfoMap keyframe_infos;

  struct Object {
    PoseWithMotionTrajectory trajectory;
    //! Tracked object points in L (ie. ^Lm)
    TrackedPointsPerObject::mapped_type points_m_L;
  };
  struct Camera {
    PoseTrajectory trajectory;
  };

  gtsam::FastMap<ObjectId, Object> objects;
  Camera camera;
};

using PoseChangeUpdateCompleteCallback =
    std::function<void(const PoseChangeUpdateComplete&)>;

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
  //! Relative motion of the object in Lkf
  gtsam::Pose3 H_Lkf_k;
  gtsam::Pose3 L_W_k;
  //! The camera pose at KF used in the smoother when estimating the motion
  //! H_W_KF_K
  gtsam::Pose3 X_W_KF;

  ObjectKeyFrameStatus keyframe_status{ObjectKeyFrameStatus::NonKeyFrame};
  ObjectTrackingStatus tracking_status{};

  bool isKeyFrame() const {
    return keyframe_status != ObjectKeyFrameStatus::NonKeyFrame;
  }
};

using ObjectPoseChangeInfoMap = gtsam::FastMap<ObjectId, ObjectPoseChangeInfo>;

std::ostream& operator<<(std::ostream& os, const ObjectKeyFrameStatus& status);

}  // namespace dyno
