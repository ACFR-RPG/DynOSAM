#pragma once

#include <unordered_map>

#include "dynosam_common/Trajectories.hpp"
#include "dynosam_common/Types.hpp"

namespace dyno {
// TODO: MotionTrackingQuality vs FeatureTrackingQuality

/// @brief Simple enum to denote the tracking quality (from OKVIS2!)
enum class TrackingQuality {
  Good,      ///< Good quality for at least 30% of the image containing tracks.
  Marginal,  ///< Marginal quality below that 30% level.
  Lost       ///< Lost, if only one or less keypoints are matched per image.
};

/// @brief Dynamic object keyframe type
enum class DoKeyframeType {
  NonKeyFrame = 0,
  RegularKeyFrame = 1,
  AnchorKeyFrame = 2
};

struct ViTrackingState {
  FrameId frame_id;
  bool is_keyframe{false};
};

struct DoTrackingState {
  FrameId frame_id;
  ObjectId object_id;
  DoKeyframeType keyframe_type{DoKeyframeType::NonKeyFrame};

  bool isKeyframe() const {
    return keyframe_type != DoKeyframeType::NonKeyFrame;
  }
};

struct DynoStateTrajectories {
  // trajectories
  PoseTrajectory camera_trajectory;
  MultiObjectTrajectories object_trajectories;
};

// really this is just Dyno1State!
// TODO: move to dynosam(1)!
struct DynoState : public DynoStateTrajectories {
  DYNO_POINTER_TYPEDEFS(DynoState)

  //! Must be filled or we get weird behaviour in visualisation
  FrameId frame_id;
  Timestamp timestamp;

  // if nothing in the trajectory this segfaults :/
  gtsam::Pose3 cameraPose() const;
  MotionEstimateMap objectMotions() const;

  //! Expect both to be in the world frame as defined by the camera pose
  //! at k
  StatusLandmarkVector static_map;
  StatusLandmarkVector dynamic_map;

  ViTrackingState vi_tracking_state;
  std::vector<DoTrackingState> do_tracking_states;
};

}  // namespace dyno
