#pragma once
#include "dynosam_common/MotionKeyFrame.hpp"
#include "dynosam_common/Trajectories.hpp"

namespace dyno {

struct DynoStateTrajectories {
  // trajectories
  PoseTrajectory camera_trajectory;
  MultiObjectTrajectories object_trajectories;
};

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

  //! Integration of keyframe based system (For older formulations this will
  //! always be empty). Used for visualisation
  KeyFrameInfoMap keyframe_infos;
};

}  // namespace dyno
