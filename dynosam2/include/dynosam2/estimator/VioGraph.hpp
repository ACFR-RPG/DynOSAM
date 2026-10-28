
#pragma once

#include <gtsam/navigation/NavState.h>

#include "dynosam2/estimator/BaGraph.hpp"
#include "dynosam2/frontend/Frame.hpp"
#include "dynosam_opt/NonlinearOptimizer.hpp"
#include "dynosam_sensors/Camera.hpp"
#include "dynosam_sensors/RGBDCamera.hpp"

namespace dyno2 {

// TODO: hold factors so we dont need to rebuild them!?
struct VioGraphState {
  dyno::Timestamp timestamp{0.0};
  dyno::FrameId frame_id{0};
  // Viostate!
  gtsam::Pose3 pose_W;
  //! Velocity in world frame
  Velocity3 velocity_W;
  // IMU link
  bool is_keyframe{false};

  inline gtsam::NavState navState() const {
    return gtsam::NavState{pose_W, velocity_W.linear};
  }
};

class VioGraph : public BaGraphStates<VioGraphState> {
 public:
  typedef BaGraphStates<VioGraphState> Base;
  typedef VioGraphState State;

  DYNO_POINTER_TYPEDEFS(VioGraph)
  VioGraph(dyno::Camera::Ptr camera);
  virtual ~VioGraph() = default;

  // get landmark in estimated reference frame not map frame
  using LandmarkMap::getLandmark;

  void initaliseStates(ViFrame::Ptr frame, const gtsam::Pose3& pose);
  void initaliseStates(ViFrame::Ptr frame, const gtsam::Pose3& pose,
                       const Velocity3& velocity_W);

  inline ViFrame::Ptr getFramePtr(dyno::FrameId frame_id) const {
    return this->getFrameAs<ViFrame>(
        dyno::TemporalObjectId{frame_id, dyno::background_label});
    // return this->getFrameAs<ViFrame>(frame_id);
  }

  void setPose(dyno::FrameId frame_id, const gtsam::Pose3& pose);
  void setVelocity(dyno::FrameId frame_id, const Velocity3& velocity_W);
  void setVelocityFromPose(dyno::FrameId frame_id);

  const VioGraphState& getVioState(dyno::FrameId frame_id) const;

  const gtsam::Pose3& getPose(dyno::FrameId frame_id) const;

  /// @brief Get velocity in the world frame
  /// @param frame_id
  /// @return
  const Velocity3& getVelocity(dyno::FrameId frame_id) const;

  /// @brief Get velocity in the body frame of the camera
  /// @param frame_id
  /// @return
  // const Velocity3& getBodyVelocity(dyno::FrameId frame_id) const;

  bool getLandmark(dyno::TrackletId id, MapPoint& map_point) const;
  size_t getLandmarks(MapPointVector& map_points) const;
  size_t getLandmarks(MapPoints& map_points) const;

  bool poseExists(dyno::FrameId frame_id) const;
  // currently no removal of keyframe
  bool setKeyframe(dyno::FrameId id, bool is_keyframe);
  bool isKeyframe(dyno::FrameId id) const;

  // active keyframes
  const std::set<dyno::FrameId>& keyFrames() const;

  bool pruneKeyframes(size_t num_keyframes, size_t num_motion_frames);

 private:
  VioGraphState& initaliseStatesWithPose(ViFrame::Ptr frame,
                                         const gtsam::Pose3& pose);

  void setStateVelocityFromPose(VioGraphState& current_state);

 protected:
  //! Active keyframes used for selection and bundle adjustmement!
  std::set<dyno::FrameId> keyframes_;
};

class VioGraphEstimator : public VioGraph, public BaProblemInterface {
 public:
  DYNO_POINTER_TYPEDEFS(VioGraphEstimator)
  VioGraphEstimator(dyno::Camera::Ptr camera) : VioGraph(camera) {}

  void optimize(dyno::FrameId frame_id,
                std::vector<dyno::FrameId>& frames_affected,
                bool only_states = false);

 private:
};

}  // namespace dyno2
