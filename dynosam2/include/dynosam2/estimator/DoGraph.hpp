#pragma once

#include "dynosam2/estimator/BaGraph.hpp"
#include "dynosam2/estimator/VioGraph.hpp"
#include "dynosam2/frontend/Frame.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_core/DynoState.hpp"
#include "dynosam_sensors/Camera.hpp"
#include "dynosam_sensors/RGBDCamera.hpp"

namespace dyno2 {

// remember this is a motion from a frame to a frame
// TODO: hold factors so we dont need to rebuild them!?
struct DOGraphState {
  dyno::Timestamp timestamp{0.0};
  //! Object pose
  gtsam::Pose3 L_W_k;
  //! Object motion in W from k-1 to k
  dyno::Motion3ReferenceFrame H_W_km1_k;
  //! Velocity in world frame
  Velocity3 velocity_W;
  // motion from anchor frame to current frame
  // this matches what is estimated by the full system
  dyno::Motion3ReferenceFrame H_W_KF_k;
  dyno::DoKeyframeType keyframe_status{dyno::DoKeyframeType::NonKeyFrame};

  /// @brief The current frame id, equivalent to the "to" frame of the contained
  /// motion(s)
  /// @return
  inline dyno::FrameId frameId() const {
    CHECK_EQ(H_W_KF_k.to(), H_W_KF_k.to());
    return H_W_KF_k.to();
  }

  inline dyno::FrameId anchorFrameId() const { return H_W_KF_k.from(); }
};

class DoGraph : public BaGraphStates<DOGraphState> {
 public:
  typedef BaGraphStates<DOGraphState> Base;
  typedef DOGraphState State;

  DYNO_POINTER_TYPEDEFS(DoGraph)
  DoGraph(dyno::Camera::Ptr camera, dyno::ObjectId object_id);
  virtual ~DoGraph() = default;

  inline dyno::ObjectId objectId() const noexcept { return object_id_; }

  // initalise of frame without state - this is the case as we need at least two
  // frames
  // before createing a state!
  FrameId initaliseFrame(DoFrame::Ptr frame);

  void initalizeNewAnchorState(DoFrame::Ptr do_frame, const gtsam::Pose3 L);

  // where j is current frame frame
  void initalizeStatesPropogate(DoFrame::Ptr do_frame_j,
                                const dyno::Motion3ReferenceFrame& G_i_ij,
                                const gtsam::Pose3 X_W_j,
                                const gtsam::Pose3 X_W_i);

  inline DoFrame::Ptr getFramePtr(dyno::FrameId frame_id) const {
    return this->getFrameAs<DoFrame>(
        dyno::TemporalObjectId{frame_id, object_id_});
    // return this->getFrameAs<DoFrame>(frame_id);
  }

  void setRegularKeyframe(dyno::FrameId frame_id);

  bool getLandmark(dyno::TrackletId id, dyno::FrameId frame_id,
                   MapPoint& map_point) const;
  size_t getLandmarks(dyno::FrameId frame_id, MapPointVector& map_points) const;

  const dyno::Motion3ReferenceFrame& keyframeMotion(
      dyno::FrameId frame_id) const;

  dyno::DoKeyframeType keyframeStatus(dyno::FrameId frame_id) const;

  const std::set<dyno::FrameId> activeKeyframes() const {
    return active_keyframes_;
  }

  const gtsam::Pose3& anchorPose(dyno::FrameId frame_id) const;
  const gtsam::Pose3& getPose(dyno::FrameId frame_id) const;

  /// @brief Get the object motion in W from k-1 to k.
  /// @param frame_id
  /// @return
  const dyno::Motion3ReferenceFrame& getMotion(dyno::FrameId frame_id) const;

  /// @brief Get velocity in the world frame
  /// @param frame_id
  /// @return
  const Velocity3& getVelocity(dyno::FrameId frame_id) const;

 protected:
  /** Adds a frame if it does not exist, otherwise checks that the incoming
   * frame is the same as the existing one.
   * Used in the initalizeNewAnchorState which often is called on the previous
   * frame id and therefore already has a DoFrame initalized!
   */
  FrameId maybeInitaliseFrame(DoFrame::Ptr frame);

  /// @brief Sets the state's motion (H_W_km1_k) and velocity given the state's
  /// current pose the and the previous state's pose
  /// @param current_state
  void setStateMotionAndVelocityFromPose(DOGraphState& current_state);

 protected:
  dyno::ObjectId object_id_;
  //! Keyframes since the last anchor keyframe
  std::set<dyno::FrameId> active_keyframes_;
};

class DoGraphEstimator : public DoGraph, public BaProblemInterface {
 public:
  DYNO_POINTER_TYPEDEFS(DoGraphEstimator)
  DoGraphEstimator(dyno::Camera::Ptr camera, dyno::ObjectId object_id)
      : DoGraph(camera, object_id) {}
  virtual ~DoGraphEstimator() = default;

  void optimize(dyno::FrameId frame_id, const VioGraph& vi_graph,
                std::vector<dyno::FrameId>& frames_affected,
                size_t keyframes_to_include = 1);
};

}  // namespace dyno2
