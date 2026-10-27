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
  // pose and relative motion (H_W_km1_k)
  // instead hold velocity and H_W and Pose?
  dyno::PoseWithMotion state;
  dyno::DoKeyframeType keyframe_status{dyno::DoKeyframeType::NonKeyFrame};

  // motion from anchor frame to current frame
  // this matches what is estimated by the full system
  dyno::Motion3ReferenceFrame H_W_KF_k;
};

class DoGraph : public BaGraphStates<DOGraphState> {
 public:
  typedef DOGraphState State;

  DYNO_POINTER_TYPEDEFS(DoGraph)
  DoGraph(dyno::Camera::Ptr camera, dyno::ObjectId object_id);
  virtual ~DoGraph() = default;

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
        dyno::TemporalObjectIdentifier{frame_id, object_id_});
    // return this->getFrameAs<DoFrame>(frame_id);
  }

  void setRegularKeyframe(dyno::FrameId frame_id);

  bool getLandmark(dyno::TrackletId id, dyno::FrameId frame_id,
                   MapPoint& map_point) const;
  size_t getLandmarks(dyno::FrameId frame_id, MapPointVector& map_points) const;

  // I think we may in future want only pose + velocity?
  // for object odometry we want H_W_km1_k,  Vel_L_km_k, L_W_k
  // so maybe we want pose with motion + velocity?
  const dyno::PoseWithMotion& getPoseWithMotion(dyno::FrameId frame_id) const;
  const dyno::Motion3ReferenceFrame& keyframeMotion(
      dyno::FrameId frame_id) const;

  dyno::DoKeyframeType keyframeStatus(dyno::FrameId frame_id) const;

  const std::set<dyno::FrameId> activeKeyframes() const {
    return active_keyframes_;
  }

  const gtsam::Pose3& anchorPose(dyno::FrameId frame_id) const;
  const gtsam::Pose3& getPose(dyno::FrameId frame_id) const;

 protected:
  /** Adds a frame if it does not exist, otherwise checks that the incoming
   * frame is the same as the existing one.
   * Used in the initalizeNewAnchorState which often is called on the previous
   * frame id and therefore already has a DoFrame initalized!
   */
  FrameId maybeInitaliseFrame(DoFrame::Ptr frame);

  dyno::ObjectId object_id_;
  dyno::Camera::CalibrationType::shared_ptr K_;
  dyno::StereoCalibPtr K_stereo_;

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
