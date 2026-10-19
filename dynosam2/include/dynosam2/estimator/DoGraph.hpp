#pragma once

#include "dynosam2/estimator/BaGraph.hpp"
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
};

class DoGraph : public BaGraphStates<DOGraphState> {
 public:
  typedef DOGraphState State;

  DYNO_POINTER_TYPEDEFS(DoGraph)
  DoGraph(dyno::Camera::Ptr camera);
  virtual ~DoGraph() = default;

  // I think should just be initalize from motion H_W_km_1
  // and then we initally propogate!
  // // should be measurements for this frame
  // //   dyno::StereoMeasurementStatusVector&
  // // think better how to initalise!
  // void initaliseStates(
  //     dyno::FrameId frame_id,
  //     const dyno::StereoMeasurementStatusVector& measurements,
  //     const gtsam::Pose3& pose);
  void initalizeNewAnchorFrame(
      dyno::FrameId frame_id,
      const dyno::StereoMeasurementStatusVector& measurements,
      const gtsam::Pose3 L);

  // where j is current frame frame
  void initalizeStatesPropogate(
      dyno::FrameId frame_j,
      const dyno::StereoMeasurementStatusVector& measurements,
      const dyno::Motion3ReferenceFrame& G_i_ij, const gtsam::Pose3 X_W_j,
      const gtsam::Pose3 X_W_i);

  // I think we may in future want only pose + velocity?
  const dyno::PoseWithMotion& getPoseWithMotion(dyno::FrameId frame_id) const;

 protected:
  dyno::Camera::CalibrationType::shared_ptr K_;
  dyno::StereoCalibPtr K_stereo_;
};

class DoGraphEstimator : public DoGraph {
 public:
  DYNO_POINTER_TYPEDEFS(DoGraphEstimator)
  DoGraphEstimator(dyno::Camera::Ptr camera) : DoGraph(camera) {}
  virtual ~DoGraphEstimator() = default;
};

}  // namespace dyno2
