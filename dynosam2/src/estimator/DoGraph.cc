#include "dynosam2/estimator/DoGraph.hpp"

#include "dynosam_sensors/Camera.hpp"
#include "dynosam_sensors/RGBDCamera.hpp"

namespace dyno2 {

DoGraph::DoGraph(dyno::Camera::Ptr camera) {
  std::shared_ptr<dyno::RGBDCamera> rgbd_camera =
      CHECK_NOTNULL(camera->safeGetRGBDCamera());
  K_stereo_ = rgbd_camera->getFakeStereoCalib();
  CHECK_NOTNULL(K_stereo_);

  K_ = camera->getGtsamCalibration();
}

void DoGraph::initalizeNewAnchorFrame(
    dyno::FrameId frame_id,
    const dyno::StereoMeasurementStatusVector& measurements,
    const gtsam::Pose3 L) {
  this->addMeasurements(measurements);

  // update anchor point we are estimating relative to!
  DOGraphState& state = this->state(frame_id);
  state.state.motion = dyno::Motion3ReferenceFrame(
      gtsam::Pose3::Identity(), dyno::MotionRepresentationStyle::F2F,
      dyno::ReferenceFrame::GLOBAL, frame_id, frame_id);
  state.state.pose = L;
  state.keyframe_status = dyno::DoKeyframeType::AnchorKeyFrame;
}

// J is current frame
void DoGraph::initalizeStatesPropogate(
    dyno::FrameId frame_j,
    const dyno::StereoMeasurementStatusVector& measurements,
    const dyno::Motion3ReferenceFrame& G_i_ij, const gtsam::Pose3 X_W_j,
    const gtsam::Pose3 X_W_i) {
  this->addMeasurements(measurements);

  // state i
  const DOGraphState& last_state = lastState();
  const dyno::FrameId frame_i = last_state.state.motion.to();
  CHECK_EQ(frame_i, G_i_ij.from());

  // TODO: add links between motions!
  const gtsam::Pose3 H_W_ij_pose = X_W_j * G_i_ij * X_W_i.inverse();
  dyno::Motion3ReferenceFrame H_W_ij(
      H_W_ij_pose, dyno::MotionRepresentationStyle::F2F,
      dyno::ReferenceFrame::GLOBAL, frame_i, frame_j);

  gtsam::Pose3 L_W_j = H_W_ij->compose(last_state.state.pose);

  DOGraphState& state = this->state(frame_j);
  state.state.pose = L_W_j;
  state.state.motion = H_W_ij;

  // TODo: store G_i_ij as well!
}

const dyno::PoseWithMotion& DoGraph::getPoseWithMotion(
    dyno::FrameId frame_id) const {
  return states_.at(frame_id).state;
}

}  // namespace dyno2
