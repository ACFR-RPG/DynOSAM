#include "dynosam2/estimator/DoGraph.hpp"

#include "dynosam_sensors/Camera.hpp"
#include "dynosam_sensors/RGBDCamera.hpp"

namespace dyno2 {

DoGraph::DoGraph(dyno::Camera::Ptr camera, dyno::ObjectId object_id)
    : object_id_(object_id) {
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
  state.H_W_KF_k = dyno::Motion3ReferenceFrame(
      gtsam::Pose3::Identity(), dyno::MotionRepresentationStyle::KF,
      dyno::ReferenceFrame::GLOBAL, frame_id, frame_id);

  // reset active keyframes so there is no overlap in the estimator
  // and forces new keyframes at the start
  active_keyframes_.clear();
  active_keyframes_.insert(frame_id);
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
  // propogate using the frame-to-frame motion to construct a motion chain
  // from the anchor frame to the current frame
  gtsam::Pose3 H_W_KF_k = H_W_ij.estimate() * last_state.H_W_KF_k.estimate();
  state.H_W_KF_k = dyno::Motion3ReferenceFrame(
      H_W_KF_k, dyno::MotionRepresentationStyle::KF,
      dyno::ReferenceFrame::GLOBAL, last_state.H_W_KF_k.from(), frame_j);
}

void DoGraph::setRegularKeyframe(dyno::FrameId frame_id) {
  active_keyframes_.insert(frame_id);

  DOGraphState& state = this->state(frame_id);
  // check that state is not an anchor keyframe
  //  warn if so
  if (state.keyframe_status == dyno::DoKeyframeType::AnchorKeyFrame) {
    LOG(WARNING) << "Regular keyframe requested but object is already an "
                    "AnchorKeyframe!";
    return;
  }
  state.keyframe_status = dyno::DoKeyframeType::RegularKeyFrame;
}

bool DoGraph::getLandmark(dyno::TrackletId id, dyno::FrameId frame_id,
                          MapPoint& map_point) const {
  auto it = LandmarkMap::find(id);
  if (it == LandmarkMap::end()) {
    return false;
  }

  const dyno::Landmark& lmk_L = LandmarkMap::landmarkAt(it->second);
  const gtsam::Pose3 L_W_k = getPoseWithMotion(frame_id).pose;
  const dyno::Landmark lmk_W_k = L_W_k * lmk_L;

  map_point = MapPoint{lmk_W_k, object_id_, id};
}

size_t DoGraph::getLandmarks(dyno::FrameId frame_id,
                             MapPointVector& map_points) const {
  const gtsam::Pose3 L_W_k = getPoseWithMotion(frame_id).pose;
  if (LandmarkMap::isContiguous()) {
    const size_t n = LandmarkMap::size();
    map_points = MapPointVector(LandmarkMap::size());
    for (size_t i = 0; i < n; i++) {
      const auto [id, lmk_L] = LandmarkMap::at(i);
      const dyno::Landmark lmk_W_k = L_W_k * lmk_L;
      map_points[i] = MapPoint{lmk_W_k, object_id_, id};
    }
  } else {
    const size_t n = LandmarkMap::size();
    map_points = MapPointVector();
    map_points.reserve(n);

    // walk over valid indices
    for (auto it = LandmarkMap::begin(); it != LandmarkMap::end(); it++) {
      Index valid_index = it->second;
      const auto [id, lmk_L] = LandmarkMap::at(valid_index);
      const dyno::Landmark lmk_W_k = L_W_k * lmk_L;
      map_points.push_back(MapPoint{lmk_W_k, object_id_, id});
    }
  }
  return map_points.size();
}

const dyno::PoseWithMotion& DoGraph::getPoseWithMotion(
    dyno::FrameId frame_id) const {
  return states_.at(frame_id).state;
}

const dyno::Motion3ReferenceFrame& DoGraph::keyframeMotion(
    dyno::FrameId frame_id) const {
  return states_.at(frame_id).H_W_KF_k;
}

dyno::DoKeyframeType DoGraph::keyframeStatus(dyno::FrameId frame_id) const {
  return states_.at(frame_id).keyframe_status;
}

const gtsam::Pose3& DoGraph::anchorPose(dyno::FrameId frame_id) const {
  const dyno::FrameId anchor = states_.at(frame_id).H_W_KF_k.from();
  return states_.at(anchor).state.pose;
}

void DoGraphEstimator::optimize(dyno::FrameId frame_id,
                                std::vector<dyno::FrameId>& frames_affected,
                                size_t keyframes_to_include, bool only_states) {

}

}  // namespace dyno2
