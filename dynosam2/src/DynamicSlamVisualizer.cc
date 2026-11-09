#include "dynosam2/DynamicSlamVisualizer.hpp"

namespace dyno2 {

DynamicSlamVisualizer::DynamicSlamVisualizer(
    dyno::CanonicalSensorRig::Ptr sensor_rig, const SlamBackend* estimator)
    : sensor_rig_(sensor_rig),
      estimator_(CHECK_NOTNULL(estimator)),
      T_RC_(sensor_rig->getCanonicalExtrinsics()),
      reference_frames_(sensor_rig->getReferenceFrames()) {}

void DynamicSlamVisualizer::setDisplayQueue(
    dyno::ImageDisplayQueue* display_queue) {
  image_displayer_ = std::make_unique<dyno::ImageQueueDisplayer>(display_queue);
}

void DynamicSlamVisualizer::setGroundTruth(
    const dyno::SharedGroundTruth& ground_truth) {
  shared_ground_truth_ = ground_truth;
}

bool DynamicSlamVisualizer::getGroundtruthPacket(
    FrameId fid, dyno::GroundTruthInputPacket& ground_truth_packet) const {
  std::optional<GroundTruthPacketMap> maybe_ground_truth =
      shared_ground_truth_.access();
  if (maybe_ground_truth) {
    auto ground_truth = maybe_ground_truth.value();
    auto it = ground_truth.find(fid);
    if (it == ground_truth.end()) {
      return false;
    }

    ground_truth_packet = it->second;
    return true;
  }
  return false;
}

bool DynamicSlamVisualizer::updateGroundtruthTrajectories(dyno::FrameId fid) {
  dyno::GroundTruthInputPacket packet;
  if (!getGroundtruthPacket(fid, packet)) {
    return false;
  }

  const dyno::Timestamp t = packet.timestamp_;
  gtsam::Pose3 X_WR_gt = cameraToRobotFrame(packet.X_world_);
  camera_trajectory_gt_.insertOrUpdate(fid, t, X_WR_gt);

  for (const auto& object_gt : packet.object_poses_) {
    const auto object_id = object_gt.object_id_;

    // check we have a gt motion here
    // in the case that we dont, this might be the first time the object
    // appears...
    dyno::PoseWithMotion state_S;
    if (!object_gt.prev_H_current_world_) {
      state_S = dyno::PoseWithMotion{
          object_gt.L_world_,
          Motion3ReferenceFrame(gtsam::Pose3::Identity(),
                                ReferenceFrame::GLOBAL, fid, fid)};
    } else {
      state_S = dyno::PoseWithMotion{
          object_gt.L_world_,
          Motion3ReferenceFrame(object_gt.prev_H_current_world_.value(),
                                ReferenceFrame::GLOBAL, fid - 1u, fid)};
    }
    const dyno::PoseWithMotion state_R = cameraToRobotFrame(state_S);
    object_trajectories_gt_[object_id].insertOrUpdate(fid, t, state_R);
  }
  return true;
}

void DynamicSlamVisualizer::addImageDisplay(dyno::ImageDisplay::Ptr display) {
  if (image_displayer_) {
    image_displayer_->addDisplay(display);
  } else {
    LOG(WARNING) << "Attempting to add ImageDisplay to DynamicSlamVisualizer, "
                    "but setDisplayQueue has not been called!";
  }
}

void DynamicSlamVisualizer::displayImages() {
  if (image_displayer_) {
    image_displayer_->displayFromQueue(2);
  }
}

}  // namespace dyno2
