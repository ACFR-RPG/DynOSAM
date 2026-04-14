#include "dynosam/formulations/KeyFrameHybridMap.hpp"

namespace dyno {

void LandmarkKFNode::add(SharedFrame frame_node,
                         const CameraMeasurement& measurement) {
  // Add frame node and measurements to internal data-structures held in base
  // class
  Base::add(frame_node, measurement);

  // add frame to keyframe data-structure based on static/dynamic classification
  if (this->isStatic() && frame_node->isCameraKeyFrame()) {
    // sanity check all keyframes are camera keyframes
    if (!keyframes_.empty()) {
      CHECK(keyframes_.back()->isCameraKeyFrame());
    }
    keyframes_.insert(frame_node);
  } else if (!this->isStatic() &&
             frame_node->isObjectKeyFrame(this->objectId())) {
    // sanity check all keyframes are object j keyframes
    if (!keyframes_.empty()) {
      CHECK(keyframes_.back()->isObjectKeyFrame(this->objectId()));
    }

    keyframes_.insert(frame_node);
    CHECK(frame_node->objectObserved(this->object_id_));
  }
}

bool FrameKFNode::isAnyKeyFrame() const {
  return isAnyObjectKeyFrame() || isCameraKeyFrame();
}
bool FrameKFNode::isAnyObjectKeyFrame() const {
  return object_keyframes_.size() > 0u;
}

void FrameKFNode::setCameraKeyFrame() {
  is_camera_keyframe_ = true;
  updateLandmarksWithKF(this->static_landmarks_);
}

void FrameKFNode::setObjectKeyFrame(ObjectId object_id) {
  object_keyframes_.insert(object_id);
  updateLandmarksWithKF(this->dynamicLandmarks(object_id));
}

bool FrameKFNode::isCameraKeyFrame() const { return is_camera_keyframe_; }
bool FrameKFNode::isObjectKeyFrame(ObjectId object_id) const {
  return object_keyframes_.find(object_id) != object_keyframes_.end();
}

void FrameKFNode::addRelativeEgoMotion(const gtsam::Pose3& T_KF_k,
                                       FrameId frame_id_k) {
  CHECK(!T_KF_k_.exists(frame_id_k));

  T_KF_k_.insert2(frame_id_k, T_KF_k);
}

void FrameKFNode::updateLandmarksWithKF(const Base::Landmarks& lmks) {
  FrameId frame_id = this->frameId();
  for (auto& lmk : lmks) {
    // check that this landmark actually things it seens this frame
    // just for consistency!
    CHECK(lmk->seenAtAnyFrame(frame_id));
    // hacky way to get the current frame but in shared ptr form!
    auto this_frame = *lmk->frames_.find(frame_id);
    CHECK_EQ(this_frame->frameId(), frame_id);
    lmk->keyframes_.insert(this_frame);
  }
}

bool KeyFrameMap::setCameraKeyFrame(FrameId frame_id) {
  auto frame_interface = this->asFrameInterface();
  auto frame_node = frame_interface->getFrame(frame_id);

  if (!frame_node) {
    return false;
  }

  frame_node->setCameraKeyFrame();
  camera_keyframes_.insert(frame_node);
  return true;
}
bool KeyFrameMap::setObjectKeyFrame(FrameId frame_id, ObjectId object_id) {
  auto frame_interface = this->asFrameInterface();
  auto frame_node = frame_interface->getFrame(frame_id);

  if (!frame_node) {
    return false;
  }

  frame_node->setObjectKeyFrame(object_id);
  return true;
}

bool KeyFrameMap::isAnyKeyFrame(FrameId frame_id) const {
  auto frame_interface = this->asFrameInterface();
  auto frame_node = frame_interface->getFrame(frame_id);

  if (!frame_node) {
    return false;
  }

  return frame_node->isAnyKeyFrame();
}
bool KeyFrameMap::isCameraKeyFrame(FrameId frame_id) const {
  auto frame_interface = this->asFrameInterface();
  auto frame_node = frame_interface->getFrame(frame_id);

  if (!frame_node) {
    return false;
  }

  return frame_node->isCameraKeyFrame();
}
bool KeyFrameMap::isObjectKeyFrame(FrameId frame_id, ObjectId object_id) const {
  auto frame_interface = this->asFrameInterface();
  auto frame_node = frame_interface->getFrame(frame_id);

  if (!frame_node) {
    return false;
  }

  return frame_node->isObjectKeyFrame(object_id);
}

KeyFrameMap::SharedFrame KeyFrameMap::closestEarlierCameraKeyFrame(
    FrameId frame_id) const {
  // first element > query frame id
  auto it = camera_keyframes_.upper_bound(frame_id);

  if (it == camera_keyframes_.begin()) {
    return nullptr;
  } else {
    --it;
    return *it;
  }
}

}  // namespace dyno
