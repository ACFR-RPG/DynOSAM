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

ObjectIds FrameKFNode::objectKeyFrameIds() const {
  return ObjectIds{object_keyframes_.begin(), object_keyframes_.end()};
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

std::optional<FrameId> SharedModuleStates::getLatestOptimizedFrame() const {
  // shortcut to the camera (j=0)
  return getLatestOptimizedFrame(0);
}
std::optional<FrameId> SharedModuleStates::getLatestOptimizedFrame(
    ObjectId object_id) const {
  std::optional<FrameId> frame_id{std::nullopt};

  if (latest_optimized_frame_per_object_.exists(object_id)) {
    frame_id.emplace(latest_optimized_frame_per_object_.at(object_id));
  }
  return frame_id;
}

void SharedModuleStates::updateLatestOptFramePerObject(
    const gtsam::FastMap<ObjectId, FrameId>& latest_frames) {
  const std::lock_guard<std::mutex> lock(
      latest_optimized_frame_per_object_mutex_);
  for (const auto& [object_id, frame_id] : latest_frames) {
    latest_optimized_frame_per_object_[object_id] = frame_id;
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

  // record keyframe
  if (!object_keyframes_.exists(object_id)) {
    object_keyframes_.insert2(object_id, SharedFrameSet());
  }
  object_keyframes_.at(object_id).insert(frame_node);

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

std::string KeyFrameMap::verboseInfo(FrameId query_frame_id, int n) const {
  if (camera_keyframes_.empty()) {
    return "No frames.";
  }

  FrameId starting_frame =
      static_cast<FrameId>(std::max(static_cast<int>(query_frame_id) - n, 0));
  FrameId ending_frame = starting_frame + 2 * n + 1;

  std::stringstream ss;
  for (FrameId k = starting_frame; k < ending_frame; k++) {
    auto frame_node = this->getFrame(k);
    if (!frame_node) {
      continue;
    }

    if (frame_node->isAnyKeyFrame()) {
      ss << "Frame k " << k;
      if (frame_node->isCameraKeyFrame()) {
        ss << " CKF ";
      }
      auto object_keyframe_ids = frame_node->objectKeyFrameIds();
      if (!object_keyframe_ids.empty()) {
        ss << "OKFS: ";
        ss << container_to_string(object_keyframe_ids);
      }
      ss << "\n";
    }
  }

  // auto getFrame = [&]() -> KeyFrameMap::SharedFrame {
  //   if(isCameraKeyFrame(query_frame_id)) {
  //     return this->getFrame(query_frame_id);
  //   }
  //   else {
  //     return closestEarlierCameraKeyFrame(query_frame_id);
  //   }
  // };

  // KeyFrameMap::SharedFrame frame_node = getFrame();
  // auto it = camera_keyframes_.find(frame_node);

  // // Find lower bound for backward traversal
  // auto begin_it = it;
  // for (int i = 0; i < n; ++i) {
  //   if (begin_it == camera_keyframes_.begin()) {
  //     break;
  //   }
  //   --begin_it;
  // }

  // // Print from begin_it to +2n around query
  // auto curr = begin_it;
  // int count = 0;

  // std::stringstream ss;
  // while (curr != camera_keyframes_.end() && count < (2 * n + 1)) {
  //   auto frame_node = *curr;
  //   const FrameId frame_id = frame_node->frameId();
  //   // const auto& frame = curr->second;

  //   ss << (frame_id == query_frame_id ? " --> " : "     ");
  //   ss << "Frame " << frame_id;
  //   ss << "\t OKF" << container_to_string(frame_node->objectSeenIds());

  //   // Example custom frame printing
  //   // std::cout << " timestamp=" << frame.timestamp;

  //   ss << "\n";

  //   ++curr;
  //   ++count;
  // }

  return ss.str();
}

}  // namespace dyno
