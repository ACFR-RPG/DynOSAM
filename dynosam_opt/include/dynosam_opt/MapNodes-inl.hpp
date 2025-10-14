/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a
 copy
 *   of this software and associated documentation files (the "Software"), to
 deal
 *   in the Software without restriction, including without limitation the
 rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in
 all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE
 *   SOFTWARE.
 */
#pragma once

#include "dynosam_common/Map.hpp"
#include "dynosam_common/MapNodes.hpp"
#include "dynosam_opt/Symbols.hpp"

namespace dyno {

template <typename MEASUREMENT>
int FrameNode<MEASUREMENT>::getId() const {
  return (int)frame_id;
}

template <typename MEASUREMENT>
bool FrameNode<MEASUREMENT>::objectObserved(ObjectId object_id) const {
  return objects_seen.exists(object_id);
}

template <typename MEASUREMENT>
bool FrameNode<MEASUREMENT>::objectObservedInPrevious(
    ObjectId object_id) const {
  const auto frame_id_k_1 = frame_id - 1u;
  FrameNodePtr<MEASUREMENT> frame_node_k_1 =
      this->map_ptr_->template getFrame(frame_id_k_1);

  if (!frame_node_k_1) {
    return false;
  }
  return frame_node_k_1->objectObserved(object_id);
}

template <typename MEASUREMENT>
gtsam::Key FrameNode<MEASUREMENT>::makePoseKey() const {
  return CameraPoseSymbol(frame_id);
}

template <typename MEASUREMENT>
gtsam::Key FrameNode<MEASUREMENT>::makeObjectMotionKey(
    ObjectId object_id) const {
  return ObjectMotionSymbol(object_id, this->frame_id);
}

template <typename MEASUREMENT>
gtsam::Key FrameNode<MEASUREMENT>::makeObjectPoseKey(ObjectId object_id) const {
  return ObjectPoseSymbol(object_id, this->frame_id);
}

template <typename MEASUREMENT>
bool FrameNode<MEASUREMENT>::objectMotionExpected(ObjectId object_id) const {
  return objectObserved(object_id) && objectObservedInPrevious(object_id);
}

template <typename MEASUREMENT>
std::vector<typename FrameNode<MEASUREMENT>::LandmarkMeasurementPair>
FrameNode<MEASUREMENT>::getStaticMeasurements() const {
  std::vector<LandmarkMeasurementPair> measurements;
  for (const auto& lmk_ptr : static_landmarks) {
    MEASUREMENT m = lmk_ptr->getMeasurement(this->frame_id);
    measurements.push_back(std::make_pair(lmk_ptr, m));
  }
  return measurements;
}

template <typename MEASUREMENT>
std::vector<typename FrameNode<MEASUREMENT>::LandmarkMeasurementPair>
FrameNode<MEASUREMENT>::getDynamicMeasurements() const {
  std::vector<LandmarkMeasurementPair> measurements;
  for (const auto& lmk_ptr : dynamic_landmarks) {
    const MEASUREMENT m = lmk_ptr->getMeasurement(this->frame_id);
    measurements.push_back(std::make_pair(lmk_ptr, m));
  }
  return measurements;
}

template <typename MEASUREMENT>
std::vector<typename FrameNode<MEASUREMENT>::LandmarkMeasurementPair>
FrameNode<MEASUREMENT>::getDynamicMeasurements(ObjectId object_id) const {
  std::vector<LandmarkMeasurementPair> measurements;
  for (const auto& lmk_ptr : dynamic_landmarks) {
    if (lmk_ptr->getObjectId() == object_id) {
      const MEASUREMENT m = lmk_ptr->getMeasurement(this->frame_id);
      measurements.push_back(std::make_pair(lmk_ptr, m));
    }
  }
  return measurements;
}

/// LandmarkNode
template <typename MEASUREMENT>
int LandmarkNode<MEASUREMENT>::getId() const {
  return (int)tracklet_id;
}

template <typename MEASUREMENT>
ObjectId LandmarkNode<MEASUREMENT>::getObjectId() const {
  return object_id;
}

template <typename MEASUREMENT>
bool LandmarkNode<MEASUREMENT>::isStatic() const {
  return object_id == background_label;
}

template <typename MEASUREMENT>
size_t LandmarkNode<MEASUREMENT>::numObservations() const {
  return frames_seen_.size();
}

template <typename MEASUREMENT>
void LandmarkNode<MEASUREMENT>::add(FrameNodePtr<MEASUREMENT> frame_node,
                                    const MEASUREMENT& measurement) {
  frames_seen_.insert(frame_node);

  // add measurement to map
  // first check that we dont already have a measurement at this frame
  if (measurements_.exists(frame_node)) {
    throw DynosamException("Unable to add new measurement to landmark node " +
                           std::to_string(this->getId()) + " at frame " +
                           std::to_string(frame_node->getId()) +
                           " as a measurement already exists at this frame!");
  }

  measurements_.insert2(frame_node, measurement);

  CHECK_EQ(frames_seen_.size(), measurements_.size());
}

template <typename MEASUREMENT>
bool LandmarkNode<MEASUREMENT>::seenAtFrame(FrameId frame_id) const {
  return frames_seen_.exists(frame_id);
}

template <typename MEASUREMENT>
bool LandmarkNode<MEASUREMENT>::hasMeasurement(FrameId frame_id) const {
  return this->seenAtFrame(frame_id);
}

template <typename MEASUREMENT>
const MEASUREMENT& LandmarkNode<MEASUREMENT>::getMeasurement(
    FrameNodePtr<MEASUREMENT> frame_node) const {
  CHECK_NOTNULL(frame_node);
  if (!hasMeasurement(frame_node->frame_id)) {
    throw DynosamException("Missing measurement in landmark node with id " +
                           std::to_string(tracklet_id) + " at frame " +
                           std::to_string(frame_node->frame_id));
  }
  return measurements_.at(frame_node);
}

template <typename MEASUREMENT>
const MEASUREMENT& LandmarkNode<MEASUREMENT>::getMeasurement(
    FrameId frame_id) const {
  if (!seenAtFrame(frame_id)) {
    throw DynosamException("Missing measurement in landmark node with id " +
                           std::to_string(tracklet_id) + " at frame " +
                           std::to_string(frame_id));
  }
  return getMeasurement(this->map_ptr_->template getFrame(frame_id));
}

template <typename MEASUREMENT>
gtsam::Key LandmarkNode<MEASUREMENT>::makeStaticKey() const {
  const auto key = StaticLandmarkSymbol(this->tracklet_id);
  if (!this->isStatic()) {
    throw InvalidLandmarkQuery(
        key, "Static estimate requested but landmark is dynamic!");
  }
  return key;
}

template <typename MEASUREMENT>
gtsam::Key LandmarkNode<MEASUREMENT>::makeDynamicKey(FrameId frame_id) const {
  return (gtsam::Key)makeDynamicSymbol(frame_id);
}

template <typename MEASUREMENT>
DynamicPointSymbol LandmarkNode<MEASUREMENT>::makeDynamicSymbol(
    FrameId frame_id) const {
  const auto key = DynamicLandmarkSymbol(frame_id, this->tracklet_id);
  if (this->isStatic()) {
    throw InvalidLandmarkQuery(
        key, "Dynamic estimate requested but landmark is static!");
  }
  return key;
}

template <typename MEASUREMENT>
FrameIds LandmarkNode<MEASUREMENT>::getSeenFrameIds() const {
  return getSeenFrames().template collectIds<FrameId>();
}

/// ObjectNode
template <typename MEASUREMENT>
int ObjectNode<MEASUREMENT>::getId() const {
  return (int)object_id;
}

template <typename MEASUREMENT>
FrameNodePtrSet<MEASUREMENT> ObjectNode<MEASUREMENT>::getSeenFrames() const {
  FrameNodePtrSet<MEASUREMENT> seen_frames;
  for (const auto& lmks : dynamic_landmarks) {
    seen_frames.merge(lmks->getSeenFrames());
  }
  return seen_frames;
}

template <typename MEASUREMENT>
FrameIds ObjectNode<MEASUREMENT>::getSeenFrameIds() const {
  return getSeenFrames().template collectIds<FrameId>();
}

template <typename MEASUREMENT>
LandmarkNodePtrSet<MEASUREMENT>
ObjectNode<MEASUREMENT>::getLandmarksSeenAtFrame(FrameId frame_id) const {
  LandmarkNodePtrSet<MEASUREMENT> seen_lmks;

  for (const auto& lmk : dynamic_landmarks) {
    // all frames this lmk was seen in
    const FrameNodePtrSet<MEASUREMENT>& frames = lmk->getSeenFrames();
    // lmk was observed at this frame
    if (frames.find(frame_id) != frames.end()) {
      seen_lmks.insert(lmk);
    }
  }
  return seen_lmks;
}

}  // namespace dyno
