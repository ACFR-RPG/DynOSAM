/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam_cv/Feature.hpp"

#include <glog/logging.h>

#include "dynosam_common/Exceptions.hpp"

namespace dyno {

Feature::Feature(const Feature& other) {
  // TODO: use both mutexs?
  std::lock_guard<std::mutex> lk(other.mutex_);
  data_ = other.data_;
}

bool Feature::operator==(const Feature& other) const {
  return data_ == other.data_;
}

Keypoint Feature::keypoint() const {
  std::lock_guard<std::mutex> lk(mutex_);
  return data_.keypoint;
}

OpticalFlow Feature::measuredFlow() const {
  std::lock_guard<std::mutex> lk(mutex_);
  return data_.measured_flow;
}

Keypoint Feature::predictedKeypoint() const {
  std::lock_guard<std::mutex> lk(mutex_);
  return data_.predicted_keypoint;
}

size_t Feature::age() const {
  std::lock_guard<std::mutex> lk(mutex_);
  return data_.age;
}

KeyPointType Feature::keypointType() const {
  std::lock_guard<std::mutex> lk(mutex_);
  return data_.type;
}

TrackletId Feature::trackletId() const {
  std::lock_guard<std::mutex> lk(mutex_);
  return data_.tracklet_id;
}

FrameId Feature::frameId() const {
  std::lock_guard<std::mutex> lk(mutex_);
  return data_.frame_id;
}

bool Feature::inlier() const {
  std::lock_guard<std::mutex> lk(mutex_);
  return data_.inlier;
}

ObjectId Feature::objectId() const {
  std::lock_guard<std::mutex> lk(mutex_);
  return data_.tracking_label;
}

Depth Feature::depth() const {
  std::lock_guard<std::mutex> lk(mutex_);
  return data_.depth;
}

Keypoint Feature::rightKeypoint() const {
  std::lock_guard<std::mutex> lk(mutex_);
  if (data_.right_kp.has_value()) return data_.right_kp.value();
  DYNO_THROW_MSG(DynosamException)
      << "Right Kp requested for" << data_.tracklet_id << " but not set";
  return Keypoint{};
}

Keypoint Feature::CalculatePredictedKeypoint(const Keypoint& keypoint,
                                             const OpticalFlow& measured_flow) {
  return keypoint + measured_flow;
}

void Feature::setPredictedKeypoint(const OpticalFlow& measured_flow) {
  std::lock_guard<std::mutex> lk(mutex_);
  data_.measured_flow = measured_flow;
  data_.predicted_keypoint =
      CalculatePredictedKeypoint(data_.keypoint, measured_flow);
}

Feature& Feature::keypoint(const Keypoint& kp) {
  std::lock_guard<std::mutex> lk(mutex_);
  data_.keypoint = kp;
  return *this;
}

Feature& Feature::measuredFlow(const OpticalFlow& measured_flow) {
  std::lock_guard<std::mutex> lk(mutex_);
  data_.measured_flow = measured_flow;
  return *this;
}

Feature& Feature::predictedKeypoint(const Keypoint& predicted_kp) {
  std::lock_guard<std::mutex> lk(mutex_);
  data_.predicted_keypoint = predicted_kp;
  return *this;
}

Feature& Feature::age(const size_t& a) {
  std::lock_guard<std::mutex> lk(mutex_);
  data_.age = a;
  return *this;
}

Feature& Feature::keypointType(const KeyPointType& kp_type) {
  std::lock_guard<std::mutex> lk(mutex_);
  data_.type = kp_type;
  return *this;
}

Feature& Feature::trackletId(const TrackletId& tracklet_id) {
  std::lock_guard<std::mutex> lk(mutex_);
  data_.tracklet_id = tracklet_id;
  return *this;
}

Feature& Feature::frameId(const FrameId& frame_id) {
  std::lock_guard<std::mutex> lk(mutex_);
  data_.frame_id = frame_id;
  return *this;
}

Feature& Feature::objectId(ObjectId id) {
  std::lock_guard<std::mutex> lk(mutex_);
  data_.tracking_label = id;
  return *this;
}

Feature& Feature::depth(Depth d) {
  std::lock_guard<std::mutex> lk(mutex_);
  data_.depth = d;
  return *this;
}

Feature& Feature::rightKeypoint(const Keypoint& right_kp) {
  std::lock_guard<std::mutex> lk(mutex_);
  data_.right_kp = right_kp;
  return *this;
}

bool Feature::usable() const {
  std::lock_guard<std::mutex> lk(mutex_);
  return data_.inlier && (data_.tracklet_id != invalid_id);
}

bool Feature::isStatic() const {
  std::lock_guard<std::mutex> lk(mutex_);
  return data_.type == KeyPointType::STATIC;
}

Feature& Feature::markOutlier() {
  std::lock_guard<std::mutex> lk(mutex_);
  data_.inlier = false;
  return *this;
}

Feature& Feature::markInlier() {
  std::lock_guard<std::mutex> lk(mutex_);
  data_.inlier = true;
  return *this;
}

Feature& Feature::markInvalid() {
  std::lock_guard<std::mutex> lk(mutex_);
  data_.tracklet_id = invalid_id;
  return *this;
}

bool Feature::hasDepth() const {
  std::lock_guard<std::mutex> lk(mutex_);
  return !std::isnan(data_.depth);
}

bool Feature::hasRightKeypoint() const {
  std::lock_guard<std::mutex> lk(mutex_);
  return data_.right_kp.has_value();
}

bool Feature::stereoPoint(gtsam::StereoPoint2& stereo) const {
  if (!hasRightKeypoint()) {
    return false;
  }

  const Keypoint& L = this->keypoint();
  const Keypoint& R = this->rightKeypoint();
  stereo = gtsam::StereoPoint2(L(0), R(0), L(1));
  return true;
}

FeatureContainer::FeatureContainer()
    : feature_map_(),
      object_feature_map_(),
      empty_object_feature_view_(0, this) {}

FeatureContainer::FeatureContainer(const FeatureContainer& other)
    : feature_map_(other.feature_map_),
      object_feature_map_(other.object_feature_map_),
      empty_object_feature_view_(0, this) {
  rebindObjectFeatureViews();
}

FeatureContainer& FeatureContainer::operator=(const FeatureContainer& other) {
  if (this != &other) {
    feature_map_ = other.feature_map_;
    object_feature_map_ = other.object_feature_map_;
    rebindObjectFeatureViews();
  }
  return *this;
}

FeatureContainer::FeatureContainer(FeatureContainer&& other) noexcept
    : feature_map_(std::move(other.feature_map_)),
      object_feature_map_(std::move(other.object_feature_map_)),
      empty_object_feature_view_(0, this) {
  rebindObjectFeatureViews();
}

FeatureContainer& FeatureContainer::operator=(
    FeatureContainer&& other) noexcept {
  if (this != &other) {
    feature_map_ = std::move(other.feature_map_);
    object_feature_map_ = std::move(other.object_feature_map_);
    rebindObjectFeatureViews();
  }
  return *this;
}

FeatureContainer::FeatureContainer(const FeaturePtrs& feature_vector)
    : FeatureContainer() {
  for (size_t i = 0; i < feature_vector.size(); i++) {
    add(feature_vector.at(i));
  }
}

void FeatureContainer::add(const Feature& feature) {
  auto feature_ptr = std::make_shared<Feature>(feature);
  add(feature_ptr);
}

void FeatureContainer::add(Feature::Ptr feature) {
  const TrackletId tracklet_id = feature->trackletId();
  CHECK(!exists(tracklet_id))
      << "Feailure in FeatureContainer::add - Tracklet Id " << tracklet_id
      << " already exists";
  feature_map_[tracklet_id] = feature;

  const ObjectId object_id = feature->objectId();
  // TODO: what if object id is not valid!
  if (!object_feature_map_.exists(object_id)) {
    object_feature_map_.insert2(object_id, ObjectFeatureView(object_id, this));
  }
  object_feature_map_.at(object_id).insert(tracklet_id);
}

// TODO: test
// this will mess up any iterator that currently has a reference to any
// feature_map_ (so any of the filters)
void FeatureContainer::remove(TrackletId tracklet_id) {
  if (!exists(tracklet_id)) {
    throw std::runtime_error("Cannot remove feature with tracklet id " +
                             std::to_string(tracklet_id) +
                             " as feature does not exist!");
  }

  auto feature = this->getByTrackletId(tracklet_id);
  const auto object_id = feature->objectId();
  // check object id exists in map
  CHECK(object_feature_map_.exists(object_id));
  // remove tracklet id from tracklet set and if the set is now empty
  // remove object id entirely
  auto& tracklets_per_object = object_feature_map_.at(object_id).tracklets_;
  tracklets_per_object.erase(tracklet_id);

  if (tracklets_per_object.empty()) {
    object_feature_map_.erase(object_id);
  }

  // remove from main feature map
  feature_map_.erase(tracklet_id);
}

void FeatureContainer::removeByObjectId(ObjectId object_id) {
  if (!object_feature_map_.exists(object_id)) {
    return;
  }

  // this->remove will act on the same tracklet object
  //  make copy of tracklets to ensure correct iteration
  const auto tracklets_to_remove = object_feature_map_.at(object_id).tracklets_;
  for (const auto tracklet_id : tracklets_to_remove) {
    this->remove(tracklet_id);
  }

  // remove from object_feature_map_
  object_feature_map_.erase(object_id);
}

void FeatureContainer::clear() {
  feature_map_.clear();
  object_feature_map_.clear();
}

// TODO: else if logic needs test!!
TrackletIds FeatureContainer::collectTracklets(bool only_usable) const {
  TrackletIds tracklets;
  for (const auto& feature : *this) {
    if (only_usable && feature->usable()) {
      tracklets.push_back(feature->trackletId());
    } else if (!only_usable) {
      tracklets.push_back(feature->trackletId());
    }
  }

  return tracklets;
}

void FeatureContainer::markOutliers(const TrackletIds& outliers) {
  for (TrackletId tracklet_id : outliers) {
    CHECK(exists(tracklet_id));

    getByTrackletId(tracklet_id)->markOutlier();
  }
}

size_t FeatureContainer::size() const { return feature_map_.size(); }

size_t FeatureContainer::size(ObjectId object_id) const {
  if (object_feature_map_.exists(object_id)) {
    return object_feature_map_.at(object_id).size();
  }
  return 0;
}

Feature::Ptr FeatureContainer::getByTrackletId(TrackletId tracklet_id) const {
  if (!exists(tracklet_id)) {
    return nullptr;
  }
  return feature_map_.at(tracklet_id);
}

bool FeatureContainer::exists(TrackletId tracklet_id) const {
  return feature_map_.find(tracklet_id) != feature_map_.end();
}

TrackletIds FeatureContainer::getByObject(ObjectId object_id) const {
  if (object_feature_map_.exists(object_id)) {
    const auto& tracklets = object_feature_map_.at(object_id).tracklets();
    // convert unordered_set to vector
    return TrackletIds(tracklets.begin(), tracklets.end());
  } else {
    return TrackletIds{};
  }
}

const FeatureContainer::ObjectFeatureView& FeatureContainer::featuresByObject(
    ObjectId object_id) const {
  checkAndThrow(
      hasObject(object_id),
      "FeatureContainer::featuresByObject error: no features with object " +
          std::to_string(object_id));
  return object_feature_map_.at(object_id);
}

FeatureContainer::ObjectFeatureView& FeatureContainer::featuresByObject(
    ObjectId object_id) {
  checkAndThrow(
      hasObject(object_id),
      "FeatureContainer::featuresByObject error: no features with object " +
          std::to_string(object_id));
  return object_feature_map_.at(object_id);
}

FeatureContainer::ObjectToFeatureMap::iterator
FeatureContainer::beginObjectIterator() {
  return object_feature_map_.begin();
}
FeatureContainer::ObjectToFeatureMap::iterator
FeatureContainer::endObjectIterator() {
  return object_feature_map_.end();
}

FeatureContainer::ObjectToFeatureMap::const_iterator
FeatureContainer::beginObjectIterator() const {
  return object_feature_map_.begin();
}
FeatureContainer::ObjectToFeatureMap::const_iterator
FeatureContainer::endObjectIterator() const {
  return object_feature_map_.end();
}

FeatureContainer::FastUsableObjectIterator FeatureContainer::usableIterator(
    ObjectId object_id) {
  if (hasObject(object_id)) {
    return FastUsableObjectIterator(object_feature_map_.at(object_id));
  } else {
    return FastUsableObjectIterator(empty_object_feature_view_);
  }
}

FeatureContainer::ConstFastUsableObjectIterator
FeatureContainer::usableIterator(ObjectId object_id) const {
  if (hasObject(object_id)) {
    return ConstFastUsableObjectIterator(object_feature_map_.at(object_id));
  } else {
    return ConstFastUsableObjectIterator(empty_object_feature_view_);
  }
}

// TODO: else if logic needs test!!
std::vector<cv::Point2f> FeatureContainer::toOpenCV(TrackletIds* tracklet_ids,
                                                    bool only_inliers) const {
  if (tracklet_ids) tracklet_ids->clear();

  std::vector<cv::Point2f> keypoints_cv;
  for (const auto& feature : *this) {
    const Keypoint& kp = feature->keypoint();

    float x = static_cast<float>(kp(0));
    float y = static_cast<float>(kp(1));

    if (only_inliers && feature->usable()) {
      keypoints_cv.push_back(cv::Point2f(x, y));
      if (tracklet_ids) tracklet_ids->push_back(feature->trackletId());
    } else if (!only_inliers) {
      keypoints_cv.push_back(cv::Point2f(x, y));
      if (tracklet_ids) tracklet_ids->push_back(feature->trackletId());
    }
  }
  return keypoints_cv;
}

}  // namespace dyno
