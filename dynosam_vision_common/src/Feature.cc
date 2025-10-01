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

#include "dynosam_vision_common/Feature.hpp"

#include <glog/logging.h>

namespace dyno {

FeatureContainer::FeatureContainer() : feature_map_() {}

FeatureContainer::FeatureContainer(const FeaturePtrs feature_vector) {
  for (size_t i = 0; i < feature_vector.size(); i++) {
    add(feature_vector.at(i));
  }
}

void FeatureContainer::add(const Feature& feature) {
  auto feature_ptr = std::make_shared<Feature>(feature);
  add(feature_ptr);
}

void FeatureContainer::add(Feature::Ptr feature) {
  CHECK(!exists(feature->trackletId()))
      << "Feailure in FeatureContainer::add - Tracklet Id "
      << feature->trackletId() << " already exists";
  feature_map_[feature->trackletId()] = feature;

  // TODO: what if object id is not valid!
  if (!object_feature_map_.exists(feature->objectId())) {
    object_feature_map_.insert2(feature->objectId(),
                                std::unordered_set<TrackletId>{});
  }
  object_feature_map_[feature->objectId()].insert(feature->trackletId());
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
  auto& tracklets_per_object = object_feature_map_.at(object_id);
  tracklets_per_object.erase(tracklet_id);

  if (tracklets_per_object.empty()) {
    object_feature_map_.erase(object_id);
  }

  // remove from main feature map
  feature_map_.erase(tracklet_id);
}

void FeatureContainer::removeByObjectId(ObjectId object_id) {
  // collect all tracklets
  auto itr = FilterIterator(*this, [object_id](const Feature::Ptr& f) -> bool {
    return f->objectId() == object_id;
  });

  TrackletIds tracklets_to_remove;
  for (const auto& feature : itr) {
    CHECK_EQ(feature->objectId(), object_id);
    tracklets_to_remove.push_back(feature->trackletId());
  }

  for (const auto tracklet_id : tracklets_to_remove) {
    this->remove(tracklet_id);
  }

  // TODO: remove from object_feature_map_
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
  auto itr =
      ConstFilterIterator(*this, [object_id](const Feature::Ptr& f) -> bool {
        return f->objectId() == object_id;
      });

  return static_cast<size_t>(std::distance(itr.begin(), itr.end()));
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
    const auto& tracklets = object_feature_map_.at(object_id);
    return TrackletIds(tracklets.begin(), tracklets.end());
  } else {
    return TrackletIds{};
  }
}

FeatureContainer::FilterIterator FeatureContainer::beginUsable() {
  return FilterIterator(*this, [](const Feature::Ptr& f) -> bool {
    return Feature::IsUsable(f);
  });
}

FeatureContainer::FilterIterator FeatureContainer::beginUsable() const {
  FeatureContainer& t = const_cast<FeatureContainer&>(*this);
  return FilterIterator(
      t, [](const Feature::Ptr& f) -> bool { return Feature::IsUsable(f); });
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
