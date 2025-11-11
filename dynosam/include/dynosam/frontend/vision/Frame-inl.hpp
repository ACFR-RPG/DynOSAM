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

#pragma once

#include "dynosam/frontend/vision/Frame.hpp"

namespace dyno {

template <typename RefType, typename CurType>
bool Frame::getCorrespondences(
    GenericCorrespondences<RefType, CurType>& correspondences,
    const Frame& previous_frame, KeyPointType kp_type,
    const ConstructCorrespondanceFunc<RefType, CurType>& func) const {
  correspondences.clear();
  FeaturePairs feature_correspondences;
  const bool result =
      getCorrespondences(feature_correspondences, previous_frame, kp_type);
  if (!result) {
    return false;
  }

  for (const auto& pair : feature_correspondences) {
    const Feature::Ptr& prev_feature = pair.first;
    const Feature::Ptr& curr_feature = pair.second;

    CHECK(prev_feature);
    CHECK(curr_feature);

    CHECK_EQ(prev_feature->trackletId(), curr_feature->trackletId());
    CHECK_EQ(prev_feature->frameId(), previous_frame.getFrameId());
    CHECK_EQ(curr_feature->frameId(), getFrameId());

    correspondences.push_back(func(previous_frame, prev_feature, curr_feature));
  }
  return true;
}

template <typename RefType, typename CurType>
bool Frame::getDynamicCorrespondences(
    GenericCorrespondences<RefType, CurType>& correspondences,
    const Frame& previous_frame, ObjectId object_id,
    const ConstructCorrespondanceFunc<RefType, CurType>& func) const {
  FeaturePairs feature_correspondences;
  const bool result = getDynamicCorrespondences(feature_correspondences,
                                                previous_frame, object_id);
  if (!result) {
    return false;
  }

  // unncessary but just for sanity check
  for (const auto& feature_pairs : feature_correspondences) {
    const Feature::Ptr& prev_feature = feature_pairs.first;
    const Feature::Ptr& curr_feature = feature_pairs.second;

    CHECK_EQ(feature_pairs.first->objectId(), object_id);
    CHECK(feature_pairs.first->usable());

    CHECK_EQ(feature_pairs.second->objectId(), object_id);
    CHECK(feature_pairs.second->usable());

    CHECK_EQ(feature_pairs.second->trackletId(),
             feature_pairs.first->trackletId());

    correspondences.push_back(func(previous_frame, prev_feature, curr_feature));
  }
  return true;
}

}  // namespace dyno
