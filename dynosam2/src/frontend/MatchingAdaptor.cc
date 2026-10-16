#include "dynosam2/frontend/MatchingAdaptor.hpp"

#include "dynosam2/frontend/Utils.hpp"

namespace dyno2 {

FrameAbsoluteAdaptor::FrameAbsoluteAdaptor(
    const dyno::Camera::Ptr camera, FeatureBlockGeometry& local_geometry,
    const LandmarkMap& reference_geometry, FeatureBlockContainer& features)
    : local_geometry_(local_geometry),
      reference_geometry_(reference_geometry),
      features_(features) {
  const size_t num_features = local_geometry.ids.size();
  matched_landmarks_ref_.reserve(num_features);
  bearings_.reserve(num_features);

  ComputeBearing compute_bearing(camera->getParams());

  size_t adaptor_index{0};
  for (size_t curr_idx = 0; curr_idx < num_features; curr_idx++) {
    const TrackletId tracklet_id = local_geometry.ids[curr_idx];
    const Index fc_curr_idx = local_geometry.fc_indices[curr_idx];

    size_t age = features_.age[fc_curr_idx];
    // seen at least twice (ie tracked from previous to current and not a new
    // feature!)
    if (age < 2) {
      continue;
    }

    if (!features_.inlier[fc_curr_idx]) {
      continue;
    }

    auto it = reference_geometry.find(tracklet_id);
    // we have a match in the local map!
    if (it != reference_geometry.end()) {
      // check that the landmarks have the same id!
      const Index ref_index = it->second;
      matches_[adaptor_index] = std::make_pair(curr_idx, ref_index);
      matched_landmarks_ref_.push_back(
          reference_geometry_.landmarkAt(ref_index));

      const gtsam::Point2& kp = local_geometry_.left_kps[curr_idx];
      bearings_.push_back(compute_bearing(kp));
      ++adaptor_index;
    }
  }
}

}  // namespace dyno2
