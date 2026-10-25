#include "dynosam2/common/Types.hpp"

#include "dynosam_common/Transforms.hpp"
#include "dynosam_common/Types.hpp"

namespace dyno2 {

LandmarkMap::LandmarkMap(size_t N) { reserve(N); }

void LandmarkMap::reserve(size_t N) {
  landmarks_.reserve(N);
  ids_.reserve(N);
  landmark_indices_.reserve(N);
}

void LandmarkMap::clear() {
  landmarks_.clear();
  ids_.clear();
  landmark_indices_.clear();
  is_contiguous_ = true;
}

bool LandmarkMap::landmarkExists(dyno::TrackletId tracklet_id) const {
  return landmark_indices_.find(tracklet_id) != landmark_indices_.end();
}
const dyno::Landmark& LandmarkMap::getLandmark(
    dyno::TrackletId tracklet_id) const {
  auto it = find(tracklet_id);
  CHECK(it != end()) << tracklet_id << " is missing!";
  return landmarks_[it->second];
}

bool LandmarkMap::setLandmark(dyno::TrackletId tracklet_id,
                              const dyno::Landmark& lmk) {
  auto it = landmark_indices_.find(tracklet_id);
  if (it == landmark_indices_.end()) {
    const Index new_idx = landmarks_.size();
    landmark_indices_[tracklet_id] = new_idx;
    landmarks_.push_back(lmk);
    ids_.push_back(tracklet_id);
    return true;
  } else {
    const Index index = it->second;
    landmarks_.at(index) = lmk;
    return false;
  }
}

LandmarkMap LandmarkMap::transformTo(const gtsam::Pose3& pose) {
  TrackletIndices indices = landmark_indices_;
  dyno::TrackletIds tracklet_ids = ids_;
  gtsam::Point3Vector landmarks;

  dyno::transformTo(pose, landmarks_, landmarks);
  return LandmarkMap(indices, landmarks, tracklet_ids, is_contiguous_);
}

}  // namespace dyno2

// void reduce();
