#include "dynosam2/common/Types.hpp"

#include "dynosam_common/Transforms.hpp"
#include "dynosam_common/Types.hpp"

namespace dyno2 {

UpdatedStates& UpdatedStates::merge(const UpdatedStates other) {
  vo.insert(vo.begin(), other.vo.begin(), other.vo.end());
  objects.insert(objects.begin(), other.objects.begin(), other.objects.end());
  return *this;
}

LandmarkMap::LandmarkMap(size_t N) { reserveLandmarks(N); }

void LandmarkMap::reserveLandmarks(size_t N) { data_.reserve(N); }

void LandmarkMap::clearLandmarks() { data_.clear(); }

bool LandmarkMap::landmarkExists(dyno::TrackletId tracklet_id) const {
  return data_.find(tracklet_id) != data_.end();
}
const dyno::Landmark& LandmarkMap::getLandmark(
    dyno::TrackletId tracklet_id) const {
  auto it = find(tracklet_id);
  CHECK(it != end()) << tracklet_id << " is missing!";
  return it->second;
}

bool LandmarkMap::setLandmark(dyno::TrackletId tracklet_id,
                              const dyno::Landmark& lmk) {
  const auto [_, inserted] = data_.insert_or_assign(tracklet_id, lmk);
  return inserted;
}

}  // namespace dyno2

// void reduce();
