#pragma once

// TODO: unfortunately for Jetpack 6 boost::unordered_flat_set is unavailable
// since we only
//  have access to boost 1.74 and boost > 1.8 is needed. Instead we fall back to
//  a 3rd party implementation these macros are inherited from dynosam_opt
#if DYNOSAM_USE_BOOST_FLAT_CONTAINERS
#include <boost/unordered/unordered_flat_map.hpp>
template <typename Key, typename Value>
using UnorderedFlatMap = boost::unordered_flat_map<Key, Value>;
#pragma message("Using boost::unordered_flat_map")
#elif DYNOSAM_HAS_PHMAP
#include <parallel_hashmap/phmap.h>
template <typename Key, typename Value>
using UnorderedFlatMap = phmap::parallel_flat_hash_map<Key, Value>;
#pragma message("Using phmap::parallel_flat_hash_map")
#else
#error "No supported unordered container implementation"
#endif

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "dynosam_common/SensorModels.hpp"
#include "dynosam_common/Types.hpp"

namespace dyno2 {

typedef std::size_t Index;
typedef std::pair<Index, Index> IndexMatch;
typedef std::vector<IndexMatch> IndexMatches;
typedef std::unordered_map<Index, Index> IndexMapping;

struct Velocity3 {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // Linear velocity [m/s]
  Eigen::Vector3d linear{0.0, 0.0, 0.0};
  // Angular velocity [rad/s]
  Eigen::Vector3d angular{0.0, 0.0, 0.0};
};

struct PoseVelocity3 {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  gtsam::Pose3 pose{};
  Velocity3 velocity{};
};

struct UpdatedStates {
  std::vector<dyno::FrameId> vo;
  std::vector<dyno::TemporalObjectId> objects;

  UpdatedStates& merge(const UpdatedStates other);
};

class LandmarkMap {
 public:
  // typedef boost::unordered_flat_map<dyno::TrackletId, dyno::Landmark>
  //     InternalData;
  typedef UnorderedFlatMap<dyno::TrackletId, dyno::Landmark> InternalData;

  LandmarkMap() = default;

  /// @brief Constructor with N reservations
  /// @param N
  LandmarkMap(size_t N);
  virtual ~LandmarkMap() = default;

  inline size_t numLandmarks() const noexcept { return data_.size(); }
  void reserveLandmarks(size_t N);
  void clearLandmarks();

  bool landmarkExists(dyno::TrackletId tracklet_id) const;
  const dyno::Landmark& getLandmark(dyno::TrackletId tracklet_id) const;

  bool setLandmark(dyno::TrackletId tracklet_id, const dyno::Landmark& lmk);
  bool removeLandmark(dyno::TrackletId tracklet_id);

  inline bool getLandmark(dyno::TrackletId id, dyno::Landmark& lmk) const {
    auto it = data_.find(id);
    if (it == data_.end()) {
      return false;
    }
    lmk = it->second;
    return true;
  }

  inline InternalData::const_iterator find(dyno::TrackletId id) const {
    return data_.find(id);
  }

  /* Walk over existing tracklets -> stored index from the beginning */
  inline InternalData::const_iterator begin() const { return data_.begin(); }

  inline InternalData::const_iterator end() const { return data_.end(); }

 private:
  InternalData data_;
};

struct MapPoint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  gtsam::Point3 p_W;
  dyno::ObjectId object_id;
  dyno::TrackletId id;
};

typedef dyno::AlignedVector<MapPoint> MapPointVector;
typedef dyno::AlignedMap<dyno::TrackletId, MapPoint> MapPoints;

typedef dyno::HeavyReferenceFrameValue<dyno::Pose3Measurement>
    HeavyPose3Measurement;

}  // namespace dyno2
