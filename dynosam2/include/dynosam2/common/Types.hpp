#pragma once

#include <unordered_map>
#include <utility>
#include <vector>

#include "dynosam_common/Types.hpp"

namespace dyno2 {

typedef std::size_t Index;
typedef std::pair<Index, Index> IndexMatch;
typedef std::vector<IndexMatch> IndexMatches;
typedef std::unordered_map<Index, Index> IndexMapping;

class LandmarkMap {
 public:
  typedef std::unordered_map<dyno::TrackletId, Index> TrackletIndices;
  LandmarkMap() = default;

  /// @brief Constructor with N reservations
  /// @param N
  LandmarkMap(size_t N);
  virtual ~LandmarkMap() = default;

  // NOTE: the size will be somewhat misleading when is_contiguous_ is false
  // but we need the size of the vector to iterate over all landmarks even if
  // some 'dont exist'
  inline size_t size() const noexcept { return landmarks_.size(); }
  void reserve(size_t N);

  bool landmarkExists(dyno::TrackletId tracklet_id) const;
  const dyno::Landmark& getLandmark(dyno::TrackletId tracklet_id) const;

  bool setLandmark(dyno::TrackletId tracklet_id, const dyno::Landmark& lmk);

  // fast transform of landmarks p = (R*p + t)
  // TODO: flag to indicate whether we should reduce before transforming
  // so the returned map is is_contiguous_
  LandmarkMap transformTo(const gtsam::Pose3& pose);

  inline std::pair<dyno::TrackletId, dyno::Landmark> at(Index index) const {
    return {ids_[index], landmarks_[index]};
  }

  const dyno::Landmark& landmarkAt(Index index) const {
    return landmarks_[index];
  }
  const gtsam::Point3Vector& landmarks() const { return landmarks_; }
  const dyno::TrackletIds& ids() const { return ids_; }

  inline TrackletIndices::const_iterator find(dyno::TrackletId id) const {
    return landmark_indices_.find(id);
  }

  /* Walk over existing tracklets -> stored index from the beginning */
  inline TrackletIndices::const_iterator begin() const {
    return landmark_indices_.begin();
  }

  inline TrackletIndices::const_iterator end() const {
    return landmark_indices_.end();
  }

  /**
   * @brief If true indicates that all contiguously allocated landmarks/ids are
   * valid and there has been no deletion i.e all landmarks_ appear in
   * landmark_indices_. However, when a deletion occurs, the index is removed
   * from landmark_indices_ but not necessarily from landmarks_/ids since this
   * is espensive and is_contiguous_ is marked as false. In this case, the user
   * must check landmarkExists to ensure that all landmarks are valid or call
   * reduce to reallocate memory to only the valid landmarks.
   *
   * @return true
   * @return false
   */
  inline bool isContiguous() const { return is_contiguous_; }

  /**
   * @brief Reduce the memory allocation to only the existing landmarks
   * i.e those that exist in landmark_indices_. After reduction isContiguous
   * will return true
   *
   */
  // void reduce();

 private:
  LandmarkMap(const TrackletIndices& indices, const gtsam::Point3Vector& lmks,
              const dyno::TrackletIds& tracklet_ids, const bool is_contiguous)
      : landmark_indices_(indices),
        landmarks_(lmks),
        ids_(tracklet_ids),
        is_contiguous_(is_contiguous) {}

 protected:
  TrackletIndices landmark_indices_;
  gtsam::Point3Vector landmarks_;
  dyno::TrackletIds ids_;

  //! Indicates that a deletion has taken place and not contiguously
  //! allocated points are valid (ie. a landmark in landmarks_ does NOT appear
  //! in landmark_indices_)
  bool is_contiguous_{true};
};

struct MapPoint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  gtsam::Point3 p_W;
  dyno::ObjectId object_id;
  dyno::TrackletId id;
};

typedef dyno::AlignedVector<MapPoint> MapPointVector;
typedef dyno::AlignedMap<dyno::TrackletId, MapPoint> MapPoints;

}  // namespace dyno2
