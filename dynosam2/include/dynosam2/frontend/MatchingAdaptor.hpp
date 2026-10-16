#pragma once

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>

#include "dynosam2/common/Types.hpp"
#include "dynosam2/frontend/FeatureBlockContainer.hpp"
#include "dynosam2/frontend/Frame.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_sensors/Camera.hpp"

namespace dyno2 {

/**
 * @brief Finds correspondants between a FeatureBlockGeometry and a reference
 * map. Similarily acts as an adatptor for different geometric solving
 * algorithms
 *
 */
class FrameAbsoluteAdaptor : public opengv::absolute_pose::AbsoluteAdapterBase {
 public:
  FrameAbsoluteAdaptor(const dyno::Camera::Ptr camera,
                       FeatureBlockGeometry& local_geometry,
                       const LandmarkMap& reference_geometry,
                       FeatureBlockContainer& features);

  virtual ~FrameAbsoluteAdaptor() = default;

  inline size_t getNumberCorrespondences() const override {
    return matches_.size();
  }

  inline const gtsam::Point2& keypointPrev(size_t i) const {
    return local_geometry_.left_kps_previous.at(localIndex(i));
  }

  inline const gtsam::Point2& keypoint(size_t i) const {
    return local_geometry_.left_kps.at(localIndex(i));
  }

  inline void keypoint(size_t i, const gtsam::Point2& keypoint) {
    auto local_index = localIndex(i);
    local_geometry_.left_kps[local_index] = keypoint;

    auto fc_index = local_geometry_.fc_indices[local_index];
    features_.points[fc_index] = dyno::utils::gtsamPointToCv<float>(keypoint);
  }

  inline void getCvKeypoint(size_t i, cv::Point2f& pt) const {
    auto fc_index = local_geometry_.fc_indices[localIndex(i)];
    pt = features_.points[fc_index];
  }

  inline dyno::TrackletId trackletId(size_t i) const {
    return local_geometry_.ids.at(localIndex(i));
  }

  inline const gtsam::Point3& landmark(size_t i) const {
    return reference_geometry_.landmarkAt(referenceIndex(i));
  }

  /// @brief Direct access to the inlier via pointer access for match i
  /// @param i
  /// @return
  uchar* inlierPtr(size_t i) const {
    auto fc_index = local_geometry_.fc_indices.at(localIndex(i));
    return &features_.inlier[fc_index];
  }

  inline dyno::ObjectId objectId(size_t i) const {
    auto fc_index = local_geometry_.fc_indices.at(localIndex(i));
    return features_.object_ids[fc_index];
  }

  inline bool isInlier(size_t i) const {
    auto fc_index = local_geometry_.fc_indices.at(localIndex(i));
    return static_cast<bool>(features_.inlier[fc_index]);
  }

  inline opengv::bearingVector_t getBearingVector(size_t index) const override {
    // bearings are made only for matches so we can index directly
    return bearings_[index];
  }
  inline double getWeight(size_t) const override { return 1.0; }
  inline opengv::translation_t getCamOffset(size_t) const override {
    return Eigen::Vector3d::Zero();
  }

  inline opengv::rotation_t getCamRotation(size_t) const override {
    return Eigen::Matrix3d::Identity();
  }

  inline opengv::point_t getPoint(size_t index) const override {
    return this->landmark(index);
  }

 private:
  inline Index localIndex(size_t i) const { return matches_.at(i).first; }
  inline Index referenceIndex(size_t i) const { return matches_.at(i).second; }

 private:
  FeatureBlockGeometry& local_geometry_;
  const LandmarkMap& reference_geometry_;
  FeatureBlockContainer& features_;

  //! Cached matched landmarks
  gtsam::Point3Vector matched_landmarks_ref_;
  gtsam::Point3Vector bearings_;

  // ! Index of matches between local geometry <-> reference geometry
  // IndexMatches matches_;
  std::unordered_map<Index, IndexMatch> matches_;
};

}  // namespace dyno2
