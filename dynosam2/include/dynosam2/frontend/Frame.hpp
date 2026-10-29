#pragma once

#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/StereoPoint2.h>

#include "dynosam2/common/Types.hpp"
#include "dynosam2/frontend/FeatureBlockContainer.hpp"
#include "dynosam_common/StructuredContainers.hpp"
#include "dynosam_common/Types.hpp"

using namespace dyno;

namespace dyno2 {

/**
 * @brief Intermediate data-structure representing the 3D geometry of a set of
 * featues currently in a FeatureBlockContainer.
 * This data-structure is used to hold tracking and geoemtry as stereo and is
 * updated during processing
 *
 */
struct FeatureBlockGeometry {
  gtsam::Point3Vector lmks_C;
  gtsam::Point2Vector left_kps;
  gtsam::Point2Vector left_kps_previous;
  //! Stereo correspondent in the right camera (x coordinate)
  std::vector<double> right_pixel;
  TrackletIds ids;
  //! Index in the corresponding parent FeatureBlockContainer
  std::vector<Index> fc_indices;
  //! Tracklet ids -> index for this set of vectors
  // ie. to get the lmk of tracklet id i -> lmks_C[local_indices[i]]
  std::unordered_map<TrackletId, Index> local_indices;

  inline Index getFeatureContainerIndex(TrackletId tracklet_id) const {
    return fc_indices.at(local_indices.at(tracklet_id));
  }

  inline const gtsam::Point3& getLandmark(TrackletId tracklet_id) const {
    return lmks_C.at(local_indices.at(tracklet_id));
  }

  inline gtsam::StereoPoint2 getStereoPoint(TrackletId tracklet_id) const {
    Index local_index = local_indices.at(tracklet_id);
    const auto left_kp = left_kps.at(local_index);
    const auto uR = right_pixel.at(local_index);
    return gtsam::StereoPoint2(left_kp(0), uR, left_kp(1));
  }
};
typedef dyno::FastUnorderedMap<ObjectId, FeatureBlockGeometry>
    FeatureBlockGeometryMap;

/**
 * @brief All (valid) stereo measurements taken for a single frame.
 * Constructed from the intermediate FeatureBlockGeometry which represents
 * all raw left keypoints (and depth) measurements prior to
 * construction of the local geometry (ie the 3D point map).
 * The final (ie refined and inlier) 3D geometry is then used to generate a
 * Frame
 *
 *
 */
struct Frame {
  DYNO_POINTER_TYPEDEFS(Frame)

  TrackletIds ids;
  gtsam::Point3Vector lmks_C;
  std::vector<gtsam::StereoPoint2> measurements;
  std::unordered_map<TrackletId, Index> local_indices;

  FrameId frame_id;
  Timestamp timestamp;
  ObjectId object_id;

  // Probavly make protected?
  Frame(FrameId fi = 0, Timestamp ts = 0, ObjectId o = 0)
      : frame_id(fi), timestamp(ts), object_id(o) {}
  virtual ~Frame() = default;

  inline FrameId getFrameId() const { return frame_id; }
  inline Timestamp getTimestamp() const { return timestamp; }
  inline ObjectId getObjectId() const { return object_id; }
  inline dyno::TemporalObjectId temporalObjectId() const {
    return dyno::TemporalObjectId(frame_id, object_id);
  }

  size_t numKeypoints() const noexcept { return ids.size(); }

  inline void reserve(size_t n) {
    ids.reserve(n);
    lmks_C.reserve(n);
    measurements.reserve(n);
  }

  inline void getKeypointByIndex(Index index, Keypoint& keypoint) const {
    keypoint = measurements[index].point2();
  }

  inline void getCvKeypointByIndex(Index index, cv::Point2f& keypoint) const {
    keypoint = utils::gtsamPointToCv<float>(measurements[index].point2());
  }

  inline void getPointByIndex(Index index, gtsam::Point3& point) const {
    point = lmks_C.at(local_indices.at(index));
  }

  inline dyno::Keypoint keypointById(TrackletId id) const {
    return measurements.at(local_indices.at(id)).point2();
  }

  inline const gtsam::StereoPoint2& stereoPointById(TrackletId id) const {
    return measurements.at(local_indices.at(id));
  }

  /* Indicates that the landmark was observed */
  inline bool observedLandmark(TrackletId id) const {
    return local_indices.find(id) != local_indices.end();
  }

  void fill(const FeatureBlockGeometry& local_geometry,
            const FeatureBlockContainer& features) {
    size_t n = local_geometry.ids.size();
    reserve(n);

    for (size_t i = 0; i < n; i++) {
      TrackletId id = local_geometry.ids[i];
      Index fc_index = local_geometry.getFeatureContainerIndex(id);

      if (!features.inlier[fc_index]) {
        continue;
      }

      size_t index = numKeypoints();
      ids.push_back(id);
      lmks_C.push_back(local_geometry.getLandmark(id));
      measurements.push_back(local_geometry.getStereoPoint(id));
      local_indices[id] = index;
    }
  }
};

class ViFrame : public Frame {
 public:
  DYNO_POINTER_TYPEDEFS(ViFrame)
  ViFrame(FrameId fi, Timestamp ts) : Frame(fi, ts, dyno::background_label) {}

  // TODO: contain IMU + T_ij measurement
  const Motion3ReferenceFrame& getRelativePose() const { return T_ij_; }
  void setRelativePose(const Motion3ReferenceFrame& T_ij);

 private:
  //! Measured relative camera moition
  Motion3ReferenceFrame T_ij_;
};

class DoFrame : public Frame {
 public:
  DYNO_POINTER_TYPEDEFS(DoFrame)
  DoFrame(FrameId fi, Timestamp ts, ObjectId o) : Frame(fi, ts, o) {}

  // contain G_i_ij/T_i_ij measurement!
};

}  // namespace dyno2
