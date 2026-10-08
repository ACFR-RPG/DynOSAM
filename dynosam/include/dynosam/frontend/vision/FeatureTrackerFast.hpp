#pragma once

#include <tbb/task_group.h>

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>

#include "dynosam/frontend/FrontendParams.hpp"
#include "dynosam/frontend/vision/FeatureTrackerBase.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam/frontend/vision/StaticFeatureTracker.hpp"
#include "dynosam_nn/ObjectDetector.hpp"
#include "dynosam_sensors/Camera.hpp"
#include "dynosam_sensors/Feature.hpp"

namespace dyno {

typedef std::size_t Index;
typedef std::pair<Index, Index> IndexMatch;
typedef std::vector<IndexMatch> IndexMatches;
typedef std::unordered_map<Index, Index> IndexMapping;

// TODO: can do this (?) as a vector operation using unart expression and Batch
// operator
//  to a Eigen::Map of Point's for extra speed!
/**
 * @brief Super fast implementation of K^{-1} * kp, returning normalized bearing
 * vector. Basic benchmarking shows at least a 7x speed up over the raw eigen
 * version: (K_inv * gtsam::Vector3(kp(0), kp(1), 1.0)).normzlied().
 *
 * Implementation takes advantage of the structure of the K matrix.
 *
 * @param fx
 * @param fy
 * @param cx
 * @param cy
 * @param kp
 * @return gtsam::Vector3
 */
inline gtsam::Vector3 bearingOptimized(double fx, double fy, double cx,
                                       double cy, const gtsam::Point2& kp) {
  const double nx = (kp.x() - cx) / fx;
  const double ny = (kp.y() - cy) / fy;

  const double inv_norm = 1.0 / std::sqrt(nx * nx + ny * ny + 1.0);

  return gtsam::Vector3(nx * inv_norm, ny * inv_norm, inv_norm);
}

inline bool checkBounds(const cv::Point2f& point, int rows, int cols) {
  const int x = cvRound(point.x);
  const int y = cvRound(point.y);
  return x >= 0 && x < cols && y >= 0 && y < rows;
}

inline bool checkBoundsAndLabel(const cv::Point2f& point,
                                const cv::Mat& object_masks,
                                ObjectId expected_id, int rows, int cols) {
  const int x = cvRound(point.x);
  const int y = cvRound(point.y);
  return x >= 0 && x < cols && y >= 0 && y < rows &&
         object_masks.at<ObjectId>(y, x) == expected_id;
}

// should be of type double and of size 3xN
template <typename Derived>
inline void transformTo(const gtsam::Pose3& T_ij,
                        Eigen::MatrixBase<Derived>& P) {
  static_assert(std::is_same_v<typename Derived::Scalar, double>);
  static_assert(Derived::RowsAtCompileTime == 3);

  const gtsam::Matrix33& R = T_ij.rotation().matrix();
  const gtsam::Vector3& t = T_ij.translation();

  P = R * P;
  P.colwise() += t;
}

/**
 * @brief Super fast batch implementation of pose * points using Eigen map
 * operations rather than linear operations.
 *
 * From simple benchmarks we get 60% improvements in speed vs a loop for vectors
 * as small as 200!
 *
 * @param T_ij transform from j to i (that is p_i = T_ij * p_j)
 * @param points
 * @return gtsam::Point3Vector
 */
inline void transformTo(const gtsam::Pose3& T_ij,
                        const gtsam::Point3Vector& points_j,
                        gtsam::Point3Vector& points_i) {
  if (points_j.empty()) {
    return;
  }
  points_i = points_j;
  Eigen::Map<Eigen::Matrix3Xd> P(points_i[0].data(), 3,
                                 static_cast<Eigen::Index>(points_i.size()));

  transformTo(T_ij, P);
}

class FeatureBlockContainer {
 public:
  struct FeatureData {
    std::vector<cv::Point2f> points;
    std::vector<cv::Point2f> previous_points;
    std::vector<TrackletId> ids;
    std::vector<size_t> age;
    // TODO: actually dont need inliers as part of feature data
    //  as we assume all provided featues ARE inliers!
    std::vector<uchar> inlier;
    std::vector<float> errors;

    inline size_t size() const { return points.size(); }
    inline bool empty() const { return points.empty(); }
    void checkSizes() const;
    void reserve(size_t n);
    void resize(size_t n);
  };

  /**
   * @brief Specifies the number of features for an object id.
   *
   * Used to generated a block of contiguous memory
   *
   */
  struct BlockDim {
    ObjectId object_id;
    size_t size{0};
  };

 private:
  /**
   * @brief Internal storage for memory layout
   *
   */
  struct BlockLayout {
    ObjectId object_id;
    size_t begin{0};
    size_t end{0};

    size_t size() const { return end - begin; }
  };

 public:
  /**
   * @brief View to a block of features pertaining to a single object.
   *
   * Access is via raw pointers for speed.
   *
   * NOTE: After doing performance profiling it is very important to keep all
   * these functions in the header file to make them inline. We loose about 12%
   * performance on access compared to direct SoA access if they are not inline!
   *
   */
  class BlockView {
   public:
    inline size_t size() const { return layout_.size(); }
    inline ObjectId objectId() const { return layout_.object_id; }

    inline cv::Point2f* points() {
      return features_->points.data() + layout_.begin;
    }
    inline const cv::Point2f* points() const {
      return features_->points.data() + layout_.begin;
    }

    // potentiall dangerous as we could modify the points mat!
    inline cv::Mat pointsMat() {
      return cv::Mat(static_cast<int>(size()), 1, CV_32FC2, points());
    }
    // dangerous as not actually const as cv::Mat will mantain a non-const
    // pointer to the raw data
    inline cv::Mat pointsMat() const {
      return cv::Mat(static_cast<int>(size()), 1, CV_32FC2,
                     const_cast<cv::Point2f*>(points()));
    }

    inline cv::Point2f* previousPoints() {
      return features_->previous_points.data() + layout_.begin;
    }
    inline const cv::Point2f* previousPoints() const {
      return features_->previous_points.data() + layout_.begin;
    }

    inline TrackletId* ids() { return features_->ids.data() + layout_.begin; }
    inline const TrackletId* ids() const {
      return features_->ids.data() + layout_.begin;
    }

    inline ObjectId* objectIds() {
      return features_->object_ids.data() + layout_.begin;
    }
    inline const ObjectId* objectIds() const {
      return features_->object_ids.data() + layout_.begin;
    }

    inline uchar* inlier() { return features_->inlier.data() + layout_.begin; }
    inline const uchar* inlier() const {
      return features_->inlier.data() + layout_.begin;
    }

    inline float* errors() { return features_->errors.data() + layout_.begin; }
    inline const float* errors() const {
      return features_->errors.data() + layout_.begin;
    }

    inline size_t* age() { return features_->age.data() + layout_.begin; }
    inline const size_t* age() const {
      return features_->age.data() + layout_.begin;
    }

    inline const BlockLayout& layout() const { return layout_; }

    // ---------------------------------------------------------------------
    // Convenient bulk assignment
    // ---------------------------------------------------------------------

    void copyFrom(const FeatureData& data);

   private:
    friend class FeatureBlockContainer;

    BlockView(FeatureBlockContainer* features, ObjectId object_id, size_t begin,
              size_t end)
        : features_(features), layout_{object_id, begin, end} {}

    BlockView(FeatureBlockContainer* features, const BlockLayout& layout)
        : features_(features), layout_(layout) {}

    template <typename T>
    static void copyBlock(T* destination, const T* source, size_t count) {
      static_assert(std::is_trivially_copyable<T>::value,
                    "BlockView fields must be trivially copyable");

      if (count > 0) {
        std::memcpy(destination, source, count * sizeof(T));
      }
    }

    // in reality might be pointer to const FeatureSet.
    // TODO: redesign with template as before
    FeatureBlockContainer* features_;
    // Layout for this feature blocks
    BlockLayout layout_;
  };

  // empty initaliser
  FeatureBlockContainer() {}

  FeatureBlockContainer(std::initializer_list<BlockDim> specs);
  explicit FeatureBlockContainer(const std::vector<BlockDim>& specs);

  //@tparam TERMS A container whose value type is std::pair<ObjectId,
  // FeatureData>
  template <typename TERMS>
  explicit FeatureBlockContainer(const TERMS& terms) {
    std::vector<BlockDim> specs;
    specs.reserve(terms.size());
    for (typename TERMS::const_iterator it = terms.begin(); it != terms.end();
         ++it) {
      const auto& term = *it;

      const ObjectId object_id = term.first;
      const FeatureData& data = term.second;

      data.checkSizes();
      specs.push_back({object_id, data.size()});
    }

    initialize(specs.begin(), specs.end());

    for (typename TERMS::const_iterator it = terms.begin(); it != terms.end();
         ++it) {
      const auto& term = *it;
      const FeatureData& source = term.second;

      BlockView destination = objectView(term.first);
      destination.copyFrom(source);
    }

    checkInvariants();
  }

  /// @brief Number of total features (all blocks)
  /// @return size_t
  size_t size() const;

  /// @brief Number of memory blocks representing number of objects stored
  /// @return
  size_t objectCount() const;
  /**
   * @brief Get views for all feature blocks, one for each object
   *
   * @return std::vector<BlockView>
   */
  std::vector<BlockView> objectViews() const;

  /**
   * @brief If an object exists in the container.
   *
   * (ie. a block of contiguous memory has been allocated for this object)
   *
   * @param object_id ObjectId
   * @return true
   * @return false
   */
  bool containsObject(ObjectId object_id) const;

  FeatureBlockContainer merge(const FeatureBlockContainer& other) const;

  BlockView objectView(ObjectId object_id);
  const BlockView objectView(ObjectId object_id) const;

  // mapping is from the original container to the new container!
  FeatureBlockContainer reduceToInliers(
      IndexMapping* index_mapping = nullptr) const;
  FeatureBlockContainer& reduceToInliersInplace(
      IndexMapping* index_mapping = nullptr);

  std::vector<cv::Point2f> points;
  // TODO: not sure if we actually want previous points!
  std::vector<cv::Point2f> previous_points;
  std::vector<TrackletId> ids;
  std::vector<size_t> age;
  std::vector<ObjectId> object_ids;
  std::vector<uchar> inlier;
  std::vector<float> errors;

  std::string debugInfoString() const;

  void checkInvariants() const;

 private:
  template <typename T>
  static void copyBlock(T* destination, const T* source, size_t count) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "FeatureBlockContainer fields must be trivially copyable");

    if (count > 0) {
      std::memcpy(destination, source, count * sizeof(T));
    }
  }

  static void copyFeatures(BlockView destination, const BlockView& source,
                           size_t destination_offset = 0);

  // =========================================================================
  // Layout construction
  // =========================================================================

  template <typename Iterator>
  void initialize(Iterator begin, Iterator end) {
    const size_t object_count = static_cast<size_t>(std::distance(begin, end));

    block_layout_.reserve(object_count);
    object_lookup_.reserve(object_count);

    size_t total_size = 0;

    for (Iterator it = begin; it != end; ++it) {
      const BlockDim& spec = *it;

      if (object_lookup_.find(spec.object_id) != object_lookup_.end()) {
        throw std::invalid_argument(
            "FeatureBlockContainer: duplicate object ID " +
            std::to_string(spec.object_id));
      }

      const size_t object_begin = total_size;
      const size_t object_end = total_size + spec.size;

      object_lookup_.emplace(spec.object_id, block_layout_.size());

      block_layout_.push_back(
          BlockLayout{spec.object_id, object_begin, object_end});

      total_size = object_end;
    }

    points.resize(total_size);
    previous_points.resize(total_size);
    ids.resize(total_size);
    age.resize(total_size);
    object_ids.resize(total_size);
    inlier.resize(total_size);
    errors.resize(total_size);

    // TODO: add depth!

    // Initialise object IDs immediately so the FeatureSet is valid even
    // before the caller fills the feature data.
    for (const BlockLayout& object : block_layout_) {
      std::fill(object_ids.begin() + object.begin,
                object_ids.begin() + object.end, object.object_id);
    }

    checkInvariants();
  }

  // TODO: change name!
  const BlockLayout& objectMetadata(ObjectId object_id) const {
    auto it = object_lookup_.find(object_id);

    if (it == object_lookup_.end()) {
      throw std::out_of_range("FeatureSet: unknown object ID " +
                              std::to_string(object_id));
    }

    return block_layout_[it->second];
  }

  //! Memory layout of each block, where each block represents contiguous memory
  //! block per object
  std::vector<BlockLayout> block_layout_;
  //! Index of object id -> index in the block_layout vector
  std::unordered_map<ObjectId, size_t> object_lookup_;
};

/// @brief Alias to FeatureBlockContainer::BlockDim
using FeatureBlockDim = FeatureBlockContainer::BlockDim;
/// @brief Alias to FeatureBlockContainer::BlockView
using FeatureBlockView = FeatureBlockContainer::BlockView;

cv::Mat drawBatchedFeatures(const cv::Mat& image,
                            const FeatureBlockContainer& batchedFeatures);

// local map structurew mantained for geometric visual/object odometry tracking
// should only really contain the last N keyframes or the active tracks
//  add indicator of which frame was observed in so we can access observations
//  of it!
// TODO: tbh this just could be a map dirctly since we only ever access (I
// think)
//  we dont need contiguous memory
//  becuase at some point we need to delete lmks (defintiely)
//  and this is going to be slow compared to a map!
class LandmarkMap {
 public:
  typedef dyno::FastUnorderedMap<TrackletId, Index> TrackletIndices;

  LandmarkMap() = default;
  virtual ~LandmarkMap() = default;

  void setLandmark(TrackletId tracklet_id, const Landmark& lmk) {
    auto it = landmark_indices_.find(tracklet_id);
    if (it == landmark_indices_.end()) {
      Index new_idx = landmarks_.size();
      landmark_indices_[tracklet_id] = new_idx;
      landmarks_.push_back(lmk);
      ids_.push_back(tracklet_id);
    } else {
      Index index = it->second;
      landmarks_.at(index) = lmk;
    }
  }

  inline size_t size() const { return landmarks_.size(); }

  bool landmarkExists(TrackletId tracklet_id) const {
    return landmark_indices_.exists(tracklet_id);
  }

  const Landmark& getLandmark(TrackletId tracklet_id) const {
    return landmarks_[landmark_indices_.at(tracklet_id)];
  }

  // index is between 0 and size() - 1 and is the index of the stored position
  // if the landmark
  inline const Landmark& getLandmarkByIndex(Index index) const {
    return landmarks_[index];
  }

  inline TrackletId getIdByIndex(Index index) const { return ids_[index]; }

  const gtsam::Point3Vector& getLandmarks() const { return landmarks_; }
  const TrackletIds& getTrackletIds() const { return ids_; }

  // fast transform of landmarks p = P*p (R*p + t)
  LandmarkMap transformTo(const gtsam::Pose3& pose) const {
    TrackletIndices indices = landmark_indices_;
    TrackletIds tracklet_ids = ids_;
    gtsam::Point3Vector landmarks;

    dyno::transformTo(pose, landmarks_, landmarks);
    return LandmarkMap(indices, landmarks, tracklet_ids);
  }

  inline TrackletIndices::const_iterator find(TrackletId id) const {
    return landmark_indices_.find(id);
  }

  /* Walk over the tracklets -> stored index from the beginning */
  inline TrackletIndices::const_iterator begin() const {
    return landmark_indices_.begin();
  }

  inline TrackletIndices::const_iterator end() const {
    return landmark_indices_.end();
  }

 private:
  LandmarkMap(const TrackletIndices& indices, const gtsam::Point3Vector& lmks,
              const TrackletIds& tracklet_ids)
      : landmark_indices_(indices), landmarks_(lmks), ids_(tracklet_ids) {}

 protected:
  TrackletIndices landmark_indices_;
  gtsam::Point3Vector landmarks_;
  TrackletIds ids_;
};

// from here onwards we operate in the land of doubles
// as all the geometric solvers operate using gtsam/egien double types!
// these must be separately synchronized/maintaied with the
// FeatureBlockContainer
// TODO: better name again is FeatureGeometry as this contains the referecnes to
// the
// feature container index. Therefor all other contains need to refer back to
// this when something in the current container updates. Previous feature
// containers should never need to be updated!
// could carry object ids here as well? but the point of the frame geometry
// is that it captures one object!
struct FrameGeometry {
  // note the use of double here
  // points in the camera frame
  // todo: eventually dont need this for all points as we will use the initial
  // from the local map - but as we need to comptute the right pixel anyway we
  // have to comptue the landmark somehow so might as well have it here!
  gtsam::Point3Vector lmks_C;
  gtsam::Point2Vector left_kps;
  gtsam::Point2Vector left_kps_previous;
  //! Stereo correspondent in the right camera (x coordinate)
  std::vector<double> right_pixel;
  TrackletIds ids;
  // Index value in the original FeatureBlockContainer
  // only valid while the contianer remains unchanged
  // after a container is reduced to inliers the fc_indices must be updatd
  // TODO: why we dont have view_indices here as well?
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

// A class that acts as an adaptor to match between the current frame geoemtry
// and a reference frame defined by a set of LandmarkMap
// holds references to the input objects so lifetime must be managed
class MatchingAdaptorBase {
 public:
  MatchingAdaptorBase(FrameGeometry& local_geometry,
                      const LandmarkMap& reference_geometry,
                      FeatureBlockContainer& features);

  virtual ~MatchingAdaptorBase() = default;

  inline size_t numMatches() const { return matches_.size(); }

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
    features_.points[fc_index] = utils::gtsamPointToCv<float>(keypoint);
  }

  /// @brief Reference landmark
  /// @param i
  /// @return
  inline const gtsam::Point3& landmark(size_t i) const {
    return reference_geometry_.getLandmarkByIndex(referenceIndex(i));
  }

  inline TrackletId trackletId(size_t i) const {
    return local_geometry_.ids.at(localIndex(i));
  }

  inline bool isInlier(size_t i) const {
    auto fc_index = local_geometry_.fc_indices.at(localIndex(i));
    return static_cast<bool>(features_.inlier[fc_index]);
  }

  inline const gtsam::Point3Vector& referenceLandmarks() const {
    return matched_landmarks_ref_;
  }

  /// @brief Direct access to the inlier via pointer access for match i
  /// @param i
  /// @return
  uchar* inlierPtr(size_t i) const {
    auto fc_index = local_geometry_.fc_indices.at(localIndex(i));
    return &features_.inlier[fc_index];
  }

  inline Index localIndex(size_t i) const { return matches_.at(i).first; }
  inline Index referenceIndex(size_t i) const { return matches_.at(i).second; }

  // TODO: pretty sure not used!
  /// @brief Recompute matches_ based on new inliers
  void recompute();

 protected:
  /// @brief Recompute any cached variables based on the updated matches_
  inline virtual void recomputeCache() {}

 protected:
  FrameGeometry& local_geometry_;
  const LandmarkMap& reference_geometry_;
  FeatureBlockContainer& features_;

  //! Cached matched landmarks
  gtsam::Point3Vector matched_landmarks_ref_;

  // ! Index of matches between local geometry <-> reference geometry
  // IndexMatches matches_;
  std::unordered_map<Index, IndexMatch> matches_;
};

class OpenGVCentralAbsolutePoseAdaptor
    : public opengv::absolute_pose::AbsoluteAdapterBase,
      public MatchingAdaptorBase {
 public:
  OpenGVCentralAbsolutePoseAdaptor(const Camera::Ptr camera,
                                   FrameGeometry& local_geometry,
                                   const LandmarkMap& reference_geometry,
                                   FeatureBlockContainer& features)
      : opengv::absolute_pose::AbsoluteAdapterBase(),
        MatchingAdaptorBase(local_geometry, reference_geometry, features),
        camera_(camera) {
    const auto& camera_params = camera_->getParams();
    const double fx = camera_params.fx();
    const double fy = camera_params.fy();
    const double cx = camera_params.cu();
    const double cy = camera_params.cv();

    size_t num_matches = this->numMatches();
    bearings_local_.reserve(num_matches);
    for (size_t i = 0; i < num_matches; i++) {
      bearings_local_.push_back(
          bearingOptimized(fx, fy, cx, cy, this->keypoint(i)));
    }
  }

  virtual ~OpenGVCentralAbsolutePoseAdaptor() override = default;

  inline opengv::bearingVector_t getBearingVector(size_t index) const override {
    return bearings_local_[index];
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

  inline size_t getNumberCorrespondences() const override {
    return this->numMatches();
  }

 private:
  Camera::Ptr camera_;
  gtsam::Point3Vector bearings_local_;
};

class ViFrame {
 public:
  TrackletIds ids;
  gtsam::Point3Vector lmks_C;
  std::vector<gtsam::StereoPoint2> measurements;
  std::unordered_map<TrackletId, Index> local_indices;

  size_t numKeypoints() const noexcept { return ids.size(); }

  inline void getKeypointByIndex(Index index, Keypoint& keypoint) const {
    keypoint = measurements[index].point2();
  }

  inline void getCvKeypointByIndex(Index index, cv::Point2f& keypoint) const {
    keypoint = utils::gtsamPointToCv<float>(measurements[index].point2());
  }

  /* Indicates that the landmark was observed */
  inline bool observedLandmark(TrackletId id) const {
    return local_indices.find(id) != local_indices.end();
  }

  FrameId frame_id;
  Timestamp timestamp;
};
typedef dyno::FastUnorderedMap<FrameId, ViFrame> ViFrames;

typedef gtsam::FastMap<ObjectId, FrameGeometry> FrameGeometryMap;

class DepthUpdaterFast {
 public:
  //! need more than 8 points for fundamental matrix calc with ransac
  constexpr static size_t kMinStereoMatches{8};

  DepthUpdaterFast(const DepthThresholds& params, Camera::Ptr camera,
                   const ImageContainer& images,
                   FeatureBlockContainer& features);

  // TODO: return outliers and mark features separately!
  void calcPoints(FrameGeometryMap& point_map);
  // void calcPoints(const std::vector<Index>& indicies,
  //                 FrameGeometryMap& point_map);

  // void updateGeometry(FrameGeometry& local_geometry, const
  // std::vector<Index>& local_indices);

  // update geoemtry based on new pixel location in the provided feature
  // geometry
  void updateGeometry(FrameGeometry& local_geometry);

 private:
  void calcPointsRGBD(FrameGeometryMap& point_map);
  void calcPointsStereo(FrameGeometryMap& point_map);

  void updateGeometryRGBD(FrameGeometry& local_geometry);
  void updateGeometryStereo(FrameGeometry& local_geometry);

  inline bool checkTwoViewGeometry(const gtsam::Point3& point_left,
                                   const gtsam::Point2& left_kp,
                                   const gtsam::Point2& right_kp,
                                   double max_reprojection_error = 2.0,
                                   double min_parallax = 1.0 * M_PI / 180.0) {
    if (!point_left.allFinite() || point_left.z() <= 0.0) {
      return false;
    }

    if (!std::isfinite(fx_) || !std::isfinite(fy_) || fx_ <= 0.0 ||
        fy_ <= 0.0 || !std::isfinite(baseline_) || baseline_ <= 0.0) {
      throw DynosamException(
          "checkTwoViewGeometry failed: invalid camera params!");
    }

    // Left-camera reprojection.
    const double u_left = fx_ * point_left.x() / point_left.z() + cu_;

    const double v_left = fy_ * point_left.y() / point_left.z() + cv_;

    double reprojection_error_left =
        std::hypot(u_left - left_kp.x(), v_left - left_kp.y());

    // Right-camera point.
    const Eigen::Vector3d point_right(point_left.x() - baseline_,
                                      point_left.y(), point_left.z());

    if (!point_right.allFinite() || point_right.z() <= 0.0) {
      return false;
    }

    const double u_right = fx_ * point_right.x() / point_right.z() + cu_;

    const double v_right = fy_ * point_right.y() / point_right.z() + cv_;

    double reprojection_error_right =
        std::hypot(u_right - right_kp.x(), v_right - right_kp.y());

    return reprojection_error_left <= max_reprojection_error &&
           reprojection_error_right <= max_reprojection_error;
  }

  inline bool checkStereoDepth(double disparity, double depth,
                               ObjectId object_id, double min_depth = 0.1,
                               double min_disparity = 1.0) const {
    const Depth max_depth = (object_id == background_label)
                                ? params_.max_background
                                : params_.max_object;

    return std::isfinite(disparity) && disparity >= min_disparity &&
           std::isfinite(depth) && depth > min_depth && depth <= max_depth;
  }

  inline bool checkRGBDDepth(Depth depth, ObjectId object_id,
                             double min_depth = 0.1) const {
    const Depth max_depth = (object_id == background_label)
                                ? params_.max_background
                                : params_.max_object;

    return std::isfinite(depth) && depth >= min_depth && depth <= max_depth;
  }

  // assume all are part of the same object and t
  // matched* vectors will all have the same size (may be < left_kps)
  // as well matched_indexs and will say which index in the original left_kps
  // the matching is for!
  bool computeStereoMatching(const std::vector<cv::Point2f>& left_kps,
                             std::vector<cv::Point2f>& matched_right_kps,
                             std::vector<size_t>& matched_indexs);

  DepthThresholds params_;
  Camera::Ptr camera_;
  ImageContainer images_;
  FeatureBlockContainer& features_;

  // cached camera paramters
  double fx_, fy_, cu_, cv_;
  double baseline_;
};

// TODO: no depth updateer here as we want to operate (once again)
//  on the contiguous memory block so we can do flow tracking for stereo
//  this means we need to wait till after all the geometric solves?
//  ah but for the object motion solves this is a problem since we use stereo
//  measurements which need to be updated!!
//  we will deal with this later!
class FlowRefinement {
 public:
  FlowRefinement(const Camera::Ptr camera, const ImageContainer& images,
                 MatchingAdaptorBase& adaptor);
  ~FlowRefinement();

  // will update the pixel values in the local geometry but NOT the 3d geometry
  // will mark values in features as outliers but not update the memory space!
  void refine(const OpticalFlowAndPoseSolverParams& params,
              const gtsam::Pose3& pose_in, gtsam::Pose3& pose_out);

 private:
  Camera::Ptr camera_;
  ImageContainer images_;
  MatchingAdaptorBase& adaptor_;

  struct ImplOptimizer;
  std::unique_ptr<ImplOptimizer> impl_;
};

struct TrackingResult {
  ObjectDetectionResult object_detection;
  // note is a reference!!!
  // TODO: comment as to why!
  FeatureBlockContainer& featues;
};

template <>
struct measurement_traits<StereoMeasurement> {
  static StereoMeasurement::Optional stereo(
      const StereoMeasurement& measurement) {
    return measurement;
  }
};

using StereoMap = RegularMap<StereoMeasurement>;

class LocalBAGraph : public LandmarkMap {
 public:
  DYNO_POINTER_TYPEDEFS(LocalBAGraph)

  LocalBAGraph() : observations_(StereoMap::create()) {}
  virtual ~LocalBAGraph() = default;

  inline void addMeasurements(
      const StereoMeasurementStatusVector& measurements) {
    observations_->updateObservations(measurements);
  }

  const StereoMap& getObservations() const { return *observations_; }
  const std::set<FrameId>& keyFrames() const { return keyframes_; }

  // currently no removal of keyframe
  bool setKeyframe(FrameId id, bool is_keyframe) {
    CHECK(is_keyframe);
    keyframes_.insert(id);
    return true;
  }

 protected:
  StereoMap::Ptr observations_;

  std::set<FrameId> keyframes_;

  // gtsam::FastMap<FrameId, gtsam::FastMap<FrameId, int>>
  // co_observation_counts_;

  // poses?
};

class LocalVIOGraph : public LocalBAGraph {
 public:
  DYNO_POINTER_TYPEDEFS(LocalVIOGraph)
  LocalVIOGraph(Camera::Ptr camera);

  void optimize(FrameId frame_id,
                std::vector<FrameId>* frames_affected = nullptr);

  void setPose(FrameId frame_id, const gtsam::Pose3& pose) {
    states_[frame_id] = pose;
  }

  bool poseExists(FrameId frame_id) const { return states_.exists(frame_id); }

  const gtsam::Pose3& getPose(FrameId frame_id) const {
    return states_.at(frame_id);
  }

 private:
  gtsam::FastMap<FrameId, gtsam::Pose3> states_;

  Camera::CalibrationType::shared_ptr K_;
  StereoCalibPtr K_stereo_;
};

// dynamic object graph
class DOGraph : public LocalBAGraph {
 public:
  DYNO_POINTER_TYPEDEFS(DOGraph)
};

class MultiBAGraph {
 public:
  typedef gtsam::FastMap<ObjectId, LocalBAGraph::Ptr> BaGraphs;

  MultiBAGraph() = default;
  // virtual ~MultiBaGraph() = default;

  bool exists(ObjectId object_id) const { return maps_.exists(object_id); }

  LocalBAGraph::Ptr get(ObjectId object_id) const {
    auto map = maps_.at(object_id);
    CHECK_NOTNULL(map);
    return map;
  }

  BaGraphs::const_iterator begin() const { return maps_.begin(); }

  BaGraphs::const_iterator end() const { return maps_.end(); }

 protected:
  void add(ObjectId object_id, LocalBAGraph::Ptr map) {
    maps_[object_id] = map;
  }

  template <typename T>
  std::shared_ptr<T> getAs(ObjectId object_id) const {
    auto map_base = this->get(object_id);
    std::shared_ptr<T> map_derived = std::dynamic_pointer_cast<T>(map_base);
    CHECK_NOTNULL(map_derived);
    return map_derived;
  }

 protected:
  BaGraphs maps_;
};

class DynamicSlamMap : public MultiBAGraph {
 public:
  DynamicSlamMap(Camera::Ptr camera) : camera_(camera) {
    maps_[background_label] = std::make_shared<LocalVIOGraph>(camera_);
  }

  bool hasStaticMap() const { return this->exists(background_label); }
  bool hasDynamicMap(ObjectId object_id) {
    CHECK_GT(object_id, background_label);
    return this->exists(object_id);
  }

  LocalVIOGraph::Ptr getStaticMap() const {
    return this->getAs<LocalVIOGraph>(background_label);
  }

  DOGraph::Ptr getDynamicObjectMap(ObjectId object_id) const {
    return this->getAs<DOGraph>(object_id);
  }

  LocalBAGraph::Ptr add(ObjectId object_id) {
    if (!exists(object_id)) {
      CHECK_GT(object_id, background_label);
      auto map = std::make_shared<DOGraph>();
      maps_[object_id] = map;
      return map;
    } else {
      return getStaticMap();
    }
  }

 private:
  Camera::Ptr camera_;
};

struct OpticalFlowLK {
  static constexpr float kMaxErr = 20.0f;

  const cv::Size win_size;
  const int max_level;
  const cv::TermCriteria criteria;

  OpticalFlowLK(int iterations)
      : win_size(24, 24),
        max_level(4),
        criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, iterations,
                 0.001) {}

  OpticalFlowLK() : OpticalFlowLK(50) {}

  struct ImplResult {
    std::vector<uchar> status;
    std::vector<float> error;
    std::vector<cv::Point2f> prev;
    std::vector<cv::Point2f> next;

    size_t size() const { return status.size(); }
  };

  struct Result {
    const ImplResult forward_result;
    const ImplResult reverse_result;

    Result(const ImplResult& forward, const ImplResult& reverse)
        : forward_result(forward), reverse_result(reverse) {}

    const std::vector<cv::Point2f>& predictedPoints() const {
      return forward_result.next;
    }

    const std::vector<cv::Point2f>& fromPoints() const {
      return forward_result.prev;
    }

    const std::vector<float>& error() const { return forward_result.error; }

    size_t size() const { return forward_result.size(); }

    bool isGood(size_t i) const {
      const bool both_status_good =
          forward_result.status.at(i) && reverse_result.status.at(i);

      // compare distance betwen original pixel local and reverse predicted
      // pixel location (which should be the same)
      const bool within_distance =
          utils::distance(forward_result.prev.at(i),
                          reverse_result.next.at(i)) <= 0.5;

      const bool within_error = forward_result.error[i] < kMaxErr &&
                                reverse_result.error[i] < kMaxErr;

      return both_status_good && within_distance && within_error;
    }
  };

  Result operator()(const std::vector<cv::Mat>& prev_pyr,
                    const std::vector<cv::Mat>& next_pyr,
                    const std::vector<cv::Point2f>& prev_points) const;
};

// Should just be called tracker or something as also does object tracking!
class FeatureTrackerFast : public FeatureTrackerBase {
 public:
  DYNO_POINTER_TYPEDEFS(FeatureTrackerFast)

  FeatureTrackerFast(const FrontendParams& params, Camera::Ptr camera,
                     ImageDisplayQueue* display_queue = nullptr);
  virtual ~FeatureTrackerFast() {}

  // just for now!!!! Lets soo what kind of speed gains we get!
  // TODO: eventually we need to return a pointer or a reference to features
  // becuase we will need to update the block outside with inlier/outliers
  // and refine the pixel values
  // and we want the next iteration of track to use these!
  TrackingResult track(FrameId frame_id, Timestamp timestamp,
                       const ImageContainer& image_container,
                       const std::optional<gtsam::Rot3>& R_km1_k = {});

 private:
  struct ObjectDetectionImpl {
    FeatureTrackerFast* parent;

    ObjectDetectionImpl(FeatureTrackerFast* parent_);

    ObjectDetectionResult detectAndTrack(
        const ImageContainer& image_container) const;
    ObjectDetectionResult detectionViaObjectMask(
        const ImageContainer& image_container) const;
    ObjectDetectionResult detectionViaInference(
        const ImageContainer& image_container) const;
  };

  ObjectDetectionImpl object_detection_impl_;
  ObjectDetectionEngine::Ptr object_detection_engine_;

  // using DetailedCornerTracksPerObject = gtsam::FastMap<ObjectId,
  // DetailedCornerTracks>; DetailedCornerTracksPerObject calcKLTBatch(
  //     const cv::Mat& mono,
  //     const cv::Mat& object_mask);

 private:
  const FrontendParams frontend_params_;
  TrackletIdManager& tracklet_id_manager;

  std::vector<cv::Mat> curr_mono_pyr_;
  cv::Mat prev_mono_;
  //! Image pyramid for the previous mono frame
  std::vector<cv::Mat> prev_mono_pyr_;
  gtsam::FastMap<ObjectId, cv::Mat> previous_object_detection_masks_;
  //! Object mask for the previous frame
  cv::Mat prev_object_mask_;

  FeatureBlockContainer previous_features_;
  FeatureBlockContainer previous_tracked_features_;

  struct GfttDetector {
    struct Param {
      ObjectId object_id;
      //! Binary object/detection mask
      cv::Mat mask;
      cv::Rect bbox;
      // num total corners to extract
      int max_corners;
      // int corners_after_anms;
      float min_distance;
      //! Number of final corners after extraction and pruning
      //! If set to zero, assume same as max_corners
      int num_corners_needed{0};
    };

    struct Params : public std::vector<Param> {
      using Base = std::vector<Param>;
      using Base::Base;

      float quality_level{0.01};
      int block_size{3};
    };

    GfttDetector(const cv::Size& size);
    FeatureBlockContainer calc(const cv::Mat& mono, const cv::Mat& object_mask,
                               const Params& params);

    TrackletIdManager& tracklet_id_manager_;

    //! Page locked host allocations which own the image memory. Owns memory
    cv::cuda::HostMem h_mono_;
    cv::cuda::HostMem h_eig_;
    //! cv::Mat headers pointing to the page-locked allocations above. Does not
    //! own memory
    cv::Mat mono_;
    cv::Mat eig_;

    cv::cuda::GpuMat d_mono_;
    cv::cuda::GpuMat d_eig_;

    cv::cuda::Stream stream_;
    // impl detector for min-eigenvalue corner response.
    cv::Ptr<cv::cuda::CornernessCriteria> detector_;
  };

  void fillDetectionParam(ObjectId object_id, const cv::Mat& mask,
                          const cv::Rect& bounding_box, int current_tracks,
                          GfttDetector::Param& detection_param) const;

  GfttDetector feature_detector_;

  struct FlowTrackingStats {
    //! Valid features tracked by LKT
    size_t tracked_after_flow{0};
    //! Num tracks after outlier rejection (ie. verification)
    size_t tracked_after_or{0};
    //! Number of tracks in the previous frame
    size_t num_previous_tracks{0};

    float survivalRatio() const {
      if (num_previous_tracks > 0 && tracked_after_or > 0) {
        return (float)tracked_after_or / num_previous_tracks;
      } else {
        return 0.0;
      }
    }
  };
  using FlowTrackingStatsMap = gtsam::FastMap<ObjectId, FlowTrackingStats>;

  std::pair<FeatureBlockContainer, FlowTrackingStatsMap> trackGfftBatched(
      const cv::Mat& mono, const cv::Mat& object_mask);

  // TODo: assumes prev pyramid, prev detection masks and curr pyramid have been
  // set have been set correctly!
  FeatureBlockContainer trackRetroactively(
      const FeatureBlockContainer& detected_features,
      const cv::Mat& object_mask);

  OpticalFlowLK optical_flow_impl_;

  /**
   * @brief Get the desired minimum distance between features for detection,
   * depending on if the object is static (object_id = 0) or dynamic.
   *
   * @param object_id
   * @return int
   */
  inline int getMinFeatureDistance(ObjectId object_id) const {
    return object_id > background_label
               ? params_.min_distance_btw_tracked_and_detected_dynamic_features
               : params_.min_distance_btw_tracked_and_detected_static_features;
  }

  /**
   * @brief Get the maximum desired corners to be extracted depending on if the
   * object is static (object_id = 0) or dynamic.
   *
   * @param object_id
   * @return int
   */
  inline int getMaxDetectionCorners(ObjectId object_id) const {
    return object_id > background_label ? params_.max_dynamic_features_per_frame
                                        : params_.max_nr_keypoints_before_anms;
  }

  inline int getMaxTrackingCorners(ObjectId object_id) const {
    return object_id > background_label ? params_.max_dynamic_features_per_frame
                                        : params_.max_features_per_frame;
  }

  inline int getMinAllowableTracks(ObjectId object_id) const {
    return object_id > background_label ? params_.min_dynamic_tracks
                                        : params_.min_features_per_frame;
  }

  inline int getMaxFeatureAge(ObjectId object_id) const {
    return object_id > background_label ? params_.max_feature_track_age
                                        : params_.max_dynamic_feature_age;
  }
};

}  // namespace dyno
