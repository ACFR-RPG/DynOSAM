#pragma once

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

// from here onwards we operate in the land of doubles
// as all the geometric solvers operate using gtsam/egien double types!
// these must be separately synchronized/maintaied with the
// FeatureBlockContainer
struct LocalPoints {
  // note the use of double here
  // points in the camera frame
  gtsam::Point3Vector lmks_C;
  gtsam::Point2Vector left_kps;
  //! Stereo correspondent in the right camera (x coordinate)
  std::vector<double> right_pixel;
  TrackletIds ids;
  // Index value in the original FeatureBlockContainer
  // only valid while the contianer remains unchanged
  // after a container is reduced to inliers the fc_indices must be updatd
  std::vector<Index> fc_indices;

  //! Tracklet ids -> index for this set of vectors
  // ie. to get the lmk of tracklet id i -> lmks_C[local_indices[i]]
  std::unordered_map<TrackletId, Index> local_indices;
};

typedef gtsam::FastMap<ObjectId, LocalPoints> LocalPointMap;

// for solving we need a points in local of previous frame (could be map or
// previous frame) and bearing vector for PnP and pixel for of flow
// TODO: for of flow can we use 3d point from anywhere? (technically yes, but
// have not tried!) what about 2d2d matching?
struct RelativePoseMatches {
  gtsam::Point3Vector lmks_C_ref;
  std::vector<Index> fc_indices_ref;

  // TODO: cant just have this as we need pixel coodinates for OF refinement!
  gtsam::Point3Vector bearing_vecs_curr;
  std::vector<Index> fc_indices_curr;

  TrackletIds ids;
};
typedef gtsam::FastMap<ObjectId, RelativePoseMatches> RelativePoseMatchesMap;

class DepthUpdaterFast {
 public:
  DepthUpdaterFast(const FrontendParams& params, Camera::Ptr camera,
                   const ImageContainer& images,
                   FeatureBlockContainer& features);

  // TODO: return outliers and mark features separately!
  void calcPoints(LocalPointMap& point_map);
  void calcPoints(const std::vector<Index>& indicies, LocalPointMap& point_map);

 private:
  FrontendParams params_;
  Camera::Ptr camera_;
  ImageContainer images_;
  FeatureBlockContainer& features_;
};

class FrameFast {
 public:
  DYNO_POINTER_TYPEDEFS(FrameFast)

  FrameFast(Camera::Ptr camera, const ImageContainer& image_container,
            const ObjectDetectionResult& object_detection,
            const FeatureBlockContainer& features);

  FrameId frameId() const { return images_.frameId(); }
  Timestamp timestamp() const { return images_.timestamp(); }
  const ImageContainer& images() const { return images_; }

  const FeatureBlockContainer& features() const { return features_; }
  FeatureBlockContainer& features() { return features_; }

  /// @brief NOTE: checks features not the object detection altthough these
  /// should be the same!
  /// @param object_id
  /// @return
  inline bool containsObject(ObjectId object_id) const {
    return features_.containsObject(object_id);
  }

  static bool findMatches(const FrameFast& first, const FrameFast& second,
                          IndexMatches& matches);
  static bool findMatches(const FrameFast& first, const FrameFast& second,
                          ObjectId object_id, IndexMatches& matches);

 private:
  void computeDepths();

 private:
  Camera::Ptr camera_;
  ImageContainer images_;
  //! The raw object detections
  ObjectDetectionResult object_detection_;
  FeatureBlockContainer features_;
};

struct TrackingResult {
  ObjectDetectionResult object_detection;
  // note is a reference!!!
  // TODO: comment as to why!
  FeatureBlockContainer& featues;
  cv::Mat viz;
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
    FeatureBlockContainer calc(const cv::Mat& mono, const Params& params);

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

  void buildOpticalFlowPyramid(const cv::Mat& mono,
                               std::vector<cv::Mat>& pyramid) const;

  // optical flow params
  const cv::Size win_size_;
  const int max_level_;
  const cv::TermCriteria criteria_;

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

  cv::Mat drawBatchedFeatures(
      const cv::Mat& image, const FeatureBlockContainer& batchedFeatures) const;
};

}  // namespace dyno
