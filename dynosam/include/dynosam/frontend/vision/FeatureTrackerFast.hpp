#pragma once

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "dynosam/frontend/FrontendParams.hpp"
#include "dynosam/frontend/vision/FeatureTrackerBase.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam/frontend/vision/StaticFeatureTracker.hpp"
#include "dynosam_nn/ObjectDetector.hpp"
#include "dynosam_sensors/Camera.hpp"
#include "dynosam_sensors/Feature.hpp"

namespace dyno {

class FeatureBlockContainer {
 public:
  struct FeatureData {
    std::vector<cv::Point2f> points;
    std::vector<cv::Point2f> previous_points;
    std::vector<int> ids;
    std::vector<uchar> status;
    std::vector<float> errors;

    size_t size() const { return points.size(); }

    bool empty() const { return points.empty(); }

    void checkSizes() const {
      const size_t n = points.size();

      if (previous_points.size() != n || ids.size() != n ||
          status.size() != n || errors.size() != n) {
        throw std::invalid_argument(
            "FeatureData: all feature arrays must have the same size");
      }
    }
  };

  struct FeatureBlockDim {
    ObjectId object_id;
    size_t size{0};
  };

 private:
  struct FeatureBlockLayout {
    ObjectId object_id;
    size_t begin{0};
    size_t end{0};

    size_t size() const { return end - begin; }
  };

 public:
  class FeatureBlockView {
   public:
    size_t size() const { return end_ - begin_; }

    ObjectId objectId() const { return object_id_; }

    cv::Point2f* points() { return features_->points.data() + begin_; }

    const cv::Point2f* points() const {
      return features_->points.data() + begin_;
    }

    cv::Point2f* previousPoints() {
      return features_->previous_points.data() + begin_;
    }

    const cv::Point2f* previousPoints() const {
      return features_->previous_points.data() + begin_;
    }

    int* ids() { return features_->ids.data() + begin_; }

    const int* ids() const { return features_->ids.data() + begin_; }

    int* objectIds() { return features_->object_ids.data() + begin_; }

    const int* objectIds() const {
      return features_->object_ids.data() + begin_;
    }

    uchar* status() { return features_->status.data() + begin_; }

    const uchar* status() const { return features_->status.data() + begin_; }

    float* errors() { return features_->errors.data() + begin_; }

    const float* errors() const { return features_->errors.data() + begin_; }

    // ---------------------------------------------------------------------
    // Convenient bulk assignment
    // ---------------------------------------------------------------------

    void copyFrom(const FeatureData& data) {
      data.checkSizes();

      if (data.size() != size()) {
        throw std::invalid_argument(
            "FeatureSet::FeatureBlockView::copyFrom: "
            "FeatureData size does not match object size");
      }

      copyBlock(points(), data.points.data(), size());
      copyBlock(previousPoints(), data.previous_points.data(), size());
      copyBlock(ids(), data.ids.data(), size());
      copyBlock(status(), data.status.data(), size());
      copyBlock(errors(), data.errors.data(), size());
    }

   private:
    friend class FeatureBlockContainer;

    FeatureBlockView(FeatureBlockContainer* features, ObjectId object_id,
                     size_t begin, size_t end)
        : features_(features),
          object_id_(object_id),
          begin_(begin),
          end_(end) {}

    template <typename T>
    static void copyBlock(T* destination, const T* source, size_t count) {
      static_assert(std::is_trivially_copyable<T>::value,
                    "FeatureBlockView fields must be trivially copyable");

      if (count > 0) {
        std::memcpy(destination, source, count * sizeof(T));
      }
    }

    // in reality might be pointer to const FeatureSet.
    // TODO: redesign with template as before
    FeatureBlockContainer* features_;
    // TODO: make FeatureBlockLayout
    ObjectId object_id_;
    size_t begin_;
    size_t end_;
  };

  FeatureBlockContainer(std::initializer_list<FeatureBlockDim> specs) {
    initialize(specs.begin(), specs.end());
  }

  explicit FeatureBlockContainer(const std::vector<FeatureBlockDim>& specs) {
    initialize(specs.begin(), specs.end());
  }

  //@tparam TERMS A container whose value type is std::pair<ObjectId,
  // FeatureData>
  template <typename TERMS>
  explicit FeatureBlockContainer(const TERMS& terms) {
    std::vector<FeatureBlockDim> specs;
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

      const ObjectId object_id = term.first;
      const FeatureData& source = term.second;

      FeatureBlockView destination = objectView(term.first);
      destination.copyFrom(source);
    }

    checkInvariants();
  }

  size_t size() const { return points.size(); }

  size_t objectCount() const { return objects_.size(); }

  bool containsObject(ObjectId object_id) const {
    return object_lookup_.find(object_id) != object_lookup_.end();
  }

  FeatureBlockContainer merge(const FeatureBlockContainer& other) const {
    // check for tracklet ids dupliactes
    // TODO: comment out for now - this makes creating empty or initalised but
    // inassigned FeatureSets invalid becuase all objectids/trackletids will
    // have the same id!
    //  std::unordered_set<int> feature_ids;
    //  feature_ids.reserve(ids.size() + other.ids.size());

    // for (const int id : ids)
    // {
    //     feature_ids.insert(id);
    // }

    // for (const int id : other.ids)
    // {
    //     if (!feature_ids.insert(id).second)
    //     {
    //         throw std::invalid_argument(
    //             "FeatureSet::merge: duplicate feature ID " +
    //             std::to_string(id));
    //     }
    // }

    // -------------------------------------------------------------------------
    // Build the resulting object layout.
    //
    // Existing objects retain their order.
    // New objects from `other` are appended in `other`'s order.
    // -------------------------------------------------------------------------

    std::vector<FeatureBlockDim> specs;
    specs.reserve(objects_.size() + other.objects_.size());

    // Existing objects.
    for (const FeatureBlockLayout& object : objects_) {
      const auto other_it = other.object_lookup_.find(object.object_id);

      const size_t other_size = other_it != other.object_lookup_.end()
                                    ? other.objects_[other_it->second].size()
                                    : 0;

      specs.push_back({object.object_id, object.size() + other_size});
    }

    // Objects which only exist in `other`.
    for (const FeatureBlockLayout& object : other.objects_) {
      if (object_lookup_.find(object.object_id) == object_lookup_.end()) {
        specs.push_back({object.object_id, object.size()});
      }
    }

    // -------------------------------------------------------------------------
    // Allocate the final FeatureSet exactly once.
    // -------------------------------------------------------------------------

    FeatureBlockContainer result(specs);

    // -------------------------------------------------------------------------
    // Copy the existing features into their final locations.
    // -------------------------------------------------------------------------

    for (const FeatureBlockLayout& object : objects_) {
      const FeatureBlockView source = objectView(object.object_id);

      FeatureBlockView destination = result.objectView(object.object_id);

      copyFeatures(destination, source);
    }

    // -------------------------------------------------------------------------
    // Append features from `other`.
    //
    // Existing objects are appended after their existing features.
    // New-only objects are copied starting at offset zero.
    // -------------------------------------------------------------------------

    for (const FeatureBlockLayout& other_object : other.objects_) {
      const FeatureBlockView source = other.objectView(other_object.object_id);

      FeatureBlockView destination = result.objectView(other_object.object_id);

      const auto existing_it = object_lookup_.find(other_object.object_id);

      const size_t destination_offset =
          existing_it != object_lookup_.end()
              ? objects_[existing_it->second].size()
              : 0;

      if (source.size() == 0) continue;

      copyFeatures(destination, source, destination_offset);
    }

    // object_ids are established by the FeatureSet constructor and therefore
    // don't need to be copied during the merge.

    result.checkInvariants();

    return result;
  }

  FeatureBlockView objectView(int object_id) {
    const FeatureBlockLayout& metadata = objectMetadata(object_id);

    return FeatureBlockView(this, metadata.object_id, metadata.begin,
                            metadata.end);
  }

  const FeatureBlockView objectView(int object_id) const {
    const FeatureBlockLayout& metadata = objectMetadata(object_id);

    return FeatureBlockView(const_cast<FeatureBlockContainer*>(this),
                            metadata.object_id, metadata.begin, metadata.end);
  }

  std::vector<cv::Point2f> points;
  std::vector<cv::Point2f> previous_points;
  std::vector<int> ids;
  std::vector<int> object_ids;
  std::vector<uchar> status;
  std::vector<float> errors;

  void checkInvariants() const {
#ifndef NDEBUG
    const size_t n = points.size();

    assert(previous_points.size() == n);
    assert(ids.size() == n);
    assert(object_ids.size() == n);
    assert(status.size() == n);
    assert(errors.size() == n);

    size_t expected_begin = 0;

    for (const FeatureBlockLayout& object : objects_) {
      assert(object.begin == expected_begin);
      assert(object.begin <= object.end);
      assert(object.end <= n);

      for (size_t i = object.begin; i < object.end; ++i) {
        assert(object_ids[i] == object.object_id);
      }

      expected_begin = object.end;
    }

    assert(expected_begin == n);
#endif
  }

 private:
  template <typename T>
  static void copyBlock(T* destination, const T* source, size_t count) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "FeatureBlockContainer fields must be trivially copyable");

    if (count > 0) {
      std::memcpy(destination, source, count * sizeof(T));
    }
  }

  static void copyFeatures(FeatureBlockView destination,
                           const FeatureBlockView& source,
                           size_t destination_offset = 0) {
    copyBlock(destination.points() + destination_offset, source.points(),
              source.size());

    copyBlock(destination.previousPoints() + destination_offset,
              source.previousPoints(), source.size());

    copyBlock(destination.ids() + destination_offset, source.ids(),
              source.size());

    copyBlock(destination.status() + destination_offset, source.status(),
              source.size());

    copyBlock(destination.errors() + destination_offset, source.errors(),
              source.size());
  }

  // =========================================================================
  // Layout construction
  // =========================================================================

  template <typename Iterator>
  void initialize(Iterator begin, Iterator end) {
    const size_t object_count = static_cast<size_t>(std::distance(begin, end));

    objects_.reserve(object_count);
    object_lookup_.reserve(object_count);

    size_t total_size = 0;

    for (Iterator it = begin; it != end; ++it) {
      const FeatureBlockDim& spec = *it;

      if (object_lookup_.find(spec.object_id) != object_lookup_.end()) {
        throw std::invalid_argument(
            "FeatureBlockContainer: duplicate object ID " +
            std::to_string(spec.object_id));
      }

      const size_t object_begin = total_size;
      const size_t object_end = total_size + spec.size;

      object_lookup_.emplace(spec.object_id, objects_.size());

      objects_.push_back(
          FeatureBlockLayout{spec.object_id, object_begin, object_end});

      total_size = object_end;
    }

    points.resize(total_size);
    previous_points.resize(total_size);
    ids.resize(total_size);
    object_ids.resize(total_size);
    status.resize(total_size);
    errors.resize(total_size);

    // Initialise object IDs immediately so the FeatureSet is valid even
    // before the caller fills the feature data.
    for (const FeatureBlockLayout& object : objects_) {
      std::fill(object_ids.begin() + object.begin,
                object_ids.begin() + object.end, object.object_id);
    }

    checkInvariants();
  }

  // TODO: change name!
  const FeatureBlockLayout& objectMetadata(ObjectId object_id) const {
    auto it = object_lookup_.find(object_id);

    if (it == object_lookup_.end()) {
      throw std::out_of_range("FeatureSet: unknown object ID " +
                              std::to_string(object_id));
    }

    return objects_[it->second];
  }

  // TODO: change name
  std::vector<FeatureBlockLayout> objects_;
  std::unordered_map<ObjectId, size_t> object_lookup_;
};

// Should just be called tracker or something as also does object tracking!
class FeatureTrackerFast {
 public:
  DYNO_POINTER_TYPEDEFS(FeatureTrackerFast)

  FeatureTrackerFast(const FrontendParams& params, Camera::Ptr camera,
                     ImageDisplayQueue* display_queue = nullptr);
  virtual ~FeatureTrackerFast() {}

  Frame::Ptr track(FrameId frame_id, Timestamp timestamp,
                   const ImageContainer& image_container,
                   const std::optional<gtsam::Rot3>& R_km1_k = {});

 private:
  void initDeviceMemory(const cv::Size& size);

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

  // void buildOpticalFlowPyramid(const cv::Mat& mono,
  //                         std::vector<cv::Mat>& pyramid) const;

 private:
  const FrontendParams frontend_params_;
  TrackletIdManager& tracklet_id_manager;

  Frame::Ptr prev_frame_;
  cv::Mat prev_mono_;
  //! Image pyramid for the previous mono frame
  std::vector<cv::Mat> prev_mono_pyr_;
  // //! Binary image used feature detection mask for the previous frame
  // cv::Mat prev_detection_mask_;
  //! Object mask for the previous frame
  cv::Mat prev_object_mask_;

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
  cv::Ptr<cv::cuda::CornernessCriteria> detector_;

  //   struct TrackingInfo {
  //     ObjectId object_id{};
  //     FrameId last_detection{};
  //     size_t num_last_detected_features{0};
  //     size_t num_last_tracked{0};
  //   };

  //   gtsam::FastMap<ObjectId, CornerTracks> previous_tracks_;

  struct CornersPerDetectionParams {
    ObjectId object_id;
    //! Binary object/detection mask
    cv::Mat mask;
    cv::Rect bbox;
    // num total corners to extract
    int max_corners;
    // int corners_after_anms;
    float min_distance;
  };

  // optical flow params
  const cv::Size win_size_;
  const int max_level_;
  const cv::TermCriteria criteria_;
};

}  // namespace dyno
