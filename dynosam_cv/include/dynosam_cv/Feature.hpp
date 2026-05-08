/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a
 copy
 *   of this software and associated documentation files (the "Software"), to
 deal
 *   in the Software without restriction, including without limitation the
 rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in
 all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE
 *   SOFTWARE.
 */
#pragma once

#include <map>
#include <mutex>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "dynosam_common/StructuredContainers.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/GtsamUtils.hpp"
#include "dynosam_common/utils/Numerical.hpp"
#include "dynosam_common/utils/OpenCVUtils.hpp"
#include "dynosam_cv/ImageContainer.hpp"

namespace dyno {

struct functional_keypoint {
  template <typename T = int>
  static inline T u(const Keypoint& kp) {
    return static_cast<T>(kp(0));
  }

  template <typename T = int>
  static inline int v(const Keypoint& kp) {
    return static_cast<T>(kp(1));
  }

  template <typename Tp>
  static Tp at(const Keypoint& kp, const cv::Mat& img) {
    const int x = functional_keypoint::u<int>(kp);
    const int y = functional_keypoint::v<int>(kp);

    if (!utils::matContains(img, x, y)) {
      DYNO_THROW_MSG(DynosamException)
          << "Keypoint x: " << x << ", y: " << y
          << " out of bounds for image of size " << to_string(img.size());
      throw;
    }

    return img.at<Tp>(y, x);
  }
};

// DEBUG_CVMAT_AT()

// /// @brief adaptor struct to allow types to act like a cv::KeyPoint
// template<typename T>
// struct cv_keypoint_adaptor;

/**
 * @brief 2D tracking and id information for a feature observation at a single
 * frame.
 *
 * Modifcation and access to Feature is thread safe.
 *
 */
class Feature {
 public:
  DYNO_POINTER_TYPEDEFS(Feature)

  constexpr static TrackletId invalid_id =
      -1;  //! can refer to and invalid id label (of type int)
           //!  including tracklet id, instance and tracking label

  constexpr static auto invalid_depth =
      NaN;  //! nan is used to indicate the absense of a depth value (since we
            //! have double)

  Feature() : data_() {}

  Feature(const Feature& other);

  bool operator==(const Feature& other) const;

  /**
   * @brief Gets keypoint observation.
   *
   * @return Keypoint
   */
  Keypoint keypoint() const;

  /**
   * @brief Gets the measured flow.
   *
   * @return OpticalFlow
   */
  OpticalFlow measuredFlow() const;

  /**
   * @brief Gets the predicted keypoint.
   *
   * NOTE: due to historical implementations, the optical-flow used to track the
   * (dynamic) points was k to k+1 which is how we are able to get a predicted
   * keypoint. In the current implementation, this is still how we track the
   * dynamic points and for static points we just fill in the predicted keypoint
   * once we actually track the keypoint from k-1 to k.
   *
   * @return Keypoint
   */
  Keypoint predictedKeypoint() const;

  /**
   * @brief Get the number of consequative frames this feature has been
   * successfully tracked in.
   *
   * @return size_t
   */
  size_t age() const;

  /**
   * @brief Get keypoint type (static or dynamci).
   *
   * @return KeyPointType
   */
  KeyPointType keypointType() const;

  /**
   * @brief Get the keypoints unique tracklet id (ie. i).
   *
   * @return TrackletId
   */
  TrackletId trackletId() const;

  /**
   * @brief Get the frame id this feature was observed in (ie. k).
   *
   * @return FrameId
   */
  FrameId frameId() const;
  /**
   * @brief If the feature is an inlier.
   *
   * @return true
   * @return false
   */
  bool inlier() const;

  /**
   * @brief Get the object id associated with this feature (0 if
   * background, 1...N for object. ie. j)
   *
   * @return ObjectId
   */
  ObjectId objectId() const;

  /**
   * @brief Depth of this keypoint (will be Feature::invalid_depth is depth is
   * not set).
   *
   * @return Depth
   */
  Depth depth() const;

  /**
   * @brief Gets the right keypoint. The value will be
   * std::nullopt if not set.
   *
   * @return Keypoint
   */
  Keypoint rightKeypoint() const;

  static Keypoint CalculatePredictedKeypoint(const Keypoint& keypoint,
                                             const OpticalFlow& measured_flow);

  /**
   * @brief Sets the measured optical flow (which should start at the features
   * keypoint) and updates the predicted keypoint using the flow:
   * predicted_keypoint_ = keypoint_ + measured_flow_;
   *
   * Uses the internal Feature::keypoint_ value
   *
   * @param measured_flow
   */
  void setPredictedKeypoint(const OpticalFlow& measured_flow);

  Feature& keypoint(const Keypoint& kp);

  Feature& measuredFlow(const OpticalFlow& measured_flow);

  Feature& predictedKeypoint(const Keypoint& predicted_kp);

  Feature& age(const size_t& a);

  Feature& keypointType(const KeyPointType& kp_type);

  Feature& trackletId(const TrackletId& tracklet_id);

  Feature& frameId(const FrameId& frame_id);

  Feature& objectId(ObjectId id);

  Feature& depth(Depth d);

  Feature& rightKeypoint(const Keypoint& right_kp);

  /**
   * @brief If the feature is valid - a combination of inlier and if the
   * tracklet Id != -1
   *
   * To make a feature invalid, set tracklet_id == -1
   *
   * @return true
   * @return false
   */
  bool usable() const;

  bool isStatic() const;

  Feature& markOutlier();

  Feature& markInlier();

  Feature& markInvalid();

  bool hasDepth() const;

  bool hasRightKeypoint() const;

  bool stereoPoint(gtsam::StereoPoint2& stereo) const;

  inline static bool IsUsable(const Feature::Ptr& f) { return f->usable(); }

  inline static bool IsNotNull(const Feature::Ptr& f) { return f != nullptr; }

 private:
  struct impl {
    Keypoint keypoint;            //! u,v keypoint at this frame (frame_id)
    OpticalFlow measured_flow;    //! Observed optical flow that. The predicted
                                  //! keypoint is calculated as keypoint + flow
    Keypoint predicted_keypoint;  //! from optical flow
    size_t age{0u};
    KeyPointType type{KeyPointType::STATIC};  //! starts STATIC
    TrackletId tracklet_id{invalid_id};       // starts invalid
    FrameId frame_id{0u};
    bool inlier{true};  //! Starts as inlier
    ObjectId instance_label{
        invalid_id};  //! instance label as provided by the input mask
    ObjectId tracking_label{
        invalid_id};  //! object tracking label that should indicate the same
                      //! tracked object between frames
    Depth depth{invalid_depth};  //! Depth as provided by a depth image (not Z).
                                 //! Initalised as invalid_depth (NaN)
    //! Possible stereo keypoint in the right image
    std::optional<Keypoint> right_kp = {};

    bool operator==(const impl& other) const {
      // TODO: lock?
      return gtsam::equal_with_abs_tol(keypoint, other.keypoint) &&
             gtsam::equal_with_abs_tol(measured_flow, other.measured_flow) &&
             gtsam::equal_with_abs_tol(predicted_keypoint,
                                       other.predicted_keypoint) &&
             age == other.age && type == other.type &&
             tracklet_id == other.tracklet_id && frame_id == other.frame_id &&
             inlier == other.inlier && instance_label == other.instance_label &&
             tracking_label == other.tracking_label &&
             fpEqual(depth, other.depth) &&
             utils::equateGtsamOptionalValues(right_kp, other.right_kp);
    }
  };

  impl data_;
  mutable std::mutex mutex_;
};

// template<>
// struct cv_keypoint_adaptor<Feature> {
//   static float x(const Feature& f) { return
//   functional_keypoint::u<float>(f.keypoint()); } static float y(const
//   Feature& f) { return functional_keypoint::v<float>(f.keypoint()); } static
//   float response(const Feature& f) { return 0; }
// };

using FeaturePtrs = std::vector<Feature::Ptr>;
using FeaturePair =
    std::pair<Feature::Ptr,
              Feature::Ptr>;  //! Pair of feature (shared) pointers
using FeaturePairs = std::vector<FeaturePair>;  //! Vector of feature pairs

struct UsableFeaturePredicate {
  bool operator()(const Feature::Ptr& f) const { return Feature::IsUsable(f); }
};

struct UsableObjectLabelPredicate {
  ObjectId object_id;
  UsableObjectLabelPredicate(const ObjectId j) : object_id(j) {}

  bool operator()(const Feature::Ptr& f) const {
    return Feature::IsUsable(f) && f->objectId() == object_id;
  }
};

/**
 * @brief Basic container mapping tracklet id's to a feature (pointer).
 *
 * Unlike a regular std::map, this container allows direct iteration over the
 * feature's and modification of inliers/outliers etc.
 *
 * Used regular in frames and other data-structures to add and maintain features
 * in different stages of their construction.
 *
 */
class FeatureContainer {
 public:
  using TrackletToFeatureMap = std::unordered_map<TrackletId, Feature::Ptr>;

  using TrackletIdSet = std::unordered_set<TrackletId>;
  // using ObjectToFeatureMap = FastUnorderedMap<ObjectId, TrackletIdSet>;

  /// @brief Internal typedefs to allow FeatureContainer to satisfy the
  /// definitions of a std::iterator See:
  /// https://en.cppreference.com/w/cpp/iterator/iterator_traits
  using iterator =
      vector_iterator_base<TrackletToFeatureMap::iterator, Feature::Ptr>;
  using pointer = iterator::pointer;
  using const_iterator =
      vector_iterator_base<TrackletToFeatureMap::const_iterator,
                           const Feature::Ptr>;
  using const_pointer = const_iterator::pointer;
  using value_type = Feature::Ptr;
  using reference = Feature::Ptr&;
  using const_reference = const Feature::Ptr&;
  using difference_type = std::ptrdiff_t;

  // Common filter view types
  using UsableFeatureIterator =
      internal::FilterView<FeatureContainer, UsableFeaturePredicate>;
  using UsableObjectFeatureIterator =
      internal::FilterView<FeatureContainer, UsableObjectLabelPredicate>;

  using ConstUsableFeatureIterator =
      internal::FilterView<const FeatureContainer, UsableFeaturePredicate>;
  using ConstUsableObjectFeatureIterator =
      internal::FilterView<const FeatureContainer, UsableObjectLabelPredicate>;

  template <typename FeatureT>
  class ObjectFeatureViewT {
   private:
    //! Tight integration with the parent feature container is necessary
    friend class FeatureContainer;

    using This = ObjectFeatureViewT<FeatureT>;
    ObjectId object_id_;
    FeatureContainer* container_ = nullptr;
    //! Set of tracklet ids for object id j
    TrackletIdSet tracklets_{};

    ObjectFeatureViewT(const ObjectId j, FeatureContainer* c)
        : object_id_(j), container_(c) {}

    void rebindContainer(FeatureContainer* new_owner) {
      CHECK_NOTNULL(new_owner);
      container_ = new_owner;
    }

    void insert(const TrackletId& tracklet_id) {
      tracklets_.insert(tracklet_id);
    }

   public:
    size_t size() const { return tracklets_.size(); }
    ObjectId objectId() const { return object_id_; }
    const TrackletIdSet& tracklets() const { return tracklets_; }

    struct iterator {
     public:
      using iterator_type = TrackletIdSet::const_iterator;
      // FeatureT determines if const Feature::Ptr or Feature::ConstPtr
      using value_type = FeatureT;
      using reference = value_type&;
      using const_reference = const reference;
      using pointer = value_type*;
      using difference_type = std::ptrdiff_t;
      using iterator_category = std::forward_iterator_tag;

     private:
      //! Tight integration with the parent view
      friend class ObjectFeatureViewT;

      iterator_type it_;
      //! Somewhat unsafe immutable pointer to the feature container
      FeatureContainer* container_;
      iterator(iterator_type it, FeatureContainer* container)
          : it_(it), container_(CHECK_NOTNULL(container)) {}

     public:
      // NOTE: we return the actual value from the feature map rather than using
      // the more safe container_->getByTrackletId() this is becuase we need to
      // return a reference so that the view can operate like a real iterator
      // and getByTrackletId returns value value
      const_reference operator*() const {
        auto t = *it_;
        CHECK(container_->feature_map_.find(t) !=
              container_->feature_map_.end())
            << "Feature not available i= " << t;
        return container_->feature_map_.at(t);
      }

      reference operator*() {
        auto t = *it_;
        CHECK(container_->feature_map_.find(t) !=
              container_->feature_map_.end())
            << "Feature not available i= " << t;
        return container_->feature_map_.at(t);
      }

      bool operator==(const iterator& other) const { return it_ == other.it_; }
      bool operator!=(const iterator& other) const { return it_ != other.it_; }

      bool operator==(const iterator_type& other) const { return it_ == other; }
      bool operator!=(const iterator_type& other) const { return it_ != other; }

      iterator& operator++() {
        ++it_;
        return *this;
      }
    };

    iterator begin() const { return iterator(tracklets_.begin(), container_); }
    iterator end() const { return iterator(tracklets_.end(), container_); }
    iterator begin() { return iterator(tracklets_.begin(), container_); }
    iterator end() { return iterator(tracklets_.end(), container_); }
  };

  using ObjectFeatureView = ObjectFeatureViewT<Feature::Ptr>;
  using ObjectToFeatureMap = FastUnorderedMap<ObjectId, ObjectFeatureView>;

  // Fast becuase this iterator iteratores directly over the set of tracklets
  // for a requested object j and therefore has many less features too look at
  using FastUsableObjectIterator =
      internal::FilterView<ObjectFeatureView, UsableFeaturePredicate>;
  using ConstFastUsableObjectIterator =
      internal::FilterView<const ObjectFeatureView, UsableFeaturePredicate>;

  FeatureContainer();
  FeatureContainer(const FeaturePtrs& feature_vector);
  // Copy constructor
  FeatureContainer(const FeatureContainer& other);
  // Copy assignment
  FeatureContainer& operator=(const FeatureContainer& other);
  // Move constructor
  FeatureContainer(FeatureContainer&& other) noexcept;
  // Move assignment
  FeatureContainer& operator=(FeatureContainer&& other) noexcept;

  template <typename Iterator>
  FeatureContainer(Iterator itr) : FeatureContainer() {
    for (const auto& feature : itr) {
      this->add(feature);
    }
  }

  /**
   * @brief Adds a new feature to the container.
   * Uses feature.trackletId() to set the tracklet key.
   *
   * Takes a copy of the feature.
   *
   * @param feature const Feature&
   */
  void add(const Feature& feature);

  /**
   * @brief Adds a new feature to the container.
   * Uses feature->trackletId() to set the tracklet key.
   *
   * @param feature Feature::Ptr feature
   */
  void add(Feature::Ptr feature);

  /**
   * @brief Removes a feature by tracklet id by using std::map::erase.
   *
   * NOTE: This will directly modify the internal map, messing up any iterator
   * that currently has a reference to this container (so any of the
   * FilterIterator)!!
   *
   *
   * @param tracklet_id TrackletId
   */
  void remove(TrackletId tracklet_id);

  /**
   * @brief Clears the entire container
   *
   */
  void clear();

  /**
   * @brief Removes all features with a particular object id.
   *
   * @param object_id ObjectId
   */
  void removeByObjectId(ObjectId object_id);

  bool hasObject(ObjectId object_id) const {
    return object_feature_map_.exists(object_id);
  }

  /**
   * @brief Collects all feature tracklets.
   * If only_usable is True, only features with Feature::usable() == true will
   * be included. This is a short-hand way of collecting only inlier tracklets!
   * Else, all features in the container will be included.
   *
   * @param only_usable bool. Defaults to true.
   * @return TrackletIds
   */
  TrackletIds collectTracklets(bool only_usable = true) const;

  /**
   * @brief If the container is empty.
   *
   * @return true
   * @return false
   */
  inline bool empty() const { return size() == 0u; }

  /**
   * @brief Mark all features with the provided tracklet ids as outliers.
   * If the present feature is already an outlier or does not exist, nothing
   * happens ;)!
   *
   * @param outliers const TrackletIds&
   */
  void markOutliers(const TrackletIds& outliers);

  /**
   * @brief Returns the number of features in the container.
   *
   * @return size_t
   */
  size_t size() const;

  // TODO: could now use the object_feature_map_ to directtly get the size but
  // currently no tests
  /**
   * @brief Returns number of features in the container per object id.
   *
   * @param object_id ObjectId
   * @return size_t
   */
  size_t size(ObjectId object_id) const;

  /**
   * @brief Gets a feature given its tracklet id.
   * If the feature does not exist, nullptr is returned.
   *
   * @param tracklet_id TrackletId
   * @return Feature::Ptr
   */
  Feature::Ptr getByTrackletId(TrackletId tracklet_id) const;

  /**
   * @brief Returns true if a feature with the given tracklet id exists.
   *
   * @param tracklet_id TrackletId
   * @return true
   * @return false
   */
  bool exists(TrackletId tracklet_id) const;

  FeatureContainer& operator+=(const FeatureContainer& other) {
    feature_map_.insert(other.feature_map_.begin(), other.feature_map_.end());
    // cannot insert manually since this will mean that the other object
    // features will still internally point to the other container! for now
    // manually overwrite!
    for (const auto& [object_id, other_feature_view] :
         other.object_feature_map_) {
      if (!object_feature_map_.exists(object_id)) {
        object_feature_map_.insert2(object_id,
                                    ObjectFeatureView(object_id, this));
      }
      const auto& oth_tracklets = other_feature_view.tracklets();
      // insert new tracklets per object if necessary
      object_feature_map_.at(object_id).tracklets_.insert(oth_tracklets.begin(),
                                                          oth_tracklets.end());
    }
    // object_feature_map_.insert(other.object_feature_map_.begin(),
    //                            other.object_feature_map_.end());
    return *this;
  }

  TrackletIds getByObject(ObjectId) const;

  // vector begin
  inline iterator begin() { return iterator(feature_map_.begin()); }
  inline const_iterator begin() const {
    return const_iterator(feature_map_.cbegin());
  }

  /* Get the a feature view over all features associated with object id. Use
   * with hasObject */
  const ObjectFeatureView& featuresByObject(ObjectId object_id) const;
  /* Get the a feature view over all features associated with object id. Use
     with hasObject. The view itself is immutable due to private access but the
     internal features may be modified
  */
  ObjectFeatureView& featuresByObject(ObjectId object_id);

  // vector end
  inline iterator end() { return iterator(feature_map_.end()); }
  inline const_iterator end() const {
    return const_iterator(feature_map_.cend());
  }

  UsableFeatureIterator usableIterator() {
    return UsableFeatureIterator(*this, UsableFeaturePredicate());
  }
  ConstUsableFeatureIterator usableIterator() const {
    return ConstUsableFeatureIterator(*this, UsableFeaturePredicate());
  }

  ObjectToFeatureMap::iterator beginObjectIterator();
  ObjectToFeatureMap::iterator endObjectIterator();

  ObjectToFeatureMap::const_iterator beginObjectIterator() const;
  ObjectToFeatureMap::const_iterator endObjectIterator() const;

  FastUsableObjectIterator usableIterator(ObjectId object_id);
  ConstFastUsableObjectIterator usableIterator(ObjectId object_id) const;
  // ConstFastUsableObjectIterator usableIterator(ObjectId object_id) const;

  /**
   * @brief Converts the keypoints of all features in the container to
   * cv::Point2f representation. This makes them compatible with OpenCV
   * functions.
   *
   * If the argument TrackletIds* is provided (ie, tracklet_ids != nullptr), the
   * vector will be filled with the associated tracklet'id of each feature. This
   * will be a 1-to-1 match with the output vector, allowing the keypoints to be
   * associated with their tracklet id.
   *
   * @param tracklet_ids TrackletIds*. Defaults to nullptr
   * @param only_inliers bool. Defaults to false
   * @return std::vector<cv::Point2f>
   */
  std::vector<cv::Point2f> toOpenCV(TrackletIds* tracklet_ids = nullptr,
                                    bool only_inliers = false) const;

 private:
  void rebindObjectFeatureViews() {
    for (auto& [id, view] : object_feature_map_) {
      view.rebindContainer(this);
    }
  }

 private:
  TrackletToFeatureMap feature_map_;
  ObjectToFeatureMap object_feature_map_;

  // Empty object iterators used as dummies
  // Must be in persistant memory so that the object iterator
  // can take a reference
  // cannot be static becuase each view requires a pointer to this
  // initalised with 0 object id becuase it should not matter
  ObjectFeatureView empty_object_feature_view_;
};

/**
 * @brief Alias to a FilterView over a feature container with some Predicate.
 *
 * @tparam Predicate
 */
template <typename Predicate>
using FeatureIterator = internal::FilterView<FeatureContainer, Predicate>;

/**
 * @brief Alias to a FilterView over a (const) feature container with some
 * Predicate.
 *
 * @tparam Predicate
 */
template <typename Predicate>
using ConstFeatureIterator =
    internal::FilterView<const FeatureContainer, Predicate>;

}  // namespace dyno
