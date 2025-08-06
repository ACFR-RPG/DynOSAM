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
#include <vector>

#include "dynosam/common/ImageContainer.hpp"
#include "dynosam/common/StructuredContainers.hpp"
#include "dynosam/common/Types.hpp"
#include "dynosam/utils/GtsamUtils.hpp"
#include "dynosam/utils/Numerical.hpp"

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

  // ImageWrapperType should be a ImageType, eg ImageType::RGBMono etc...
  template <typename ImageWrapperType,
            typename AccessType = typename ImageWrapperType::OpenCVType>
  static AccessType at(const Keypoint& kp,
                       const ImageWrapper<ImageWrapperType>& image_wrapper) {
    return at<AccessType>(kp, static_cast<const cv::Mat&>(image_wrapper));
  }

  template <typename AccessType>
  static AccessType at(const Keypoint& kp, const cv::Mat& img) {
    const int x = functional_keypoint::u<int>(kp);
    const int y = functional_keypoint::v<int>(kp);
    return img.at<AccessType>(y, x);
  }
};

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

  Feature(const Feature& other) {
    // TODO: use both mutexs?
    std::lock_guard<std::mutex> lk(other.mutex_);
    data_ = other.data_;
  }

  bool operator==(const Feature& other) const { return data_ == other.data_; }

  /**
   * @brief Gets keypoint observation.
   *
   * @return Keypoint
   */
  Keypoint keypoint() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return data_.keypoint;
  }

  /**
   * @brief Gets the measured flow.
   *
   * @return OpticalFlow
   */
  OpticalFlow measuredFlow() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return data_.measured_flow;
  }

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
  Keypoint predictedKeypoint() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return data_.predicted_keypoint;
  }

  /**
   * @brief Get the number of consequative frames this feature has been
   * successfully tracked in.
   *
   * @return size_t
   */
  size_t age() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return data_.age;
  }

  /**
   * @brief Get keypoint type (static or dynamci).
   *
   * @return KeyPointType
   */
  KeyPointType keypointType() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return data_.type;
  }

  /**
   * @brief Get the keypoints unique tracklet id (ie. i).
   *
   * @return TrackletId
   */
  TrackletId trackletId() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return data_.tracklet_id;
  }

  /**
   * @brief Get the frame id this feature was observed in (ie. k).
   *
   * @return FrameId
   */
  FrameId frameId() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return data_.frame_id;
  }

  /**
   * @brief If the feature is an inlier.
   *
   * @return true
   * @return false
   */
  bool inlier() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return data_.inlier;
  }

  /**
   * @brief Get the object id associated with this feature (0 if
   * background, 1...N for object. ie. j)
   *
   * @return ObjectId
   */
  ObjectId objectId() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return data_.tracking_label;
  }

  /**
   * @brief Depth of this keypoint (will be Feature::invalid_depth is depth is
   * not set).
   *
   * @return Depth
   */
  Depth depth() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return data_.depth;
  }

  /**
   * @brief Gets the right keypoint. The value will be
   * std::nullopt if not set.
   *
   * @return Keypoint
   */
  Keypoint rightKeypoint() const {
    std::lock_guard<std::mutex> lk(mutex_);
    if (data_.right_kp.has_value()) return data_.right_kp.value();
    DYNO_THROW_MSG(DynosamException)
        << "Right Kp requested for" << data_.tracklet_id << " but not set";
  }

  static Keypoint CalculatePredictedKeypoint(const Keypoint& keypoint,
                                             const OpticalFlow& measured_flow) {
    return keypoint + measured_flow;
  }

  /**
   * @brief Sets the measured optical flow (which should start at the features
   * keypoint) and updates the predicted keypoint using the flow:
   * predicted_keypoint_ = keypoint_ + measured_flow_;
   *
   * Uses the internal Feature::keypoint_ value
   *
   * @param measured_flow
   */
  void setPredictedKeypoint(const OpticalFlow& measured_flow) {
    std::lock_guard<std::mutex> lk(mutex_);
    data_.measured_flow = measured_flow;
    data_.predicted_keypoint =
        CalculatePredictedKeypoint(data_.keypoint, measured_flow);
  }

  Feature& keypoint(const Keypoint& kp) {
    std::lock_guard<std::mutex> lk(mutex_);
    data_.keypoint = kp;
    return *this;
  }

  Feature& measuredFlow(const OpticalFlow& measured_flow) {
    std::lock_guard<std::mutex> lk(mutex_);
    data_.measured_flow = measured_flow;
    return *this;
  }

  Feature& predictedKeypoint(const Keypoint& predicted_kp) {
    std::lock_guard<std::mutex> lk(mutex_);
    data_.predicted_keypoint = predicted_kp;
    return *this;
  }

  Feature& age(const size_t& a) {
    std::lock_guard<std::mutex> lk(mutex_);
    data_.age = a;
    return *this;
  }

  Feature& keypointType(const KeyPointType& kp_type) {
    std::lock_guard<std::mutex> lk(mutex_);
    data_.type = kp_type;
    return *this;
  }

  Feature& trackletId(const TrackletId& tracklet_id) {
    std::lock_guard<std::mutex> lk(mutex_);
    data_.tracklet_id = tracklet_id;
    return *this;
  }

  Feature& frameId(const FrameId& frame_id) {
    std::lock_guard<std::mutex> lk(mutex_);
    data_.frame_id = frame_id;
    return *this;
  }

  Feature& objectId(ObjectId id) {
    std::lock_guard<std::mutex> lk(mutex_);
    data_.tracking_label = id;
    return *this;
  }

  Feature& depth(Depth d) {
    std::lock_guard<std::mutex> lk(mutex_);
    data_.depth = d;
    return *this;
  }

  Feature& rightKeypoint(const Keypoint& right_kp) {
    std::lock_guard<std::mutex> lk(mutex_);
    data_.right_kp = right_kp;
    return *this;
  }

  /**
   * @brief If the feature is valid - a combination of inlier and if the
   * tracklet Id != -1
   *
   * To make a feature invalid, set tracklet_id == -1
   *
   * @return true
   * @return false
   */
  inline bool usable() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return data_.inlier && data_.tracklet_id != invalid_id;
  }

  inline bool isStatic() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return data_.type == KeyPointType::STATIC;
  }

  Feature& markOutlier() {
    std::lock_guard<std::mutex> lk(mutex_);
    data_.inlier = false;
    return *this;
  }

  Feature& markInlier() {
    std::lock_guard<std::mutex> lk(mutex_);
    data_.inlier = true;
    return *this;
  }

  Feature& markInvalid() {
    std::lock_guard<std::mutex> lk(mutex_);
    data_.tracklet_id = invalid_id;
    return *this;
  }

  inline bool hasDepth() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return !std::isnan(data_.depth);
  }

  inline bool hasRightKeypoint() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return data_.right_kp.has_value();
  }

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

// some typedefs and trait types
namespace internal {

/// @brief Alias for checking if the parsed iterator has a value_type equivalent
/// to Feature::Ptr, in other words, that the template iterates over Feature
/// Ptr's.
/// @tparam Iter
template <typename Iter>
using is_feature_ptr_iterator =
    std::is_same<Feature::Ptr, typename Iter::value_type>;

template <typename Iter>
using enable_if_feature_ptr_iterator =
    typename std::enable_if<is_feature_ptr_iterator<Iter>::value, void>::type;

}  // namespace internal

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
  using TrackletToFeatureMap = std::map<TrackletId, Feature::Ptr>;

  /**
   * @brief Internal iterator type allowing iteration over the features directly
   * e.g for(Feature::Ptr : container). This type satisfies constraints for a
   * filter_iterator_base as well as an std::iterator
   *
   * @tparam MapIterator The internal iterator to use
   * @tparam MappedType The type we are iterating over (either Feature::Ptr or
   * const Feature::Ptr )
   */
  template <typename MapIterator, typename MappedType>
  struct vector_iterator_base {
    using iterator_type = MapIterator;

    using value_type = MappedType;
    using reference = value_type&;
    using pointer = value_type*;

    iterator_type it_;
    vector_iterator_base(iterator_type it) : it_(it) {}

    reference operator*() { return it_->second; }
    reference operator->() { return it_->second; }

    bool operator==(const vector_iterator_base& other) const {
      return it_ == other.it_;
    }
    bool operator!=(const vector_iterator_base& other) const {
      return it_ != other.it_;
    }

    bool operator==(const iterator_type& other) const { return it_ == other; }
    bool operator!=(const iterator_type& other) const { return it_ != other; }

    vector_iterator_base& operator++() {
      ++it_;
      return *this;
    }
  };

  /// @brief Vector-style iterator definition
  using vector_iterator =
      vector_iterator_base<TrackletToFeatureMap::iterator, Feature::Ptr>;
  /// @brief Vector-style const iterator definition
  using const_vector_iterator =
      vector_iterator_base<TrackletToFeatureMap::const_iterator,
                           const Feature::Ptr>;

  /// @brief Internal typedefs to allow FeatureContainer to satisfy the
  /// definitions of a std::iterator See:
  /// https://en.cppreference.com/w/cpp/iterator/iterator_traits
  using iterator = vector_iterator;
  using pointer = iterator::pointer;
  using const_iterator = const_vector_iterator;
  using const_pointer = const_iterator::pointer;
  using value_type = Feature::Ptr;
  using reference = Feature::Ptr&;
  using const_reference = const Feature::Ptr&;
  using difference_type = std::ptrdiff_t;

  /// @brief Typedefs for Filter iteratorss for the FeatureContainer allowing
  /// conditional iterators to be defined. Must be defined after the internal
  /// iterator, pointer.... typedefs as internal::filter_iterator<> expects the
  /// type defined to have a valid iterator.
  using FilterIterator = internal::filter_iterator<FeatureContainer>;
  using ConstFilterIterator =
      internal::filter_const_iterator<const FeatureContainer>;

  FeatureContainer();
  FeatureContainer(const FeaturePtrs feature_vector);

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
    return *this;
  }

  // vector begin
  inline vector_iterator begin() {
    return vector_iterator(feature_map_.begin());
  }
  inline const_vector_iterator begin() const {
    return const_vector_iterator(feature_map_.cbegin());
  }

  // vector end
  inline vector_iterator end() { return vector_iterator(feature_map_.end()); }
  inline const_vector_iterator end() const {
    return const_vector_iterator(feature_map_.cend());
  }

  FilterIterator beginUsable();
  FilterIterator beginUsable() const;

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
  TrackletToFeatureMap feature_map_;
};

/// @brief filter iterator over a FeatureContainer class
using FeatureFilterIterator = FeatureContainer::FilterIterator;
using ConstFeatureFilterIterator = FeatureContainer::ConstFilterIterator;

}  // namespace dyno

// add iterator traits so we can use smart thigns on the FeatureFilterIterator
// like std::count, std::distance...
template <>
struct std::iterator_traits<dyno::FeatureFilterIterator>
    : public dyno::internal::filter_iterator_detail<
          dyno::FeatureFilterIterator::pointer> {};

template <>
struct std::iterator_traits<dyno::ConstFeatureFilterIterator>
    : public dyno::internal::filter_iterator_detail<
          dyno::FeatureFilterIterator::pointer> {};

template <>
struct std::iterator_traits<dyno::FeatureContainer::vector_iterator>
    : public dyno::internal::filter_iterator_detail<
          dyno::FeatureContainer::vector_iterator::pointer> {};

template <>
struct std::iterator_traits<dyno::FeatureContainer::const_vector_iterator>
    : public dyno::internal::filter_iterator_detail<
          dyno::FeatureContainer::const_vector_iterator::pointer> {};

template <>
struct std::iterator_traits<dyno::FeatureContainer>
    : public dyno::internal::filter_iterator_detail<
          dyno::FeatureContainer::pointer> {};

template <>
struct std::iterator_traits<const dyno::FeatureContainer>
    : public dyno::internal::filter_iterator_detail<
          dyno::FeatureContainer::const_pointer> {};

template <>
struct std::iterator_traits<dyno::FeaturePtrs>
    : public dyno::internal::filter_iterator_detail<
          dyno::FeaturePtrs::pointer> {};

template <>
struct std::iterator_traits<const dyno::FeaturePtrs>
    : public dyno::internal::filter_iterator_detail<
          dyno::FeaturePtrs::const_pointer> {};
