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

#include <glog/logging.h>
#include <gtsam/base/FastMap.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/Pose3.h>

#include <boost/optional.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <optional>
#include <string_view>
#include <type_traits>
#include <vector>

#include "dynosam_common/Cuda.hpp"
#include "dynosam_common/Flags.hpp"  //for common glags DECLARATIONS
#include "dynosam_common/utils/Macros.hpp"

/** \mainpage C++ API documentation
 *  DynoSAM is a C++ library for Dynamic Visual SLAM.
 *
 * See:
 *  - <a href="https://acfr-rpg.github.io/DynOSAM/">DynoSAM Project Page</a>
 *  - <a href="https://github.com/ACFR-RPG/DynOSAM">Github</a>
 *
 */

namespace dyno {

template <typename T>
struct traits;

static constexpr auto NaN = std::numeric_limits<double>::quiet_NaN();

using Timestamp = double;
using Timestamps = Eigen::Matrix<Timestamp, 1, Eigen::Dynamic>;
constexpr static Timestamp InvalidTimestamp = 0;

/// @brief Discrete object id, j.
using ObjectId = int;
using ObjectIds = std::vector<ObjectId>;

/// @brief Discrete time-step k.
using FrameId = size_t;
using FrameIds = std::vector<FrameId>;

using Depth = double;
using Depths = std::vector<double>;

/// @brief Unique tracklet/landmark id, i.
using TrackletId = long int;  // -1 for invalid landmarks. // int would be too
                              // small if it is 16 bits!
using TrackletIds = std::vector<TrackletId>;

using BearingVectors =
    std::vector<gtsam::Vector3, Eigen::aligned_allocator<gtsam::Vector3>>;

using Landmark = gtsam::Point3;
using Landmarks = gtsam::Point3Vector;  //! Vector of Landmarks using gtsam's
                                        //! definition for allocation
using LandmarkMap = gtsam::FastMap<TrackletId, Landmark>;

using OpticalFlow =
    gtsam::Point2;  //! Observed optical flow vector (ordered x, y)

using Keypoint = gtsam::Point2;
using Keypoints = gtsam::Point2Vector;  //! Vector of 2D keypoints using gtsam's
                                        //! definition for allocation

//! For RGBD/StereoCamera
using Baseline = double;

using KeypointCV = cv::KeyPoint;
using KeypointsCV = std::vector<KeypointCV>;

using Motion3 = gtsam::Pose3;
using MotionMap =
    gtsam::FastMap<TrackletId,
                   Motion3>;  //! Map of tracklet ids to Motion3 (gtsam::Pose3)

/// @brief Map of FrameIds (k) to Timestamps
using FrameIdTimestampMap = gtsam::FastMap<FrameId, Timestamp>;

/// @brief Map of Object Ids to CLass Labels
using ObjectIdClassMap = gtsam::FastMap<ObjectId, std::string>;

// T is expected to have (at least) bitwise | (OR) support
template <typename T>
class Flags {
 public:
  using Storage = std::underlying_type_t<T>;

  Flags() : bits_(0) {}
  explicit Flags(T value) : bits_(static_cast<Storage>(value)) {}

  Flags& set(T value) {
    bits_ |= static_cast<Storage>(value);
    return *this;
  }

  bool has(T value) const { return (bits_ & static_cast<Storage>(value)) != 0; }

  bool hasAll(T value) const {
    return (bits_ & static_cast<Storage>(value)) == static_cast<Storage>(value);
  }

  bool hasAny(T value) const {
    return has(value);  // same logic
  }

  bool operator==(const Flags& other) const { return bits_ == other.bits_; }

  bool operator==(const T& value) const { return has(value); }

  bool operator!=(const Flags& other) const { return !(*this == other); }

  bool operator!=(const T& value) const { return !(*this == value); }

  Flags& operator=(const T& value) {
    bits_ = static_cast<Storage>(value);
    return *this;
  }

  Storage raw() const { return bits_; }

 private:
  Storage bits_;
};

/**
 * @brief Get demangled class name
 *
 * @param name
 * @return std::string
 */
std::string demangle(const char* name);

/**
 * @brief Get demangled class name of type T from an instance
 * @tparam T
 * @param t
 * @return std::string
 */
template <class T>
std::string type_name(const T& t) {
  return demangle(typeid(t).name());
}

/**
 * @brief Get a demangled class name of type T from a templated type
 *
 * @tparam T
 * @return std::string
 */
template <class T>
constexpr std::string type_name() {
  return demangle(typeid(T).name());
}

//! Expected label for the background in a semantic or motion mask
constexpr static ObjectId background_label = 0u;

enum KeyPointType { STATIC, DYNAMIC };

enum ReferenceFrame { GLOBAL, LOCAL, OBJECT };

/**
 * @brief Estimate with a reference frame and operator casting
 *
 * If the estimate E should be const, template on const E
 * If estimate E should be a reference, can template on E&
 *
 * @tparam E
 */
template <typename E>
struct ReferenceFrameValue {
  using Estimate = E;
  using This = ReferenceFrameValue<E>;

  using ConstEstimate =
      std::add_const_t<Estimate>;  //!	Const qualification of M. Regardless
                                   //! whether M is aleady const qualified.

  Estimate estimate_;
  ReferenceFrame frame_;

  ReferenceFrameValue() {}
  ReferenceFrameValue(ConstEstimate& estimate, ReferenceFrame frame)
      : estimate_(estimate), frame_(frame) {}

  operator Estimate&() { return estimate_; }
  operator const Estimate&() const { return estimate_; }
  operator const ReferenceFrame&() const { return frame_; }

  const ReferenceFrame& frame() const { return frame_; }
  const Estimate& estimate() const { return estimate_; }

  friend std::ostream& operator<<(std::ostream& os, const This& t) {
    os << type_name<Estimate>() << ": " << t.estimate() << "\n";
    os << "frame: " << to_string(t.frame()) << "\n";
    return os;
  }
};

enum MotionRepresentationStyle {
  F2F,  // frame-to-frame
  KF    // keyframe representation of motion
};

template <typename E>
struct MotionReferenceFrame : public ReferenceFrameValue<E> {
  using This = MotionReferenceFrame<E>;
  using Base = ReferenceFrameValue<E>;
  using ConstEstimate = typename Base::ConstEstimate;
  using Estimate = typename Base::Estimate;

  // explicitly inherit casting operators
  using Base::operator Estimate&;
  using Base::operator const Estimate&;
  using Base::operator const ReferenceFrame&;

  // forward the style ;)
  using Style = MotionRepresentationStyle;

  MotionRepresentationStyle style_;
  FrameId from_;
  FrameId to_;

  inline FrameId from() const { return from_; }
  inline FrameId to() const { return to_; }
  inline const MotionRepresentationStyle& style() const { return style_; }
  inline const ReferenceFrame& origin() const { return Base::frame_; }

  MotionReferenceFrame() {}
  MotionReferenceFrame(ConstEstimate& estimate, const Style& style,
                       ReferenceFrame frame, FrameId from, FrameId to)
      : Base(estimate, frame), style_(style), from_(from), to_(to) {}

  // really for seralization
  MotionReferenceFrame(const Base& base, const Style& style, FrameId from,
                       FrameId to)
      : Base(base), style_(style), from_(from), to_(to) {}

  friend std::ostream& operator<<(std::ostream& os, const This& t) {
    os << static_cast<const Base&>(t);
    os << "from: " << t.from() << "\n";
    os << "to: " << t.to() << "\n";
    os << "style : " << to_string(t.style()) << "\n";
    return os;
  }
};

template <typename VALUE>
class TrackedValueStatus {
 public:
  using Value = VALUE;
  using This = TrackedValueStatus<Value>;

  // Constexpr value used for the frame_id when it is NA (not applicable)
  // this may be the case when the TrackedValueStatus object represents a
  // time-invariant value (e.g a static point) and the value of the frame_id is
  // therefore meaingless NOTE:this is possibly dangerous if we are not careful
  // with casting (e.g. int to FrameId) since the overflow coulld land us at a
  // meaningless frame
  static constexpr auto MeaninglessFrame = std::numeric_limits<FrameId>::max();

  // for IO
  TrackedValueStatus() {}

  // not we will have implicit casting of all other frame_ids to int here which
  // could be dangerous
  TrackedValueStatus(const Value& value, FrameId frame_id,
                     TrackletId tracklet_id, ObjectId label,
                     ReferenceFrame reference_frame)
      : value_(value, reference_frame),
        frame_id_(frame_id),
        tracklet_id_(tracklet_id),
        label_(label) {}

  virtual ~TrackedValueStatus() = default;

  /**
   * @brief Allows the casting of the internal value type to another using
   * static casting.
   *
   * This allows sub measurements to be extracted if T has explict cast
   * operators. e.g. struct Foo { operator Bar&(); }
   *
   * we can use as type to construct a TrackedValueStatus<Bar> from
   * TrackedValueStatus<Foo>: TrackedValueStatus<Foo> foo(...);
   * TrackedValueStatus<Bar> bar = foo.asType<Bar>();
   *
   * This will copy over all the meta-data (e.g. frame id, label etc) but
   * replace the value with the cast Boo type. This is only possible since Foo
   * contains an operator for Bar, allowing static casting to work.
   *
   * This function exists to allow nested data-types for VALUE to be used, where
   * each sub-value can be extracted using asType(), as long as the casting
   * operating works.
   *
   * @tparam U
   * @return TrackedValueStatus<U>
   */
  template <typename U>
  TrackedValueStatus<U> asType() const {
    const U& new_measurement = static_cast<const U&>(this->value());
    return TrackedValueStatus<U>(new_measurement, this->frameId(),
                                 this->trackletId(), this->objectId(),
                                 this->referenceFrame());
  }

  /**
   * @brief Get value. Const version.
   *
   * @return const Value&
   */
  const Value& value() const { return value_; }

  /**
   * @brief Get value. Reference version.
   *
   * @return Value&
   */
  Value& value() { return value_; }

  /**
   * @brief Get tracklet id (i) for this status.
   *
   * @return TrackletId
   */
  TrackletId trackletId() const { return tracklet_id_; }

  /**
   * @brief Get object id (j) for this status.
   *
   * @return ObjectId
   */
  ObjectId objectId() const { return label_; }

  /**
   * @brief Get frame id (k) for this status.
   *
   * @return FrameId
   */
  FrameId frameId() const { return frame_id_; }

  /**
   * @brief Get the reference frame the value is in. Const version.
   *
   * @return const ReferenceFrame&
   */
  const ReferenceFrame& referenceFrame() const { return value_; }

  /**
   * @brief Get the reference frame the value is in.
   *
   * @return ReferenceFrame&
   */
  ReferenceFrame& referenceFrame() { return value_; }

  /**
   * @brief Get the full reference frame and value.
   *
   * @return ReferenceFrameValue<Value>&
   */
  ReferenceFrameValue<Value>& referenceFrameValue() { return value_; }

  /**
   * @brief Get the full reference frame and value. Const version.
   *
   * @return const ReferenceFrameValue<Value>&
   */
  const ReferenceFrameValue<Value>& referenceFrameValue() const {
    return value_;
  }

  bool operator==(const This& other) const {
    return gtsam::traits<Value>::Equals(this->value(), other.value()) &&
           frame_id_ == other.frame_id_ && tracklet_id_ == other.tracklet_id_ &&
           label_ == other.label_ &&
           this->referenceFrame() == other.referenceFrame();
  }

  friend std::ostream& operator<<(std::ostream& os, const This& t) {
    os << type_name<Value>() << ": " << t.value() << "\n";
    os << "frame id: " << t.frameId() << "\n";
    os << "tracklet id: " << t.trackletId() << "\n";
    os << "object id: " << t.objectId() << "\n";
    return os;
  }

  /**
   * @brief Returns true if the object lavel is equal to dyno::background_label.
   *
   * @return true
   * @return false
   */
  inline bool isStatic() const { return label_ == background_label; }

  /**
   * @brief Returns true if frame_id is equal to MeaninglessFrame (e.g.
   * std::numeric_limits<FrameId>::quiet_NaN) and should indicatate that the
   * value represented by this TrackedValueStatus is time-invariant, e.g. for a
   * static point which does not change overtime.
   * @return true
   * @return false
   */
  inline bool isTimeInvariant() const { return frame_id_ == MeaninglessFrame; }

 protected:
  ReferenceFrameValue<Value> value_;
  FrameId frame_id_;
  TrackletId tracklet_id_;
  ObjectId label_;  //! Will be 0 if background
};

/// @brief Check if derived DERIVEDSTATUS us in factor derived from
/// Status<Value>
/// @tparam DERIVEDSTATUS derived type
/// @tparam VALUE expected value to templated base Status on
template <typename DERIVEDSTATUS, typename VALUE>
inline constexpr bool IsDerivedTrackedValueStatus =
    std::is_base_of_v<TrackedValueStatus<VALUE>, DERIVEDSTATUS>;

template <class DERIVEDSTATUS, typename VALUE = typename DERIVEDSTATUS::Value>
struct IsStatus {
  static_assert(IsDerivedTrackedValueStatus<DERIVEDSTATUS, VALUE>,
                "DERIVEDSTATUS does not derive from Status<Value>");
  using type = DERIVEDSTATUS;
  using value = VALUE;
};

template <typename DERIVEDSTATUS,
          typename VALUE = typename DERIVEDSTATUS::Value>
class GenericTrackedStatusVector : public std::vector<DERIVEDSTATUS> {
 public:
  // check if the DERIVEDSTATUS meets requirements
  // and alias to the value type of the status
  using Value = typename IsStatus<DERIVEDSTATUS, VALUE>::value;
  using This = GenericTrackedStatusVector<DERIVEDSTATUS, VALUE>;

  using Base = std::vector<DERIVEDSTATUS>;
  using Base::Base;

  // for IO
  GenericTrackedStatusVector() {}

  /** Conversion to a standard STL container */
  operator std::vector<DERIVEDSTATUS>() const {
    return std::vector<DERIVEDSTATUS>(this->begin(), this->end());
  }

  This& operator+=(const GenericTrackedStatusVector& rhs) {
    this->insert(this->end(), rhs.begin(), rhs.end());
    return *this;
  }
};

/**
 * @brief Map of key to an estimate containting a reference frame
 *
 * @tparam Key
 * @tparam Estimate
 */
template <typename Key, typename Estimate>
using EstimateMap = gtsam::FastMap<Key, ReferenceFrameValue<Estimate>>;

/// @brief Map of key to a MotionReferenceFrame. Used specifically for motions.
/// @tparam Key
/// @tparam Estimate
template <typename Key, typename Estimate>
using MotionReferenceEstimateMap =
    gtsam::FastMap<Key, MotionReferenceFrame<Estimate>>;

/// @brief Map of object id's to MotionReferenceFrame with Motion3
using MotionEstimateMap = MotionReferenceEstimateMap<ObjectId, Motion3>;

/// @brief Alias of a MotionReferenceFrame using Motion3 (Pose3)
using Motion3ReferenceFrame = MotionReferenceFrame<Motion3>;

/// @brief Map of object id's to gtsam::Pose3
using PoseEstimateMap = gtsam::FastMap<ObjectId, gtsam::Pose3>;

// TODO: outer layer should be a unorderd map!
// TODO: should be more generic than encoding a FrameId!
/**
 * @brief Generic mapping of Object Id to FrameId to VALUE within a nested
 * gtsam::FastMap structure. This is a common datastrcture used to
 * store temporal information about each object.
 *
 * We call it ObjectCentric map as we order by ObjectId first.
 *
 * @tparam VALUE Value type to be stored
 */
template <typename VALUE>
class GenericObjectCentricMap
    : public gtsam::FastMap<ObjectId, gtsam::FastMap<FrameId, VALUE>> {
 public:
  using Base = gtsam::FastMap<ObjectId, gtsam::FastMap<FrameId, VALUE>>;
  using NestedBase = gtsam::FastMap<FrameId, VALUE>;
  using EstimateMap = gtsam::FastMap<ObjectId, VALUE>;

  using This = GenericObjectCentricMap<VALUE>;
  using Value = VALUE;

  using Base::at;
  using Base::Base;  // all the stl map stuff
  using Base::exists;
  using Base::insert2;

  /** Conversion to a gtsam::FastMap container */
  operator Base() const { return Base(this->begin(), this->end()); }

  operator typename Base::Base() const {
    return typename Base::Base(this->begin(), this->end());
  }

  /**
   * @brief Handy insert function allowing direct insertion to the nested map
   * structure.
   *
   * @param object_id
   * @param frame_id
   * @param value
   * @return true
   * @return false
   */
  bool insert22(ObjectId object_id, FrameId frame_id, const Value& value) {
    if (!this->exists(object_id)) {
      this->insert2(object_id, NestedBase{});
    }

    NestedBase& frame_map = this->at(object_id);
    return frame_map.insert2(frame_id, value);
  }

  // construct from an estimate map
  bool insert2(FrameId frame_id, const EstimateMap& estimate_map) {
    bool result = true;
    for (const auto& [object_id, estimate_value] : estimate_map) {
      result &= this->insert22(object_id, frame_id, estimate_value);
    }
    return result;
  }

  bool exists(ObjectId object_id, FrameId frame_id) const {
    static size_t out_of_range_flag;
    return existsImpl(object_id, frame_id, out_of_range_flag);
  }

  const Value& at(ObjectId object_id, FrameId frame_id) const {
    return atImpl<const This, const Value&>(this, object_id, frame_id);
  }

  Value& at(ObjectId object_id, FrameId frame_id) {
    return atImpl<This, Value&>(const_cast<This*>(this), object_id, frame_id);
  }

  // wont update key if exists?
  This& operator+=(const This& rhs) {
    for (const auto& [key, value] : rhs) {
      this->operator[](key).insert(value.begin(), value.end());
    }
    return *this;
  }

  /**
   * @brief Collect all objects that appear in the query frame, as well as their
   * value.
   *
   * @param frame_id FrameId
   * @return gtsam::FastMap<ObjectId, Value>
   */
  gtsam::FastMap<ObjectId, Value> collectByFrame(FrameId frame_id) const {
    return this->toEstimateMap(frame_id);
  }

  EstimateMap toEstimateMap(FrameId frame_id) const {
    EstimateMap object_map;
    for (const auto& [object_id, frame_map] : *this) {
      if (frame_map.exists(frame_id)) {
        object_map.insert2(object_id, frame_map.at(frame_id));
      }
    }
    return object_map;
  }

  ObjectIds gatherInvolvedObjects() const {
    ObjectIds object_ids;
    object_ids.reserve(this->size());
    for (const auto& [object_id, _] : *this) {
      object_ids.emplace_back(object_id);
    }
    return object_ids;
  }

  // maybe slow!!
  // TODO: maybe remove and cache frames somewhere else if necessary!!!
  FrameIds getInvovledFrameIds() const {
    FrameIds frame_ids;
    for (const auto& [_, frame_map] : *this) {
      for (const auto& [frame_id, _] : frame_map) {
        frame_ids.push_back(frame_id);
      }
    }
    return frame_ids;
  }

 private:
  template <typename Container, typename Return>
  static Return atImpl(Container* container, ObjectId object_id,
                       FrameId frame_id) {
    size_t out_of_range_flag;
    const bool result =
        container->existsImpl(object_id, frame_id, out_of_range_flag);
    if (result) {
      CHECK_EQ(out_of_range_flag, 2u);
      return container->at(object_id).at(frame_id);
    } else {
      std::stringstream ss;
      ss << "Index out of range: "
         << ((out_of_range_flag == 0) ? " object id " : " frame id")
         << " missing. Full query - (object id " << object_id << ", frame id "
         << frame_id << ").";
      throw std::out_of_range(ss.str());
    }
  }

  /**
   * @brief Helper function to determine if the query exists.
   * Operates like a regular exists function but also sets out_of_range_flag
   * to indicate which query (object_id or frame_id) is out of range:
   * out_of_range_flag = 0, object_id out of range
   * out_of_range_flag = 1, frame_id out of range
   * out_of_range_flag = 2 both queries exist and the function should return
   * true
   *
   * @param object_id
   * @param frame_id
   * @param out_of_range_flag
   * @return true
   * @return false
   */
  bool existsImpl(ObjectId object_id, FrameId frame_id,
                  size_t& out_of_range_flag) const {
    if (!this->exists(object_id)) {
      out_of_range_flag = 0;
      return false;
    }

    const auto& frame_map = this->at(object_id);
    if (!frame_map.exists(frame_id)) {
      out_of_range_flag = 1;
      return false;
    } else {
      out_of_range_flag = 2;
      return true;
    }
  }
};

/// @brief Map of object poses per object per frame
using ObjectPoseMap = GenericObjectCentricMap<gtsam::Pose3>;
/// @brief Map of object motions per object per frame
using ObjectMotionMap = GenericObjectCentricMap<Motion3ReferenceFrame>;

// Optional string that can be modified directly (similar to old-stype
// boost::optional) to access the mutable reference the internal string must be
// accessed with get()
//  e.g. optional->get() = "updated string value";
// This is to overcome the fact that the stdlib does not support
// std::optional<T&> directly
using OptionalString = std::optional<std::reference_wrapper<std::string>>;

template <typename T>
std::string to_string(const T& t);

// template <typename T>
// std::string to_string(const T& t) {
//     std::ostringstream t;
//     oss << value;
//     return oss.str();
// }

template <typename Input, typename Output>
bool convert(const Input&, Output&);

// TODO: adds delimiter at end of string too!
template <class Container>
inline std::string container_to_string(const Container& container,
                                       const std::string& delimiter = " ") {
  std::stringstream ss;
  for (const auto& c : container) {
    ss << c << delimiter;
  }
  return ss.str();
}

inline std::string info_string(FrameId frame_id, ObjectId object_id) {
  return "j= " + std::to_string(object_id) + ", k=" + std::to_string(frame_id);
}

// template<typename T>
// struct io_traits {

//   //T must be inside dyno namespace to work
//   static std::string ToString(const T& t, const std::string& s = "") {
//     using std::to_string;
//     // This calls std::to_string if std::to_string is defined;
//     // Otherwise it uses ADL to find the correct to_string function
//     const std::string repr = to_string(t);

//     if(s.empty()) return repr;
//     else return s + " " + repr;
//   }

//   static void Print(const T& t, const std::string& s = "") {
//     const std::string str = ToString(t, s);
//     std::cout << str << std::endl;
//   }

// };

// template<typename T>
// struct traits : public io_traits<T> {};

}  // namespace dyno

namespace std {
template <typename T>
struct underlying_type<dyno::Flags<T>> {
  using type = typename dyno::Flags<T>::Storage;
};
}  // namespace std

#include "dynosam_common/SensorModels.hpp"
#include "dynosam_common/Types-inl.hpp"
