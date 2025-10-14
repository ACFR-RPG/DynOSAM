/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
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

#include <gtsam/base/FastSet.h>
#include <pcl/common/centroid.h>  //for compute centroid

#include <memory>
#include <set>
#include <type_traits>

#include "dynosam_common/Types.hpp"
#include "dynosam_opt/Symbols.hpp"

namespace dyno {

// forward declare map
template <typename MEASUREMENT>
class Map;

struct InvalidMapException : public DynosamException {
  InvalidMapException()
      : DynosamException(
            "The Map could not be accessed as it is no longer valid. Has the "
            "object gone out of scope?") {}
};

template <typename MEASUREMENT>
class MapNodeBase {
 public:
  MapNodeBase(const std::shared_ptr<Map<MEASUREMENT>>& map) : map_ptr_(map) {}
  virtual ~MapNodeBase() = default;
  virtual int getId() const = 0;
  virtual std::string toString() const = 0;

 protected:
  std::shared_ptr<Map<MEASUREMENT>> map_ptr_;
};

// // TODO: unused due to circular dependancies
// template <typename NODE>
// struct IsMapNode {
//   using UnderlyingType =
//       typename NODE::element_type;  // how to check NODE is a shared_ptr?
//   static_assert(std::is_base_of_v<MapNodeBase, UnderlyingType>);
// };

template <typename NODE>
struct MapNodePtrComparison {
  bool operator()(const NODE& f1, const NODE& f2) const {
    return f1->getId() < f2->getId();
  }
};

/**
 * @brief Data structuring for fast access of map nodes, sorted by their id.
 * We use a set to prevent objects being added multiple times.
 *
 * @tparam NODE Must be a pointer to a MapNodeBase (or equivalent concept with a
 * getId()) function.
 */
template <typename NODE>
class FastMapNodeSet
    : public std::set<
          NODE, MapNodePtrComparison<NODE>,
          typename gtsam::internal::FastDefaultAllocator<NODE>::type> {
 public:
  typedef std::set<NODE, MapNodePtrComparison<NODE>,
                   typename gtsam::internal::FastDefaultAllocator<NODE>::type>
      Base;

  using Base::Base;  // Inherit the set constructors

  FastMapNodeSet() = default;  ///< Default constructor

  /**
   * @brief Constructor from a iterable container, passes through to base class.
   *
   * @tparam INPUTCONTAINER
   * @param container const INPUTCONTAINER&
   */
  template <typename INPUTCONTAINER>
  explicit FastMapNodeSet(const INPUTCONTAINER& container)
      : Base(container.begin(), container.end()) {}

  /**
   * @brief Copy constructor from another FastMapNodeSet, passes through to base
   * class.
   *
   * @param x const FastMapNodeSet<NODE>&
   */
  FastMapNodeSet(const FastMapNodeSet<NODE>& x) : Base(x) {}

  /**
   * @brief Copy constructor from the base set class, passes through to base
   * class.
   *
   * @param x Base
   */
  FastMapNodeSet(const Base& x) : Base(x) {}

  /**
   * @brief Returns all id's from the contained nodes.
   *
   * Recalculates each time O(N)
   *
   * @tparam Index
   * @return std::vector<Index>
   */
  template <typename Index = int>
  std::vector<Index> collectIds() const {
    std::vector<Index> ids;
    for (const auto& node : *this) {
      // ducktyping for node to have a getId function
      ids.push_back(static_cast<Index>(node->getId()));
    }

    return ids;
  }

  /**
   * @brief Get the first (smallest) index stored.
   *
   * @tparam Index
   * @return Index
   */
  template <typename Index = int>
  Index getFirstIndex() const {
    const NODE& node = *Base::cbegin();
    return getIndexSafe<Index>(node);
  }

  /**
   * @brief Get the last (largest) index stored
   *
   * @tparam Index
   * @return Index
   */
  template <typename Index = int>
  Index getLastIndex() const {
    const NODE& node = *Base::crbegin();
    return getIndexSafe<Index>(node);
  }

  /**
   * @brief Find a NODE based on their index. Returns this->end() if not found.
   * Non const version.
   *
   * @tparam Index
   * @param index
   * @return Base::iterator
   */
  template <typename Index = int>
  typename Base::iterator find(Index index) {
    return std::find_if(Base::begin(), Base::cend(), [index](const NODE& node) {
      return getIndexSafe<Index>(node) == index;
    });
  }

  /**
   * @brief Find a NODE based on their index. Returns this->end() if not found.
   * Const version.
   *
   * @tparam Index
   * @param index
   * @return Base::const_iterator
   */
  template <typename Index = int>
  typename Base::const_iterator find(Index index) const {
    return std::find_if(Base::cbegin(), Base::cend(),
                        [index](const NODE& node) {
                          return getIndexSafe<Index>(node) == index;
                        });
  }

  /**
   * @brief Check if a node with the given ID exists in the container.
   *
   * @tparam Index
   * @param index
   * @return true
   * @return false
   */
  template <typename Index = int>
  bool exists(Index index) const {
    return this->find(index) != this->end();
  }

  /**
   * @brief Merges this container with another. Will automatically be sorted.
   *
   * @param other FastMapNodeSet<NODE>&
   */
  void merge(const FastMapNodeSet<NODE>& other) {
    Base::insert(other.begin(), other.end());
  }

  /**
   * @brief Checks if the ID's of all the nodes are containuously ascending (ie.
   * there is no missing ID)
   *
   * @return true
   * @return false
   */
  bool continuousAscendingIndex() const {
    auto ids = collectIds();
    if (ids.size() < 2) return true;

    bool is_continuous = true;
    for (size_t i = 1; i < ids.size(); i++) {
      const auto prev = ids.at(i - 1);
      const auto curr = ids.at(i);
      is_continuous &= (prev + 1 == curr);
    }
    return is_continuous;
  }

 private:
  template <typename Index>
  static inline Index getIndexSafe(const NODE& node) {
    return static_cast<Index>(node->getId());
  }
};

// forward delcare structs
template <typename MEASUREMENT>
struct FrameNode;
template <typename MEASUREMENT>
struct ObjectNode;
template <typename MEASUREMENT>
struct LandmarkNode;

template <typename MEASUREMENT>
using FrameNodePtr = std::shared_ptr<FrameNode<MEASUREMENT>>;
template <typename MEASUREMENT>
using ObjectNodePtr = std::shared_ptr<ObjectNode<MEASUREMENT>>;
template <typename MEASUREMENT>
using LandmarkNodePtr = std::shared_ptr<LandmarkNode<MEASUREMENT>>;

template <typename MEASUREMENT>
using FrameNodePtrSet = FastMapNodeSet<FrameNodePtr<MEASUREMENT>>;
template <typename MEASUREMENT>
using LandmarkNodePtrSet = FastMapNodeSet<LandmarkNodePtr<MEASUREMENT>>;
template <typename MEASUREMENT>
using ObjectNodePtrSet = FastMapNodeSet<ObjectNodePtr<MEASUREMENT>>;

/// @brief Status for a StateQuery<> type. This is defined outside the
/// StateQuery class so that the type of status is independant of the StateQuery
/// template.
enum StateQueryStatus { VALID, NOT_IN_MAP, WAS_IN_MAP, INVALID_MAP };
/**
 * @brief Represents an optional value with meta-data that is retrieved from the
 * map.
 *
 * @tparam ValueType A value type stored in gtsam::Values
 */
template <typename ValueType>
class StateQuery : public std::optional<ValueType> {
 public:
  //! Base type representing the existance of the query value
  using Base = std::optional<ValueType>;
  using Base::value_or;

  //! Status of the state query
  using Status = StateQueryStatus;
  gtsam::Key key_;
  Status status_;

  inline gtsam::Key key() const { return key_; }
  inline Status status() const { return status_; }

  StateQuery() {}
  StateQuery(gtsam::Key key, const ValueType& v) : key_(key), status_(VALID) {
    Base::emplace(v);
  }
  StateQuery(gtsam::Key key, Status status) : key_(key), status_(status) {}

  // template<typename OtherValueType, typename...Args>

  const ValueType& get() const {
    if (!Base::has_value())
      throw DynosamException("StateQuery has no value for query type " +
                             type_name<ValueType>() + " with key " +
                             DynosamKeyFormatter(key_));
    return Base::value();
  }

  bool isValid() const { return status_ == VALID; }

  static StateQuery InvalidMap() {
    return StateQuery(gtsam::Key{}, INVALID_MAP);
  }
  static StateQuery NotInMap(gtsam::Key key) {
    return StateQuery(key, NOT_IN_MAP);
  }
  static StateQuery WasInMap(gtsam::Key key) {
    return StateQuery(key, WAS_IN_MAP);
  }

 private:
  // template<typename ValueType, typename OtherValueType, typename...Args>
  //   struct TemplateHelper {
  //     using ValueQuery = StateQuery<ValueType>;
  //     using OtherValueQuery = StateQuery<OtherValueType>;

  //       static OtherValueQuery create(const Args&&... args) {
  //         ValueQuery& query_input =
  //         std::get<ValueQuery>(std::forward_as_tuple(std::forward<T>(args)...));

  //         if(!query_input) {
  //           return OtherValueQuery(query_input.key_, query_input.status_);
  //         }
  //         else {
  //           return
  //         }
  //       }

  //   }
};

/**
 * @brief Safe getter to a StateQuery object.
 * If the StateQuery is successful, result is set to the value of the query and
 * true is returned. Else, result is set to to the default value and false is
 * returned.
 *
 * @tparam ValueType
 * @param result
 * @param query
 * @param default_value
 * @return true
 * @return false
 */
template <typename ValueType>
bool getSafeQuery(ValueType& result, const StateQuery<ValueType>& query,
                  const ValueType& default_value) {
  if (query) {
    result = query.get();
    return true;
  } else {
    result = default_value;
    return false;
  }
}

struct InvalidLandmarkException : public DynosamException {
  InvalidLandmarkException(TrackletId tracklet_id,
                           const std::string& reason = std::string())
      : DynosamException("Landmark with tracklet id" +
                         std::to_string(tracklet_id) + " is invalid" +
                         (reason.empty() ? "." : " with reason " + reason)) {}
};

struct MissingLandmarkException : InvalidLandmarkException {
  MissingLandmarkException(TrackletId tracklet_id, FrameId frame_id,
                           bool is_static)
      : InvalidLandmarkException(
            tracklet_id, (is_static ? "static" : "dynamic") +
                             std::string(" landmark is missing from frame ") +
                             std::to_string(frame_id)) {}
};

/**
 * @brief Frame node representing all temporal data at a frame id (k) in the
 * map-graph.
 *
 * @tparam MEASUREMENT
 */
template <typename MEASUREMENT>
class FrameNode : public MapNodeBase<MEASUREMENT> {
 public:
  using Base = MapNodeBase<MEASUREMENT>;
  using This = FrameNode<MEASUREMENT>;
  DYNO_POINTER_TYPEDEFS(This)

  FrameNode(const std::shared_ptr<Map<MEASUREMENT>>& map)
      : MapNodeBase<MEASUREMENT>(map) {}

  /// @brief FrameId (k)
  FrameId frame_id;
  /// @brief All dynamic landmarks observed at this frame
  LandmarkNodePtrSet<MEASUREMENT> dynamic_landmarks;
  /// @brief All static landmarks observed at this frame
  LandmarkNodePtrSet<MEASUREMENT> static_landmarks;
  /// @brief All objects seen at this frame. NOTE that this means that we have
  /// point observations at this frame, but not necessarily a motion (e.g. if we
  /// only observe this object once)
  ObjectNodePtrSet<MEASUREMENT> objects_seen;

  /// @brief Optional initial camera pose in world, provided by the front-end
  std::optional<Pose3Measurement> X_world;
  /// @brief Optional initial object motions in the world, provided by the
  /// front-end
  std::optional<MotionEstimateMap> motions;

  /**
   * @brief Returns the frame_id
   *
   * @return int
   */
  int getId() const override;
  std::string toString() const override {
    std::stringstream ss;
    ss << "Frame Id: " << frame_id << "\n";
    ss << "Objects seen: " << container_to_string(getObservedObjects()) << "\n";
    ss << "Num dynamic measurements: " << numDynamicMeasurements() << "\n";
    ss << "Num static measurements: " << numStaticMeasurements();
    return ss.str();
  }

  /**
   * @brief True if the requested object was observed in this frame.
   *
   * @param object_id ObjectId
   * @return true
   * @return false
   */
  bool objectObserved(ObjectId object_id) const;

  /**
   * @brief True if the requested object was observed at the previous frame.
   *
   * @param object_id ObjectId
   * @return true
   * @return false
   */
  bool objectObservedInPrevious(ObjectId object_id) const;

  /**
   * @brief True if the object appears at this (k) and the previous frame (k-1)
   * and therefore we expect a motion to exist taking us from k-1 to k.
   *
   * @param object_id ObjectId
   * @return true
   * @return false
   */
  bool objectMotionExpected(ObjectId object_id) const;

  /**
   * @brief Constructs a robot/sensor pose key.
   *
   * @return gtsam::Key
   */
  gtsam::Key makePoseKey() const;

  /**
   * @brief Consturcts an object motion key. The associated motion will be from
   * k-1 to k.
   *
   * @param object_id ObjectId
   * @return gtsam::Key
   */
  gtsam::Key makeObjectMotionKey(ObjectId object_id) const;

  /**
   * @brief Construct an object pose key. The associated pose will be for frame
   * k.
   *
   * @param object_id ObjectId
   * @return gtsam::Key
   */
  gtsam::Key makeObjectPoseKey(ObjectId object_id) const;

  /// @brief Const LandmarkNodePtr with corresponding Measurement value
  using LandmarkMeasurementPair =
      std::pair<const LandmarkNodePtr<MEASUREMENT>, MEASUREMENT>;

  /**
   * @brief Get all static measurements for this frame.
   *
   * @return std::vector<LandmarkMeasurementPair>
   */
  std::vector<LandmarkMeasurementPair> getStaticMeasurements() const;

  /**
   * @brief Get all (over all objects) dynamic measurements for this frame.
   *
   * @return std::vector<LandmarkMeasurementPair>
   */
  std::vector<LandmarkMeasurementPair> getDynamicMeasurements() const;

  /**
   * @brief Get the Dynamic Measurements object
   *
   * @param object_id
   * @return std::vector<LandmarkMeasurementPair>
   */
  std::vector<LandmarkMeasurementPair> getDynamicMeasurements(
      ObjectId object_id) const;

  /**
   * @brief Number of observed objects at this frame.
   *
   * @return size_t
   */
  inline size_t numObjects() const { return objects_seen.size(); }

  /**
   * @brief Get the ids (j) of objects observed at this frame.
   *
   * @return ObjectIds
   */
  ObjectIds getObservedObjects() const {
    return objects_seen.template collectIds<ObjectId>();
  }

  /**
   * @brief Number of dynamic measurements for this frame.
   *
   * @return size_t
   */
  inline size_t numDynamicMeasurements() const {
    return dynamic_landmarks.size();
  }

  /**
   * @brief Number of static measurements for this frame.
   *
   * @return size_t
   */
  inline size_t numStaticMeasurements() const {
    return static_landmarks.size();
  }
};

/**
 * @brief Object node representing all object related data (j) over all time(k
 * to K) in the map-graph.
 *
 * @tparam MEASUREMENT
 */
template <typename MEASUREMENT>
class ObjectNode : public MapNodeBase<MEASUREMENT> {
 public:
  using Base = MapNodeBase<MEASUREMENT>;
  using This = ObjectNode<MEASUREMENT>;
  DYNO_POINTER_TYPEDEFS(This)

  ObjectNode(const std::shared_ptr<Map<MEASUREMENT>>& map)
      : MapNodeBase<MEASUREMENT>(map) {}

  /// @brief Object label (j)
  ObjectId object_id;
  /// @brief All landmarks associated with the object over time
  LandmarkNodePtrSet<MEASUREMENT> dynamic_landmarks;

  /**
   * @brief Returns the object_id
   *
   * @return int
   */
  int getId() const override;
  std::string toString() const override {
    std::stringstream ss;
    ss << "Object Id: " << getId() << "\n";
    ss << "First seen: " << getFirstSeenFrame() << " last seen "
       << getLastSeenFrame() << "\n";
    ss << "Seen continuously: " << std::boolalpha
       << getSeenFrames().continuousAscendingIndex() << "\n";
    ss << "Total landmarks: " << dynamic_landmarks.size();
    return ss.str();
  }

  /**
   * @brief Get the first frame this object was observed in.
   *
   * @return FrameId
   */
  inline FrameId getFirstSeenFrame() const {
    return getSeenFrames().template getFirstIndex<FrameId>();
  }

  /**
   * @brief Get the last frame this object was observed in.
   *
   * @return FrameId
   */
  inline FrameId getLastSeenFrame() const {
    return getSeenFrames().template getLastIndex<FrameId>();
  }

  /**
   * @brief Get all the frame nodes that have observed this object.
   *
   * @return FrameNodePtrSet<MEASUREMENT>
   */
  FrameNodePtrSet<MEASUREMENT> getSeenFrames() const;

  /**
   * @brief Get all the frame ids that have observed this object.
   *
   * @return FrameIds
   */
  FrameIds getSeenFrameIds() const;

  /**
   * @brief Get all the landmarks that observed this object at a particular
   * frame id. This should be a subset of the dynamic_landmarks stored.
   *
   * @param frame_id FrameId
   * @return LandmarkNodePtrSet<MEASUREMENT>
   */
  LandmarkNodePtrSet<MEASUREMENT> getLandmarksSeenAtFrame(
      FrameId frame_id) const;
};

struct InvalidLandmarkQuery : public DynosamException {
  InvalidLandmarkQuery(gtsam::Key key, const std::string& string)
      : DynosamException("Landmark estimate query failed with key " +
                         DynosamKeyFormatter(key) + ", reason: " + string) {}
};

/**
 * @brief Landmark node representing all the measurements of a particular
 * landmark (i) over all time (k to K) in the map-graph.
 *
 * The node can represent either a static or dynamic landmark and has multiple
 * measurements associated with it.
 *
 * @tparam MEASUREMENT
 */
template <typename MEASUREMENT>
class LandmarkNode : public MapNodeBase<MEASUREMENT> {
 public:
  using Base = MapNodeBase<MEASUREMENT>;
  using This = LandmarkNode<MEASUREMENT>;

  // Map of measurements, via the frame this measurement was seen in
  using Measurements = gtsam::FastMap<FrameNodePtr<MEASUREMENT>, MEASUREMENT>;
  DYNO_POINTER_TYPEDEFS(This)

  LandmarkNode(const std::shared_ptr<Map<MEASUREMENT>>& map)
      : MapNodeBase<MEASUREMENT>(map) {}
  virtual ~LandmarkNode() = default;

  /// @brief Unique tracklet (i) of the landmark/
  TrackletId tracklet_id;
  /// @brief Object label (j) of the landmark.
  ObjectId object_id;

  /**
   * @brief Returns the tracklet_id
   *
   * @return int
   */
  int getId() const override;

  std::string toString() const override {
    std::stringstream ss;
    if (isStatic()) {
      ss << "Static point: ";
    } else {
      ss << "Dynamic point (" << object_id << "): ";
    }
    // tracklet ID
    ss << getId() << "\n";
    ss << "Num obs: " << numObservations();
    return ss.str();
  }

  ObjectId getObjectId() const;

  /**
   * @brief Returns true if the landmark is static.
   *
   * Simply checks the background label, so could be dangerous if the label
   * changes (but this should never happen!!)
   *
   * @return true
   * @return false
   */
  bool isStatic() const;

  /**
   * @brief Number of observations of this landmark.
   *
   * @return size_t
   */
  size_t numObservations() const;

  /**
   * @brief Get all the frame nodes this landmark was observed in.
   *
   * @return const FrameNodePtrSet<MEASUREMENT>&
   */
  inline const FrameNodePtrSet<MEASUREMENT>& getSeenFrames() const {
    return frames_seen_;
  }

  /**
   * @brief Get all the frame id's this landmark was observed in.
   *
   * @return FrameIds
   */
  FrameIds getSeenFrameIds() const;

  /**
   * @brief Get all the measurements for this landmark.
   *
   * @return const Measurements&
   */
  inline const Measurements& getMeasurements() const { return measurements_; }

  /**
   * @brief True if the landmark was observed at the requested frame.
   *
   * @param frame_id FrameId
   * @return true
   * @return false
   */
  bool seenAtFrame(FrameId frame_id) const;

  /**
   * @brief True if the landmark was a measurement at the requested frame.
   * Should be true if seenAtFrame() is also true.
   *
   * @param frame_id FrameId
   * @return true
   * @return false
   */
  bool hasMeasurement(FrameId frame_id) const;

  /**
   * @brief Get the measurement at the requested frame node.
   * Throws DynosamException if no measurement existd at this frame; use with
   * seenAtFrame or hasMeasurement.
   *
   * @param frame_node FrameNodePtr<MEASUREMENT>
   * @return const MEASUREMENT&
   */
  const MEASUREMENT& getMeasurement(FrameNodePtr<MEASUREMENT> frame_node) const;

  /**
   * @brief Get the measurement at the requested frame id.
   * Throws DynosamException if no measurement existd at this frame; use with
   * seenAtFrame or hasMeasurement.
   *
   * @param frame_id FrameId
   * @return const MEASUREMENT&
   */
  const MEASUREMENT& getMeasurement(FrameId frame_id) const;

  /**
   * @brief Adds a measurement with the associated frame id.
   *
   * @param frame_node FrameNodePtr<MEASUREMENT>
   * @param measurement const MEASUREMENT&
   */
  void add(FrameNodePtr<MEASUREMENT> frame_node,
           const MEASUREMENT& measurement);

  /**
   * @brief Construcs a static landmark key for this landmark. The tracklet id
   * will be used to construct a unique key.
   *
   * @exception DynosamException if the landmark is not static.
   *
   *
   * @return gtsam::Key
   */
  gtsam::Key makeStaticKey() const;

  /**
   * @brief Construcs a dynamic landmark key for this landmark.
   * @see LandmarkNode<MEASUREMENT>#makeDynamicSymbol
   *
   * @param frame_id
   * @return gtsam::Key
   */
  gtsam::Key makeDynamicKey(FrameId frame_id) const;

  /**
   * @brief Construcs a dynamic landmark symbol for this landmark. The frame id
   * and tracklet id will be used to construct a unique key.
   * @exception DynosamException if the landmark is not dynamic.
   *
   * @param frame_id FrameId
   * @return gtsam::Key DynamicPointSymbol
   */
  DynamicPointSymbol makeDynamicSymbol(FrameId frame_id) const;

 protected:
  FrameNodePtrSet<MEASUREMENT> frames_seen_;
  Measurements measurements_;
};

}  // namespace dyno

#include "dynosam_common/MapNodes-inl.hpp"
