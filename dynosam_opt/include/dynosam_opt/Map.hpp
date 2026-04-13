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

#include <gtsam/base/FastMap.h>

#include <memory>

#include "dynosam_common/StructuredContainers.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_opt/Symbols.hpp"

namespace dyno {

struct InvalidLandmarkQuery : public DynosamException {
  InvalidLandmarkQuery(gtsam::Key key, const std::string& string)
      : DynosamException("Landmark estimate query failed with key " +
                         DynosamKeyFormatter(key) + ", reason: " + string) {}
};

namespace internal {

// Type traits to check the existance of the getId function for a Node
template <typename, typename = std::void_t<>>
struct HasGetId : std::false_type {};

template <typename T>
struct HasGetId<T, std::void_t<decltype(std::declval<const T&>().getId())>>
    : std::true_type {};
}  // namespace internal

template <typename Key, typename NODE>
struct NodeTraits {
  //! Storage key type, may be different for each node
  //! Used for comparison operator in SharedNodeSet and
  //! as the key_type in MapInterface<>##SharedNodes
  //! Must match the return value type of Node##getId()
  typedef NODE Node;
  typedef Key KeyType;
  typedef std::shared_ptr<Node> SharedNode;
  typedef gtsam::FastMap<KeyType, SharedNode> SharedNodeMap;

  static KeyType getKey(const Node& node) { return node.getId(); }

  static KeyType getKey(const SharedNode& node) { return getKey(*node); }

  struct Compare {
    // enables heterogeneous lookup
    using is_transparent = void;

    bool operator()(const SharedNode& a, const SharedNode& b) const {
      return a->getId() < b->getId();
    }

    bool operator()(const SharedNode& a, KeyType id) const {
      return a->getId() < id;
    }
    bool operator()(KeyType id, const SharedNode& a) const {
      return id < a->getId();
    }
  };

  /** Define a specalist node set that also has some specific functionality */
  class SharedNodeSet : public dyno::FastSet<SharedNode, Compare> {
   public:
    typedef dyno::FastSet<SharedNode, Compare> Base;
    using Base::Base;

    SharedNodeSet() = default;

    /** Additional exists function */
    bool exists(KeyType key) const { return this->find(key) != this->end(); }

    std::vector<KeyType> collectKeys() const {
      std::vector<KeyType> keys;
      keys.reserve(this->size());

      for (const auto& node : *this) {
        keys.push_back(NodeTraits::getKey(node));
      }
      return keys;
    }
  };
};

template <typename KeyType, typename Node>
class MapInterfaceBase {
 public:
  typedef NodeTraits<KeyType, Node> NodeTraitsT;
  //! Shared pointer to the Node
  typedef typename NodeTraitsT::SharedNode SharedNode;
  //! Fast Map of KeyType to shared node pointer
  typedef typename NodeTraitsT::SharedNodeMap SharedNodeMap;
  typedef typename SharedNodeMap::iterator iterator;
  typedef typename SharedNodeMap::const_iterator const_iterator;

  const SharedNodeMap& getNodes() const { return nodes_; }
  SharedNodeMap& getNodes() { return nodes_; }

  template <typename... Args>
  void emplace_shared(Args&&... args) {
    this->push_back(std::make_shared<Node>(std::forward<Args>(args)...));
  }

  void push_back(SharedNode node) {
    nodes_.insert2(NodeTraitsT::getKey(node), node);
  }

  bool exists(KeyType key) const { return nodes_.exists(key); }

  const SharedNode& at(KeyType key) const { return nodes_.at(key); }

  SharedNode& at(KeyType key) { return nodes_.at(key); }

  size_t size() const { return nodes_.size(); }
  bool empty() const { return nodes_.empty(); }

  std::vector<KeyType> collectKeys() const {
    std::vector<KeyType> keys;
    keys.reserve(this->size());

    for (const auto& [_, node] : nodes_) {
      keys.push_back(NodeTraitsT::getKey(node));
    }
    return keys;
  }

  const_iterator begin() const { return nodes_.begin(); }
  const_iterator end() const { return nodes_.end(); }

  /** Get the first node ordered by Node##getId */
  SharedNode front() const { return nodes_.front(); }

  /** Get the last node ordered by Node##getId */
  SharedNode back() const { return nodes_.back(); }

  /** non-const STL-style begin() */
  iterator begin() { return nodes_.begin(); }

  /** non-const STL-style end() */
  iterator end() { return nodes_.end(); }

 protected:
  //! FastMap of SharedNodes
  SharedNodeMap nodes_;
};

template <typename FrameNode>
class FrameNodeInterface : public MapInterfaceBase<FrameId, FrameNode> {
 public:
  using Base = MapInterfaceBase<FrameId, FrameNode>;
  using SharedNode = typename Base::SharedNode;
  using SharedNodeMap = typename Base::SharedNodeMap;

  size_t numFrames() const { return this->size(); }

  bool frameExists(FrameId frame_id) const {
    return this->template exists(frame_id);
  }

  SharedNode getFrame(FrameId frame_id) const {
    if (!frameExists(frame_id)) {
      return nullptr;
    }
    return this->template at(frame_id);
  }

  SharedNode lastFrame() const { return this->nodes_.crbegin()->second; }
  SharedNode firstFrame() const { return this->nodes_.cbegin()->second; }
  FrameId lastFrameId() const { return lastFrame()->frameId(); }
  FrameId firstFrameId() const { return firstFrame()->frameId(); }
  Timestamp lastTimestamp() const { return lastFrame()->timestamp(); }
  Timestamp firstTimestamp() const { return firstFrame()->timestamp(); }

  FrameIds getFrameIds() const { return this->template collectKeys(); }

  const SharedNodeMap& getFrames() const { return this->template getNodes(); }
  SharedNodeMap& getFrames() { return this->template getNodes(); }

  /* Get all tracklets for static landmarks observed at this frame */
  TrackletIds staticTrackletsByFrame(FrameId frame_id) const {
    if (!frameExists(frame_id)) {
      return TrackletIds{};
    }

    const SharedNode frame_node = getFrame(frame_id);
    const auto& static_landmarks = frame_node->staticLandmarks();

    TrackletIds tracklet_ids;
    tracklet_ids.reserve(static_landmarks.size());
    for (const auto& landmark_node : static_landmarks) {
      tracklet_ids.push_back(landmark_node->trackletId());
    }
    return tracklet_ids;
  }
};

template <typename LandmarkNode>
class LandmarkNodeInterface
    : public MapInterfaceBase<TrackletId, LandmarkNode> {
 public:
  using Base = MapInterfaceBase<TrackletId, LandmarkNode>;
  using SharedNode = typename Base::SharedNode;
  using SharedNodeMap = typename Base::SharedNodeMap;

  bool landmarkExists(TrackletId tracklet_id) const {
    return this->template exists(tracklet_id);
  }

  SharedNode getLandmark(TrackletId tracklet_id) const {
    if (!landmarkExists(tracklet_id)) {
      return nullptr;
    }
    return this->template at(tracklet_id);
  }

  const SharedNodeMap& getLandmarks() const {
    return this->template getNodes();
  }
  SharedNodeMap& getLandmarks() { return this->template getNodes(); }
};

template <typename ObjectNode>
class ObjectNodeInterface : public MapInterfaceBase<ObjectId, ObjectNode> {
 public:
  using Base = MapInterfaceBase<ObjectId, ObjectNode>;
  using SharedNode = typename Base::SharedNode;
  using SharedNodeMap = typename Base::SharedNodeMap;

  bool objectExists(ObjectId object_id) const {
    return this->template exists(object_id);
  }

  SharedNode getObject(ObjectId object_id) const {
    if (!objectExists(object_id)) {
      return nullptr;
    }
    return this->template at(object_id);
  }

  ObjectIds getObjectIds() const { return this->template collectKeys(); }

  /**
   * @brief Get number of objects seen
   *
   * @return size_t
   */
  size_t numObjectsSeen() const { return this->size(); }

  const SharedNodeMap& getObjects() const { return this->template getNodes(); }
  SharedNodeMap& getObjects() { return this->template getNodes(); }
};

template <typename LandmarkNode>
using LandmarkNodeTraits = NodeTraits<TrackletId, LandmarkNode>;

template <typename ObjectNode>
using ObjectNodeTraits = NodeTraits<ObjectId, ObjectNode>;

template <typename FrameNode>
using FrameNodeTraits = NodeTraits<FrameId, FrameNode>;

template <typename NodeTypes>
class ObjectNodeBase {
 public:
  using M = typename NodeTypes::Measurement;
  using FrameNode = typename NodeTypes::FrameNodeT;
  using LandmarkNode = typename NodeTypes::LandmarkNodeT;

  typedef LandmarkNodeTraits<LandmarkNode> LandmarkNodeTraitsT;
  typedef typename LandmarkNodeTraitsT::SharedNodeSet Landmarks;

  typedef FrameNodeTraits<FrameNode> FrameNodeTraitsT;
  typedef typename FrameNodeTraitsT::SharedNodeSet Frames;

  ObjectNodeBase(ObjectId object_id) : object_id_(object_id) {}

  ObjectId getId() const { return objectId(); }
  ObjectId objectId() const { return object_id_; }

  // change this behaviour in deriving classes to change which frames are
  // counted
  Frames getSeenFrames() const {
    Frames seen_frames;
    for (const auto& lmk : dynamic_landmarks_) {
      seen_frames.merge(lmk->getSeenFrames());
    }
    return seen_frames;
  }

  FrameIds getSeenFrameIds() const {
    // slow call as we basically iterate over the frames twice
    // but idea is to reuse the getSeenFrames() function as often as
    // possible
    Frames seen_frames = getSeenFrames();
    return seen_frames.template collectKeys();
  }

  FrameId getFirstSeenFrame() const {
    const auto first_frame = getSeenFrames().front();
    return first_frame->frameId();
  }

  FrameId getLastSeenFrame() const {
    const auto last_frame = getSeenFrames().back();
    return last_frame->frameId();
  }

  Landmarks landmarksSeenAtFrame(FrameId frame_id) const {
    Landmarks seen_lmks;
    for (const auto& lmk : dynamic_landmarks_) {
      // all frames this lmk was seen in
      const Frames& frames = lmk->getSeenFrames();
      // lmk was observed at this frame
      if (frames.find(frame_id) != frames.end()) {
        seen_lmks.insert(lmk);
      }
    }
    return seen_lmks;
  }

  /**
   * @brief Gets the frame id seen immediately before the latest one!
   * If the object has only been seen once, return false
   *
   * @param frame_id
   * @return true
   * @return false
   */
  bool previouslySeenFrame(FrameId* frame_id = nullptr) const {
    const Frames all_frames_seen = this->getSeenFrames();
    if (all_frames_seen.size() < 2) {
      return false;
    }

    if (frame_id) {
      const auto& prev_frame = *std::next(all_frames_seen.crbegin());
      *frame_id = prev_frame->frameId();
    }
    return true;
  }

  /**
   * @brief If the object has been seen before current frame, and if so,
   * at which frame (previous frame).
   *
   * May not be immediately before if there is a jump in the trajectory.
   *
   * Returns true if the object was observed at current frame and
   * has a previous observation.
   *
   * @param current_frame FrameId
   * @param previous_frame FrameId*
   * @return true
   * @return false
   */
  bool previouslySeenFrame(FrameId current_frame,
                           FrameId* previous_frame = nullptr) const {
    const Frames all_frames_seen = this->getSeenFrames();
    if (all_frames_seen.size() < 2) {
      return false;
    }

    auto current_frame_itr = all_frames_seen.find(current_frame);

    // Object not seen at current frame
    if (current_frame_itr == all_frames_seen.end()) {
      return false;
    }

    // If this is the first (smallest) frame, there is no previous
    if (current_frame_itr == all_frames_seen.begin()) {
      return false;
    }

    // Move iterator one step back
    auto previous_frame_itr = std::prev(current_frame_itr);

    if (previous_frame) {
      *previous_frame = (*previous_frame_itr)->frameId();
    }

    return true;
  }

  const Landmarks& landmarks() const { return dynamic_landmarks_; }
  Landmarks& landmarks() { return dynamic_landmarks_; }

  /* Get tracklet ids for all landmarks */
  TrackletIds trackletIds() const {
    return dynamic_landmarks_.template collectKeys();
  }

 protected:
  ObjectId object_id_;
  Landmarks dynamic_landmarks_;
};

template <typename NodeTypes>
class FrameNodeBase {
 public:
  using M = typename NodeTypes::Measurement;
  using LandmarkNode = typename NodeTypes::LandmarkNodeT;
  using ObjectNode = typename NodeTypes::ObjectNodeT;

  typedef LandmarkNodeTraits<LandmarkNode> LandmarkNodeTraitsT;
  typedef ObjectNodeTraits<ObjectNode> ObjectNodeTraitsT;

  typedef typename LandmarkNodeTraitsT::SharedNode SharedLandmark;

  typedef typename LandmarkNodeTraitsT::SharedNodeSet Landmarks;
  typedef typename ObjectNodeTraitsT::SharedNodeSet Objects;

  FrameNodeBase(FrameId frame_id, Timestamp timestamp)
      : frame_id_(frame_id), timestamp_(timestamp) {}

  FrameId getId() const { return frameId(); }
  FrameId frameId() const { return frame_id_; }
  Timestamp timestamp() const { return timestamp_; }

  /**
   * @brief True if the requested object was observed in this frame.
   *
   * @param object_id ObjectId
   * @return true
   * @return false
   */
  bool objectObserved(ObjectId object_id) const {
    return objects_.exists(object_id);
  }

  gtsam::Key makePoseKey() const { return CameraPoseSymbol(frame_id_); }
  /**
   * @brief Consturcts an object motion key.
   * The associated motion will be from k-1 to k.
   *
   * @param object_id ObjectId
   * @return gtsam::Key
   */
  gtsam::Key makeObjectMotionKey(ObjectId object_id) const {
    return ObjectMotionSymbol(object_id, frame_id_);
  }

  /**
   * @brief Construct an object pose key.
   * The associated pose will be for frame k.
   *
   * @param object_id ObjectId
   * @return gtsam::Key
   */
  gtsam::Key makeObjectPoseKey(ObjectId object_id) const {
    return ObjectPoseSymbol(object_id, frame_id_);
  }

  const Objects& objectsSeen() const { return objects_; }
  Objects& objectsSeen() { return objects_; }

  ObjectIds objectSeenIds() const {
    return objectsSeen().template collectKeys();
  }

  void setInitialSensorPose(const Pose3Measurement& X_W_k) { X_W_k_ = X_W_k; }

  /* Safe gets the initial sensor pose if availble by out-arg */
  bool getInitialSensorPose(Pose3Measurement& X_W_k) const {
    if (!X_W_k_) {
      return false;
    }
    X_W_k = X_W_k_.value();
    return true;
  }

  /* Returns the initial sensor pose if available, throws DynosamException if
   * not */
  Pose3Measurement initialSensorPose() const {
    Pose3Measurement X_W_k;
    if (!getInitialSensorPose(X_W_k)) {
      DYNO_THROW_MSG(DynosamException)
          << "No initial sensor pose for FrameNode k=" << this->frameId();
    }
    return X_W_k;
  }

  const Landmarks& dynamicLandmarks() const { return dynamic_landmarks_; }
  Landmarks& dynamicLandmarks() { return dynamic_landmarks_; }

  const Landmarks& staticLandmarks() const { return static_landmarks_; }
  Landmarks& staticLandmarks() { return static_landmarks_; }

  Landmarks dynamicLandmarks(ObjectId object_id) const {
    Landmarks landmarks_j;

    if (!objectObserved(object_id)) {
      return landmarks_j;
    }

    for (const auto& lmk_node : dynamic_landmarks_) {
      if (lmk_node->objectId() == object_id) {
        landmarks_j.insert(lmk_node);
      }
    }
    return landmarks_j;
  }

  /// @brief Const SharedLandmarkNode with corresponding Measurement value
  using LandmarkMeasurementPair = std::pair<const SharedLandmark, M>;

  std::vector<LandmarkMeasurementPair> staticMeasurements() const {
    return measurementsFromLandmarks(this->static_landmarks_);
  }

  std::vector<LandmarkMeasurementPair> dynamicMeasurements() const {
    return measurementsFromLandmarks(this->dynamic_landmarks_);
  }

  std::vector<LandmarkMeasurementPair> dynamicMeasurements(
      ObjectId object_id) const {
    const Landmarks lmks_j = dynamicLandmarks(object_id);
    return measurementsFromLandmarks(lmks_j);
  }

 private:
  /** Construct a vector of LandmarkMeasurementPair from a set of input
   * Landmarks using this frame for measurements
   */
  std::vector<LandmarkMeasurementPair> measurementsFromLandmarks(
      const Landmarks& landmarks) const {
    std::vector<LandmarkMeasurementPair> measurements;
    measurements.reserve(landmarks.size());

    for (const auto& lmk_node : landmarks) {
      const M& m = lmk_node->getMeasurement(this->frameId());
      measurements.push_back(std::make_pair(lmk_node, m));
    }
    return measurements;
  }

 protected:
  FrameId frame_id_;
  Timestamp timestamp_;

  Landmarks dynamic_landmarks_;
  Landmarks static_landmarks_;

  Objects objects_;

  /// @brief Optional initial camera pose in world, provided by the front-end
  std::optional<Pose3Measurement> X_W_k_;
};

// by using derived, the only functions that need changing are getSeenFrames but
// also add
template <typename NodeTypes>
class LandmarkNodeBase {
 protected:
  using Derived = typename NodeTypes::LandmarkNodeT;

 public:
  using M = typename NodeTypes::Measurement;
  using FrameNode = typename NodeTypes::FrameNodeT;
  using ObjectNode = typename NodeTypes::ObjectNodeT;

  typedef FrameNodeTraits<FrameNode> FrameNodeTraitsT;
  typedef typename FrameNodeTraitsT::SharedNode SharedFrame;
  typedef typename FrameNodeTraitsT::SharedNodeSet Frames;

  // Map of measurements, via the frame this measurement was seen in
  using Measurements = gtsam::FastMap<SharedFrame, M>;

  LandmarkNodeBase(TrackletId tracklet_id, ObjectId object_id)
      : tracklet_id_(tracklet_id), object_id_(object_id) {}

  TrackletId getId() const { return trackletId(); }
  TrackletId trackletId() const { return tracklet_id_; }
  ObjectId objectId() const { return object_id_; }

  /**
   * @brief Returns true if the landmark is static.
   *
   * Simply checks the background label, so could be dangerous if the label
   * changes (but this should never happen!!)
   *
   * @return true
   * @return false
   */
  bool isStatic() const { return object_id_ == background_label; }

  /**
   * @brief Adds a measurement with the associated frame id.
   *
   * @param frame_node SharedFrame
   * @param measurement const M&
   */
  void add(SharedFrame frame_node, const M& measurement) {
    // add measurement to map
    // first check that we dont already have a measurement at this frame
    if (measurements_.exists(frame_node)) {
      const std::string info =
          this->isStatic()
              ? "Static"
              : std::string("Dynamic j=" + std::to_string(object_id_));
      DYNO_THROW_MSG(DynosamException)
          << "Unable to add new measurement to landmark node "
          << "i= " << tracklet_id_ << " k=" << frame_node->frameId() << " ("
          << info << ")"
          << " as a measurement already exists at this frame!";
      throw;
    }

    frames_.insert(frame_node);
    measurements_.insert2(frame_node, measurement);
    CHECK_EQ(frames_.size(), measurements_.size());
  }

  /**
   * @brief True if the landmark was observed at the requested frame.
   *
   * @param frame_id FrameId
   * @return true
   * @return false
   */
  bool seenAtFrame(FrameId frame_id) const {
    return asDerived().getSeenFrames().exists(frame_id);
  }

  /**
   * @brief Get the measurement at the requested frame node.
   * Throws DynosamException if no measurement existd at this frame; use with
   * seenAtFrame or hasMeasurement.
   *
   * @param frame_node SharedFrame
   * @return const M&
   */
  const M& getMeasurement(SharedFrame frame_node) const {
    CHECK_NOTNULL(frame_node);
    if (!asDerived().seenAtFrame(frame_node->frameId())) {
      throw DynosamException("Missing measurement in landmark node with id " +
                             std::to_string(tracklet_id_) + " at frame " +
                             std::to_string(frame_node->frameId()));
    }
    return measurements_.at(frame_node);
  }

  const Measurements& getMeasurements() const { return measurements_; }

  /**
   * @brief Get the measurement at the requested frame id.
   * Throws DynosamException if no measurement existd at this frame; use with
   * seenAtFrame or hasMeasurement.
   *
   * @param frame_id FrameId
   * @return const M&
   */
  const M& getMeasurement(FrameId frame_id) const {
    if (!asDerived().seenAtFrame(frame_id)) {
      DYNO_THROW_MSG(DynosamException)
          << "Missing measurement in landmark node "
          << "i=" << tracklet_id_ << " k=" << frame_id;
    }

    const auto frames = asDerived().getSeenFrames();
    SharedFrame frame = *frames.find(frame_id);
    return getMeasurement(frame);
  }

  const Frames& getSeenFrames() const { return frames_; }
  Frames& getSeenFrames() { return frames_; }

  FrameIds getSeenFrameIds() const {
    // slow call as we basically iterate over the frames twice
    // but idea is to reuse the getSeenFrames() function as often as
    // possible
    const Frames& seen_frames = asDerived().getSeenFrames();
    return seen_frames.template collectKeys();
  }

  /** Return number of frames landmark is observed in */
  size_t numObservations() const { return asDerived().getSeenFrames().size(); }

  /**
   * @brief Construcs a static landmark key for this landmark. The tracklet id
   * will be used to construct a unique key.
   *
   * @exception DynosamException if the landmark is not static.
   *
   *
   * @return gtsam::Key
   */
  gtsam::Key makeStaticKey() const {
    const auto key = StaticLandmarkSymbol(tracklet_id_);
    if (!this->isStatic()) {
      throw InvalidLandmarkQuery(
          key, "Static estimate requested but landmark is dynamic!");
    }
    return key;
  }

  /**
   * @brief Construcs a dynamic landmark key for this landmark.
   * @see LandmarkNode<MEASUREMENT>#makeDynamicSymbol
   *
   * @param frame_id
   * @return gtsam::Key
   */
  gtsam::Key makeDynamicKey(FrameId frame_id) const {
    const auto key = DynamicLandmarkSymbol(frame_id, tracklet_id_);
    if (this->isStatic()) {
      throw InvalidLandmarkQuery(
          key, "Dynamic estimate requested but landmark is static!");
    }
    return key;
  }

 protected:
 private:
  const Derived& asDerived() const {
    return static_cast<const Derived&>(*this);
  }

  Derived& asDerived() { return static_cast<Derived&>(*this); }

 protected:
  TrackletId tracklet_id_;
  ObjectId object_id_;

  Frames frames_;
  Measurements measurements_;

  // bool inlier_{true};
  // bool in_optimisation
};

template <typename NodeTypes>
class RegularObjectNode : public ObjectNodeBase<NodeTypes> {
 public:
  typedef RegularObjectNode<NodeTypes> This;
  typedef ObjectNodeBase<NodeTypes> Base;
  DYNO_POINTER_TYPEDEFS(This)

  RegularObjectNode(ObjectId object_id) : Base(object_id) {}
};

template <typename NodeTypes>
class RegularFrameNode : public FrameNodeBase<NodeTypes> {
 public:
  typedef RegularFrameNode<NodeTypes> This;
  typedef FrameNodeBase<NodeTypes> Base;
  DYNO_POINTER_TYPEDEFS(This)

  RegularFrameNode(FrameId frame_id, Timestamp timestamp)
      : Base(frame_id, timestamp) {}

  void setInitialObjectMotions(const MotionEstimateMap& motions) {
    H_W_km1_ks_ = motions;
  }

  bool getInitialObjectMotion(ObjectId object_id,
                              Motion3ReferenceFrame& motion) const {
    if (!H_W_km1_ks_ || !this->objectObserved(object_id)) {
      return false;
    }

    const MotionEstimateMap& motions = H_W_km1_ks_.value();
    if (!motions.exists(object_id)) {
      return false;
    }

    motion = motions.at(object_id);
    return true;
  }

  Motion3ReferenceFrame initialObjectMotion(ObjectId object_id) {
    Motion3ReferenceFrame H;
    if (!getInitialObjectMotion(object_id, H)) {
      DYNO_THROW_MSG(DynosamException)
          << "No initial object motion for FrameNode k=" << this->frameId()
          << " at requested object j=" << object_id;
    }
    return H;
  }

 protected:
  /// @brief Optional initial object motions in the world, provided by the
  /// front-end
  std::optional<MotionEstimateMap> H_W_km1_ks_;
};

template <typename NodeTypes>
class RegularLandmarkNode : public LandmarkNodeBase<NodeTypes> {
 public:
  typedef RegularLandmarkNode<NodeTypes> This;
  typedef LandmarkNodeBase<NodeTypes> Base;
  DYNO_POINTER_TYPEDEFS(This)

  RegularLandmarkNode(TrackletId tracklet_id, ObjectId object_id)
      : Base(tracklet_id, object_id) {}
};

template <typename Measurement_>
struct RegularNodeTypes {
  using Measurement = Measurement_;
  using ObjectNodeT = RegularObjectNode<RegularNodeTypes>;
  using FrameNodeT = RegularFrameNode<RegularNodeTypes>;
  using LandmarkNodeT = RegularLandmarkNode<RegularNodeTypes>;
};

template <typename NodeTypes = RegularNodeTypes<Keypoint>>
class Map : public FrameNodeInterface<typename NodeTypes::FrameNodeT>,
            public LandmarkNodeInterface<typename NodeTypes::LandmarkNodeT>,
            public ObjectNodeInterface<typename NodeTypes::ObjectNodeT>,
            public std::enable_shared_from_this<Map<NodeTypes>> {
 public:
  /* Public constructor mostly for testing but should not really be used
   * except with std::make_shared to ensure proper memory ownership
   * with enable_shared_from_this!
   */
  Map() = default;
  virtual ~Map() = default;

  typedef typename NodeTypes::FrameNodeT FrameNodeT;
  typedef typename NodeTypes::LandmarkNodeT LandmarkNodeT;
  typedef typename NodeTypes::ObjectNodeT ObjectNodeT;

  typedef typename NodeTypes::Measurement Measurement;

  typedef FrameNodeInterface<FrameNodeT> FrameNodeInterfaceT;
  typedef LandmarkNodeInterface<LandmarkNodeT> LandmarkNodeInterfaceT;
  typedef ObjectNodeInterface<ObjectNodeT> ObjectNodeInterfaceT;

  typedef FrameNodeTraits<FrameNodeT> FrameNodeTraitsT;
  typedef LandmarkNodeTraits<LandmarkNodeT> LandmarkNodeTraitsT;
  typedef ObjectNodeTraits<ObjectNodeT> ObjectNodeTraitsT;

  typedef typename FrameNodeTraitsT::SharedNode SharedFrameNodeT;
  typedef typename LandmarkNodeTraitsT::SharedNode SharedLandmarkNodeT;
  typedef typename ObjectNodeTraitsT::SharedNode SharedObjectNodeT;

  typedef typename FrameNodeTraitsT::SharedNodeSet SharedFrameSet;
  typedef typename LandmarkNodeTraitsT::SharedNodeSet SharedLandmarkSet;
  typedef typename ObjectNodeTraitsT::SharedNodeSet SharedObjectSet;

 private:
  /* Simple struct to validate the internals of a node (as we're in C++17 no
   * Concepts) */
  template <typename NodeTraits>
  struct NodeTraitsValidator {
    using KeyType = typename NodeTraits::KeyType;
    using Node = typename NodeTraits::Node;
    static_assert(internal::HasGetId<Node>::value, "Node must have getId()");
    static_assert(
        std::is_same_v<decltype(std::declval<const Node&>().getId()), KeyType>,
        "getId() must return the type specified by Node::KeyType");

    // additionally all nodes must at least inherit from the XNodeBase class
    // ie. LandmarkNodeBase<>, ObjectNodeBase<> and FrameNodeBase<>
    // as the map class depends on these functionalities to exist!
  };

  typedef NodeTraitsValidator<FrameNodeTraitsT> FrameTraitsValidator;
  typedef NodeTraitsValidator<LandmarkNodeTraitsT> LandmarkTraitsValidator;
  typedef NodeTraitsValidator<ObjectNodeTraitsT> ObjectTraitsValidator;

 public:
  /// @brief Alias to a GenericTrackedStatusVector using the templated
  /// Measurement type, specifying that StatusVector must contain the desired
  /// measurement type
  /// @tparam DERIVEDSTATUS
  template <typename DERIVEDSTATUS>
  using MeasurementStatusVector =
      GenericTrackedStatusVector<DERIVEDSTATUS, Measurement>;

  typedef Map<NodeTypes> This;

  std::shared_ptr<const This> getPtr() const {
    return this->shared_from_this();
  }
  std::shared_ptr<This> getPtr() { return this->shared_from_this(); }

  std::shared_ptr<const FrameNodeInterfaceT> asFrameInterface() const {
    return std::dynamic_pointer_cast<const FrameNodeInterfaceT>(this->getPtr());
  }

  std::shared_ptr<const LandmarkNodeInterfaceT> asLandmarkInterface() const {
    return std::dynamic_pointer_cast<const LandmarkNodeInterfaceT>(
        this->getPtr());
  }

  std::shared_ptr<const ObjectNodeInterfaceT> asObjectInterface() const {
    return std::dynamic_pointer_cast<const ObjectNodeInterfaceT>(
        this->getPtr());
  }

  std::shared_ptr<FrameNodeInterfaceT> asFrameInterface() {
    return std::dynamic_pointer_cast<FrameNodeInterfaceT>(this->getPtr());
  }

  std::shared_ptr<LandmarkNodeInterfaceT> asLandmarkInterface() {
    return std::dynamic_pointer_cast<LandmarkNodeInterfaceT>(this->getPtr());
  }

  std::shared_ptr<ObjectNodeInterfaceT> asObjectInterface() {
    return std::dynamic_pointer_cast<ObjectNodeInterfaceT>(this->getPtr());
  }

  template <typename DERIVEDSTATUS>
  void updateObservations(
      const GenericTrackedStatusVector<DERIVEDSTATUS>& measurements) {
    using DerivedMeasurement =
        typename GenericTrackedStatusVector<DERIVEDSTATUS>::Value;

    for (const DERIVEDSTATUS& status_measurement : measurements) {
      const GenericValueTrack<DerivedMeasurement>& derived_status =
          static_cast<const GenericValueTrack<DerivedMeasurement>&>(
              status_measurement);
      const GenericValueTrack<Measurement>& track =
          derived_status.template asType<Measurement>();
      // thread safe update
      updateFromTrack(track);
    }
  }

  template <typename DERIVEDSTATUS>
  void updateObservations(const DERIVEDSTATUS& derived_status) {
    updateObservations(
        GenericTrackedStatusVector<DERIVEDSTATUS>({derived_status}));
  }

  /* Sets initial camera pose (X_W_k) and creates FrameNode if does not exist */
  void setInitialSensorPose(FrameId frame_id, Timestamp timestamp,
                            const Pose3Measurement& X_W_k) {
    auto frame_interface = this->asFrameInterface();

    if (!frame_interface->frameExists(frame_id)) {
      frame_interface->emplace_shared(frame_id, timestamp);
    }

    auto frame_node = frame_interface->getFrame(frame_id);
    frame_node->setInitialSensorPose(X_W_k);
  }

 private:
  typedef GenericValueTrack<Measurement> GenericValueTrackT;

  void updateFromTrack(const GenericValueTrackT& track) {
    const Measurement& measurement = track.value();
    const TrackletId tracklet_id = track.trackletId();
    const FrameId frame_id = track.frameId();
    const Timestamp timestamp = track.timestamp();
    const ObjectId object_id = track.objectId();
    const bool is_static = track.isStatic();

    CHECK((is_static && object_id == background_label) ||
          (!is_static && object_id != background_label));

    auto frame_interface = this->asFrameInterface();
    auto landmark_interface = this->asLandmarkInterface();

    if (!landmark_interface->landmarkExists(tracklet_id)) {
      landmark_interface->push_back(
          std::make_shared<LandmarkNodeT>(tracklet_id, object_id));
    }

    if (!frame_interface->frameExists(frame_id)) {
      frame_interface->push_back(
          std::make_shared<FrameNodeT>(frame_id, timestamp));
    }

    SharedLandmarkNodeT landmark_node =
        landmark_interface->getLandmark(tracklet_id);
    SharedFrameNodeT frame_node = frame_interface->getFrame(frame_id);

    CHECK_NOTNULL(landmark_node);
    CHECK_NOTNULL(frame_node);

    CHECK_EQ(landmark_node->trackletId(), tracklet_id);
    CHECK_EQ(frame_node->frameId(), frame_id);

    // this might fail of a tracklet get associated with a different object
    if (landmark_node->objectId() != object_id) {
      DYNO_THROW_MSG(DynosamException)
          << "Trackings inconsistency detected: "
          << " Landmark i=" << tracklet_id
          << " with j=" << landmark_node->objectId()
          << " has new measurement with different object id " << object_id;
      throw;
    }

    landmark_node->add(frame_node, measurement);

    if (is_static) {
      frame_node->staticLandmarks().insert(landmark_node);
    } else {
      CHECK(object_id != background_label);

      auto object_interface = this->asObjectInterface();
      if (!object_interface->objectExists(object_id)) {
        object_interface->push_back(std::make_shared<ObjectNodeT>(object_id));
      }

      SharedObjectNodeT object_node = object_interface->getObject(object_id);
      CHECK_NOTNULL(object_node);

      object_node->landmarks().insert(landmark_node);
      frame_node->dynamicLandmarks().insert(landmark_node);
      frame_node->objectsSeen().insert(object_node);
    }
  }
};

template <typename M>
class RegularMap : public Map<RegularNodeTypes<M>> {
  struct Private {};

 public:
  using This = RegularMap<M>;
  using Base = Map<RegularNodeTypes<M>>;
  DYNO_POINTER_TYPEDEFS(This)

  // Constructor is only usable by this class
  RegularMap(Private) {}

  static std::shared_ptr<This> create() {
    return std::make_shared<This>(Private());
  }

  void setInitialObjectMotions(FrameId frame_id,
                               const MotionEstimateMap& motions) {
    auto frame_interface = this->asFrameInterface();
    auto frame_node = frame_interface->getFrame(frame_id);
    CHECK_NOTNULL(frame_node);
    frame_node->setInitialObjectMotions(motions);
  }

  bool getInitialObjectMotion(FrameId frame_id, ObjectId object_id,
                              Motion3ReferenceFrame& motion_frame) const {
    auto frame_interface = this->asFrameInterface();
    if (!frame_interface->frameExists(frame_id)) {
      return false;
    }

    auto frame_node = frame_interface->getFrame(frame_id);
    return frame_node->getInitialObjectMotion(object_id, motion_frame);
  }

  Motion3ReferenceFrame initialObjectMotion(FrameId frame_id,
                                            ObjectId object_id) const {
    auto frame_interface = this->asFrameInterface();
    if (!frame_interface->frameExists(frame_id)) {
      DYNO_THROW_MSG(DynosamException)
          << "No initial object motion for FrameNode k=" << frame_id
          << " at requested object j=" << object_id
          << " as frame does not exist!";
    }

    auto frame_node = frame_interface->getFrame(frame_id);
    return frame_node->initialObjectMotion(object_id);
  }
};

using Map3d = RegularMap<Landmark>;
using ObjectNode3d = Map3d::ObjectNodeT;
using LandmarkNode3d = Map3d::LandmarkNodeT;
using FrameNode3d = Map3d::FrameNodeT;

using Map2d = RegularMap<Keypoint>;
using ObjectNode2d = Map2d::ObjectNodeT;
using LandmarkNode2d = Map2d::LandmarkNodeT;
using FrameNode2d = Map2d::FrameNodeT;

using MapVision = RegularMap<CameraMeasurement>;
using ObjectNodeV = MapVision::ObjectNodeT;
using LandmarkNodeV = MapVision::LandmarkNodeT;
using FrameNodeV = MapVision::FrameNodeT;

}  // namespace dyno
