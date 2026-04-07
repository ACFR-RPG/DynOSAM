#pragma once

#include <gtsam/base/FastMap.h>

#include <memory>

#include "dynosam_common/StructuredContainers.hpp"
#include "dynosam_common/Types.hpp"

using namespace dyno;

namespace dyno_testing {

template <typename Node>
struct NodeInterface {
  struct NodeCompare {
    // enables heterogeneous lookup
    using is_transparent = void;

    bool operator()(const Node& a, const Node& b) const {
      return a.getId() < b.getId();
    }

    bool operator()(const Node& a, int id) const { return a.getId() < id; }

    bool operator()(int id, const Node& a) const { return id < a.getId(); }
  };

  typedef std::shared_ptr<Node> SharedNode;
  typedef dyno::FastSet<Node, NodeCompare> SharedNodeSet;

  template <typename Key>
  static inline Key getKey(const Node& node) {
    return static_cast<Key>(node.getId());
  }

  template <typename Key>
  static inline Key getKey(const SharedNode& node) {
    return getKey<Key>(*node);
  }

  template <typename Key>
  static inline std::vector<Key> collectKeys(const SharedNodeSet& nodes) {
    std::vector<Key> keys;
    keys.resize(nodes.size());

    for (const auto& node : nodes) {
      keys.push_back(getKey<Key>(node));
    }
    return keys;
  }
};

template <typename Node>
class MapInterfaceBase {
 public:
  typedef std::shared_ptr<Node> SharedNode;
  // typedef gtsam::FastSet<Node> SharedNodeSet;
  typedef gtsam::FastMap<int, SharedNode> SharedNodes;
  typedef typename SharedNodes::iterator iterator;
  typedef typename SharedNodes::const_iterator const_iterator;

  typedef NodeInterface<Node> NodeInterfaceT;

  const SharedNodes& getNodes() const { return nodes_; }
  SharedNodes& getNodes() { return nodes_; }

  void add(SharedNode node) {
    nodes_.insert2(NodeInterfaceT::template getKey<int>(node), node);
  }

  template <typename Key>
  bool exists(Key key) const {
    return nodes_.exists(static_cast<int>(key));
  }

  bool exists(int key) const { return nodes_.exists(key); }

  template <typename Key>
  const SharedNode& at(Key key) const {
    return nodes_.at(static_cast<int>(key));
  }

  template <typename Key>
  SharedNode& at(Key key) {
    return nodes_.at(static_cast<int>(key));
  }

  size_t size() const { return nodes_.size(); }
  bool empty() const { return nodes_.empty(); }

  template <typename Key>
  std::vector<Key> collectKeys() const {
    std::vector<Key> keys;
    keys.resize(this->size());

    for (const auto& [_, node] : nodes_) {
      keys.push_back(NodeInterfaceT::template getKey<Key>(node));
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
  //! FastMap of SharedNode
  SharedNodes nodes_;
};

template <typename FrameNode>
class FrameNodeInterface : protected MapInterfaceBase<FrameNode> {
 public:
  using Base = MapInterfaceBase<FrameNode>;
  using SharedNode = typename Base::SharedNode;

  size_t numFrames() const { return this->size(); }

  bool frameExists(FrameId frame_id) const {
    return this->template exists<FrameId>(frame_id);
  }

  SharedNode getFrame(FrameId frame_id) const {
    return this->template at<FrameId>(frame_id);
  }

  SharedNode lastFrame() const { return this->nodes_.crbegin()->second; }
  SharedNode firstFrame() const { return this->nodes_.cbegin()->second; }
  FrameId lastFrameId() const { return lastFrame()->frameId(); }
  FrameId firstFrameId() const { return firstFrame()->frameId(); }
  Timestamp lastTimestamp() const { return lastFrame()->timestamp(); }
  Timestamp firstTimestamp() const { return firstFrame()->timestamp(); }

  FrameIds getAllFrameIds() const {
    return this->template collectKeys<FrameId>();
  }

  decltype(auto) getFrames() const { return this->template getNodes(); }
  decltype(auto) getFrames() { return this->template getNodes(); }
};

template <typename LandmarkNode>
class LandmarkNodeInterface : protected MapInterfaceBase<LandmarkNode> {
 public:
  using Base = MapInterfaceBase<LandmarkNode>;
  using SharedNode = typename Base::SharedNode;

  bool landmarkExists(TrackletId tracklet_id) const {
    return this->template exists<TrackletId>(tracklet_id);
  }

  SharedNode getLandmark(TrackletId tracklet_id) const {
    return this->template at<TrackletId>(tracklet_id);
  }

  decltype(auto) getLandmarks() const { return this->template getNodes(); }
  decltype(auto) getLandmarks() { return this->template getNodes(); }
};

template <typename ObjectNode>
class ObjectNodeInterface : protected MapInterfaceBase<ObjectNode> {
 public:
  using Base = MapInterfaceBase<ObjectNode>;
  using SharedNode = typename Base::SharedNode;

  bool objectExists(ObjectId object_id) const {
    return this->template exists<ObjectId>(object_id);
  }

  SharedNode getObject(ObjectId object_id) const {
    return this->template at<ObjectId>(object_id);
  }

  ObjectIds getAllObjectIds() const {
    return this->template collectKeys<ObjectId>();
  }

  /**
   * @brief Get number of objects seen
   *
   * @return size_t
   */
  size_t numObjectsSeen() const { return this->size(); }

  decltype(auto) getObjects() const { return this->template getNodes(); }
  decltype(auto) getObjects() { return this->template getNodes(); }
};

template <typename NodeTypes>
class ObjectNodeBase {
 public:
  using M = typename NodeTypes::Measurement;
  using FrameNode = typename NodeTypes::FrameNodeT;
  using LandmarkNode = typename NodeTypes::LandmarkNodeT;

  typedef NodeInterface<LandmarkNode> LandmarkNodeInterfaceT;
  typedef typename LandmarkNodeInterfaceT::SharedNodeSet Landmarks;

  ObjectNodeBase(ObjectId object_id) : object_id_(object_id) {}

  int getId() const { return static_cast<int>(object_id_); }
  ObjectId objectId() const { return object_id_; }

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

  typedef NodeInterface<LandmarkNode> LandmarkNodeInterfaceT;
  typedef NodeInterface<ObjectNode> ObjectNodeInterfaceT;

  typedef typename LandmarkNodeInterfaceT::SharedNodeSet Landmarks;
  typedef typename ObjectNodeInterfaceT::SharedNodeSet Objects;

  FrameNodeBase(FrameId frame_id, Timestamp timestamp)
      : frame_id_(frame_id), timestamp_(timestamp) {}

  int getId() const { return static_cast<int>(frame_id_); }
  FrameId frameId() const { return frame_id_; }
  Timestamp timestamp() const { return timestamp_; }

 protected:
  FrameId frame_id_;
  Timestamp timestamp_;

  Landmarks dynamic_landmarks_;
  Landmarks static_landmarks_;

  Objects objects_;
};

template <typename NodeTypes>
class LandmarkNodeBase {
 public:
  using M = typename NodeTypes::Measurement;
  using FrameNode = typename NodeTypes::FrameNodeT;
  using ObjectNode = typename NodeTypes::ObjectNodeT;

  typedef NodeInterface<FrameNode> FrameNodeInterfaceT;
  typedef typename FrameNodeInterfaceT::SharedNode SharedFrame;
  typedef typename FrameNodeInterfaceT::SharedNodeSet Frames;

  // Map of measurements, via the frame this measurement was seen in
  using Measurements = gtsam::FastMap<SharedFrame, M>;

  LandmarkNodeBase(TrackletId tracklet_id, ObjectId object_id)
      : tracklet_id_(tracklet_id), object_id_(object_id) {}

  int getId() const { return static_cast<int>(tracklet_id_); }
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
  bool seenAtFrame(FrameId frame_id) const { return frames_.exists(frame_id); }

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
    if (!seenAtFrame(frame_node->frameId())) {
      throw DynosamException("Missing measurement in landmark node with id " +
                             std::to_string(tracklet_id_) + " at frame " +
                             std::to_string(frame_node->frameId()));
    }
    return measurements_.at(frame_node);
  }

  /**
   * @brief Get the measurement at the requested frame id.
   * Throws DynosamException if no measurement existd at this frame; use with
   * seenAtFrame or hasMeasurement.
   *
   * @param frame_id FrameId
   * @return const M&
   */
  const M& getMeasurement(FrameId frame_id) const {
    if (!seenAtFrame(frame_id)) {
      throw DynosamException("Missing measurement in landmark node with id " +
                             std::to_string(tracklet_id_) + " at frame " +
                             std::to_string(frame_id));
    }

    SharedFrame frame = *frames_.find(frame_id);
    return getMeasurement(frame);
  }

 protected:
  TrackletId tracklet_id_;
  ObjectId object_id_;

  Frames frames_;
  Measurements measurements_;
};

// do these nodes need to know about the other object types?
template <typename NodeTypes>
class DefaultObjectNode : public ObjectNodeBase<NodeTypes> {
 public:
  using This = DefaultObjectNode<NodeTypes>;
  DYNO_POINTER_TYPEDEFS(This)
};

template <typename NodeTypes>
class DefaultFrameNode : public FrameNodeBase<NodeTypes> {
 public:
  using This = DefaultFrameNode<NodeTypes>;
  DYNO_POINTER_TYPEDEFS(This)
};

template <typename NodeTypes>
class DefaultLandmarkNode : public LandmarkNodeBase<NodeTypes> {
 public:
  using This = DefaultLandmarkNode<NodeTypes>;
  DYNO_POINTER_TYPEDEFS(This)
};

template <typename Measurement_>
struct DefaultNodeTypes {
  using Measurement = Measurement_;
  using ObjectNodeT = DefaultObjectNode<DefaultNodeTypes>;
  using FrameNodeT = DefaultFrameNode<DefaultNodeTypes>;
  using LandmarkNodeT = DefaultLandmarkNode<DefaultNodeTypes>;
};

template <typename NodeTypes = DefaultNodeTypes<Keypoint>>
class Map : public FrameNodeInterface<typename NodeTypes::FrameNodeT>,
            public LandmarkNodeInterface<typename NodeTypes::LandmarkNodeT>,
            public ObjectNodeInterface<typename NodeTypes::ObjectNodeT>,
            public std::enable_shared_from_this<Map<NodeTypes>> {
  struct Private {};

 public:
  typedef typename NodeTypes::FrameNodeT FrameNodeT;
  typedef typename NodeTypes::LandmarkNodeT LandmarkNodeT;
  typedef typename NodeTypes::ObjectNodeT ObjectNodeT;

  typedef typename NodeInterface<FrameNodeT>::SharedNode SharedFrameNodeT;
  typedef typename NodeInterface<LandmarkNodeT>::SharedNode SharedLandmarkNodeT;
  typedef typename NodeInterface<ObjectNodeT>::SharedNode SharedObjectNodeT;

  typedef typename NodeTypes::Measurement Measurement;

  typedef FrameNodeInterface<FrameNodeT> FrameNodeInterfaceT;
  typedef LandmarkNodeInterface<LandmarkNodeT> LandmarkNodeInterfaceT;
  typedef ObjectNodeInterface<ObjectNodeT> ObjectNodeInterfaceT;

  /// @brief Alias to a GenericTrackedStatusVector using the templated
  /// Measurement type, specifying that StatusVector must contain the desired
  /// measurement type
  /// @tparam DERIVEDSTATUS
  template <typename DERIVEDSTATUS>
  using MeasurementStatusVector =
      GenericTrackedStatusVector<DERIVEDSTATUS, Measurement>;

  typedef Map<NodeTypes> This;
  DYNO_POINTER_TYPEDEFS(This)

  Map(Private) {}

  static std::shared_ptr<This> create() {
    return std::make_shared<This>(Private());
  }

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

    auto object_interface = this->asObjectInterface();
    auto frame_interface = this->asFrameInterface();
    auto landmark_interface = this->asLandmarkInterface();

    if (!landmark_interface->landmarkExists(tracklet_id)) {
      landmark_interface->add(
          std::make_shared<LandmarkNodeT>(tracklet_id, object_id));
    }

    if (!frame_interface->frameExists(frame_id)) {
      frame_interface->add(std::make_shared<FrameNodeT>(frame_id, timestamp));
    }

    SharedLandmarkNodeT landmark_node =
        landmark_interface->getLandmark(tracklet_id);
    SharedFrameNodeT frame_node = frame_interface->getFrame(frame_id);

    CHECK_NOTNULL(landmark_node);
    CHECK_NOTNULL(frame_node);

    CHECK_EQ(landmark_node->trackletId(), tracklet_id);
    // this might fail of a tracklet get associated with a different object
    CHECK_EQ(landmark_node->objectId(), object_id);
    CHECK_EQ(frame_node->frameId(), frame_id);

    landmark_node->add(frame_node, measurement);

    if (is_static) {
      frame_node->static_landmarks.insert(landmark_node);
    } else {
      CHECK(object_id != background_label);

      if (!object_interface->objectExists(object_id)) {
        object_interface->add(std::make_shared<ObjectNodeT>(object_id));
      }

      SharedObjectNodeT object_node = object_interface->getObject(object_id);
      CHECK_NOTNULL(object_node);

      object_node->dynamic_landmarks.insert(landmark_node);
      frame_node->dynamic_landmarks.insert(landmark_node);
      frame_node->objects_seen.insert(object_node);
    }
  }
};

}  // namespace dyno_testing
