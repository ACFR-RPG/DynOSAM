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

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <type_traits>

#include "dynosam_common/DynamicObjects.hpp"
#include "dynosam_common/MapNodes.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/logger/Logger.hpp"
#include "dynosam_common/utils/GtsamUtils.hpp"
#include "dynosam_opt/Symbols.hpp"  //for all the chr's used in the keys

namespace dyno {

/**
 * @brief A container that holds all connected information about static and
 * dynamic entities that is build from input measurements.
 *
 * Structured as a undirected graph, this map structure holds temporal,
 * observation and measurement information about frames, objects and landmarks
 * which are stored as nodes
 *
 * From each node relevant information can be querired. No optimsiation
 * information is held in the map, this is just used to hold the structure of
 * information that can then be used to build a factor graph.
 *
 * So that nodes can carry a pointer to the map, we structure the map using
 * enable_shared_from_this
 * (https://en.cppreference.com/w/cpp/memory/enable_shared_from_this), so that
 * constructor is only usable by this class.
 *
 * @tparam MEASUREMENT measurement type made by the frontend.
 */
template <typename MEASUREMENT = Keypoint>
class Map : public std::enable_shared_from_this<Map<MEASUREMENT>> {
  struct Private {};

 public:
  using Measurement = MEASUREMENT;
  using This = Map<Measurement>;

  /// @brief Alias to a GenericTrackedStatusVector using the templated
  /// Measurement type, specifying that StatusVector must contain the desired
  /// measurement type
  /// @tparam DERIVEDSTATUS
  template <typename DERIVEDSTATUS>
  using MeasurementStatusVector =
      GenericTrackedStatusVector<DERIVEDSTATUS, Measurement>;

  /// @brief Alias to an object node with typedefed measurement
  using ObjectNodeM = ObjectNode<Measurement>;
  /// @brief Alias to an frame node with typedefed measurement
  using FrameNodeM = FrameNode<Measurement>;
  /// @brief Alias to an landmark node with typedefed measurement
  using LandmarkNodeM = LandmarkNode<Measurement>;

  DYNO_POINTER_TYPEDEFS(This)

  // Constructor is only usable by this class
  Map(Private) {}

  static std::shared_ptr<This> create() {
    return std::make_shared<This>(Private());
  }

  std::shared_ptr<This> getptr() {
    // dependant base so need to qualify name lookup with this
    return this->shared_from_this();
  }

  /**
   * @brief Update the map structure given new measurements.
   *
   * @tparam DERIVEDSTATUS
   * @param measurements const GenericTrackedStatusVector<DERIVEDSTATUS>&
   */
  template <typename DERIVEDSTATUS>
  void updateObservations(
      const GenericTrackedStatusVector<DERIVEDSTATUS>& measurements) {
    using DerivedMeasurement =
        typename GenericTrackedStatusVector<DERIVEDSTATUS>::Value;
    for (const DERIVEDSTATUS& status_measurement : measurements) {
      const TrackedValueStatus<DerivedMeasurement>& derived_status =
          static_cast<const TrackedValueStatus<DerivedMeasurement>&>(
              status_measurement);
      const TrackedValueStatus<MEASUREMENT>& status =
          derived_status.template asType<MEASUREMENT>();
      const MEASUREMENT& measurement = status_measurement.value();
      const TrackletId tracklet_id = status.trackletId();
      const FrameId frame_id = status.frameId();
      const ObjectId object_id = status.objectId();
      const bool is_static = status.isStatic();
      addOrUpdateMapStructures(measurement, tracklet_id, frame_id, object_id,
                               is_static);
    }
  }

  /**
   * @brief Updates sensor pose for the given frame.
   *
   * Overwrites value if already exists. Makes a new frame node in the map the
   * frame is new!
   *
   * @param frame_id FrameId
   * @param X const Pose3Measurement&
   */
  void updateSensorPoseMeasurement(FrameId frame_id,
                                   const Pose3Measurement& X) {
    auto frame_node = this->getFrame(frame_id);
    if (!frame_node) {
      frame_node = std::make_shared<FrameNodeM>(getptr());
      frame_node->frame_id = frame_id;
      frames_.insert2(frame_id, frame_node);
    }
    frame_node->X_world = X;
  }

  /**
   * @brief Updates object motion measurements for the given frame.
   * Does not make a new frame node if it does not exist and overwrites the
   * existing motions if already set.
   *
   * @param frame_id FrameId
   * @param motions const MotionEstimateMap&
   */
  void updateObjectMotionMeasurements(FrameId frame_id,
                                      const MotionEstimateMap& motions) {
    auto frame_node = this->getFrame(frame_id);
    CHECK_NOTNULL(frame_node);
    frame_node->motions = motions;
  }

  /**
   * @brief Check a frame exists at the requested frame id.
   *
   * @param frame_id FrameId
   * @return true
   * @return false
   */
  inline bool frameExists(FrameId frame_id) const {
    return frames_.exists(frame_id);
  }

  /**
   * @brief Check a landmark exists with the requested tracklet id.
   *
   * @param tracklet_id TrackletId
   * @return true
   * @return false
   */
  inline bool landmarkExists(TrackletId tracklet_id) const {
    return landmarks_.exists(tracklet_id);
  }

  /**
   * @brief Check that an object exists with the requested object id.
   *
   * @param object_id ObjectId
   * @return true
   * @return false
   */
  inline bool objectExists(ObjectId object_id) const {
    return objects_.exists(object_id);
  }

  /**
   * @brief Get the Object object.
   * If the object does not exist, return nullptr
   *
   * @param object_id ObjectId
   * @return ObjectNodeM::Ptr
   */
  typename ObjectNodeM::Ptr getObject(ObjectId object_id) const {
    if (objectExists(object_id)) {
      return objects_.at(object_id);
    }
    return nullptr;
  }

  /**
   * @brief Get the Frame object.
   * If the frame does not exist, return nullptr.
   *
   * @param frame_id FrameId
   * @return FrameNodeM::Ptr
   */
  typename FrameNodeM::Ptr getFrame(FrameId frame_id) const {
    if (frameExists(frame_id)) {
      return frames_.at(frame_id);
    }
    return nullptr;
  }

  /**
   * @brief Get the Landmark object.
   * If the landmark does not exist, return nullptr.
   *
   * @param tracklet_id TrackletId
   * @return LandmarkNodeM::Ptr
   */
  typename LandmarkNodeM::Ptr getLandmark(TrackletId tracklet_id) const {
    if (landmarkExists(tracklet_id)) {
      return landmarks_.at(tracklet_id);
    }
    return nullptr;
  }

  /**
   * @brief Get the mapping of all tracklet id's to landmarks.
   *
   * @return const gtsam::FastMap<TrackletId, typename LandmarkNodeM::Ptr>&
   */
  const gtsam::FastMap<TrackletId, typename LandmarkNodeM::Ptr>& getLandmarks()
      const {
    return landmarks_;
  }

  /**
   * @brief Get all static tracklet's that were observed at the requested frame
   * id.
   *
   * @param frame_id FrameId
   * @return TrackletIds
   */
  TrackletIds getStaticTrackletsByFrame(FrameId frame_id) const {
    // if frame does not exist?
    TrackletIds tracklet_ids;
    auto frame_node = frames_.at(frame_id);
    for (const auto& landmark_node : frame_node->static_landmarks) {
      tracklet_ids.push_back(landmark_node->tracklet_id);
    }

    return tracklet_ids;
  }

  bool hasInitialObjectMotion(
      FrameId frame_id, ObjectId object_id,
      Motion3ReferenceFrame* motion_frame = nullptr) const {
    auto frame_node = this->getFrame(frame_id);
    if (!frame_node) {
      return false;
    }

    // motion map has not been set
    if (frame_node->motions) {
      const auto& motions = *frame_node->motions;

      // no motion for this object in map
      if (!motions.exists(object_id)) {
        return false;
      }

      if (motion_frame) {
        *motion_frame = motions.at(object_id);
      }
      return true;
    } else {
      return false;
    }
  }

  bool hasInitialObjectMotion(FrameId frame_id, ObjectId object_id,
                              Motion3* motion) const {
    Motion3ReferenceFrame frame;
    if (hasInitialObjectMotion(frame_id, object_id, &frame)) {
      if (motion) *motion = frame;  // implicit cast
      return true;
    }
    return false;
  }

  // TODO: test
  bool hasInitialSensorPose(FrameId frame_id,
                            Pose3Measurement* X = nullptr) const {
    auto frame_node = this->getFrame(frame_id);
    if (!frame_node) {
      return false;
    }

    if (frame_node->X_world) {
      const auto& pose = *frame_node->X_world;
      if (X) *X = pose;
      return true;
    } else {
      return false;
    }
  }

  bool hasInitialSensorPose(FrameId frame_id, gtsam::Pose3* X = nullptr) const {
    Pose3Measurement frame;
    if (hasInitialSensorPose(frame_id, &frame)) {
      if (X) *X = frame;  // implicit cast
      return true;
    }
    return false;
  }

  /**
   * @brief Get number of objects seen
   *
   * @return size_t
   */
  inline size_t numObjectsSeen() const { return objects_.size(); }

  /**
   * @brief Get the most recent frame object.
   *
   * @return const FrameNodeM::Ptr
   */
  const typename FrameNodeM::Ptr lastFrame() const {
    return frames_.crbegin()->second;
  }

  /**
   * @brief Get the earliest (first) frame object.
   *
   * @return const FrameNodeM::Ptr
   */
  const typename FrameNodeM::Ptr firstFrame() const {
    return frames_.cbegin()->second;
  }

  /**
   * @brief Get the most recent frame id.
   *
   * @return FrameId
   */
  FrameId lastFrameId() const { return this->lastFrame()->frame_id; }

  /**
   * @brief Get the earliest (first) frame id.
   *
   * @return FrameId
   */
  FrameId firstFrameId() const { return this->firstFrame()->frame_id; }

  // TODO: depricate
  inline FrameId lastEstimateUpdate() const { return last_estimate_update_; }

  /**
   * @brief Get the (by output-argument) object id for some tracked point.
   * Function returns false if the landmark does not exist.
   *
   * @param object_id ObjectId& (output-argument)
   * @param tracklet_id TrackletId
   * @return true
   * @return false
   */
  bool getLandmarkObjectId(ObjectId& object_id, TrackletId tracklet_id) const {
    const auto lmk = getLandmark(tracklet_id);
    if (!lmk) {
      return false;
    }

    object_id = lmk->getObjectId();
    return true;
  }

  /**
   * @brief Get all object ids.
   *
   * @return ObjectIds
   */
  ObjectIds getObjectIds() const {
    ObjectIds object_ids;
    for (const auto& [object_id, _] : objects_) {
      object_ids.push_back(object_id);
    }
    return object_ids;
  }

  /**
   * @brief Get all frame numbers.
   *
   * @return FrameIds
   */
  FrameIds getFrameIds() const {
    FrameIds frame_ids;
    for (const auto& [frame_id, _] : frames_) {
      frame_ids.push_back(frame_id);
    }
    return frame_ids;
  }

  const auto getFrames() const { return frames_; }

 private:
  void addOrUpdateMapStructures(const Measurement& measurement,
                                TrackletId tracklet_id, FrameId frame_id,
                                ObjectId object_id, bool is_static) {
    typename LandmarkNodeM::Ptr landmark_node = nullptr;
    typename FrameNodeM::Ptr frame_node = nullptr;

    CHECK((is_static && object_id == background_label) ||
          (!is_static && object_id != background_label));

    if (!landmarkExists(tracklet_id)) {
      landmark_node = std::make_shared<LandmarkNodeM>(getptr());
      CHECK_NOTNULL(landmark_node);
      landmark_node->tracklet_id = tracklet_id;
      landmark_node->object_id = object_id;
      landmarks_.insert2(tracklet_id, landmark_node);
    }

    if (!frameExists(frame_id)) {
      frame_node = std::make_shared<FrameNodeM>(getptr());
      frame_node->frame_id = frame_id;
      frames_.insert2(frame_id, frame_node);
    }

    landmark_node = getLandmark(tracklet_id);
    frame_node = getFrame(frame_id);

    CHECK(landmark_node);
    CHECK(frame_node);

    CHECK_EQ(landmark_node->tracklet_id, tracklet_id);
    // this might fail of a tracklet get associated with a different object
    CHECK_EQ(landmark_node->object_id, object_id);
    CHECK_EQ(frame_node->frame_id, frame_id);

    landmark_node->add(frame_node, measurement);

    // add to frame
    if (is_static) {
      frame_node->static_landmarks.insert(landmark_node);
    } else {
      CHECK(object_id != background_label);

      typename ObjectNodeM::Ptr object_node = nullptr;
      if (!objectExists(object_id)) {
        object_node = std::make_shared<ObjectNodeM>(getptr());
        object_node->object_id = object_id;
        objects_.insert2(object_id, object_node);
      }

      object_node = getObject(object_id);
      CHECK(object_node);

      object_node->dynamic_landmarks.insert(landmark_node);

      frame_node->dynamic_landmarks.insert(landmark_node);
      frame_node->objects_seen.insert(object_node);
    }
  }

 private:
  // nodes
  gtsam::FastMap<FrameId, typename FrameNodeM::Ptr> frames_;
  gtsam::FastMap<TrackletId, typename LandmarkNodeM::Ptr> landmarks_;
  gtsam::FastMap<ObjectId, typename ObjectNodeM::Ptr> objects_;

  FrameId last_estimate_update_{0};
};

// TODO: can depricate most of these...
using Map3d = Map<Landmark>;
using ObjectNode3d = Map3d::ObjectNodeM;
using LandmarkNode3d = Map3d::LandmarkNodeM;
using FrameNode3d = Map3d::FrameNodeM;

using Map2d = Map<Keypoint>;
using ObjectNode2d = Map2d::ObjectNodeM;
using LandmarkNode2d = Map2d::LandmarkNodeM;
using FrameNode2d = Map2d::FrameNodeM;

using MapVision = Map<CameraMeasurement>;
using ObjectNodeV = MapVision::ObjectNodeM;
using LandmarkNodeV = MapVision::LandmarkNodeM;
using FrameNodeV = MapVision::FrameNodeM;

}  // namespace dyno
