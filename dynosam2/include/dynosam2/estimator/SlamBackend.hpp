#pragma once

#include <atomic>
#include <unordered_map>

#include "dynosam2/estimator/DoGraph.hpp"
#include "dynosam2/estimator/FullGraph.hpp"
#include "dynosam2/estimator/VioGraph.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_sensors/Camera.hpp"

namespace dyno2 {

/// @brief Main estimator class, holding all current state information
class SlamBackend {
 public:
  DYNO_POINTER_TYPEDEFS(SlamBackend)
  SlamBackend(dyno::Camera::Ptr camera);

  const VioGraphEstimator& vioEstimator() const { return vio_estimator_; }
  VioGraphEstimator& vioEstimator() { return vio_estimator_; }

  DoGraphEstimator& allocateDoEstimator(dyno::ObjectId object_id);
  const DoGraphEstimator& doEstimator(dyno::ObjectId object_id) const {
    return object_estimators_.at(object_id);
  }
  DoGraphEstimator& doEstimator(dyno::ObjectId object_id) {
    return object_estimators_.at(object_id);
  }

  bool hasObjectEstimator(dyno::ObjectId object_id) const;

  const gtsam::Pose3& getVioPose(dyno::FrameId frame_id) const;

  /// @brief Get the object pose in W
  /// @param id
  /// @return
  const gtsam::Pose3& getDooPose(dyno::TemporalObjectId id) const;

  /// @brief Get the motion for object j in W from k-1 to k.
  /// @param id
  /// @return
  const dyno::Motion3ReferenceFrame& getDooMotion(
      dyno::TemporalObjectId id) const;

  /// @brief Get the velocity in W for object j
  /// @param id
  /// @return
  const Velocity3& getDooVelocity(dyno::TemporalObjectId id) const;

  bool isCameraKeyframe(dyno::FrameId frame_id) const;

  bool isObjectKeyframe(dyno::TemporalObjectId id,
                        dyno::DoKeyframeType& keyframe_type) const;

  const dyno::PoseWithMotion& getDooEntry(dyno::TemporalObjectId id) const;

  // get all landmarks for all objects
  size_t getLandmarks(MapPointVector& map_points) const;

  /// @brief Get static landmarks and object landmarks for requesed objects
  /// @param map_points
  /// @param objects
  /// @return
  size_t getLandmarks(MapPointVector& map_points,
                      const dyno::ObjectIds& objects) const;

  // all VoStates should have a timestamp?

  // add to full problem!
  void addStaticMeasurements(
      const std::vector<KeypointIdentifier>& vo_measurements) {
    full_estimator_.addStaticMeasurements(vio_estimator_, vo_measurements);
  }

  inline bool needsFullGraphOptimisation() const {
    return full_estimator_.needsFullGraphOptimisation();
  }

  inline bool isRunningFullGraphOptimisation() const {
    return full_estimator_.isRunningFullGraphOptimisation();
  }

  inline bool updateAvaialbleFromFullGraph() const {
    return full_estimator_.isUpdateAvailable();
  }

  void startFullGraphOptimisation() { full_estimator_.optimize(); }

  bool synchroniseRealtimeAndFullGraph(UpdatedStates& involved_states);

 private:
  dyno::Camera::Ptr camera_;
  //! Local realtime vio estimator and holder of all valid states
  VioGraphEstimator vio_estimator_;
  FullGraph full_estimator_;
  std::unordered_map<dyno::ObjectId, DoGraphEstimator> object_estimators_;
};

};  // namespace dyno2
