#pragma once

#include <unordered_map>

#include "dynosam2/estimator/DoGraph.hpp"
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
  const gtsam::Pose3& getDooPose(dyno::TemporalObjectIdentifier id) const;

  const dyno::PoseWithMotion& getDooEntry(
      dyno::TemporalObjectIdentifier id) const;

  // all VoStates should have a timestamp?

 private:
  dyno::Camera::Ptr camera_;
  //! Local realtime vio estimator and holder of all valid states
  VioGraphEstimator vio_estimator_;
  std::unordered_map<dyno::ObjectId, DoGraphEstimator> object_estimators_;
};

};  // namespace dyno2
