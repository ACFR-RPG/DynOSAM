#include "dynosam2/estimator/SlamBackend.hpp"

namespace dyno2 {

SlamBackend::SlamBackend(dyno::Camera::Ptr camera)
    : camera_(camera), vio_estimator_(camera), object_estimators_() {}

DoGraphEstimator& SlamBackend::allocateDoEstimator(dyno::ObjectId object_id) {
  CHECK(!hasObjectEstimator(object_id));
  object_estimators_.emplace(object_id, camera_);
  return object_estimators_.at(object_id);
}

bool SlamBackend::hasObjectEstimator(dyno::ObjectId object_id) const {
  return object_estimators_.find(object_id) != object_estimators_.end();
}

const gtsam::Pose3& SlamBackend::getVioPose(dyno::FrameId frame_id) const {
  return vio_estimator_.getPose(frame_id);
}

const gtsam::Pose3& SlamBackend::getDooPose(
    dyno::TemporalObjectIdentifier id) const {
  CHECK(hasObjectEstimator(id.object_id));
  return object_estimators_.at(id.object_id)
      .getPoseWithMotion(id.frame_id)
      .pose;
}

const dyno::PoseWithMotion& SlamBackend::getDooEntry(
    dyno::TemporalObjectIdentifier id) const {
  CHECK(hasObjectEstimator(id.object_id));
  return object_estimators_.at(id.object_id).getPoseWithMotion(id.frame_id);
}

}  // namespace dyno2
