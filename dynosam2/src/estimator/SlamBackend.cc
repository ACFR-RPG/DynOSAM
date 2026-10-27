#include "dynosam2/estimator/SlamBackend.hpp"

namespace dyno2 {

SlamBackend::SlamBackend(dyno::Camera::Ptr camera)
    : camera_(camera),
      vio_estimator_(camera),
      full_estimator_(camera_),
      object_estimators_(),
      is_running_full_opt_(false),
      needs_full_graph_opt_(false) {}

DoGraphEstimator& SlamBackend::allocateDoEstimator(dyno::ObjectId object_id) {
  CHECK(!hasObjectEstimator(object_id));
  object_estimators_.try_emplace(object_id, camera_, object_id);
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

bool SlamBackend::isCameraKeyframe(dyno::FrameId frame_id) const {
  return vio_estimator_.isKeyframe(frame_id);
}

bool SlamBackend::isObjectKeyframe(dyno::TemporalObjectIdentifier id,
                                   dyno::DoKeyframeType& keyframe_type) const {
  CHECK(hasObjectEstimator(id.object_id));
  keyframe_type =
      object_estimators_.at(id.object_id).keyframeStatus(id.frame_id);
  return keyframe_type != dyno::DoKeyframeType::NonKeyFrame;
}

const dyno::PoseWithMotion& SlamBackend::getDooEntry(
    dyno::TemporalObjectIdentifier id) const {
  CHECK(hasObjectEstimator(id.object_id));
  return object_estimators_.at(id.object_id).getPoseWithMotion(id.frame_id);
}

size_t SlamBackend::getLandmarks(MapPointVector& map_points) const {
  vio_estimator_.getLandmarks(map_points);

  for (const auto& [object_id, estimator] : object_estimators_) {
    MapPointVector object_map_points;
    estimator.getLandmarks(estimator.currentFrameId(), object_map_points);

    map_points.insert(map_points.begin(), object_map_points.begin(),
                      object_map_points.end());
  }
  return map_points.size();
}

size_t SlamBackend::getLandmarks(MapPointVector& map_points,
                                 const dyno::ObjectIds& objects) const {
  vio_estimator_.getLandmarks(map_points);

  for (auto object_id : objects) {
    const DoGraphEstimator& estimator = object_estimators_.at(object_id);

    MapPointVector object_map_points;
    estimator.getLandmarks(estimator.currentFrameId(), object_map_points);
    map_points.insert(map_points.begin(), object_map_points.begin(),
                      object_map_points.end());
  }
  return map_points.size();
}

}  // namespace dyno2
