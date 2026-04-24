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

#include <glog/logging.h>

#include "dynosam/backend/Accessor.hpp"

namespace dyno {

template <class MAP, class DerivedAccessor>
template <typename... DerivedArgs>
AccessorT<MAP, DerivedAccessor>::AccessorT(
    const SharedFormulationData::Ptr& shared_data, typename Map::Ptr map,
    DerivedArgs&&... derived_args)
    : DerivedAccessor(std::forward<DerivedArgs>(derived_args)...),
      shared_data_(CHECK_NOTNULL(shared_data)),
      map_(map) {}

template <class MAP, class DerivedAccessor>
StateQuery<gtsam::Point3> AccessorT<MAP, DerivedAccessor>::getStaticLandmark(
    TrackletId tracklet_id) const {
  const auto lmk = CHECK_NOTNULL(map()->getLandmark(tracklet_id));
  return this->template query<gtsam::Point3>(lmk->makeStaticKey());
}

template <class MAP, class DerivedAccessor>
MotionEstimateMap AccessorT<MAP, DerivedAccessor>::getObjectMotions(
    FrameId frame_id) const {
  MotionEstimateMap motion_estimates;

  const auto frame_node = map()->getFrame(frame_id);
  if (!frame_node) {
    return motion_estimates;
  }

  // const auto object_seen =
  //     frame_node->objects_seen.template collectIds<ObjectId>();
  const auto& objects_seen = frame_node->objectsSeen();
  for (const auto& object_node : objects_seen) {
    const ObjectId object_id = object_node->objectId();

    StateQuery<Motion3ReferenceFrame> motion_query =
        this->getObjectMotionReferenceFrame(frame_id, object_id);
    if (motion_query) {
      motion_estimates.insert2(object_id, motion_query.get());
    }
  }
  return motion_estimates;
}

template <class MAP, class DerivedAccessor>
EstimateMap<ObjectId, gtsam::Pose3>
AccessorT<MAP, DerivedAccessor>::getObjectPoses(FrameId frame_id) const {
  EstimateMap<ObjectId, gtsam::Pose3> pose_estimates;

  const auto frame_node = map()->getFrame(frame_id);
  if (!frame_node) {
    return pose_estimates;
  }

  const auto& objects_seen = frame_node->objectsSeen();
  for (const auto& object_node : objects_seen) {
    const ObjectId object_id = object_node->objectId();
    StateQuery<gtsam::Pose3> object_pose =
        this->getObjectPose(frame_id, object_id);
    if (object_pose) {
      pose_estimates.insert2(
          object_id, ReferenceFrameValue<gtsam::Pose3>(object_pose.get(),
                                                       ReferenceFrame::GLOBAL));
    }
  }
  return pose_estimates;
}

template <class MAP, class DerivedAccessor>
StatusLandmarkVector
AccessorT<MAP, DerivedAccessor>::getDynamicLandmarkEstimates(
    FrameId frame_id) const {
  const auto frame_node = map()->getFrame(frame_id);
  CHECK_NOTNULL(frame_node);

  StatusLandmarkVector estimates;
  // const auto object_seen =
  //     frame_node->objects_seen.template collectIds<ObjectId>();
  const auto& objects_seen = frame_node->objectsSeen();
  for (const auto& object_node : objects_seen) {
    estimates +=
        this->getDynamicLandmarkEstimates(frame_id, object_node->objectId());
  }
  return estimates;
}

template <class MAP, class DerivedAccessor>
StatusLandmarkVector
AccessorT<MAP, DerivedAccessor>::getDynamicLandmarkEstimates(
    FrameId frame_id, ObjectId object_id) const {
  const auto frame_node = map()->getFrame(frame_id);
  CHECK_NOTNULL(frame_node);

  if (!frame_node->objectObserved(object_id)) {
    return StatusLandmarkVector{};
  }

  const auto timestamp = frame_node->timestamp();

  StatusLandmarkVector estimates;
  const auto& dynamic_landmarks = frame_node->dynamicLandmarks();
  for (const auto& lmk_node : dynamic_landmarks) {
    const auto tracklet_id = lmk_node->trackletId();

    if (object_id != lmk_node->objectId()) {
      continue;
    }

    // user defined function should put point in the world frame
    StateQuery<gtsam::Point3> lmk_query =
        this->getDynamicLandmark(frame_id, tracklet_id);
    if (lmk_query) {
      estimates.push_back(LandmarkStatus::DynamicInGlobal(
          Point3Measurement(lmk_query.get()), frame_id, timestamp, tracklet_id,
          object_id));
    }
  }
  return estimates;
}

template <class MAP, class DerivedAccessor>
ObjectIds AccessorT<MAP, DerivedAccessor>::getObjectIds() const {
  return map()->getObjectIds();
}

template <class MAP, class DerivedAccessor>
FrameIds AccessorT<MAP, DerivedAccessor>::getFrameIds() const {
  return map()->getFrameIds();
}

template <class MAP, class DerivedAccessor>
Timestamp AccessorT<MAP, DerivedAccessor>::getTimestamp(
    FrameId frame_id) const {
  auto frame_node = map()->getFrame(frame_id);

  if (!frame_node) {
    DYNO_THROW_MSG(DynosamException)
        << "Cannot query timestamp for k=" << frame_id
        << ": frame node is null";
    throw;
  }

  return frame_node->timestamp();
}

template <class MAP, class DerivedAccessor>
PoseTrajectory AccessorT<MAP, DerivedAccessor>::getCameraTrajectory() const {
  PoseTrajectory pose_trajectory;

  for (const auto& [frame_id, frame_node] : map()->getFrames()) {
    const Timestamp timestamp = frame_node->timestamp();

    const gtsam::Pose3 X_W_k =
        DYNO_GET_QUERY_DEBUG(this->getSensorPose(frame_id));
    pose_trajectory.insert(frame_id, timestamp, X_W_k);
  }

  return pose_trajectory;
}

template <class MAP, class DerivedAccessor>
PoseTrajectory AccessorT<MAP, DerivedAccessor>::getObjectPoseTrajectory(
    ObjectId object_id) const {
  const auto object_node = map()->getObject(object_id);

  if (!object_node) {
    return PoseTrajectory{};
  }

  PoseTrajectory pose_trajectory;
  for (const auto& frame_node : object_node->getSeenFrames()) {
    const FrameId frame_id = frame_node->frameId();
    const Timestamp timestamp = frame_node->timestamp();

    StateQuery<gtsam::Pose3> object_pose =
        this->getObjectPose(frame_id, object_id);

    if (object_pose) {
      pose_trajectory.insert(frame_id, timestamp, object_pose.get());
    }
  }

  return pose_trajectory;
}

template <class MAP, class DerivedAccessor>
MotionTrajetory AccessorT<MAP, DerivedAccessor>::getObjectMotionTrajectory(
    ObjectId object_id) const {
  const auto object_node = map()->getObject(object_id);

  if (!object_node) {
    return MotionTrajetory{};
  }

  MotionTrajetory motion_trajectory;
  for (const auto& frame_node : object_node->getSeenFrames()) {
    const FrameId frame_id = frame_node->frameId();
    const Timestamp timestamp = frame_node->timestamp();

    StateQuery<Motion3ReferenceFrame> object_motion =
        this->getObjectMotionReferenceFrame(frame_id, object_id);

    if (object_motion) {
      motion_trajectory.insert(frame_id, timestamp, object_motion.get());
    }
  }

  return motion_trajectory;
}

template <class MAP, class DerivedAccessor>
StatusLandmarkVector
AccessorT<MAP, DerivedAccessor>::getStaticLandmarkEstimates(
    FrameId frame_id) const {
  // dont go over the frames as this contains references to the landmarks
  // multiple times
  // e.g. the ones seen in that frame
  StatusLandmarkVector estimates;

  const auto frame_node = map()->getFrame(frame_id);
  CHECK_NOTNULL(frame_node);

  const auto timestamp = frame_node->timestamp();

  for (const auto& landmark_node : frame_node->staticLandmarks()) {
    if (landmark_node->isStatic()) {
      StateQuery<gtsam::Point3> lmk_query =
          getStaticLandmark(landmark_node->trackletId());
      if (lmk_query) {
        estimates.push_back(LandmarkStatus::StaticInGlobal(
            Point3Measurement(lmk_query.get()),
            LandmarkStatus::MeaninglessFrame, timestamp,
            landmark_node->trackletId()));
      }
    }
  }
  return estimates;
}

template <class MAP, class DerivedAccessor>
StatusLandmarkVector AccessorT<MAP, DerivedAccessor>::getFullStaticMap() const {
  // dont go over the frames as this contains references to the landmarks
  // multiple times e.g. the ones seen in that frame
  StatusLandmarkVector estimates;
  const auto landmarks = map()->getLandmarks();

  for (const auto& [_, landmark_node] : landmarks) {
    if (landmark_node->isStatic()) {
      StateQuery<gtsam::Point3> lmk_query =
          getStaticLandmark(landmark_node->trackletId());
      if (lmk_query) {
        estimates.push_back(LandmarkStatus::StaticInGlobal(
            Point3Measurement(lmk_query.get()),  // estimate
            LandmarkStatus::MeaninglessFrame, NaN,
            landmark_node->trackletId()));
      }
    }
  }
  return estimates;
}

template <class MAP, class DerivedAccessor>
bool AccessorT<MAP, DerivedAccessor>::hasObjectMotionEstimate(
    FrameId frame_id, ObjectId object_id, Motion3* motion) const {
  const auto frame_node = map()->getFrame(frame_id);
  StateQuery<Motion3> motion_query = this->getObjectMotion(frame_id, object_id);

  if (motion_query) {
    if (motion) {
      *motion = motion_query.get();
    }
    return true;
  }
  return false;
}

template <class MAP, class DerivedAccessor>
bool AccessorT<MAP, DerivedAccessor>::hasObjectPoseEstimate(
    FrameId frame_id, ObjectId object_id, gtsam::Pose3* pose) const {
  const auto frame_node = map()->getFrame(frame_id);
  StateQuery<gtsam::Pose3> pose_query =
      this->getObjectPose(frame_id, object_id);

  if (pose_query) {
    if (pose) {
      *pose = pose_query.get();
    }
    return true;
  }
  return false;
}

template <class MAP, class DerivedAccessor>
gtsam::FastMap<ObjectId, gtsam::Point3>
AccessorT<MAP, DerivedAccessor>::computeObjectCentroids(
    FrameId frame_id) const {
  gtsam::FastMap<ObjectId, gtsam::Point3> centroids;

  const auto frame_node = map()->getFrame(frame_id);
  if (!frame_node) {
    return centroids;
  }

  // const auto object_seen =
  //     frame_node->objects_seen.template collectIds<ObjectId>();
  const auto& objects_seen = frame_node->objectsSeen();
  for (const auto& object_node : objects_seen) {
    const ObjectId object_id = object_node->objectId();
    const auto [centroid, result] =
        this->computeObjectCentroid(frame_id, object_id);

    if (result) {
      centroids.insert2(object_id, centroid);
    }
  }
  return centroids;
}

template <class MAP, class DerivedAccessor>
boost::optional<const gtsam::Value&>
AccessorT<MAP, DerivedAccessor>::getValueImpl(const gtsam::Key key) const {
  const std::lock_guard<std::mutex> lock(shared_data_->theta_mutex);
  const auto& theta = shared_data_->theta;
  if (theta.exists(key)) {
    return theta.at(key);
  }
  return boost::none;
}

}  // namespace dyno
