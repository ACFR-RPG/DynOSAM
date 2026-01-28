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
    const SharedFormulationData& shared_data, typename Map::Ptr map,
    DerivedArgs&&... derived_args)
    : DerivedAccessor(std::forward<DerivedArgs>(derived_args)...),
      shared_data_(shared_data),
      map_(map) {
  CHECK_NOTNULL(shared_data.values);
  CHECK_NOTNULL(shared_data.hooks);
}

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

  const auto object_seen =
      frame_node->objects_seen.template collectIds<ObjectId>();
  for (ObjectId object_id : object_seen) {
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

  const auto object_seen =
      frame_node->objects_seen.template collectIds<ObjectId>();
  for (ObjectId object_id : object_seen) {
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
ObjectPoseMap AccessorT<MAP, DerivedAccessor>::getObjectPoses() const {
  ObjectPoseMap object_poses;
  for (FrameId frame_id : map()->getFrameIds()) {
    EstimateMap<ObjectId, gtsam::Pose3> per_object_pose =
        this->getObjectPoses(frame_id);

    for (const auto& [object_id, pose] : per_object_pose) {
      object_poses.insert22(object_id, frame_id, pose);
      // if (!object_poses.exists(object_id)) {
      //   object_poses.insert2(object_id,
      //                        gtsam::FastMap<FrameId, gtsam::Pose3>{});
      // }

      // auto& per_frame_pose = object_poses.at(object_id);
      // per_frame_pose.insert2(frame_id, pose);
    }
  }
  return object_poses;
}

template <class MAP, class DerivedAccessor>
ObjectMotionMap AccessorT<MAP, DerivedAccessor>::getObjectMotions() const {
  ObjectMotionMap object_motions;
  for (FrameId frame_id : map()->getFrameIds()) {
    MotionEstimateMap per_object_motions = this->getObjectMotions(frame_id);

    for (const auto& [object_id, motion] : per_object_motions) {
      object_motions.insert22(object_id, frame_id, motion);
      // if (!object_motions.exists(object_id)) {
      //   object_motions.insert2(object_id, ObjectMotionMap::NestedBase{});
      // }

      // auto& per_frame_motion = object_motions.at(object_id);
      // per_frame_motion.insert2(frame_id, motion);
    }
  }
  return object_motions;
}

template <class MAP, class DerivedAccessor>
StatusLandmarkVector
AccessorT<MAP, DerivedAccessor>::getDynamicLandmarkEstimates(
    FrameId frame_id) const {
  const auto frame_node = map()->getFrame(frame_id);
  CHECK_NOTNULL(frame_node);

  StatusLandmarkVector estimates;
  const auto object_seen =
      frame_node->objects_seen.template collectIds<ObjectId>();
  for (ObjectId object_id : object_seen) {
    estimates += this->getDynamicLandmarkEstimates(frame_id, object_id);
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

  StatusLandmarkVector estimates;
  const auto& dynamic_landmarks = frame_node->dynamic_landmarks;
  for (auto lmk_node : dynamic_landmarks) {
    const auto tracklet_id = lmk_node->tracklet_id;

    if (object_id != lmk_node->object_id) {
      continue;
    }

    // user defined function should put point in the world frame
    StateQuery<gtsam::Point3> lmk_query =
        this->getDynamicLandmark(frame_id, tracklet_id);
    if (lmk_query) {
      estimates.push_back(LandmarkStatus::DynamicInGLobal(
          Point3Measurement(lmk_query.get()),  // estimate
          frame_id, tracklet_id,
          object_id)  // status
      );
    }
  }
  return estimates;
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

  for (const auto& landmark_node : frame_node->static_landmarks) {
    if (landmark_node->isStatic()) {
      StateQuery<gtsam::Point3> lmk_query =
          getStaticLandmark(landmark_node->tracklet_id);
      if (lmk_query) {
        estimates.push_back(LandmarkStatus::StaticInGlobal(
            Point3Measurement(lmk_query.get()),  // estimate
            LandmarkStatus::MeaninglessFrame,
            landmark_node->getId()  // tracklet id
            )                       // status
        );
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
          getStaticLandmark(landmark_node->tracklet_id);
      if (lmk_query) {
        estimates.push_back(LandmarkStatus::StaticInGlobal(
            Point3Measurement(lmk_query.get()),  // estimate
            LandmarkStatus::MeaninglessFrame,
            landmark_node->getId()  // tracklet id
            )                       // status
        );
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

  const auto object_seen =
      frame_node->objects_seen.template collectIds<ObjectId>();
  for (ObjectId object_id : object_seen) {
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
  const gtsam::Values* theta = shared_data_.values;
  if (theta->exists(key)) {
    return theta->at(key);
  }
  return boost::none;
}

}  // namespace dyno
