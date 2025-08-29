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

template <class MAP>
Accessor<MAP>::Accessor(const SharedFormulationData& shared_data,
                        typename Map::Ptr map)
    : shared_data_(shared_data), map_(map) {
  CHECK_NOTNULL(shared_data.values);
  CHECK_NOTNULL(shared_data.hooks);
}

template <class MAP>
StateQuery<gtsam::Point3> Accessor<MAP>::getStaticLandmark(
    TrackletId tracklet_id) const {
  const auto lmk = CHECK_NOTNULL(map()->getLandmark(tracklet_id));
  return this->query<gtsam::Point3>(lmk->makeStaticKey());
}

template <class MAP>
MotionEstimateMap Accessor<MAP>::getObjectMotions(FrameId frame_id) const {
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

template <class MAP>
StateQuery<Motion3ReferenceFrame> Accessor<MAP>::getObjectMotionReferenceFrame(
    FrameId frame_id, ObjectId object_id) const {
  StateQuery<Motion3> motion_query = this->getObjectMotion(frame_id, object_id);
  if (motion_query) {
    return StateQuery<Motion3ReferenceFrame>(
        motion_query.key_,
        Motion3ReferenceFrame(motion_query.get(),
                              Motion3ReferenceFrame::Style::F2F,
                              ReferenceFrame::GLOBAL, frame_id - 1u, frame_id));
  } else {
    return StateQuery<Motion3ReferenceFrame>(motion_query.key_,
                                             motion_query.status_);
  }
}

template <class MAP>
EstimateMap<ObjectId, gtsam::Pose3> Accessor<MAP>::getObjectPoses(
    FrameId frame_id) const {
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

template <class MAP>
ObjectPoseMap Accessor<MAP>::getObjectPoses() const {
  ObjectPoseMap object_poses;
  for (FrameId frame_id : map()->getFrameIds()) {
    EstimateMap<ObjectId, gtsam::Pose3> per_object_pose =
        this->getObjectPoses(frame_id);

    for (const auto& [object_id, pose] : per_object_pose) {
      if (!object_poses.exists(object_id)) {
        object_poses.insert2(object_id,
                             gtsam::FastMap<FrameId, gtsam::Pose3>{});
      }

      auto& per_frame_pose = object_poses.at(object_id);
      per_frame_pose.insert2(frame_id, pose);
    }
  }
  return object_poses;
}

template <class MAP>
ObjectMotionMap Accessor<MAP>::getObjectMotions() const {
  ObjectMotionMap object_motions;
  for (FrameId frame_id : map()->getFrameIds()) {
    MotionEstimateMap per_object_motions = this->getObjectMotions(frame_id);

    for (const auto& [object_id, motion] : per_object_motions) {
      if (!object_motions.exists(object_id)) {
        object_motions.insert2(object_id, ObjectMotionMap::NestedBase{});
      }

      auto& per_frame_motion = object_motions.at(object_id);
      per_frame_motion.insert2(frame_id, motion);
    }
  }
  return object_motions;
}

template <class MAP>
StatusLandmarkVector Accessor<MAP>::getDynamicLandmarkEstimates(
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

template <class MAP>
StatusLandmarkVector Accessor<MAP>::getDynamicLandmarkEstimates(
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

template <class MAP>
StatusLandmarkVector Accessor<MAP>::getStaticLandmarkEstimates(
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

template <class MAP>
StatusLandmarkVector Accessor<MAP>::getFullStaticMap() const {
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

template <class MAP>
StatusLandmarkVector Accessor<MAP>::getLandmarkEstimates(
    FrameId frame_id) const {
  StatusLandmarkVector estimates;
  estimates += getStaticLandmarkEstimates(frame_id);
  estimates += getDynamicLandmarkEstimates(frame_id);
  return estimates;
}

template <class MAP>
bool Accessor<MAP>::hasObjectMotionEstimate(FrameId frame_id,
                                            ObjectId object_id,
                                            Motion3* motion) const {
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

template <class MAP>
bool Accessor<MAP>::hasObjectMotionEstimate(FrameId frame_id,
                                            ObjectId object_id,
                                            Motion3& motion) const {
  return hasObjectMotionEstimate(frame_id, object_id, &motion);
}

template <class MAP>
bool Accessor<MAP>::hasObjectPoseEstimate(FrameId frame_id, ObjectId object_id,
                                          gtsam::Pose3* pose) const {
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

template <class MAP>
bool Accessor<MAP>::hasObjectPoseEstimate(FrameId frame_id, ObjectId object_id,
                                          gtsam::Pose3& pose) const {
  return hasObjectPoseEstimate(frame_id, object_id, &pose);
}

template <class MAP>
gtsam::FastMap<ObjectId, gtsam::Point3> Accessor<MAP>::computeObjectCentroids(
    FrameId frame_id) const {
  gtsam::FastMap<ObjectId, gtsam::Point3> centroids;

  const auto frame_node = map()->getFrame(frame_id);
  if (!frame_node) {
    return centroids;
  }

  const auto object_seen =
      frame_node->objects_seen.template collectIds<ObjectId>();
  for (ObjectId object_id : object_seen) {
    const auto [centroid, result] = computeObjectCentroid(frame_id, object_id);

    if (result) {
      centroids.insert2(object_id, centroid);
    }
  }
  return centroids;
}

template <class MAP>
std::tuple<gtsam::Point3, bool> Accessor<MAP>::computeObjectCentroid(
    FrameId frame_id, ObjectId object_id) const {
  const StatusLandmarkVector& dynamic_lmks =
      this->getDynamicLandmarkEstimates(frame_id, object_id);

  // convert to point cloud - should be a map with only one map in it
  CloudPerObject object_clouds =
      groupObjectCloud(dynamic_lmks, this->getSensorPose(frame_id).get());
  if (object_clouds.size() == 0) {
    VLOG(20) << "Cannot collect object clouds from dynamic landmarks of "
             << object_id << " and frame " << frame_id << "!! "
             << " # Dynamic lmks in the map for this object at this frame was "
             << dynamic_lmks.size();  //<< " but reocrded lmks was " <<
                                      // dynamic_landmarks.size();
    return {gtsam::Point3{}, false};
  }
  CHECK_EQ(object_clouds.size(), 1);
  CHECK(object_clouds.exists(object_id));

  const auto dynamic_point_cloud = object_clouds.at(object_id);
  pcl::PointXYZ centroid;
  pcl::computeCentroid(dynamic_point_cloud, centroid);
  // TODO: outlier reject?
  gtsam::Point3 translation = pclPointToGtsam(centroid);
  return {translation, true};
}

template <class MAP>
EstimateMap<ObjectId, std::pair<Motion3, gtsam::Pose3>>
Accessor<MAP>::getRequestedObjectPosePair(FrameId motion_frame_id,
                                          FrameId pose_frame_id) const {
  const MotionEstimateMap motions_k = this->getObjectMotions(motion_frame_id);
  const EstimateMap<ObjectId, gtsam::Pose3> poses_k =
      this->getObjectPoses(pose_frame_id);

  EstimateMap<ObjectId, std::pair<Motion3, gtsam::Pose3>> result;
  // collect common objects
  for (const auto& [object_id, _] : motions_k) {
    if (poses_k.exists(object_id)) {
      auto motion_reference_values = motions_k.at(object_id);
      auto pose_reference_values = poses_k.at(object_id);

      CHECK_EQ(motion_reference_values.frame_, pose_reference_values.frame_);

      ReferenceFrameValue<std::pair<Motion3, gtsam::Pose3>> estimate(
          std::make_pair(motion_reference_values.estimate_,
                         pose_reference_values.estimate_),
          motion_reference_values.frame_);
      result.insert2(object_id, estimate);
    }
  }
  return result;
}

template <class MAP>
bool Accessor<MAP>::exists(gtsam::Key key) const {
  const gtsam::Values* theta = shared_data_.values;
  CHECK_NOTNULL(theta);
  return theta->exists(key);
}

template <class MAP>
template <typename ValueType>
StateQuery<ValueType> Accessor<MAP>::query(gtsam::Key key) const {
  const gtsam::Values* theta = shared_data_.values;
  CHECK_NOTNULL(theta);
  if (theta->exists(key)) {
    return StateQuery<ValueType>(key, theta->at<ValueType>(key));
  } else {
    return StateQuery<ValueType>::NotInMap(key);
  }
}

template <class MAP>
template <typename ValueType>
StateQuery<ValueType> Accessor<MAP>::queryWithTheta(
    gtsam::Key key, const gtsam::Values& new_values) const {
  if (StateQuery<ValueType> theta = this->query<ValueType>(key); theta) {
    return theta;
  }

  if (new_values.exists(key)) {
    return StateQuery<ValueType>(key, new_values.at<ValueType>(key));
  } else {
    return StateQuery<ValueType>::NotInMap(key);
  }
}

}  // namespace dyno
