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

#include "dynosam/backend/rgbd/WorldPoseEstimator.hpp"

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/factors/LandmarkMotionPoseFactor.hpp"
#include "dynosam/factors/LandmarkPoseSmoothingFactor.hpp"

namespace dyno {

StateQuery<gtsam::Pose3> WorldPoseAccessor::getSensorPose(
    FrameId frame_id) const {
  const auto frame_node = map()->getFrame(frame_id);
  CHECK_NOTNULL(frame_node);
  return this->query<gtsam::Pose3>(frame_node->makePoseKey());
}

StateQuery<gtsam::Pose3> WorldPoseAccessor::getObjectMotion(
    FrameId frame_id, ObjectId object_id) const {
  const auto frame_node_k = map()->getFrame(frame_id);
  CHECK_NOTNULL(frame_node_k);
  const auto object_motion_key = frame_node_k->makeObjectMotionKey(object_id);

  if (frame_id < 2) {
    // if the current frame is 1 then skip as we never get a frame that is 0
    // this is because the first frame (frame_id=0) is never sent to the backend
    // as at least two frames (0 and 1) are needed to track!
    return StateQuery<gtsam::Pose3>::NotInMap(object_motion_key);
  }

  const auto object_pose_k = this->getObjectPose(frame_id, object_id);
  const auto object_pose_k_1 = this->getObjectPose(frame_id - 1u, object_id);

  if (object_pose_k && object_pose_k_1) {
    // ^w_{k-1}H_k = ^wL_k \: ^wL_{k-1}^{-1}
    const gtsam::Pose3 motion =
        object_pose_k.get() * object_pose_k_1->inverse();
    return StateQuery<gtsam::Pose3>{object_motion_key, motion};
  }
  return StateQuery<gtsam::Pose3>::NotInMap(object_motion_key);
}
StateQuery<gtsam::Pose3> WorldPoseAccessor::getObjectPose(
    FrameId frame_id, ObjectId object_id) const {
  const auto frame_node = map()->getFrame(frame_id);
  if (!frame_node) {
    return StateQuery<gtsam::Pose3>::InvalidMap();
  }

  // CHECK(frame_node) << "Frame Id is null at frame " << frame_id;
  return this->query<gtsam::Pose3>(frame_node->makeObjectPoseKey(object_id));
}

StateQuery<gtsam::Point3> WorldPoseAccessor::getDynamicLandmark(
    FrameId frame_id, TrackletId tracklet_id) const {
  const auto lmk = map()->getLandmark(tracklet_id);
  CHECK_NOTNULL(lmk);

  return this->query<gtsam::Point3>(lmk->makeDynamicKey(frame_id));
}

void WorldPoseFormulation::dynamicPointUpdateCallback(
    const PointUpdateContextType& context, UpdateObservationResult& result,
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors) {
  const auto lmk_node = context.lmk_node;
  const auto frame_node_k_1 = context.frame_node_k_1;
  const auto frame_node_k = context.frame_node_k;

  auto dynamic_point_noise = noise_models_.dynamic_point_noise;

  auto theta_accessor = this->accessorFromTheta();

  const gtsam::Key object_point_key_k_1 =
      lmk_node->makeDynamicKey(frame_node_k_1->frame_id);
  const gtsam::Key object_point_key_k =
      lmk_node->makeDynamicKey(frame_node_k->frame_id);

  // if first motion (i.e first time we have both k-1 and k), add both at k-1
  // and k
  if (context.is_starting_motion_frame) {
    auto [measured_point_local, measurement_covariance] =
        MeasurementTraits::pointWithCovariance(
            lmk_node->getMeasurement(frame_node_k_1));

    if (params_.makeDynamicMeasurementsRobust()) {
      measurement_covariance = factor_graph_tools::robustifyHuber(
          params_.k_huber_3d_points_, measurement_covariance);
    }

    new_factors.emplace_shared<PoseToPointFactor>(
        frame_node_k_1->makePoseKey(),  // pose key at previous frames
        object_point_key_k_1, measured_point_local, measurement_covariance);
    // object_debug_info.num_dynamic_factors++;
    result.updateAffectedObject(frame_node_k_1->frame_id,
                                context.getObjectId());
    if (result.debug_info)
      result.debug_info->getObjectInfo(context.getObjectId())
          .num_dynamic_factors++;

    // add landmark at previous frame
    Landmark lmk_world_k_1;
    getSafeQuery(lmk_world_k_1,
                 theta_accessor->query<Landmark>(object_point_key_k_1),
                 gtsam::Point3(context.X_k_1_measured * measured_point_local));
    new_values.insert(object_point_key_k_1, lmk_world_k_1);
    if (result.debug_info)
      result.debug_info->getObjectInfo(context.getObjectId())
          .num_new_dynamic_points++;
  }

  // previous point must be added by the previous iteration
  CHECK(new_values.exists(object_point_key_k_1) ||
        theta_accessor->exists(object_point_key_k_1));

  // const Landmark measured_k =
  // lmk_node->getMeasurement(frame_node_k).landmark;
  auto [measured_point_local, measurement_covariance] =
      MeasurementTraits::pointWithCovariance(
          lmk_node->getMeasurement(frame_node_k));

  if (params_.makeDynamicMeasurementsRobust()) {
    measurement_covariance = factor_graph_tools::robustifyHuber(
        params_.k_huber_3d_points_, measurement_covariance);
  }

  new_factors.emplace_shared<PoseToPointFactor>(
      frame_node_k
          ->makePoseKey(),  // pose key at this (in the iteration) frames
      object_point_key_k, measured_point_local, measurement_covariance);
  if (result.debug_info)
    result.debug_info->getObjectInfo(context.getObjectId())
        .num_dynamic_factors++;
  result.updateAffectedObject(frame_node_k->frame_id, context.getObjectId());

  Landmark lmk_world_k;
  getSafeQuery(lmk_world_k, theta_accessor->query<Landmark>(object_point_key_k),
               gtsam::Point3(context.X_k_measured * measured_point_local));
  new_values.insert(object_point_key_k, lmk_world_k);
  if (result.debug_info)
    result.debug_info->getObjectInfo(context.getObjectId())
        .num_new_dynamic_points++;

  const gtsam::Key object_pose_k_1_key =
      frame_node_k_1->makeObjectPoseKey(context.getObjectId());
  const gtsam::Key object_pose_k_key =
      frame_node_k->makeObjectPoseKey(context.getObjectId());

  auto landmark_motion_noise = noise_models_.landmark_motion_noise;
  new_factors.emplace_shared<LandmarkMotionPoseFactor>(
      object_point_key_k_1, object_point_key_k, object_pose_k_1_key,
      object_pose_k_key, landmark_motion_noise);
  result.updateAffectedObject(frame_node_k_1->frame_id, context.getObjectId());
  result.updateAffectedObject(frame_node_k->frame_id, context.getObjectId());
  if (result.debug_info)
    result.debug_info->getObjectInfo(context.getObjectId())
        .num_motion_factors++;

  // mark as now in map
  is_dynamic_tracklet_in_map_.insert2(context.getTrackletId(), true);
}

void WorldPoseFormulation::objectUpdateContext(
    const ObjectUpdateContextType& context, UpdateObservationResult& result,
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors) {
  auto frame_node_k = context.frame_node_k;
  const gtsam::Key object_pose_key_k =
      frame_node_k->makeObjectPoseKey(context.getObjectId());

  auto theta_accessor = this->accessorFromTheta();

  if (!is_other_values_in_map.exists(object_pose_key_k)) {
    FrameId frame_id_k_1 = context.getFrameId() - 1u;
    // try and propogate from previous initalisation
    StateQuery<gtsam::Pose3> pose_k_1_query =
        theta_accessor->getObjectPose(frame_id_k_1, context.getObjectId());

    // first frame the object is seen in
    const FrameId first_seen_frame = context.object_node->getFirstSeenFrame();

    // takes me from k-1 to k
    Motion3 motion;
    gtsam::Pose3 object_pose_k;
    // we have a motion from k-1 to k and a pose at k
    if (map()->hasInitialObjectMotion(context.getFrameId(),
                                      context.getObjectId(), &motion) &&
        pose_k_1_query) {
      object_pose_k = motion * pose_k_1_query.get();
      CHECK_NE(first_seen_frame, context.getFrameId());

    } else {
      // no motion or previous pose, so initalise with centroid translation and
      // identity rotation
      const auto [centroid_initial, result] =
          theta_accessor->computeObjectCentroid(context.getFrameId(),
                                                context.getObjectId());
      CHECK(result);
      gtsam::Pose3 initial_object_pose(gtsam::Rot3::Identity(),
                                       centroid_initial);

      getSafeQuery(object_pose_k,
                   theta_accessor->query<gtsam::Pose3>(object_pose_key_k),
                   initial_object_pose);

      // CHECK_EQ(first_seen_frame, context.getFrameId());
    }

    // for experiments and testing
    constexpr static bool init_LL_with_identity = false;
    constexpr static bool use_identity_rot_L_for_init = false;
    constexpr static bool corrupt_L_for_init = false;
    if (init_LL_with_identity && pose_k_1_query) {
      object_pose_k = pose_k_1_query.get();
    }
    if (use_identity_rot_L_for_init) {
      auto tmp_pose =
          gtsam::Pose3(gtsam::Rot3::Identity(), object_pose_k.translation());
      object_pose_k = tmp_pose;
    }
    if (corrupt_L_for_init) {
      object_pose_k = utils::perturbWithNoise<gtsam::Pose3>(object_pose_k, 0.2);
    }

    LOG(INFO) << "Adding object pose " << object_pose_k << " object id "
              << context.getObjectId();

    new_values.insert(object_pose_key_k, object_pose_k);
    is_other_values_in_map.insert2(object_pose_key_k, true);
    VLOG(50) << "Adding object pose key "
             << DynosamKeyFormatter(object_pose_key_k);
  }

  if (FLAGS_use_smoothing_factor) {
    const auto frame_id = context.getFrameId();
    const auto object_id = context.getObjectId();
    if (frame_id < 2) return;

    auto frame_node_k_2 = map()->getFrame(frame_id - 2u);
    auto frame_node_k_1 = map()->getFrame(frame_id - 1u);

    if (!frame_node_k_2 || !frame_node_k_1) {
      return;
    }

    // pose key at k-2
    const gtsam::Symbol object_pose_key_k_2 =
        frame_node_k_2->makeObjectPoseKey(object_id);
    // pose key at previous frame (k-1)
    const gtsam::Symbol object_pose_key_k_1 =
        frame_node_k_1->makeObjectPoseKey(object_id);

    auto object_smoothing_noise = noise_models_.object_smoothing_noise;
    CHECK(object_smoothing_noise);
    CHECK_EQ(object_smoothing_noise->dim(), 6u);

    {
      ObjectId object_label_k_2, object_label_k_1, object_label_k;
      FrameId frame_id_k_2, frame_id_k_1, frame_id_k;
      CHECK(reconstructPoseInfo(object_pose_key_k_2, object_label_k_2,
                                frame_id_k_2));
      CHECK(reconstructPoseInfo(object_pose_key_k_1, object_label_k_1,
                                frame_id_k_1));
      CHECK(reconstructPoseInfo(object_pose_key_k, object_label_k, frame_id_k));
      CHECK_EQ(object_label_k_2, object_label_k);
      CHECK_EQ(object_label_k_1, object_label_k);
      CHECK_EQ(frame_id_k_1 + 1, frame_id_k);    // assumes consequative frames
      CHECK_EQ(frame_id_k_2 + 1, frame_id_k_1);  // assumes consequative frames
    }

    // if the motion key at k (motion from k-1 to k), and key at k-1 (motion
    // from k-2 to k-1) exists in the map or is about to exist via new values,
    // add the smoothing factor
    if (is_other_values_in_map.exists(object_pose_key_k_1) &&
        is_other_values_in_map.exists(object_pose_key_k) &&
        is_other_values_in_map.exists(object_pose_key_k_2)) {
      new_factors.emplace_shared<LandmarkPoseSmoothingFactor>(
          object_pose_key_k_2, object_pose_key_k_1, object_pose_key_k,
          object_smoothing_noise);
      if (result.debug_info)
        result.debug_info->getObjectInfo(object_id).smoothing_factor_added =
            true;
      VLOG(50) << "Adding smoothing "
               << DynosamKeyFormatter(object_pose_key_k_2) << " -> "
               << DynosamKeyFormatter(object_pose_key_k);
    }
  }
}

}  // namespace dyno
