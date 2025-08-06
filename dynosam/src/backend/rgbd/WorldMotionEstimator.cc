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

#include "dynosam/backend/rgbd/WorldMotionEstimator.hpp"

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/factors/LandmarkMotionTernaryFactor.hpp"

namespace dyno {

StateQuery<gtsam::Pose3> WorldMotionAccessor::getObjectMotion(
    FrameId frame_id, ObjectId object_id) const {
  const auto frame_node_k = map()->getFrame(frame_id);
  CHECK(frame_node_k);

  // from k-1 to k
  return this->query<gtsam::Pose3>(
      frame_node_k->makeObjectMotionKey(object_id));
}
StateQuery<gtsam::Pose3> WorldMotionAccessor::getObjectPose(
    FrameId frame_id, ObjectId object_id) const {
  const auto object_poses = getObjectPoses(frame_id);
  if (object_poses.exists(object_id)) {
    return StateQuery<gtsam::Pose3>(ObjectPoseSymbol(object_id, frame_id),
                                    object_poses.at(object_id));
  }
  return StateQuery<gtsam::Pose3>::InvalidMap();
}

EstimateMap<ObjectId, gtsam::Pose3> WorldMotionAccessor::getObjectPoses(
    FrameId frame_id) const {
  EstimateMap<ObjectId, gtsam::Pose3> object_poses;
  // for each object, go over the cached object poses and check if that object
  // has a pose at the query frame
  for (const auto& [object_id, pose] :
       object_pose_cache_.collectByFrame(frame_id)) {
    object_poses.insert2(object_id, ReferenceFrameValue<gtsam::Pose3>(
                                        pose, ReferenceFrame::GLOBAL));
  }
  return object_poses;
}

void WorldMotionAccessor::refreshPoseCache() {
  // this is pretty slow!!
  // update object_pose_cache_ with new values
  // this means we have to start again at the first frame and update all the
  // poses!!
  ObjectPoseMap object_poses;
  const auto frames = map()->getFrames();
  auto frame_itr = frames.begin();

  const auto& gt_packet_map = hooks().ground_truth_packets_request();

  // advance itr one so we're now at the second frame
  std::advance(frame_itr, 1);
  for (auto itr = frame_itr; itr != frames.end(); itr++) {
    auto prev_itr = itr;
    std::advance(prev_itr, -1);
    CHECK(prev_itr != frames.end());

    const auto [frame_id_k, frame_k_ptr] = *itr;
    const auto [frame_id_k_1, frame_k_1_ptr] = *prev_itr;
    CHECK_EQ(frame_id_k_1 + 1, frame_id_k);

    // collect all object centoids from the latest estimate
    gtsam::FastMap<ObjectId, gtsam::Point3> centroids_k =
        this->computeObjectCentroids(frame_id_k);
    gtsam::FastMap<ObjectId, gtsam::Point3> centroids_k_1 =
        this->computeObjectCentroids(frame_id_k_1);
    // collect motions from k-1 to k
    MotionEstimateMap motion_estimates = this->getObjectMotions(frame_id_k);

    // construct centroid vectors in object id order
    gtsam::Point3Vector object_centroids_k_1, object_centroids_k;
    // we may not have a pose for every motion, e.g. if there is a new object at
    // frame k, it wont have a motion yet! if we have a motion we MUST have a
    // pose at the previous frame!
    for (const auto& [object_id, _] : motion_estimates) {
      CHECK(centroids_k.exists(object_id));
      CHECK(centroids_k_1.exists(object_id));

      object_centroids_k.push_back(centroids_k.at(object_id));
      object_centroids_k_1.push_back(centroids_k_1.at(object_id));
    }

    PropogatePoseResult propogation_result;
    if (FLAGS_init_object_pose_from_gt) {
      if (!gt_packet_map)
        LOG(WARNING) << "FLAGS_init_object_pose_from_gt is true but gt_packet "
                        "map not provided!";
      dyno::propogateObjectPoses(
          object_poses, motion_estimates, object_centroids_k_1,
          object_centroids_k, frame_id_k, gt_packet_map, &propogation_result);
    } else {
      dyno::propogateObjectPoses(object_poses, motion_estimates,
                                 object_centroids_k_1, object_centroids_k,
                                 frame_id_k, std::nullopt, &propogation_result);
    }

    if (VLOG_IS_ON(20)) {
      // report on how many were propogated with motions for this frame only
      const auto propogation_type =
          propogation_result.collectByFrame(frame_id_k);
      std::stringstream ss;
      ss << "Propogation result: " << propogation_type.size()
         << " objects for frame " << frame_id_k;

      size_t n_used_motion = 0;
      for (const auto& [_, type] : propogation_type) {
        if (type == PropogateType::Propogate) {
          n_used_motion++;
        }
      }
      ss << " of which " << n_used_motion << " were motion propogated";
      VLOG(100) << ss.str();
    }
  }

  VLOG(50) << "Updated object pose cache";
  object_pose_cache_ = object_poses;
  LOG(INFO) << "ending MotionWorldAccessor::postUpdateCallback";
}

void WorldMotionFormulation::dynamicPointUpdateCallback(
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

  bool add_point_from_previous = context.is_starting_motion_frame;
  bool does_previous_point_exist = new_values.exists(object_point_key_k_1) ||
                                   theta_accessor->exists(object_point_key_k_1);

  // check the case that the point exists at the previous frame
  // this only happens when we have non-consequative frames ie 2 3 4 5 6 7 10 11
  // and we are at frame 11!!
  //  logically this occurs becuase of the implementation of Formulation which
  //  checks for all the frames that the point has been observed in and only
  //  sets is_starting_motion_frame = true on the first pair of motions it does
  //  not account for inconsistencies in the frame order as pointed out...
  // Jesse: honestly this should NOT happen but we handle the case anyway...
  if (!add_point_from_previous && !does_previous_point_exist) {
    // we could do many things including re-initing this point OR removing it
    // from the tracked set
    CHECK(lmk_node->seenAtFrame(frame_node_k_1->frame_id));
    // if we think we shouldn't add the previous point but the previous point
    // does not exist, add it!@!
    add_point_from_previous = true;
  }

  // if first motion (i.e first time we have both k-1 and k), add both at k-1
  // and k
  if (add_point_from_previous) {
    CHECK(!theta_accessor->exists(object_point_key_k_1));

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
    if (result.debug_info)
      result.debug_info->getObjectInfo(context.getObjectId())
          .num_dynamic_factors++;
    result.updateAffectedObject(frame_node_k_1->frame_id,
                                context.getObjectId());

    // add landmark at previous frame
    // const Landmark measured_k_1 =
    //     lmk_node->getMeasurement(frame_node_k_1->frame_id).landmark;
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
        theta_accessor->exists(object_point_key_k_1))
      << "Key: " << this->formatter()(object_point_key_k_1)
      << " and seen at frames "
      << container_to_string(lmk_node->getSeenFrameIds());

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

  const gtsam::Key object_motion_key_k =
      frame_node_k->makeObjectMotionKey(context.getObjectId());

  auto landmark_motion_noise = noise_models_.landmark_motion_noise;
  new_factors.emplace_shared<LandmarkMotionTernaryFactor>(
      object_point_key_k_1, object_point_key_k, object_motion_key_k,
      landmark_motion_noise);
  result.updateAffectedObject(frame_node_k_1->frame_id, context.getObjectId());
  result.updateAffectedObject(frame_node_k->frame_id, context.getObjectId());
  if (result.debug_info)
    result.debug_info->getObjectInfo(context.getObjectId())
        .num_motion_factors++;

  // mark as now in map
  is_dynamic_tracklet_in_map_.insert2(context.getTrackletId(), true);
}

void WorldMotionFormulation::objectUpdateContext(
    const ObjectUpdateContextType& context, UpdateObservationResult& result,
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors) {
  auto frame_node_k = context.frame_node_k;
  const gtsam::Key object_motion_key_k =
      frame_node_k->makeObjectMotionKey(context.getObjectId());

  auto theta_accessor = this->accessorFromTheta();

  // skip if no motion pair availble
  if (!context.has_motion_pair) {
    return;
  }

  const auto frame_id = context.getFrameId();
  const auto object_id = context.getObjectId();

  if (!is_other_values_in_map.exists(object_motion_key_k)) {
    // when we have an initial motion
    Motion3 initial_motion = Motion3::Identity();

    constexpr static bool init_H_with_identity = false;
    if (!init_H_with_identity) {
      map()->hasInitialObjectMotion(frame_id, object_id, &initial_motion);
      LOG(INFO) << "Using motion from frontend " << initial_motion;
      initial_motion =
          gtsam::Pose3(gtsam::Rot3::Identity(), initial_motion.translation());
    }

    new_values.insert(object_motion_key_k, initial_motion);
    is_other_values_in_map.insert2(object_motion_key_k, true);
  }

  if (frame_id < 2) return;

  auto frame_node_k_1 = map()->getFrame(frame_id - 1u);
  if (!frame_node_k_1) {
    return;
  }

  if (FLAGS_use_smoothing_factor && frame_node_k_1->objectObserved(object_id)) {
    // motion key at previous frame
    const gtsam::Symbol object_motion_key_k_1 =
        frame_node_k_1->makeObjectMotionKey(object_id);

    auto object_smoothing_noise = noise_models_.object_smoothing_noise;
    CHECK(object_smoothing_noise);
    CHECK_EQ(object_smoothing_noise->dim(), 6u);

    {
      ObjectId object_label_k_1, object_label_k;
      FrameId frame_id_k_1, frame_id_k;
      CHECK(reconstructMotionInfo(object_motion_key_k_1, object_label_k_1,
                                  frame_id_k_1));
      CHECK(reconstructMotionInfo(object_motion_key_k, object_label_k,
                                  frame_id_k));
      CHECK_EQ(object_label_k_1, object_label_k);
      CHECK_EQ(frame_id_k_1 + 1, frame_id_k);  // assumes consequative frames
    }

    // if the motion key at k (motion from k-1 to k), and key at k-1 (motion
    // from k-2 to k-1) exists in the map or is about to exist via new values,
    // add the smoothing factor
    if (is_other_values_in_map.exists(object_motion_key_k_1) &&
        is_other_values_in_map.exists(object_motion_key_k)) {
      new_factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
          object_motion_key_k_1, object_motion_key_k, gtsam::Pose3::Identity(),
          object_smoothing_noise);
      if (result.debug_info)
        result.debug_info->getObjectInfo(context.getObjectId())
            .smoothing_factor_added = true;
    }
  }
}

}  // namespace dyno
