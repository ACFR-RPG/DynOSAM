#include "dynosam/formulations/KeyFrameHybridEstimator.hpp"

#include "dynosam/factors/MotionBetweenFactor.hpp"
#include "dynosam_opt/NonlinearOptimizer.hpp"

namespace dyno {

TrackedPointsPerObject HybridFormulationKeyFrame::getObjectPoints(
    const ObjectIds& objects) const {
  auto hybrid_accessor =
      this->derivedAccessor<HybridFormulationKeyFrameAccessor>();

  TrackedPointsPerObject points_per_object;
  for (ObjectId object_id : objects) {
    StatusLandmarkVector estimates =
        hybrid_accessor->getLocalDynamicLandmarkEstimates(object_id);

    std::vector<std::pair<TrackletId, gtsam::Point3>> tracklet_pairs;
    tracklet_pairs.reserve(estimates.size());

    for (const auto& lmk_status : estimates) {
      tracklet_pairs.push_back({lmk_status.trackletId(), lmk_status.value()});
    }
    points_per_object.insert2(object_id, std::move(tracklet_pairs));
  }

  return points_per_object;
}

StateQuery<Motion3ReferenceFrame>
HybridFormulationKeyFrameAccessor::getObjectMotionReferenceFrame(
    FrameId frame_id, ObjectId object_id) const {
  using Query = StateQuery<Motion3ReferenceFrame>;

  gtsam::Key motion_key;
  gtsam::Pose3 H_W_KF_k;
  FrameId from, to;
  const StateQueryStatus status = getObjectMotionReferenceFrameHelper(
      frame_id, object_id, motion_key, H_W_KF_k, from, to);
  if (status == StateQueryStatus::VALID) {
    // in base accessor expect motion to be a genuine KF motion
    return Query(motion_key, Motion3ReferenceFrame(
                                 H_W_KF_k, Motion3ReferenceFrame::Style::KF,
                                 ReferenceFrame::GLOBAL, from, to));
  } else {
    return Query(motion_key, status);
  }
}

TrackedPointsPerObject HybridFormulationKeyFrame::getObjectPoints() const {
  auto hybrid_accessor =
      this->derivedAccessor<HybridFormulationKeyFrameAccessor>();
  return getObjectPoints(hybrid_accessor->getObjectIds());
}

TrackedPointsPerObject HybridFormulationKeyFrame::getObjectPoints(
    FrameId frame_id) const {
  auto frame_node = map()->getFrame(frame_id);
  CHECK_NOTNULL(frame_node);
  return getObjectPoints(frame_node->objectSeenIds());
}

HybridKeyFrameUpdate HybridFormulationKeyFrame::generateUpdateInfo() const {
  auto hybrid_accessor =
      this->derivedAccessor<HybridFormulationKeyFrameAccessor>();

  // what if things are optimising when we do this...?
  HybridKeyFrameUpdate info;
  info.frame_id = hybrid_accessor->getLatestFrameId();
  info.timestamp = hybrid_accessor->getLatestTimestamp();
  info.camera_trajectory = hybrid_accessor->getCameraTrajectory();

  auto object_trajectories = hybrid_accessor->getMultiObjectTrajectories();
  auto object_points = getObjectPoints();

  CHECK_EQ(object_trajectories.size(), object_points.size());

  info.object_infos.reserve(object_trajectories.size());
  for (const auto& [object_id, trajectory] : object_trajectories) {
    CHECK(object_points.exists(object_id));

    HybridKeyFrameUpdate::Object object_info;
    object_info.object_id = object_id;
    object_info.trajectory = trajectory;
    object_info.object_points = object_points.at(object_id);

    info.object_infos.push_back(std::move(object_info));
  }

  return info;
}

MultiObjectTrajectories HybridFormulationKeyFrame::refinePerFrameMotionsPGO(
    const MultiObjectTrajectories& full_trajectories) const {
  MultiObjectTrajectories full_trajectories_refined = full_trajectories;

  auto map = this->map();
  HybridFormulationKeyFrameAccessor::Ptr accessor =
      this->derivedAccessor<HybridFormulationKeyFrameAccessor>();

  for (const auto& [object_id, trajectory_j] : full_trajectories) {
    gtsam::Values values;
    gtsam::NonlinearFactorGraph graph;

    std::vector<std::pair<FrameId, FrameId>> pose_frame_pairs;

    for (const auto& entry_k : trajectory_j) {
      const Motion3ReferenceFrame& f2f_motion = entry_k.data.motion;
      const gtsam::Pose3 pose_k = entry_k.data.pose;
      const FrameId to_frame = f2f_motion.to();
      const FrameId from_frame = f2f_motion.from();
      CHECK_EQ(f2f_motion.style(), MotionRepresentationStyle::F2F);

      pose_frame_pairs.push_back(std::make_pair(from_frame, to_frame));

      // use frontend poses as initial values
      // these will then be validated against the backend keyframe pose data
      // and priors added when necessary for anchor keyframes
      gtsam::Key object_pose_key = ObjectPoseSymbol(object_id, to_frame);
      gtsam::Key object_pose_key_from = ObjectPoseSymbol(object_id, from_frame);
      values.insert(object_pose_key, pose_k);

      // // convert H_W_km1_k to relative motion constraint
      // gtsam::Pose3 L_W_from = trajectory_j.at(from_frame).pose;
      // gtsam::Pose3 H_L_from_to =
      //     L_W_from.inverse() * f2f_motion.estimate() * L_W_from;

      // // TODO: for now!!
      gtsam::SharedNoiseModel relative_noise_model =
          gtsam::noiseModel::Isotropic::Sigma(6u, 0.2);

      using BetweenMotion3Factor = MotionBetweenFactor<gtsam::Pose3>;
      auto relative_object_motion = boost::make_shared<BetweenMotion3Factor>(
          object_pose_key_from, object_pose_key, f2f_motion.estimate(),
          relative_noise_model);
      graph += relative_object_motion;

      // auto relative_object_motion =
      //     boost::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
      //         object_pose_key_from, object_pose_key, H_L_from_to,
      //         relative_noise_model);
      // graph += relative_object_motion;

      LOG(INFO) << "Adding relative motion constraint " << from_frame << " -> "
                << to_frame;

      if (isObjectKeyFrame(object_id, to_frame)) {
        LOG(INFO) << to_frame << " is KF";
        const KeyFrameMetaData& kf_data =
            key_frames_per_object_.at(object_id, to_frame);

        // sanity check frames are good
        CHECK_EQ(kf_data.H_W_lRKF_KF.to(), to_frame);

        auto object_node = map->getObject(object_id);
        CHECK_NOTNULL(object_node);

        const KeyFrameRange::ConstPtr akf_range =
            key_frame_data_.find(object_id, to_frame);

        if (!akf_range) {
          DYNO_THROW_MSG(DynosamException) << "Missing anchor frame for motion "
                                           << info_string(to_frame, object_id);
        }

        // we will validate the backend pose is the same as the frontend pose
        auto [akf_id, akf_pose_backend] = akf_range->dataPair();
        LOG(INFO) << akf_id;

        // we expect to have a refined motion from some previous anchor frame
        // to the to_frame
        // importantly this motion will inform us what the anchor frame is
        // via the "from" frame
        Motion3ReferenceFrame H_W_akf_k_refined = DYNO_GET_QUERY_DEBUG(
            accessor->getEstimatedMotion(object_id, to_frame));
        // absolutely horrible the arguments are swapped!
        gtsam::Pose3 L_W_k_refined =
            DYNO_GET_QUERY_DEBUG(accessor->getObjectPose(to_frame, object_id));

        // CHECK_EQ(H_W_akf_k_refined.to(), to_frame);

        // // convert H_W_KF_k to relative motion constraint
        // CHECK(trajectory_j.exists(akf_id));
        // gtsam::Pose3 L_W_akf = trajectory_j.at(akf_id).pose;
        // gtsam::Pose3 H_L_akf_to =
        //     L_W_akf.inverse() * H_W_akf_k_refined.estimate() * L_W_akf;

        // add keyframe motion as constraint to hold the graph together!
        // TODO: odometry should be different noise!!
        // auto refined_keyframe_motion =
        //     boost::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        //         object_pose_key_akf, object_pose_key, H_L_akf_to,
        //         this->noiseModels().odometry_noise);
        // graph += refined_keyframe_motion;

        graph.addPrior<gtsam::Pose3>(object_pose_key, L_W_k_refined,
                                     this->noiseModels().initial_pose_prior);

        LOG(INFO) << "Adding refined KF relative motion constraint " << akf_id
                  << " -> " << to_frame;

        // TODO: currently intermediate frame nodes will not exist!!

        if (kf_data.keyframe_status == ObjectKeyFrameStatus::AnchorKeyFrame) {
          // CHECK_EQ(H_W_akf_k_refined.from(), akf_id);

          // check that the per-frame motions start from the same pose as the
          // backemd although the value does not NEED to be the same, it ensures
          // that there is good synchronisation between the frontend and the
          // estimator get the pose associated with the anchor frame
          gtsam::Pose3 akf_pose_frontend = trajectory_j.at(akf_id).pose;
          CHECK(gtsam::traits<gtsam::Pose3>::Equals(akf_pose_backend,
                                                    akf_pose_frontend, 1e-4))
              << "KF pose " << akf_pose_backend << " pose " << akf_pose_frontend
              << " akf id " << akf_id;

          gtsam::Key anchor_object_pose_key =
              ObjectPoseSymbol(object_id, akf_id);
          LOG(INFO) << "Adding pose prior at KF pose " << akf_id;
          // add a strong prior to anchor pose as this will not change!
          graph.addPrior<gtsam::Pose3>(anchor_object_pose_key,
                                       akf_pose_frontend,
                                       this->noiseModels().initial_pose_prior);

        } else if (kf_data.keyframe_status ==
                   ObjectKeyFrameStatus::RegularKeyFrame) {
        } else {
          throw DynosamException("keyframe status cannot be NonKeyFrame!");
        }

        // // more sanity checks
        // right now this is a fail because only KF measurements are added to
        // the map!! CHECK_EQ(from_frame, object_node->getFirstSeenFrame());
      }
    }

    using LMOptimizer =
        dyno::NonlinearOptimizer<gtsam::LevenbergMarquardtOptimizer>;
    LMOptimizer solver(graph, values);

    NonlinearOptimizerSummary summary;
    NonlinearOptimizerOptions options;

    LOG(INFO) << "Beginning PGO j=" << object_id;
    gtsam::Values optimised_values;
    CHECK(solver.solve(optimised_values, options, &summary));

    LOG(INFO) << "Initial error: " << summary.initial_error << " final error "
              << summary.final_error << " time[s] "
              << summary.cumulative_time_in_seconds
              << " #iterations= " << summary.numIterations();

    // recover pose values and correct motions
    PoseWithMotionTrajectory& refined_trajectory_j =
        full_trajectories_refined.at(object_id);
    for (const auto& [from_frame, to_frame] : pose_frame_pairs) {
      gtsam::Key object_pose_key_from = ObjectPoseSymbol(object_id, from_frame);
      gtsam::Key object_pose_key_to = ObjectPoseSymbol(object_id, to_frame);

      CHECK(object_pose_key_from == object_pose_key_to - 1 ||
            object_pose_key_from == object_pose_key_to);

      gtsam::Pose3 L_W_from =
          optimised_values.at<gtsam::Pose3>(object_pose_key_from);
      gtsam::Pose3 L_W_to =
          optimised_values.at<gtsam::Pose3>(object_pose_key_to);
      gtsam::Pose3 H_W_from_to = L_W_to * L_W_from.inverse();

      Motion3ReferenceFrame f2f_motion_refined(
          H_W_from_to, Motion3ReferenceFrame::Style::F2F,
          ReferenceFrame::GLOBAL, from_frame, to_frame);

      PoseWithMotion refined_entry;
      refined_entry.pose = L_W_to;
      refined_entry.motion = f2f_motion_refined;

      // always update the "to" frame
      // we will have identity motions where from==to in which case we still
      // update every frame since L_W_from==L_W_to
      CHECK(refined_trajectory_j.update(to_frame, refined_entry));
    }
  }
  return full_trajectories_refined;
}

UpdateObservationResult HybridFormulationKeyFrame::updateDynamicObservations(
    FrameId frame_id_k, gtsam::Values& new_values,
    gtsam::NonlinearFactorGraph& new_factors,
    const UpdateObservationParams& update_params) {
  typename Map::Ptr map = this->map();
  auto accessor = this->accessorFromTheta();

  // keep track of the new factors added in this function
  // these are then appended to the internal factors_ and new_factors
  gtsam::NonlinearFactorGraph internal_new_factors;
  // keep track of the new values added in this function
  // these are then appended to the internal values_ and new_values
  gtsam::Values internal_new_values;

  UpdateObservationResult result(update_params);

  // starting slot number is size of new factors
  // as long as the new factor slot is calculated before adding a new factor
  const Slot starting_factor_slot = new_factors.size();

  const auto frame_node_k = map->getFrame(frame_id_k);

  for (const auto& object_node : frame_node_k->objectsSeen()) {
    const ObjectId object_id = object_node->getId();

    // depending on how the frontend is implemented we may have measurements in
    // the map at non-keyframes but only want to update objects at KFs
    if (isObjectKeyFrame(object_id, frame_id_k)) {
      Context context;
      context.object_node = object_node;
      context.frame_node = frame_node_k;
      context.X_k_measured = getInitialOrLinearizedSensorPose(frame_id_k);
      context.starting_factor_slot = starting_factor_slot;

      LOG(INFO) << "Updating object " << info_string(frame_id_k, object_id);
      updateObject(context, result, internal_new_values, internal_new_factors);
    }
  }

  if (result.debug_info && VLOG_IS_ON(20)) {
    for (const auto& [object_id, object_info] :
         result.debug_info->getObjectInfos()) {
      std::stringstream ss;
      ss << "Object id debug info: " << object_id << "\n";
      ss << object_info;
      LOG(INFO) << ss.str();
    }
  }

  factors_ += internal_new_factors;
  new_factors += internal_new_factors;
  // update internal theta and factors
  // theta_.insert(internal_new_values);
  shared_data_->threadSafeInsertTheta(internal_new_values);
  // add to the external new_values
  new_values.insert(internal_new_values);
  return result;
}

bool HybridFormulationKeyFrame::isObjectKeyFrame(ObjectId object_id,
                                                 FrameId frame_id) const {
  return key_frames_per_object_.exists(object_id, frame_id);
}

void HybridFormulationKeyFrame::updateObject(
    const Context& context, UpdateObservationResult& result,
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors) {
  auto object_node = context.object_node;
  auto frame_node_kf = context.frame_node;
  const auto frame_id_kf = context.getFrameId();
  const auto object_id = context.getObjectId();

  const gtsam::Key object_motion_key_kf =
      frame_node_kf->makeObjectMotionKey(object_id);
  const gtsam::Key pose_key_kf = frame_node_kf->makePoseKey();

  auto seen_lmks_k = object_node->landmarksSeenAtFrame(frame_id_kf);

  CHECK(!is_other_values_in_map.exists(object_motion_key_kf));
  CHECK(initial_H_W_AKF_k_.exists(object_id, frame_id_kf));

  const KeyFrameRange::ConstPtr kf_range =
      key_frame_data_.find(object_id, frame_id_kf);
  CHECK(kf_range);

  // TODO: should check if the AKF_id is different for the from and the to?
  const auto [AKF_id, AKF_pose] = kf_range->dataPair();

  Motion3ReferenceFrame H_W_AKF_k =
      initial_H_W_AKF_k_.at(object_id, frame_id_kf);
  CHECK_EQ(H_W_AKF_k.from(), AKF_id);
  CHECK_EQ(H_W_AKF_k.to(), frame_id_kf);
  // Must add measurements at both AKF and KF (ie. multi view)
  // since the motion is constructed between these two frames
  const FrameId frame_id_akf = AKF_id;

  CHECK(key_frames_per_object_.exists(object_id, frame_id_kf));
  const KeyFrameMetaData& kf_meta_data =
      key_frames_per_object_.at(object_id, frame_id_kf);
  // measured motion from the frontend.
  const Motion3ReferenceFrame& H_W_lRKF_KF = kf_meta_data.H_W_lRKF_KF;
  CHECK_EQ(H_W_lRKF_KF.to(), frame_id_kf);
  // measured from frame
  const auto& lRKF_id = H_W_lRKF_KF.from();

  typename Map::Ptr map = this->map();
  auto frame_node_akf = map->getFrame(frame_id_akf);
  CHECK(frame_node_akf);

  // frame node for last regular KF
  auto frame_node_lrkf = map->getFrame(lRKF_id);
  CHECK(frame_node_lrkf);
  // object motion key from the 'from' frame
  const gtsam::Key object_motion_key_lkf =
      frame_node_lrkf->makeObjectMotionKey(object_id);

  new_values.insert(object_motion_key_kf, H_W_AKF_k.estimate());
  is_other_values_in_map.insert2(object_motion_key_kf, true);

  result.updateAffectedObject(frame_id_kf, object_id);

  // add zero motion at AKF
  // TODO: assumes that at least SOME points were seen at AKF!
  // TODO: should only add this if weve seen some points AKF -> should enforce
  // this actually happens!!!
  const gtsam::Key object_motion_key_akf =
      frame_node_akf->makeObjectMotionKey(object_id);
  if (!is_other_values_in_map.exists(object_motion_key_akf)) {
    new_values.insert(object_motion_key_akf, gtsam::Pose3::Identity());
    is_other_values_in_map.insert2(object_motion_key_akf, true);

    // add strong prior on initaion motion (which is just identity)
    new_factors.addPrior<gtsam::Pose3>(object_motion_key_akf,
                                       gtsam::Pose3::Identity(),
                                       noise_models_.initial_pose_prior);
  }

  size_t num_points_seen_akf = 0;
  for (const auto& obj_lmk_node : seen_lmks_k) {
    CHECK_EQ(obj_lmk_node->objectId(), object_id);
    const TrackletId tracklet_id = obj_lmk_node->trackletId();
    // LOG(INFO) << "Iterating through dynamic lmk " << tracklet_id;
    const gtsam::Key point_key = this->makeDynamicKey(tracklet_id);

    // becuase we dont anchor the motion (like in the original Hybrid with an
    // identity motion) we dont always have a motion at an anchor keyframe so
    // the point doesnt need to be seen there
    // TODO: depending in implementation of regular vs anchor KF, we may expect
    // that at anchor frames a point is not necessarily
    // seen at both the AKF and the RKF but should be seen at RKF-1 and RKF-k
    // (ie. if the previous KF was only a RKF, points should be seen at both?
    // MAYBE) CHECK(obj_lmk_node->seenAtFrame(frame_id_akf)) << "Lmk i=" <<
    // tracklet_id << " Object " << object_id << " not seen at " << frame_id_akf
    // << " but this is the from motion";

    // TODO: seen ay any frame!
    CHECK(obj_lmk_node->seenAtFrame(frame_id_kf))
        << "Lmk i=" << tracklet_id << "Object " << object_id << " not seen at "
        << frame_id_kf << " but this is the to motion";

    if (!isDynamicTrackletInMap(obj_lmk_node)) {
      // we may have more seen landmarks than points in the filter
      // This "shouldn't" happen but is maybe some slightly bug in bookkeeping
      // somewhere CHECK(m_L_initial_.exists(object_id, tracklet_id)) <<
      // "Missing initalisation for j=" << object_id << " i=" << tracklet_id;
      if (!m_L_initial_.exists(object_id, tracklet_id)) {
        continue;
      }

      // uuuh need to update these becuase something in the accessor
      //  needs them!
      // TODO: double check implementation and write comment!
      is_dynamic_tracklet_in_map_.insert2(tracklet_id, AKF_id);
      all_dynamic_landmarks_.insert2(tracklet_id, AKF_id);

      factors_added_.insert2(tracklet_id, std::set<FrameId>{});

      CHECK(isDynamicTrackletInMap(obj_lmk_node));

      gtsam::Point3 m_L_initial = m_L_initial_.at(object_id, tracklet_id);
      new_values.insert(point_key, m_L_initial);

      if (result.debug_info) {
        result.debug_info->getObjectInfo(object_id).num_new_dynamic_points++;
      }

      // at measurements of point if also seen at AKF
      // most measurements wont actually be seen at the AFK
      // since this is the first frame!
      // what we really want to do is make sure we add measurements for the
      // "from" -> "to"
      // frame of the original motion (ie. the one from the frontend)
      // if(obj_lmk_node->seenAtFrame(frame_id_akf)) {
      //   addHybridMotionFactor(new_factors, pose_key_akf,
      //   object_motion_key_akf, point_key,
      //                     AKF_pose, obj_lmk_node, frame_node_akf);
      //   if (result.debug_info) {
      //     result.debug_info->getObjectInfo(context.getObjectId())
      //         .num_dynamic_factors++;
      //   }
      //   num_points_seen_akf++;
      // }

      // add measurements at both Keyframes a point is seen at
      // only needed if a point is new (I think!)
      // if (obj_lmk_node->seenAtFrame(lRKF_id)) {
      //   const gtsam::Key object_motion_key_lrkf =
      //       frame_node_lrkf->makeObjectMotionKey(object_id);
      //   const gtsam::Key pose_key_lrkf = frame_node_lrkf->makePoseKey();

      //   addHybridMotionFactor(new_factors, pose_key_lrkf,
      //                         object_motion_key_lrkf, point_key, AKF_pose,
      //                         obj_lmk_node, frame_node_lrkf);
      //   if (result.debug_info) {
      //     result.debug_info->getObjectInfo(context.getObjectId())
      //         .num_dynamic_factors++;
      //   }
      //   num_points_seen_akf++;
      // }

      // add at from frame if point is new
      //  assume that once we have seen it we only need to add measurements
      //  at the newest KF, since we will have added measurements
      //  for the previous KF last iteration (if all works well!)
      // addHybridMotionFactor(new_factors, pose_key, object_motion_key,
      // point_key,
      //                       AKF_pose, obj_lmk_node, frame_node_akf);
    }

    // check if we've added a factor at the regular from frame
    // actually could just sanity check we've added factors at any/every frame
    // that we have a motion for

    // CHECK(obj_lmk_node->seenAtFrame(lRKF_id))
    //     << info_string(lRKF_id, obj_lmk_node->object_id);
    std::set<FrameId>& frames_with_factors_added =
        factors_added_.at(tracklet_id);
    const bool factor_not_added_for_lRKF =
        frames_with_factors_added.find(lRKF_id) ==
        frames_with_factors_added.end();

    // TODO: seen ay any frame!
    if (factor_not_added_for_lRKF && obj_lmk_node->seenAtFrame(lRKF_id)) {
      const gtsam::Key object_motion_key_lrkf =
          frame_node_lrkf->makeObjectMotionKey(object_id);
      const gtsam::Key pose_key_lrkf = frame_node_lrkf->makePoseKey();

      addHybridMotionFactor(new_factors, pose_key_lrkf, object_motion_key_lrkf,
                            point_key, AKF_pose, obj_lmk_node, frame_node_lrkf);
      if (result.debug_info) {
        result.debug_info->getObjectInfo(context.getObjectId())
            .num_dynamic_factors++;
      }
      num_points_seen_akf++;
      frames_with_factors_added.insert(frame_node_lrkf->frameId());
    }

    addHybridMotionFactor(new_factors, pose_key_kf, object_motion_key_kf,
                          point_key, AKF_pose, obj_lmk_node, frame_node_kf);
    frames_with_factors_added.insert(frame_id_kf);

    // sanity check/ backwards adding of points to ensure measurements
    // are added for all possible frames
    // TODO: seenAtAnyFrame check for isKeyFrame(), add factors accordinfly!
    const FrameIds frames_with_measurements = obj_lmk_node->getSeenFrameIds();
    for (const FrameId frame_with_z : frames_with_measurements) {
      // check if this frame is in the same backend range
      // if it is not, ignore it!
      const KeyFrameRange::ConstPtr range =
          key_frame_data_.find(object_id, frame_with_z);
      CHECK(range);
      const auto [AKF_id_for_z, _] = range->dataPair();
      // if we have not added a measurement
      if (AKF_id_for_z != AKF_id) {
        continue;
      }

      // we have not added a factor at frame id with z
      // TODO: code duplication as above!!
      const bool factor_not_added_for_frame_with_z =
          frames_with_factors_added.find(frame_with_z) ==
          frames_with_factors_added.end();
      if (factor_not_added_for_frame_with_z &&
          obj_lmk_node->seenAtFrame(frame_with_z)) {
        auto frame_node_with_z = map->getFrame(frame_with_z);
        CHECK_NOTNULL(frame_node_with_z);

        const gtsam::Key object_motion_key =
            frame_node_with_z->makeObjectMotionKey(object_id);
        const gtsam::Key pose_key = frame_node_with_z->makePoseKey();

        addHybridMotionFactor(new_factors, pose_key, object_motion_key,
                              point_key, AKF_pose, obj_lmk_node,
                              frame_node_with_z);
        if (result.debug_info) {
          result.debug_info->getObjectInfo(context.getObjectId())
              .num_dynamic_factors++;
        }
        frames_with_factors_added.insert(frame_with_z);
      }
    }

    {
      // do a sanity check that we've added all meaasurement factors for this
      // point
      // const FrameIds frames_with_measurements =
      // obj_lmk_node->getSeenFrameIds(); FrameIds
      // frames_with_factors_added_vec(frames_with_factors_added.begin(),
      //                                        frames_with_factors_added.end());
      // CHECK(equals_with_abs_tol(frames_with_measurements,
      //                           frames_with_factors_added_vec))
      //   << "Frames with measurements:" <<
      //   container_to_string(frames_with_measurements)
      //   << "\n"
      //   << "Frames with factors:" <<
      //   container_to_string(frames_with_factors_added_vec);
    }

    // if(!smoothing_factors_added_.exists(object_motion_key_kf)) {
    //   // check we have the previous keyframed motion
    //   is_other_values_in_map.exists(object_motion_key_lkf);

    //   auto object_smoothing_noise = noise_models_.object_smoothing_noise;
    //   CHECK(object_smoothing_noise);
    //   CHECK_EQ(object_smoothing_noise->dim(), 6u);

    //   new_factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
    //     object_motion_key_lkf, object_motion_key_kf, H_W_lRKF_KF.estimate(),
    //     object_smoothing_noise
    //   );

    //   if (result.debug_info)
    //     result.debug_info->getObjectInfo(context.getObjectId())
    //         .smoothing_factor_added = true;

    //     // update internal containers
    //   smoothing_factors_added_.insert(object_motion_key_kf);
    // }

    if (result.debug_info) {
      result.debug_info->getObjectInfo(context.getObjectId())
          .num_dynamic_factors++;
    }
  }

  LOG(INFO) << "Num factors added for measuemenets at AKF "
            << num_points_seen_akf;
}

void HybridFormulationKeyFrame::addHybridMotionFactor(
    gtsam::NonlinearFactorGraph& new_factors, gtsam::Key pose_key,
    gtsam::Key object_motion_key, gtsam::Key point_key,
    const gtsam::Pose3& KF_pose, SharedLandmarkNode lmk_node,
    SharedFrameNode frame_node) {
  // Landmark measured_point_local;
  // gtsam::SharedNoiseModel measurement_covariance;
  // std::tie(measured_point_local, measurement_covariance) =
  //     MeasurementTraits::pointWithCovariance(
  //         lmk_node->getMeasurement(frame_node));

  auto stereo_measurement =
      MeasurementTraits::stereo(lmk_node->getMeasurement(frame_node));

  // FOR NOW
  CHECK(stereo_measurement);
  auto [measurement, noise_model] = *stereo_measurement;

  if (params_.makeDynamicMeasurementsRobust()) {
    noise_model = factor_graph_tools::robustifyHuber(params_.k_huber_3d_points_,
                                                     noise_model);
  }

  new_factors.emplace_shared<StereoHybridMotionFactor>(
      measurement, KF_pose, noise_model, rgbd_camera_->getFakeStereoCalib(),
      pose_key, object_motion_key, point_key, true /* throw cheirality*/
  );

  // new_factors.emplace_shared<HybridMotionFactor>(
  //     pose_key, object_motion_key, point_key, measured_point_local, KF_pose,
  //     noise_model);
}

// TODO: this should mark objects with keyframes!
void HybridFormulationKeyFrame::addObjects(
    FrameId frame_id, const ObjectPoseChangeInfoMap& object_motion_info) {
  auto accessor = this->derivedAccessor<HybridFormulationKeyFrameAccessor>();
  CHECK_NOTNULL(accessor);

  for (const auto& [object_id, object_info] : object_motion_info) {
    CHECK(object_info.isKeyFrame());
    // estimated keyframe motioa from the frontend
    // in this case k is the current but will now also be the latest KF
    const Motion3ReferenceFrame& H_W_RKF_k = object_info.H_W_KF_k;
    const ObjectKeyFrameStatus keyframe_status = object_info.keyframe_status;

    KeyFrameMetaData kf_data;
    kf_data.keyframe_status = keyframe_status;
    kf_data.H_W_lRKF_KF = H_W_RKF_k;

    CHECK_EQ(frame_id, H_W_RKF_k.to());

    key_frames_per_object_.insert22(object_id, frame_id, kf_data);

    LOG(INFO) << "Processing object track: " << info_string(frame_id, object_id)
              << " keyframe status: " << keyframe_status;

    // TODO: we initalie the new KF with the pose provided from the frontend
    // for consistency. This is fine when the object observations are
    // continuous
    // but at some point the KF pose in the frontend and back-end will change
    // (i.e after opt!)

    // ad new initialisation points to backend
    // misleading print as we dont add this many points! Only new ones
    VLOG(10) << "Adding initial object points of size "
             << object_info.initial_object_points.size();
    for (const auto& landmark_status : object_info.initial_object_points) {
      const TrackletId& tracklet_id = landmark_status.trackletId();
      const gtsam::Point3& m_L = landmark_status.value();
      // only add new ones?
      if (!m_L_initial_.exists(object_id, tracklet_id)) {
        // LOG(INFO) << "Making initial object points j=" << object_id << "
        // i="
        // << tracklet_id;
        m_L_initial_.insert22(object_id, tracklet_id, m_L);
      }
    }

    if (keyframe_status == ObjectKeyFrameStatus::AnchorKeyFrame) {
      key_frame_data_.startNewActiveRange(object_id, H_W_RKF_k.from(),
                                          object_info.L_W_KF);
      LOG(INFO) << "Making Anchor KF for NEW object "
                << info_string(H_W_RKF_k.from(), object_id) << " with motion "
                << H_W_RKF_k.from() << " -> " << H_W_RKF_k.to();

      // the frontend range is always "to" because it indicates the start
      // of the next range and a single motion represents one
      front_end_keyframes_.startNewActiveRange(object_id, H_W_RKF_k.from(),
                                               object_info.L_W_KF);
      LOG(INFO) << "Making Regular KF for NEW object "
                << info_string(H_W_RKF_k.from(), object_id) << " with motion "
                << H_W_RKF_k.from() << " -> " << H_W_RKF_k.to();
      initial_H_W_AKF_k_.insert22(object_id, H_W_RKF_k.to(), H_W_RKF_k);
    } else {
      CHECK_EQ(keyframe_status, ObjectKeyFrameStatus::RegularKeyFrame);

      const KeyFrameRange::ConstPtr last_frontend_range =
          front_end_keyframes_.find(object_id, frame_id);
      CHECK(last_frontend_range)
          << "Failed for tracked object " << info_string(frame_id, object_id);
      auto [lRKF_id, L_lRKF] = last_frontend_range->dataPair();

      // hopefully the last regular KF is the from frame
      lRKF_id = H_W_RKF_k.from();
      LOG(INFO) << "Last regular KF " << lRKF_id;

      // TODO: if this is regular KF then the position of this KF will change
      // according to the motion that is refined
      //  as L_W_k = L_W_KF = H_W_AKF_KF * L_AKF
      front_end_keyframes_.startNewActiveRange(object_id, H_W_RKF_k.to(),
                                               object_info.L_W_k);
      LOG(INFO) << "Making Regular KF for tracked object "
                << info_string(H_W_RKF_k.to(), object_id) << " with motion "
                << H_W_RKF_k.from() << " -> " << H_W_RKF_k.to();

      const KeyFrameRange::ConstPtr frontend_range =
          front_end_keyframes_.find(object_id, frame_id);
      CHECK(frontend_range);
      // the most recent motion added to the estimator should take us from
      // backend_kf_id to last_kf_id
      const auto [current_kf_id, current_kf_pose] = frontend_range->dataPair();
      LOG(INFO) << "Current regular KF " << current_kf_id;

      // get backend anchor point and confert motion if necessary
      const KeyFrameRange::ConstPtr backend_range =
          CHECK_NOTNULL(key_frame_data_.find(object_id, frame_id));
      const auto [backend_kf_id, backend_kf_pose] = backend_range->dataPair();
      LOG(INFO) << "Anchor KF id: " << backend_kf_id;

      LOG(INFO) << "Provided object odometry " << H_W_RKF_k.from() << " -> "
                << H_W_RKF_k.to();

      // motion from anchor point to current k
      // this value will be added to the estimator
      Motion3ReferenceFrame H_W_AKF_KF_initial;
      if (H_W_RKF_k.from() == backend_kf_id) {
        // TODO: also check pose is close?
        CHECK_EQ(H_W_RKF_k.from(), lRKF_id);
        H_W_AKF_KF_initial = H_W_RKF_k;
      } else {
        // motion does not match, so start a new starting point and transform
        // the motio update frontend range
        // NOTE: this uses the "to" motion (not the from)

        // need to transform into correct frame using (ideally the most up to
        // date, i.e estimated motion)
        LOG(INFO) << "Looking up estimated motion from " << backend_kf_id
                  << " -> " << lRKF_id;

        // TODO: eventually should come from optimizer
        CHECK(initial_H_W_AKF_k_.exists(object_id, lRKF_id));
        // from current anchor keyframe to last regular kf
        const auto H_W_AKF_lKF_initial =
            initial_H_W_AKF_k_.at(object_id, lRKF_id);

        const StateQuery<Motion3ReferenceFrame> H_W_AKF_lKF_refined =
            accessor->getEstimatedMotion(object_id, lRKF_id);

        Motion3ReferenceFrame H_W_AKF_lKF;
        // use refined motion estimate if available, otherwise fall back to
        // initial NOTE: I think the refined query should ALways be available
        // getSafeQuery(H_W_AKF_lKF, H_W_AKF_lKF_refined, H_W_AKF_lKF_initial);
        H_W_AKF_lKF = H_W_AKF_lKF_initial;

        CHECK_EQ(H_W_AKF_lKF.from(), backend_kf_id);
        CHECK_EQ(H_W_AKF_lKF.to(), lRKF_id);

        // this motion does us from the last keyframe to the current frame k
        CHECK_EQ(H_W_RKF_k.from(), lRKF_id);
        CHECK_EQ(H_W_RKF_k.to(), current_kf_id);

        H_W_AKF_KF_initial = Motion3ReferenceFrame(
            H_W_RKF_k.estimate() * H_W_AKF_lKF.estimate(),
            Motion3ReferenceFrame::Style::KF, ReferenceFrame::GLOBAL,
            backend_kf_id, current_kf_id);
      }
      initial_H_W_AKF_k_.insert22(object_id, H_W_AKF_KF_initial.to(),
                                  H_W_AKF_KF_initial);
    }
  }
}

ObjectPoseMap HybridFormulationKeyFrame::getInitialObjectPoses() const {
  ObjectPoseMap kf_poses;

  for (const auto& [object_id, H_W_AKF_k_per_frame] : initial_H_W_AKF_k_) {
    CHECK(key_frame_data_.exists(object_id));

    const auto& kf_ranges = key_frame_data_.at(object_id);

    for (const auto& [kf_frame_id, H_W_AKF_KF] : H_W_AKF_k_per_frame) {
      // should be the anchor frame
      const auto from_frame = H_W_AKF_KF.from();
      // ranges are stored by their "to" range
      const auto to_frame = H_W_AKF_KF.to();

      const auto anchor_range = kf_ranges.find(to_frame);
      CHECK(anchor_range);
      const auto [anchor_frame, anchor_pose] = anchor_range->dataPair();
      CHECK_EQ(from_frame, anchor_frame);

      gtsam::Pose3 L_w = H_W_AKF_KF.estimate() * anchor_pose;
      kf_poses.insert22(object_id, to_frame, L_w);
    }
  }

  return kf_poses;
}

}  // namespace dyno
