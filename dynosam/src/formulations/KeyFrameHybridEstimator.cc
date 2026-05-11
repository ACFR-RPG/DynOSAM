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

PoseTrajectory HybridFormulationKeyFrameAccessor::getCameraTrajectory() const {
  PoseTrajectory pose_trajectory;

  // only go up to the last frame the backend has finished processing
  const SharedModuleStates* shared_module_states =
      map_->getSharedModuleStates();
  const std::optional<FrameId> last_ckf_id_optimized =
      shared_module_states->getLatestOptimizedFrame();

  // No camera states have been optimized yet
  if (!last_ckf_id_optimized) {
    return pose_trajectory;
  }

  // sanitry check it is actually a keyframe
  CHECK(map_->isCameraKeyFrame(last_ckf_id_optimized.value()));

  for (const auto& frame_CKF : map_->getCameraKeyFrames()) {
    const FrameId frame_id_CKF = frame_CKF->frameId();
    // only include states that have been optimized
    if (frame_id_CKF > last_ckf_id_optimized.value()) {
      continue;
    }

    const Timestamp timestamp = frame_CKF->timestamp();
    const gtsam::Pose3 X_W_k =
        DYNO_GET_QUERY_DEBUG(this->getSensorPose(frame_id_CKF));
    pose_trajectory.insert(frame_id_CKF, timestamp, X_W_k);
  }

  return pose_trajectory;
}

MultiObjectTrajectories
HybridFormulationKeyFrameAccessor::getMultiObjectTrajectories() const {
  // grossly this gets the trajectory with all variables including some not
  // optimized yet but initlized from the frontend
  MultiObjectTrajectories full_trajectories =
      Base::getMultiObjectTrajectories();

  const SharedModuleStates* shared_module_states =
      map_->getSharedModuleStates();

  MultiObjectTrajectories only_optimized;
  for (const auto& [object_id, full_traj] : full_trajectories) {
    const std::optional<FrameId> last_optimized_frame =
        shared_module_states->getLatestOptimizedFrame(object_id);

    if (!last_optimized_frame) {
      // no variables for this object have been optimized yet
      continue;
    }

    // sanitry check it is actually a keyframe
    CHECK(map_->isObjectKeyFrame(last_optimized_frame.value(), object_id))
        << "Last object frame: " << last_optimized_frame.value()
        << " j= " << object_id
        << " map info: " << map_->verboseInfo(last_optimized_frame.value(), 8);

    // get subset of trajectory up to optimized point
    only_optimized.insert2(object_id,
                           full_traj.range(std::nullopt, last_optimized_frame));
  }

  return only_optimized;
};

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

// output trajectories only up to last object keyframe
MultiObjectTrajectories HybridFormulationKeyFrame::refinePerFrameMotionsPGO(
    const MultiObjectTrajectories& full_trajectories) const {
  auto map = this->map();
  HybridFormulationKeyFrameAccessor::Ptr accessor =
      this->derivedAccessor<HybridFormulationKeyFrameAccessor>();

  // only go up to the last frame the backend has finished processing
  const SharedModuleStates* shared_module_states = map->getSharedModuleStates();

  MultiObjectTrajectories full_trajectories_refined;
  for (const auto& [object_id, full_trajectory_j] : full_trajectories) {
    gtsam::Values values;
    gtsam::NonlinearFactorGraph graph;

    std::vector<std::pair<FrameId, FrameId>> pose_frame_pairs;

    // The last object keyframe that has actually been optimized
    const std::optional<FrameId> last_okf_id_optimized =
        shared_module_states->getLatestOptimizedFrame(object_id);

    // no object states for this object have been optimized yet
    if (!last_okf_id_optimized) {
      continue;
    }

    // sanitry check it is actually a keyframe
    CHECK(map->isObjectKeyFrame(last_okf_id_optimized.value(), object_id));

    // only include data up to (and including) the last object keyframe
    const auto trajectory_j =
        full_trajectory_j.range(std::nullopt, last_okf_id_optimized);
    full_trajectories_refined.insert2(object_id, trajectory_j);

    for (const auto& entry_k : trajectory_j) {
      // TODO: as this is motion from the frontend it needs to be converted into
      // the reference frame
      //  of the backend!
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
      gtsam::Pose3 L_W_from = trajectory_j.at(from_frame).pose;
      gtsam::Pose3 L_W_to = trajectory_j.at(to_frame).pose;
      gtsam::Pose3 H_L_from_to = L_W_from.inverse() * L_W_to;
      // //     L_W_from.inverse() * f2f_motion.estimate() * L_W_from;

      // // TODO: for now!!
      gtsam::SharedNoiseModel relative_noise_model =
          gtsam::noiseModel::Isotropic::Sigma(6u, 0.4);

      // using BetweenMotion3Factor = MotionBetweenFactor<gtsam::Pose3>;
      // auto relative_object_motion = boost::make_shared<BetweenMotion3Factor>(
      //     object_pose_key_from, object_pose_key, f2f_motion.estimate(),
      //     relative_noise_model);
      // graph += relative_object_motion;

      auto relative_object_motion =
          boost::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
              object_pose_key_from, object_pose_key, H_L_from_to,
              relative_noise_model);
      graph += relative_object_motion;

      // LOG(INFO) << "Adding relative motion constraint " << from_frame << " ->
      // "
      //           << to_frame << " graph error: " << graph.error(values);

      if (isObjectKeyFrame(object_id, to_frame)) {
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

        // LOG(INFO) << "Adding refined KF relative motion constraint " <<
        // akf_id
        //           << " -> " << to_frame
        //           << " graph error: " << graph.error(values);

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
          // add a strong prior to anchor pose as this will not change!
          graph.addPrior<gtsam::Pose3>(anchor_object_pose_key,
                                       akf_pose_frontend,
                                       this->noiseModels().initial_pose_prior);

          // LOG(INFO) << "Adding pose prior at KF pose " << akf_id
          //           << " graph error: " << graph.error(values);

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

bool HybridFormulationKeyFrame::matchToStaticMap(
    Frame::Ptr frame, AbsolutePoseCorrespondences& matches,
    double* tracking_quality) const {
  HybridFormulationKeyFrameAccessor::Ptr accessor =
      this->derivedAccessor<HybridFormulationKeyFrameAccessor>();

  FrameId frame_id_k = frame->getFrameId();

  auto static_feature_itr = frame->usableStaticIterator();
  for (const auto& feature : static_feature_itr) {
    const TrackletId tracklet_id = feature->trackletId();

    if (this->staticLandmarkExists(tracklet_id)) {
      Landmark lmk_W_map = this->staticLandmarkEstimate(tracklet_id);
      const Keypoint& kp = feature->keypoint();
      matches.emplace_back(tracklet_id, lmk_W_map, kp);
    }
  }

  // if tracking quality requested
  if (tracking_quality) {
    // remember matched points
    double kptradius_ = 0.09;
    int intersectionCount = 0;
    int unionCount = 0;
    int matchedPoints = 0;

    const auto& cam_params = frame->getCamera()->getParams();

    const int rows = cam_params.ImageHeight() / 10;
    const int cols = cam_params.ImageWidth() / 10;

    const double radius = double(std::min(rows, cols)) * kptradius_;

    cv::Mat matches_img = cv::Mat::zeros(rows, cols, CV_8UC1);

    for (const auto& match : matches) {
      // keypoint from measurement
      const Keypoint& kp = match.cur_;
      const TrackletId tracklet_id = match.tracklet_id_;

      // must exist if we have a match
      CHECK(this->staticLandmarkExists(tracklet_id));

      auto lmk_node = this->map_->getLandmark(tracklet_id);
      CHECK_NOTNULL(lmk_node);

      // make sure these are observed elsewhere
      for (FrameId seen_frame_id : lmk_node->getSeenFrameIds()) {
        if (seen_frame_id != frame_id_k) {
          matchedPoints++;
          cv::circle(matches_img, utils::gtsamPointToCv(kp) * 0.1, int(radius),
                     cv::Scalar(255), cv::FILLED);
          break;
        }
      }
    }

    // one point per image does not count.
    const int pointArea = int(radius * radius * M_PI);
    intersectionCount += std::max(0, cv::countNonZero(matches_img) - pointArea);
    unionCount += rows * cols - pointArea;

    *tracking_quality = matchedPoints < 8
                            ? 0.0
                            : double(intersectionCount) / double(unionCount);
  }

  return !matches.empty();
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
    // TODO: this should actually checkk the NODE now!!
    if (isObjectKeyFrame(object_id, frame_id_k)) {
      Context context;
      context.object_node = object_node;
      context.frame_node = frame_node_k;
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

  CHECK(initial_H_W_AKF_k_.exists(object_id, frame_id_kf));

  // TODO: should check if the AKF_id is different for the from and the to?
  const auto [AKF_id, AKF_pose] =
      this->getKeyframeRange(object_id, frame_id_kf);

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

  KeyFrameMap::Ptr map = this->map();
  auto frame_node_akf = map->getFrame(frame_id_akf);
  CHECK(frame_node_akf);

  // const gtsam::Key object_motion_key_kf =
  //     frame_node_kf->makeObjectMotionKey(object_id);
  // const gtsam::Key pose_key_kf = frame_node_kf->makePoseKey();

  // CHECK(!is_other_values_in_map.exists(object_motion_key_kf));

  // new_values.insert(object_motion_key_kf, H_W_AKF_k.estimate());
  // is_other_values_in_map.insert2(object_motion_key_kf, true);
  const gtsam::Key object_motion_key_kf = addNewObjectMotionVariable(
      new_values, frame_node_kf, object_id, H_W_AKF_k);

  // frame node for last regular KF
  auto frame_node_lrkf = map->getFrame(lRKF_id);
  CHECK(frame_node_lrkf);
  // object motion key from the 'from' frame
  const gtsam::Key object_motion_key_lkf =
      frame_node_lrkf->makeObjectMotionKey(object_id);

  // motion model helps soo soo soo much ;)
  if (motionIsInValues(object_motion_key_lkf)) {
    gtsam::SharedNoiseModel relative_noise_model =
        gtsam::noiseModel::Isotropic::Sigma(6u, 0.4);
    // add relative motion constraint!
    using BetweenMotion3Factor = MotionBetweenFactor<gtsam::Pose3>;
    // TODO: this is in world so unsure how it well effect covariance!
    auto relative_object_motion = boost::make_shared<BetweenMotion3Factor>(
        object_motion_key_lkf, object_motion_key_kf, H_W_lRKF_KF,
        relative_noise_model);
    // new_factors += relative_object_motion;
  }

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
  auto seen_lmks_k = object_node->landmarksSeenAtFrame(frame_id_kf);

  size_t num_factors_not_enough_obs = 0;
  size_t num_factors_added_k = 0;
  size_t num_factors_added_lrkf = 0;
  size_t num_other_factors_added = 0;
  size_t num_missing_point_init = 0;

  for (const auto& obj_lmk_node : seen_lmks_k) {
    CHECK_EQ(obj_lmk_node->objectId(), object_id);
    const TrackletId tracklet_id = obj_lmk_node->trackletId();
    const gtsam::Key point_key = this->makeDynamicKey(tracklet_id);

    // all (object keyframes) frames with measurements of this landmark
    const FrameIds frames_with_measurements = obj_lmk_node->getSeenFrameIds();

    // TODO: seen ay any frame!
    CHECK(obj_lmk_node->seenAtFrame(frame_id_kf))
        << "Lmk i=" << tracklet_id << "Object " << object_id << " not seen at "
        << frame_id_kf << " but this is the to motion";

    if (!isDynamicTrackletInMap(obj_lmk_node)) {
      // missing a point may mean its not ready yet from the frontend!
      if (!m_L_initial_.exists(object_id, tracklet_id)) {
        num_missing_point_init++;
        continue;
      }

      // TODO: bring this back!
      if (frames_with_measurements.size() < 2) {
        num_factors_not_enough_obs++;
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

      gtsam::SharedNoiseModel lmk_prior =
          gtsam::noiseModel::Isotropic::Sigma(3u, 0.3);
      // test add small prior on landmark
      // new_factors.addPrior<gtsam::Point3>(point_key, m_L_initial, lmk_prior);

      if (result.debug_info) {
        result.debug_info->getObjectInfo(object_id).num_new_dynamic_points++;
      }
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
      addHybridMotionFactor(new_factors, point_key, object_id, AKF_pose,
                            obj_lmk_node, frame_node_lrkf);
      if (result.debug_info) {
        result.debug_info->getObjectInfo(context.getObjectId())
            .num_dynamic_factors++;
      }
      num_points_seen_akf++;
      frames_with_factors_added.insert(frame_node_lrkf->frameId());
      num_factors_added_lrkf++;
    }

    addHybridMotionFactor(new_factors, point_key, object_id, AKF_pose,
                          obj_lmk_node, frame_node_kf);
    frames_with_factors_added.insert(frame_id_kf);
    num_factors_added_k++;

    if (result.debug_info) {
      result.debug_info->getObjectInfo(context.getObjectId())
          .num_dynamic_factors++;
    }

    // sanity check/ backwards adding of points to ensure measurements
    // are added for all possible frames
    // TODO: seenAtAnyFrame check for isKeyFrame(), add factors accordinfly!
    for (const FrameId frame_with_z : frames_with_measurements) {
      // check if this frame is in the same backend range
      // if it is not, ignore it!
      const auto [AKF_id_for_z, _] =
          this->getKeyframeRange(object_id, frame_with_z);
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

        addHybridMotionFactor(new_factors, point_key, object_id, AKF_pose,
                              obj_lmk_node, frame_node_with_z);

        if (result.debug_info) {
          result.debug_info->getObjectInfo(context.getObjectId())
              .num_dynamic_factors++;
        }
        frames_with_factors_added.insert(frame_with_z);
        num_other_factors_added++;
      }
    }

    if (result.debug_info) {
      result.debug_info->getObjectInfo(context.getObjectId())
          .num_dynamic_factors++;
    }
  }

  LOG(INFO) << info_string(frame_id_kf, object_id)
            << " seen lmks= " << seen_lmks_k.size()
            << " factors added at k=" << num_factors_added_k
            << " at lrkf=" << num_factors_added_lrkf << " others "
            << num_other_factors_added
            << " not enough obs: " << num_factors_not_enough_obs
            << " missing point init " << num_missing_point_init;
}

void HybridFormulationKeyFrame::addHybridMotionFactor(
    gtsam::NonlinearFactorGraph& new_factors, gtsam::Key point_key,
    ObjectId object_id, const gtsam::Pose3& KF_pose,
    SharedLandmarkNode lmk_node, SharedFrameNode frame_node) {
  auto stereo_measurement =
      MeasurementTraits::stereo(lmk_node->getMeasurement(frame_node));
  // FOR NOW
  CHECK(stereo_measurement);
  auto [z, z_model] = *stereo_measurement;

  if (frame_node->isCameraKeyFrame()) {
    addHybridMotionFactorCameraKF(new_factors, point_key, object_id, KF_pose, z,
                                  z_model, frame_node);
  } else {
    addHybridMotionFactorNonCameraKF(new_factors, point_key, object_id, KF_pose,
                                     z, z_model, frame_node);
  }
}

void HybridFormulationKeyFrame::addHybridMotionFactorCameraKF(
    gtsam::NonlinearFactorGraph& new_factors, gtsam::Key point_key,
    ObjectId object_id, const gtsam::Pose3& KF_pose,
    const gtsam::StereoPoint2& z, const gtsam::SharedNoiseModel& z_model,
    SharedFrameNode frame_node_CKF) {
  gtsam::SharedNoiseModel noise_model = z_model;
  if (params_.makeDynamicMeasurementsRobust()) {
    noise_model = factor_graph_tools::robustifyHuber(params_.k_huber_3d_points_,
                                                     noise_model);
  }
  CHECK_NOTNULL(noise_model);

  const gtsam::Key motion_key = frame_node_CKF->makeObjectMotionKey(object_id);
  const gtsam::Key camera_pose_key = frame_node_CKF->makePoseKey();

  new_factors.emplace_shared<StereoHybridMotionFactor>(
      z, KF_pose, noise_model, rgbd_camera_->getFakeStereoCalib(),
      camera_pose_key, motion_key, point_key, false /* throw cheirality*/
  );
}

void HybridFormulationKeyFrame::addHybridMotionFactorNonCameraKF(
    gtsam::NonlinearFactorGraph& new_factors, gtsam::Key point_key,
    ObjectId object_id, const gtsam::Pose3& KF_pose,
    const gtsam::StereoPoint2& z, const gtsam::SharedNoiseModel& z_model,
    SharedFrameNode frame_node_nonCKF) {
  CHECK(!frame_node_nonCKF->isCameraKeyFrame());
  const FrameId frame_id_k = frame_node_nonCKF->frameId();

  //! Closest camera keyframe before frame_id_k
  auto frame_node_closest_CKF = map_->closestEarlierCameraKeyFrame(frame_id_k);
  CHECK_NOTNULL(frame_node_closest_CKF);
  CHECK_LT(frame_node_closest_CKF->frameId(), frame_id_k);
  CHECK(frame_node_closest_CKF->isCameraKeyFrame());
  // check that a VO transform from CKF to k exists
  CHECK(frame_node_closest_CKF->hasRelativeEgoMotion(frame_id_k));
  const gtsam::Pose3 T_CKF_k =
      frame_node_closest_CKF->getRelativeEgoMotion(frame_id_k);

  gtsam::SharedNoiseModel noise_model = z_model;
  gtsam::SharedNoiseModel extrapolated_noise_model =
      factor_graph_tools::inflateNoise(noise_model, 1.0);
  if (params_.makeDynamicMeasurementsRobust()) {
    extrapolated_noise_model = factor_graph_tools::robustifyHuber(
        params_.k_huber_3d_points_, extrapolated_noise_model);
  }
  CHECK_NOTNULL(noise_model);

  // LOG(INFO) << "Making StereoHybridMotionExtrapolatedFactor factor "
  //           << " with KF camera at " << frame_node_closest_CKF->frameId()
  //           << " and motion at " << frame_id_k << "for j=" << object_id;

  const gtsam::Key motion_key =
      frame_node_nonCKF->makeObjectMotionKey(object_id);
  // here we use a different pose key, associated with closes_CKF
  const gtsam::Key camera_pose_key = frame_node_closest_CKF->makePoseKey();
  new_factors.emplace_shared<StereoHybridMotionExtrapolatedFactor>(
      z, KF_pose, T_CKF_k, extrapolated_noise_model,
      rgbd_camera_->getFakeStereoCalib(), camera_pose_key, motion_key,
      point_key, false /* throw cheirality*/
  );
}

gtsam::Key HybridFormulationKeyFrame::addNewObjectMotionVariable(
    gtsam::Values& new_values, SharedFrameNode frame_node, ObjectId object_id,
    const Motion3ReferenceFrame& motion) {
  const gtsam::Key motion_key = frame_node->makeObjectMotionKey(object_id);

  CHECK(!is_other_values_in_map.exists(motion_key));
  CHECK_EQ(motion.to(), frame_node->frameId());

  new_values.insert(motion_key, motion.estimate());
  is_other_values_in_map.insert2(motion_key, true);

  return motion_key;
}

bool HybridFormulationKeyFrame::motionIsInValues(ObjectId object_id,
                                                 FrameId frame_id) const {
  auto frame_node = this->map_->getFrame(frame_id);
  if (!frame_node) {
    return false;
  }

  const gtsam::Key motion_key = frame_node->makeObjectMotionKey(object_id);
  return motionIsInValues(motion_key);
}

bool HybridFormulationKeyFrame::motionIsInValues(const gtsam::Key key) const {
  // TODO: assert key is for a motion!
  return is_other_values_in_map.exists(key);
}

// TODO: this should mark objects with keyframes!
void HybridFormulationKeyFrame::addObjects(
    FrameId /*frame_id*/, const ObjectPoseChangeInfoMap& object_motion_info) {
  auto accessor = this->derivedAccessor<HybridFormulationKeyFrameAccessor>();
  CHECK_NOTNULL(accessor);

  // const auto maybe_latest_camera_frame =
  // this->map_->getSharedModuleStates()->getLatestOptimizedFrame();
  // CHECK(maybe_latest_camera_frame);

  for (const auto& [object_id, object_info] : object_motion_info) {
    CHECK(object_info.isKeyFrame());
    // estimated keyframe motioan from the frontend
    // in this case k is the current but will now also be the latest KF
    // const Motion3ReferenceFrame& H_W_RKF_k_frontend = object_info.H_W_KF_k;
    Motion3ReferenceFrame H_W_RKF_k = object_info.H_W_KF_k;
    const ObjectKeyFrameStatus keyframe_status = object_info.keyframe_status;

    // TODO: for now (only when solved in parallel_run=False)
    //  convert motion from frontend reference frame to backend reference frame
    FrameId from_frame_id = H_W_RKF_k.from();
    FrameId to_frame_id = H_W_RKF_k.to();
    // dont like the naming of this function
    // get the camera pose either directly from the state or approximated via
    // the VIO
    // CHECK_GE(maybe_latest_camera_frame.value(), from_frame_id);
    auto [X_W_KFm1_opt, _] = this->getBestCameraPose(from_frame_id);
    const gtsam::Pose3 X_W_KFm1_frontend = object_info.X_W_KF;

    // do weird change of basis to put the motion in the optimized camera pose
    // reference frame
    gtsam::Pose3 H_W_KF_k_in_opt = X_W_KFm1_opt * X_W_KFm1_frontend.inverse() *
                                   H_W_RKF_k.estimate() * X_W_KFm1_frontend *
                                   X_W_KFm1_opt.inverse();
    H_W_RKF_k.estimate_ = H_W_KF_k_in_opt;

    // and apply to object
    const gtsam::Pose3 L_Wfrontend_KF = object_info.L_W_KF;

    gtsam::Pose3 L_Xfrontend_KF = X_W_KFm1_frontend.inverse() * L_Wfrontend_KF;
    gtsam::Pose3 L_Wbackend_KF = X_W_KFm1_opt * L_Xfrontend_KF;

    KeyFrameMetaData kf_data;
    kf_data.keyframe_status = keyframe_status;
    kf_data.H_W_lRKF_KF = H_W_RKF_k;

    // will not be true when we have a lost object
    // CHECK_EQ(frame_id, H_W_RKF_k.to());

    key_frames_per_object_.insert22(object_id, to_frame_id, kf_data);

    LOG(INFO) << "Processing object track: "
              << info_string(to_frame_id, object_id)
              << " keyframe status: " << keyframe_status;

    // TODO: we initalie the new KF with the pose provided from the frontend
    // for consistency. This is fine when the object observations are
    // continuous
    // but at some point the KF pose in the frontend and back-end will change
    // (i.e after opt!)

    // ad new initialisation points to backend
    // misleading print as we dont add this many points! Only new ones
    size_t num_new_lmks = 0;
    for (const auto& landmark_status : object_info.initial_object_points) {
      const TrackletId& tracklet_id = landmark_status.trackletId();
      const gtsam::Point3& m_L = landmark_status.value();
      // only add new ones?
      if (!m_L_initial_.exists(object_id, tracklet_id)) {
        gtsam::Point3 m_Lbackend =
            L_Wbackend_KF.inverse() * L_Wfrontend_KF * m_L;
        m_L_initial_.insert22(object_id, tracklet_id, m_L);
        num_new_lmks++;
      }
    }

    LOG(INFO) << "Adding initial object points of size "
              << object_info.initial_object_points.size()
              << " new lmks=" << num_new_lmks;

    if (keyframe_status == ObjectKeyFrameStatus::AnchorKeyFrame) {
      key_frame_data_.startNewActiveRange(object_id, H_W_RKF_k.from(),
                                          L_Wbackend_KF);
      LOG(INFO) << "Making Anchor KF for NEW object "
                << info_string(H_W_RKF_k.from(), object_id) << " with motion "
                << H_W_RKF_k.from() << " -> " << H_W_RKF_k.to();

      // the frontend range is always "to" because it indicates the start
      // of the next range and a single motion represents one
      front_end_keyframes_.startNewActiveRange(object_id, H_W_RKF_k.from(),
                                               L_Wbackend_KF);
      LOG(INFO) << "Making Regular KF for NEW object "
                << info_string(H_W_RKF_k.from(), object_id) << " with motion "
                << H_W_RKF_k.from() << " -> " << H_W_RKF_k.to();
      initial_H_W_AKF_k_.insert22(object_id, H_W_RKF_k.to(), H_W_RKF_k);
    } else {
      CHECK_EQ(keyframe_status, ObjectKeyFrameStatus::RegularKeyFrame);

      const KeyFrameRange::ConstPtr last_frontend_range =
          front_end_keyframes_.find(object_id, to_frame_id);
      CHECK(last_frontend_range) << "Failed for tracked object "
                                 << info_string(to_frame_id, object_id);
      auto [lRKF_id, L_lRKF] = last_frontend_range->dataPair();

      // hopefully the last regular KF is the from frame
      lRKF_id = H_W_RKF_k.from();
      // LOG(INFO) << "Last regular KF " << lRKF_id;

      // TODO: if this is regular KF then the position of this KF will change
      // according to the motion that is refined
      //  as L_W_k = L_W_KF = H_W_AKF_KF * L_AKF
      front_end_keyframes_.startNewActiveRange(object_id, H_W_RKF_k.to(),
                                               object_info.L_W_k);
      VLOG(40) << "Making Regular KF for tracked object "
               << info_string(H_W_RKF_k.to(), object_id) << " with motion "
               << H_W_RKF_k.from() << " -> " << H_W_RKF_k.to();

      const KeyFrameRange::ConstPtr frontend_range =
          front_end_keyframes_.find(object_id, to_frame_id);
      CHECK(frontend_range);
      // the most recent motion added to the estimator should take us from
      // backend_kf_id to last_kf_id
      const auto [current_kf_id, current_kf_pose] = frontend_range->dataPair();
      VLOG(40) << "Current regular KF " << current_kf_id;

      // get backend anchor point and confert motion if necessary
      const KeyFrameRange::ConstPtr backend_range =
          CHECK_NOTNULL(key_frame_data_.find(object_id, to_frame_id));
      const auto [backend_kf_id, backend_kf_pose] = backend_range->dataPair();
      VLOG(40) << "Anchor KF id: " << backend_kf_id;

      VLOG(40) << "Provided object odometry " << H_W_RKF_k.from() << " -> "
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
        VLOG(40) << "Looking up estimated motion from " << backend_kf_id
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

std::pair<gtsam::Pose3, HybridFormulationKeyFrame::CameraPoseExtraction>
HybridFormulationKeyFrame::getBestCameraPose(FrameId frame_id) const {
  auto accessor = this->derivedAccessor<HybridFormulationKeyFrameAccessor>();

  if (map_->isCameraKeyFrame(frame_id)) {
    gtsam::Pose3 X_W_k =
        DYNO_GET_QUERY_DEBUG(accessor->getSensorPose(frame_id));
    return {X_W_k, CameraPoseExtraction::Keyframe};
  } else {
    auto frame_node_closest_CKF = map_->closestEarlierCameraKeyFrame(frame_id);
    CHECK_NOTNULL(frame_node_closest_CKF);
    CHECK(frame_node_closest_CKF->isCameraKeyFrame());
    // check that a VO transform from CKF to k exists
    CHECK(frame_node_closest_CKF->hasRelativeEgoMotion(frame_id));
    const gtsam::Pose3 T_CKF_k =
        frame_node_closest_CKF->getRelativeEgoMotion(frame_id);
    // get optimized CKF pose
    gtsam::Pose3 X_W_CKF = DYNO_GET_QUERY_DEBUG(
        accessor->getSensorPose(frame_node_closest_CKF->frameId()));
    gtsam::Pose3 X_W_k = X_W_CKF * T_CKF_k;
    return {X_W_k, CameraPoseExtraction::Interpolated};
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
