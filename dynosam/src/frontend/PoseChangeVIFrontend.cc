#include "dynosam/frontend/PoseChangeVIFrontend.hpp"

#include <gflags/gflags.h>

DEFINE_bool(pc_smoother_allow_backend_updates, false,
            "If updates from the backend should be received.");

DEFINE_bool(pc_log_object_kf_structure, false,
            "If the object point cloud should be logged at keyframes");

DEFINE_bool(pc_send_objects_to_backend, true,
            "If true, objects will be included in the backend optimisation");

namespace dyno {

PoseChangeVIFrontend::PoseChangeVIFrontend(
    const DynoParams& params, Camera::Ptr camera,
    HybridFormulationKeyFrame::Ptr formulation,
    ImageDisplayQueue* display_queue,
    const SharedGroundTruth& shared_ground_truth)
    : VIFrontend("pc-frontend", params, camera, display_queue,
                 shared_ground_truth),
      formulation_(CHECK_NOTNULL(formulation)),
      accessor_(CHECK_NOTNULL(
          formulation->derivedAccessor<HybridFormulationKeyFrameAccessor>())),
      map_(CHECK_NOTNULL(formulation->map())),
      tracking_viz_(params.frontend_params_.image_tracks_vis_params,
                    params.enforceRealtime()) {
  // TODo
  HybridObjectMotionSolverParams motion_params;
  motion_params.optical_flow_solver_params.use_robust = true;

  SharedGroundTruth ground_truth;
  if (FLAGS_init_object_pose_from_gt) {
    LOG(INFO) << "FLAGS_init_object_pose_from_gt is true. Object motion solver "
                 "will attempt to initalise object poses using provided ground "
                 "truth pose!";
    ground_truth = shared_ground_truth_;
  }

  object_motion_solver_ = std::make_unique<HybridObjectMotionSolver>(
      motion_params, camera_->getParams(), DepthUpdater(&tracker_),
      ground_truth);
  object_motion_solver_->enforceRealtime(params.enforceRealtime());
}

PoseChangeVIFrontend::~PoseChangeVIFrontend() { logBestEstimates(); }

void PoseChangeVIFrontend::onBackendUpdateComplete(
    const PoseChangeUpdateComplete& event) {
  // TODO: this is definitely not thread safe
  const FrameId frame_id = event.ending_frame_id;
  const FrameId starting_frame_id = event.starting_frame_id;

  LOG(INFO) << "Recieved backend update for frames " << starting_frame_id
            << " -> " << frame_id;

  has_backend_update_.store(true);

  if (FLAGS_pc_smoother_allow_backend_updates) {
    LOG(INFO) << "Recieved backend update at frame " << frame_id;
    auto event_copy = event;
    // hack and slow for now!
    // event_copy.camera.trajectory = this->refinePerFrameCameraPGO();
    object_motion_solver_->receiveUpdate(event_copy);
  }

  if (FLAGS_pc_log_object_kf_structure) {
    // TODO: now use shared module states!
    LOG(INFO) << "Logging estimated object structures...";
    // only log for object keyframes that were part of the latest batch
    // this prevents logging objects that were NOT keyframes and
    // only includes objects that have actually been optimized
    std::unordered_map<ObjectId, FrameId> latest_okf_per_object;
    std::unordered_set<ObjectId> seen;

    const auto& keyframe_infos = event.keyframe_infos;
    // iterate from largest FrameId → smallest
    for (auto it = keyframe_infos.rbegin(); it != keyframe_infos.rend(); ++it) {
      const FrameId frame_id = it->first;
      const KeyframeInfo& kf_info = it->second;

      for (const auto& motion : kf_info.object_keyframes) {
        const ObjectId obj_id = motion.object_id;

        // first time we see it = latest occurrence
        if (seen.insert(obj_id).second) {
          latest_okf_per_object[obj_id] = frame_id;
        }
      }
    }

    for (const auto& [object_id, okf_id] : latest_okf_per_object) {
      StatusLandmarkVector points_in_L =
          accessor_->getLocalDynamicLandmarkEstimates(object_id);

      if (points_in_L.empty()) {
        VLOG(20) << "No points for j=" << object_id << ": skipping logging!";
        continue;
      }

      std::string path = dyno::getOutputFilePath(
          "refined_object_map_k" + std::to_string(okf_id) + "_j" +
          std::to_string(object_id) + ".pcd");
      VLOG(10) << "Writing object map of size " << points_in_L.size() << " - "
               << path;
      saveAsPointCloud(points_in_L, path);
    }
  }
}

PoseChangeVIFrontend::SpinReturn PoseChangeVIFrontend::boostrapSpin(
    VIFrontendInput::ConstPtr input) {
  Frame::Ptr frame_k = featureTrack(input);
  const auto frame_id_k = input->getFrameId();
  const auto timestamp_k = input->getTimestamp();
  ImageContainer::Ptr image_container = input->image_container_;

  // TODO: must set depth either by update depth or by stereo match.
  //  currently updateDepths is in feature track but this function is not!
  stereoMatch(frame_k);

  gtsam::Pose3 identity_pose = gtsam::Pose3::Identity();
  gtsam::Vector3 zero_velocity(0.0, 0.0, 0.0);
  gtsam::NavState initial_state(identity_pose, zero_velocity);

  dyno_state_.camera_trajectory.insert(frame_id_k, timestamp_k, identity_pose);

  lCKF_frame_ = frame_k;

  // no motion as first frame!
  const gtsam::Pose3 I = gtsam::Pose3::Identity();

  RealtimeOutput::Ptr realtime_output = std::make_shared<RealtimeOutput>();
  realtime_output->state.frame_id = frame_id_k;
  realtime_output->state.timestamp = timestamp_k;
  realtime_output->state.camera_trajectory = dyno_state_.camera_trajectory;
  realtime_output->ground_truth = input->ground_truth_packet;

  RelEgoPoseInfo rel_egopose;
  rel_egopose.lkf_id = lCKF_frame_->getFrameId();
  rel_egopose.i_id = frame_id_k;
  rel_egopose.j_id = frame_id_k;
  rel_egopose.frame_j = frame_k;
  rel_egopose.frontend_nav_state_j = initial_state;
  rel_egopose.T_i_j = identity_pose;
  rel_egopose.T_lkf_j = identity_pose;
  rel_egopose.pim_lk_j = nullptr;
  rel_egopose.imu_measurements =
      input->imu_measurements.value_or(ImuMeasurements{});
  rel_egopose_infos_.insert2(frame_id_k, rel_egopose);

  const TemporalNavState nav_state_k{frame_id_k, timestamp_k, initial_state};
  nav_state_km1_ = nav_state_k;
  nav_state_lkf_ = nav_state_k;

  CameraMeasurementStatusVector static_measurements;
  // for the first frame the global map aligns with the local map
  StatusLandmarkVector* local_landmarks = &realtime_output->state.static_map;
  fillMeasurementsFromFeatureIterator(
      static_measurements, frame_k->usableStaticIterator(), frame_id_k,
      timestamp_k, static_pixel_sigmas_, static_point_sigma_, local_landmarks);

  // first frame is always KF
  map_->updateObservations(static_measurements);
  map_->setInitialSensorPose(frame_id_k, timestamp_k,
                             Pose3Measurement(identity_pose));
  map_->setCameraKeyFrame(frame_id_k);

  KeyframeInfo keyframe_info;
  keyframe_info.camera_keyframe = true;

  auto pc_input = std::make_shared<SinglePoseChangeInput>();
  pc_input->frame_id = frame_id_k;
  pc_input->timestamp = timestamp_k;
  pc_input->keyframe_info = keyframe_info;

  keyframe_infos_.insert2(frame_id_k, keyframe_info);

  auto& new_static_values = pc_input->new_static_fg_input.values;
  auto& new_static_factors = pc_input->new_static_fg_input.factors;

  formulation_->addStatesInitalise(new_static_values, new_static_factors,
                                   frame_id_k, timestamp_k, identity_pose,
                                   zero_velocity);

  UpdateObservationParams update_params;
  update_params.enable_debug_info = true;
  update_params.do_backtrack = false;

  PostUpdateData post_update_data(frame_id_k);

  post_update_data.static_update_result =
      formulation_->updateStaticObservations(frame_id_k, new_static_values,
                                             new_static_factors, update_params);

  logRealTimeOutput(realtime_output);

  SharedModuleStates* shared_module_states = map_->getSharedModuleStates();
  shared_module_states->current_frontend_frame = frame_id_k;

  if (withBackend()) {
    pose_change_backend_sink_(pc_input);
  }

  return {State::Nominal, realtime_output};
}

PoseChangeVIFrontend::SpinReturn PoseChangeVIFrontend::nominalSpin(
    VIFrontendInput::ConstPtr input) {
  utils::ChronoTimingStats timer(this->moduleName() + ".spin");
  const auto t1 = utils::Timer::tic();

  ImageContainer::Ptr image_container = input->image_container_;
  const auto frame_id_k = input->getFrameId();
  const auto timestamp_k = input->getTimestamp();

  // check if we have a camera pose update for the last CKF
  // what if we get an update during the consuming!!!
  checkAndConsumeUpdate(frame_id_k);

  ImuFrontend::PimPtr pim = nullptr;
  std::optional<gtsam::NavState> imu_propogated_nav_state_k =
      tryPropogateImu(input, nav_state_lkf_.state, pim);

  // if(pim) {
  //   std::stringstream ss;
  //   ss << std::setprecision(30) << " last kf time " <<
  //   nav_state_lkf_.timestamp << "\n"; ss << "last frame t=: " <<
  //   nav_state_km1_.timestamp << "\n"; auto imu_measurements =
  //   input->imu_measurements.value(); auto timestamps =
  //   imu_measurements.timestamps_; size_t num_samples = timestamps.cols(); for
  //   (size_t idx = 0u; idx < num_samples; ++idx) {
  //     ss << "IMU (" << idx << ") t=" << timestamps(idx) << "\n";
  //   }
  //   ss << "Current t=" << timestamp_k;
  //   LOG(INFO) << ss.str();
  // }

  //! Rotation from k-1 to k in k-1
  std::optional<gtsam::Rot3> R_km1_k;
  if (imu_propogated_nav_state_k) {
    CHECK(pim);
    R_km1_k = nav_state_km1_.state.attitude().inverse() *
              imu_propogated_nav_state_k->attitude();
  }

  Frame::Ptr frame_k = featureTrack(input, R_km1_k);
  Frame::Ptr frame_km1 = tracker_.getPreviousFrame();
  CHECK(frame_km1);

  VLOG(1) << to_string(tracker_.getTrackerInfo());

  RealtimeOutput::Ptr realtime_output = std::make_shared<RealtimeOutput>();
  realtime_output->state.frame_id = frame_id_k;
  realtime_output->state.timestamp = timestamp_k;

  // when providing the propogated imu state only provide if it was
  // actually filled by a prediction from the IMU - otherwise it will ne
  // nullopt. This tells the function to use a constant motion model from the
  // previous frame ie. T_km1_k_ if tracking fails
  StatusLandmarkVector& static_landmarks_used_vo =
      realtime_output->state.static_map;
  TrackingQuality camera_tracking_quality;
  // visual odometry which is explicitly solved for
  gtsam::Pose3 T_ij;
  const bool ego_motion_solve = solveAndRefineEgoMotion(
      frame_k, frame_km1, static_landmarks_used_vo, camera_tracking_quality,
      T_ij, imu_propogated_nav_state_k, R_km1_k);

  // if(input->ground_truth_packet) {
  //   frame_k->T_world_camera_ = input->ground_truth_packet->X_world_;
  // }

  // we currently use the frame pose as the nav state - this value can come from
  // either the VO OR the IMU, depending on the result from the
  // solveCameraMotion this is only relevant since we dont solve incremental so
  // the backend is not immediately updating the frontend at which point we can
  // just use the best estimate in the case of the VO, the nav_state velocity
  const TemporalNavState nav_state_k{
      frame_id_k, timestamp_k,
      gtsam::NavState(frame_k->getPose(),
                      imu_propogated_nav_state_k
                          ? imu_propogated_nav_state_k->velocity()
                          : gtsam::Vector3(0, 0, 0))};

  RelEgoPoseInfo rel_egopose;
  rel_egopose.lkf_id = lCKF_frame_->getFrameId();
  rel_egopose.i_id = frame_km1->getFrameId();
  rel_egopose.j_id = frame_id_k;
  rel_egopose.frame_j = frame_k;
  rel_egopose.frontend_nav_state_j = nav_state_k.state;
  // rel_egopose.T_i_j =
  //     nav_state_km1_.state.pose().inverse() * nav_state_k.state.pose();
  // rel_egopose.T_lkf_j =
  //     nav_state_lkf_.state.pose().inverse() * nav_state_k.state.pose();
  // rel_egopose.T_i_j = getVOTransform(nav_state_km1_.frame_id, nav_state_k);
  // rel_egopose.T_lkf_j = getVOTransform(nav_state_lkf_.frame_id, nav_state_k);

  CHECK(rel_egopose_infos_.exists(nav_state_km1_.frame_id));

  rel_egopose.T_i_j = T_ij;
  T_lkf_j_ = T_lkf_j_ * T_ij;
  // TODO: should use propogator!!
  rel_egopose.T_lkf_j = T_lkf_j_;
  rel_egopose.pim_lk_j = (pim) ? ImuFrontend::copyPim(pim) : nullptr;
  rel_egopose.imu_measurements =
      input->imu_measurements.value_or(ImuMeasurements{});
  // very important to store this
  rel_egopose_infos_.insert2(frame_id_k, rel_egopose);
  nav_state_km1_ = nav_state_k;

  dyno_state_.camera_trajectory.insert(frame_id_k, timestamp_k,
                                       nav_state_k.state.pose());

  ObjectIds objects_with_new_motions;
  ObjectPoseChangeInfoMap kf_pose_change_infos;
  ObjectTrackingStatusMap object_tracking_status;
  solveObjectMotions(dyno_state_.object_trajectories, objects_with_new_motions,
                     object_tracking_status, kf_pose_change_infos, frame_k,
                     frame_km1);

  // TODO: slow and rematching all points again!
  //  need to rematch after solving flow with objects
  stereoMatch(frame_k);

  // update full_object_trajectories_ with trajectories for objects observed at
  // this frame
  for (ObjectId j : objects_with_new_motions) {
    full_object_trajectories_[j] = dyno_state_.object_trajectories.at(j);
  }

  realtime_output->state.camera_trajectory = dyno_state_.camera_trajectory;
  realtime_output->state.object_trajectories = dyno_state_.object_trajectories;
  realtime_output->ground_truth = input->ground_truth_packet;

  CameraMeasurementStatusVector static_measurements;
  fillMeasurementsFromFeatureIterator(
      static_measurements, frame_k->usableStaticIterator(), frame_id_k,
      timestamp_k, static_pixel_sigmas_, static_point_sigma_);

  // CameraMeasurementsByObject dynamic_measurements_per_object;
  // fillMeasurementsFromFeatureIterator(
  //     dynamic_measurements_per_object, frame_k->usableDynamicIterator(),
  //     frame_id_k, timestamp_k, dynamic_pixel_sigmas_, dynamic_point_sigma_
  //     /*&realtime_output->state.dynamic_map*/);

  // fill output dynamic map with current structure
  // only display the currently viewed objects and their last segment
  MultiObjectTrajectories trajectories_to_visualise;
  for (const auto& object_id : objects_with_new_motions) {
    // assume that getObjectStructureinW does not clear the vector
    object_motion_solver_->getObjectStructureinW(
        object_id, realtime_output->state.dynamic_map);

    const auto trajectory_j = dyno_state_.object_trajectories.at(object_id);
    const auto last_segment_j = trajectory_j.segments().back();

    if (last_segment_j.trajectory.size() > 2) {
      trajectories_to_visualise[object_id] = last_segment_j.trajectory;
    }
  }

  const size_t num_object_keyframes = kf_pose_change_infos.size();

  ObjectIds objects_with_keyframes;
  objects_with_keyframes.reserve(num_object_keyframes);
  for (const auto& [object_id, _] : kf_pose_change_infos) {
    objects_with_keyframes.push_back(object_id);
  }

  if (FLAGS_pc_log_object_kf_structure) {
    logRealTimeObjectClouds(objects_with_keyframes, frame_id_k);
  }

  const bool ego_motion_keyframe = shouldFrameBeKeyFrame(frame_k, frame_km1);
  const bool any_object_keyframes =
      num_object_keyframes > 0 && FLAGS_pc_send_objects_to_backend;
  const bool is_any_keyframe = ego_motion_keyframe || any_object_keyframes;

  if (is_any_keyframe) {
    std::stringstream ss;
    if (ego_motion_keyframe) ss << "CKF";
    if (any_object_keyframes)
      ss << " OKF: " << container_to_string(objects_with_keyframes);

    LOG(INFO) << "Keyframe info k=" << frame_id_k << " " << ss.str();
  }

  // TODO: may not be used if is_any_keyframe is false
  auto pc_input = std::make_shared<SinglePoseChangeInput>();
  pc_input->frame_id = frame_id_k;
  pc_input->timestamp = timestamp_k;

  auto& new_dynamic_values = pc_input->new_dynamic_fg_input.values;
  auto& new_dynamic_factors = pc_input->new_dynamic_fg_input.factors;

  UpdateObservationParams update_params;
  update_params.do_backtrack = false;
  if (VLOG_IS_ON(10)) update_params.enable_debug_info = true;

  PostUpdateData post_update_data(frame_id_k);
  // add static measurements if this is a camera keyframe or any object motion
  // as potentially was observed if no objects are observed and not a camera
  // keyframe no need to add anything to the map
  // TODO: and if frame_km1 has objects!
  const bool any_objects_observed =
      frame_k->getObjectObservations().size() > 0u;
  if (ego_motion_keyframe || any_objects_observed) {
    map_->updateObservations(static_measurements);
  }

  // lets test make all OKF's are also camera keyframes
  // THIS IS IMPORTANT!!
  if (ego_motion_keyframe || any_object_keyframes) {
    // call also updates the keyframe info for the pc_input
    handleCameraKeyframe(rel_egopose, update_params, post_update_data,
                         pc_input);
  } else {
    // if not a camera keyframe, update KF node with relative ego motion data
    auto frame_lkf_node =
        CHECK_NOTNULL(map_->getFrame(lCKF_frame_->getFrameId()));
    // TODO: this relative motion should be consistent with the frontend
    // TODO: use IMU if available!
    frame_lkf_node->addRelativeEgoMotion(rel_egopose.T_lkf_j, frame_id_k);
  }

  if (any_object_keyframes) {
    for (const auto& [object_id, info] : kf_pose_change_infos) {
      CHECK(info.isKeyFrame());
      const auto& H_W_KF_k = info.H_W_KF_k;
      const auto frame_id_motion_from = H_W_KF_k.from();

      // we should have two cases
      // 1. Object is well tracked and therefore has a keyframe at this frame
      // 2. Is lost and therefore has an estimate in the last frame
      if (info.tracking_status == ObjectTrackingStatus::WellTracked) {
        CHECK_EQ(H_W_KF_k.to(), frame_id_k);
      } else if (info.tracking_status == ObjectTrackingStatus::Lost) {
        CHECK_EQ(H_W_KF_k.to(), frame_km1->getFrameId());
        // in this case measurements of the object will not be included in
        // dynamic_measurements_kf_k so we need to additionally add them
      } else {
        throw DynosamException(
            "To have an object keyframe status must be either WellTracked or "
            "Lost!");
      }

      // attempt measurement update at both to and from frames
      addMeasurementsForObjectKeyframe(H_W_KF_k.to(), object_id);
      addMeasurementsForObjectKeyframe(H_W_KF_k.from(), object_id);

      CHECK(map_->isObjectKeyFrame(H_W_KF_k.to(), object_id));
      CHECK(map_->isObjectKeyFrame(H_W_KF_k.from(), object_id));

      // record keyframe info for each object
      KeyframeInfo::MotionPair object_kf_info{object_id, H_W_KF_k.from(),
                                              H_W_KF_k.to()};
      pc_input->keyframe_info.object_keyframes.push_back(object_kf_info);
    }
    pc_input->kf_pose_change_infos = kf_pose_change_infos;
    // TODO: testing delayed construction of dynamic motion factors
    //  add objects to backend with initial motion estimates
    //  formulation_->addObjects(frame_id_k, kf_pose_change_infos);

    // // generate new factors for dynamic objects based on latest measurements
    // // and object keyframe states
    // post_update_data.dynamic_update_result =
    //     formulation_->updateDynamicObservations(
    //         frame_id_k, new_dynamic_values, new_dynamic_factors,
    //         update_params);
  }

  // NOTE: we check num_object_keyframes not any_object_keyframes, as this bool
  // is also conditioned on FLAGS_pc_send_objects_to_backend
  if (ego_motion_keyframe || num_object_keyframes > 0) {
    keyframe_infos_.insert2(frame_id_k, pc_input->keyframe_info);
  }
  realtime_output->state.keyframe_infos = keyframe_infos_;

  SharedModuleStates* shared_module_states = map_->getSharedModuleStates();
  shared_module_states->current_frontend_frame = frame_id_k;

  // any keyframe triggered
  if (withBackend() && is_any_keyframe) {
    pose_change_backend_sink_(pc_input);
  }

  const auto t2 = utils::Timer::toc(t1);
  const auto compute_time = utils::Timer::toSeconds(t2);

  // fillDebugImagery(realtime_output->debug_imagery, frame_k, frame_km1);
  // set only the debug tracking imagery to avoid also calling the (somewhat
  // depricated) computeTracks function from the tracker
  ViTrackingViz::Data viz_data;
  viz_data.camera_tracking_quality = camera_tracking_quality;
  viz_data.keyframe_info = pc_input->keyframe_info;
  viz_data.object_tracking_statuses = std::move(object_tracking_status);
  viz_data.time_delta = compute_time;
  realtime_output->debug_imagery.tracking_image =
      tracking_viz_.vizTracking(*frame_km1, *frame_k, viz_data);

  pushImageToDisplayQueue("Tracks",
                          realtime_output->debug_imagery.tracking_image);

  cv::Mat okf_debug_metrics = object_motion_solver_->keyframeDebugImage();
  if (!okf_debug_metrics.empty())
    pushImageToDisplayQueue("OKF Keyframe Metrics", okf_debug_metrics);

  // if (stereo_matching_result) {
  //   cv::Mat stereo_track;
  //   tracker_->drawStereoMatches(stereo_track, *frame_k);
  //   pushImageToDisplayQueue("Stereo-Matches", stereo_track);
  // }

  logRealTimeOutput(realtime_output);

  // after logging update the multi object trajectories to visualise
  realtime_output->state.object_trajectories = trajectories_to_visualise;
  return {State::Nominal, realtime_output};
}

bool PoseChangeVIFrontend::solveAndRefineEgoMotion(
    Frame::Ptr frame_k, const Frame::Ptr& frame_km1,
    StatusLandmarkVector& points_W_used, TrackingQuality& tracking_quality,
    gtsam::Pose3& T_ij, std::optional<gtsam::NavState> propogated_nav_state_k,
    std::optional<gtsam::Rot3> R_km1_k) {
  utils::ChronoTimingStats timer(this->moduleName() + ".camera_motion");
  // get matches points in the local frame of k-1
  LandmarkKeypointCorrespondences m_matches;
  double tracking_quality_cost;
  bool success = formulation_->matchToStaticMap(
      frame_k, m_matches, frame_km1->getPose(), &tracking_quality_cost);

  bool use_map = false;
  if (success) {
    // double median_repr = calculateMedian(repr_errors);
    int num_matches = m_matches.size();
    double min_matches = 40;
    // double repr_thresh = 2.0;
    // from OKVIS < 0.3 is marginal and < 0.01 is LOST
    double coverage_thresh = 0.3;

    use_map = (num_matches > min_matches) &&
              (tracking_quality_cost > coverage_thresh);
  }

  Pose3SolverResult pnp_result;
  LandmarkKeypointCorrespondences correspondences_used;
  if (use_map) {
    VLOG(5) << "Tracking against Map: k=" << frame_k->getFrameId()
            << ": matches=" << m_matches.size()
            << " tracking quality=" << tracking_quality_cost;
    // solve PnP
    pnp_result = pnp_ransac_.solve3d2d(m_matches, R_km1_k);
    correspondences_used = std::move(m_matches);
  } else {
    VLOG(5) << "Tracking aginast Previous frame";
    LandmarkKeypointCorrespondences correspondences;
    frame_k->getCorrespondences(correspondences, *frame_km1,
                                KeyPointType::STATIC,
                                frame_k->landmarkLocalKeypointCorrespondance());

    // solve PnP
    pnp_result = pnp_ransac_.solve3d2d(correspondences, R_km1_k);
    correspondences_used = std::move(correspondences);
    // sanity check
    const TrackletIds tracklets = frame_k->static_features_.collectTracklets();
    // tracklets shoudl be more (or same as) correspondances as there will be
    // new points untracked
    CHECK_GE(tracklets.size(),
             pnp_result.inliers.size() + pnp_result.outliers.size());
  }

  const FrameId frame_id_k = frame_k->getFrameId();
  const Timestamp timestamp_k = frame_k->getTimestamp();

  // // collect points that were used for VO tracking
  // // used for frontend display
  // points_W_used.reserve(correspondences_used.size());
  // for (const auto& corr : correspondences_used) {
  //   points_W_used.push_back(LandmarkStatus::StaticInGlobal(
  //       corr.ref_, frame_id_k, timestamp_k, corr.tracklet_id_));
  // }

  frame_k->static_features_.markOutliers(pnp_result.outliers);

  if (pnp_result.status != TrackingStatus::VALID ||
      pnp_result.inliers.size() < 30) {
    // try propogate pose with available models
    if (propogated_nav_state_k) {
      frame_k->T_world_camera_ = propogated_nav_state_k->pose();
      VLOG(10) << "Number usable features invalid or too few at k= "
               << frame_k->getFrameId()
               << " - using IMU propogated pose to set camera pose!";
    } else {
      auto getLastRelativeEgoMotion = [&]() -> const gtsam::Pose3& {
        return rel_egopose_infos_.rbegin()->second.T_i_j;
      };
      // T_i_j of the camera pose representng our latest motion model of the
      // camera assume constant velocity model to propogate the model
      //  ie T_km2_km1 = T_km1_k
      const gtsam::Pose3& best_relative_motion = getLastRelativeEgoMotion();
      frame_k->T_world_camera_ =
          nav_state_km1_.state.pose() * best_relative_motion;
      VLOG(10) << "Number usable features invalid or too few at k= "
               << frame_k->getFrameId()
               << " - using constant velocity model to propogated camera pose!";
    }
    tracking_quality = TrackingQuality::Lost;

    return false;
  } else {
    // update camera pose
    T_ij = pnp_result.best_result;
    gtsam::Pose3 X_Wj = frame_km1->getPose() * T_ij;

    frame_k->T_world_camera_ = X_Wj;
    tracking_quality =
        use_map ? TrackingQuality::Good : TrackingQuality::Marginal;

    const auto& frontend_params = dyno_params_.frontend_params_;
    if (frontend_params.refine_camera_pose_with_joint_of) {
      VLOG(10) << "Refining camera pose with joint optical-flow";

      utils::ChronoTimingStats timer(this->moduleName() +
                                     ".camera_motion.refine");

      // this is actually doing most of the heavy lifting in terms of making the
      // VO smooth so would be nice to refine the flow w.r.t the map
      const auto refinement_result =
          optical_flow_pose_solver_.optimizeAndUpdate(
              frame_km1, frame_k, pnp_result.inliers, pnp_result.best_result,
              ReferenceFrame::LOCAL);

      T_ij = refinement_result.best_result.refined_pose;
      X_Wj = frame_km1->getPose() * T_ij;
      frame_k->T_world_camera_ = X_Wj;

      // VLOG(15) << "Refined camera pose with optical flow - error before: "
      //          << refinement_result.error_before.value_or(NaN)
      //          << " error_after: "
      //          << refinement_result.error_after.value_or(NaN);
    }

    points_W_used.reserve(correspondences_used.size());
    for (const auto& corr : correspondences_used) {
      gtsam::Point3 mW = frame_k->T_world_camera_ * corr.ref_;
      points_W_used.push_back(LandmarkStatus::StaticInGlobal(
          mW, frame_id_k, timestamp_k, corr.tracklet_id_));
    }

    return true;
  }
}

bool PoseChangeVIFrontend::addMeasurementsForObjectKeyframe(
    FrameId frame_id, ObjectId object_id) {
  const RelEgoPoseInfo& rel_egopose_lkf_j = rel_egopose_infos_.at(frame_id);
  CHECK_EQ(rel_egopose_lkf_j.j_id, frame_id);

  // assume measurements have been added if already keyframe
  if (map_->isObjectKeyFrame(frame_id, object_id)) {
    return false;
  }

  // add measurements at from frame for object motion
  CameraMeasurementStatusVector dynamic_measurements_kf;
  fillMeasurementsFromFeatureIterator(
      dynamic_measurements_kf,
      rel_egopose_lkf_j.frame_j->usableDynamicIterator(object_id),
      rel_egopose_lkf_j.j_id, rel_egopose_lkf_j.frame_j->getTimestamp(),
      dynamic_pixel_sigmas_, dynamic_point_sigma_);

  // update map after collecting all measurements for this frame
  map_->updateObservations(dynamic_measurements_kf);

  // mark object as keyframe for both the from and to (this frame) frames
  // this indicates that a motion variable exists at both frames
  CHECK(map_->setObjectKeyFrame(frame_id, object_id));
  return true;
}

void PoseChangeVIFrontend::solveObjectMotions(
    MultiObjectTrajectories& trajectories, ObjectIds& object_with_new_motions,
    ObjectTrackingStatusMap& object_tracking_status,
    ObjectPoseChangeInfoMap& infos, Frame::Ptr frame_k, Frame::Ptr frame_km1) {
  utils::ChronoTimingStats timer(this->moduleName() + ".object_motions");

  MotionEstimateMap estimated_motions;

  constexpr static bool kParallelSolve = true;
  // solved trajectories will have frame-to-frame motion
  object_motion_solver_->solve(frame_k, frame_km1, trajectories,
                               estimated_motions, kParallelSolve);

  object_with_new_motions.reserve(estimated_motions.size());
  for (const auto& [object_id, _] : estimated_motions) {
    object_with_new_motions.push_back(object_id);
  }

  object_tracking_status =
      object_motion_solver_->currentObjectTrackingStatuses();
  // only keyframes!!
  infos = std::move(object_motion_solver_->poseChangeInfoMap());
}

bool PoseChangeVIFrontend::shouldFrameBeKeyFrame(Frame::Ptr frame_k,
                                                 Frame::Ptr frame_km1) const {
  if (frame_k->getFrameId() < 4) {
    // just starting, so yes, we need this as a new keyframe
    return true;
  }

  const auto& cam_params = frame_k->getCamera()->getParams();

  const int rows = cam_params.ImageHeight() / 10;
  const int cols = cam_params.ImageWidth() / 10;

  const double kptradius_ = 0.09;
  const double radius = double(std::min(rows, cols)) * kptradius_;

  cv::Mat matches = cv::Mat::zeros(rows, cols, CV_8UC1);
  cv::Mat detections = cv::Mat::zeros(rows, cols, CV_8UC1);

  const FeatureContainer& static_features_lCKF = lCKF_frame_->static_features_;

  // For parallax
  std::vector<double> displacements;

  int num_detections = 0;
  int num_matches = 0;

  auto static_feature_itr = frame_k->usableStaticIterator();

  for (const auto& feature : static_feature_itr) {
    const TrackletId tracklet_id = feature->trackletId();
    const Keypoint& kp = feature->keypoint();

    const cv::Point2f pt = utils::gtsamPointToCv(kp) * 0.1;

    // --- detections (denominator proxy)
    cv::circle(detections, pt, int(radius), cv::Scalar(255), cv::FILLED);
    num_detections++;

    // --- matches (tracked from last keyframe)
    if (static_features_lCKF.exists(tracklet_id)) {
      cv::circle(matches, pt, int(radius), cv::Scalar(255), cv::FILLED);
      num_matches++;

      // --- compute displacement (parallax proxy)
      const auto& kp_kf =
          static_features_lCKF.getByTrackletId(tracklet_id)->keypoint();
      const cv::Point2f pt_kf = utils::gtsamPointToCv(kp_kf) * 0.1;

      double disp = cv::norm(pt - pt_kf);
      displacements.push_back(disp);
    }
  }

  // --- safety
  if (num_detections < 20) {
    // not enough features → don't create KF (tracking issue)
    return false;
  }

  // ===============================
  // 1. Coverage (IoU-style)
  // ===============================
  cv::Mat intersectionMask, unionMask;
  cv::bitwise_and(matches, detections, intersectionMask);
  cv::bitwise_or(matches, detections, unionMask);

  double intersection = double(cv::countNonZero(intersectionMask));
  double union_area = double(cv::countNonZero(unionMask));

  double overlap = double(intersection) / double(union_area);

  // ===============================
  // 2. Parallax (median displacement)
  // ===============================
  double median_disp = calculateMedian(displacements);
  // ===============================
  // 3. Track retention (optional but useful)
  // ===============================
  double retention = double(num_matches) / double(num_detections);

  // ===============================
  // 4. Decision thresholds
  // ===============================
  // const double overlap_thresh = 0.55;   // spatial redundancy
  const double overlap_thresh = 0.65;   // spatial redundancy
  const double disp_thresh = 15.0;      // pixels (tune)
  const double retention_thresh = 0.5;  // optional

  // ===============================
  // 5. Final decision
  // ===============================

  // Case A: not well explained by keyframe → new content
  if (overlap < overlap_thresh) {
    return true;
  }

  // more than 10 seconds since last keyframe?
  // since we've improved tracking less CKF's are made. Could also inforce this
  // with setting max_tracklet_id back to around 30
  // if(k_time - lkf_time > 10.0) {
  //   return true;
  // }

  // // Case B: strong motion → useful geometry
  // if (median_disp > disp_thresh) {
  //   return true;
  // }

  // // Optional: tracking degrading relative to KF
  if (retention < retention_thresh) {
    return true;
  }

  // Otherwise: redundant frame
  return false;
}

void PoseChangeVIFrontend::handleCameraKeyframe(
    const RelEgoPoseInfo& rel_lkf_k,
    const UpdateObservationParams& update_params,
    PostUpdateData& post_update_data, SinglePoseChangeInput::Ptr pc_input) {
  Frame::Ptr frame_k = rel_lkf_k.frame_j;
  const FrameId frame_id_k = frame_k->getFrameId();
  const Timestamp timestamp_k = frame_k->getTimestamp();

  CHECK_EQ(rel_lkf_k.j_id, frame_id_k);
  CHECK_EQ(rel_lkf_k.lkf_id, lCKF_frame_->getFrameId());
  CHECK_EQ(formulation_->getLastPropogatedFrame(), lCKF_frame_->getFrameId());
  CHECK(map_->isCameraKeyFrame(lCKF_frame_->getFrameId()));

  const gtsam::Pose3& T_lk_k = rel_lkf_k.T_lkf_j;
  ImuFrontend::PimPtr pim = rel_lkf_k.pim_lk_j;

  auto& new_static_values = pc_input->new_static_fg_input.values;
  auto& new_static_factors = pc_input->new_static_fg_input.factors;

  LOG(INFO) << "New Camera Keyframe (CKF) at k=" << frame_id_k;
  const gtsam::NavState predicted_nav_state =
      formulation_->addStatesPropogate(new_static_values, new_static_factors,
                                       frame_id_k, timestamp_k, T_lk_k, pim);

  map_->setCameraKeyFrame(frame_id_k);
  pc_input->keyframe_info.camera_keyframe = true;

  // NOTE: this is different from the nav state that is mantained in the
  // frontend so the initial states may be slightly different (only if IMU)
  // NOTE: must be after the updateObs -> these create new frames with the
  // correct attrivutes (ie. timestamp) while setInitialSensorPose
  // creates a new frame id necessary but does not populdate with timestamp!!
  // this is a known bufg!!
  map_->setInitialSensorPose(frame_id_k, timestamp_k,
                             Pose3Measurement(predicted_nav_state.pose()));

  post_update_data.static_update_result =
      formulation_->updateStaticObservations(frame_id_k, new_static_values,
                                             new_static_factors, update_params);

  imu_frontend_.resetIntegration();
  // use the frontend state but eventually this will get updated by the backend
  const gtsam::NavState& nav_state_k = rel_lkf_k.frontend_nav_state_j;
  nav_state_lkf_ = TemporalNavState{frame_id_k, timestamp_k, nav_state_k};
  lCKF_frame_ = frame_k;
  T_lkf_j_ = gtsam::Pose3::Identity();
}

void PoseChangeVIFrontend::logBestEstimates() const {
  VLOG(20) << "Logging test estimates from PoseChange frontend";

  // Use the presence of the backend sink function as a proxy to
  // indicate if the backend was running!
  if (!withBackend()) {
    return;
  }

  MultiObjectTrajectories full_object_trajectories_refined =
      formulation_->refinePerFrameMotionsPGO(full_object_trajectories_);

  const PoseTrajectory& camera_trajectory = accessor_->getCameraTrajectory();
  // const MultiObjectTrajectories& object_trajectories =
  // accessor->getMultiObjectTrajectories();
  auto logger = std::make_unique<VIFrontendLogger>("pc-pgo-estimations");
  auto ground_truths = shared_ground_truth_.access();

  logger->logCameraPose(camera_trajectory, ground_truths);

  logger->logObjectTrajectory(full_object_trajectories_refined, ground_truths);

  // right now output of refinePerFrameMotionsPGO only goes up to last object
  // keyframe
  for (ObjectId object_id : full_object_trajectories_refined.objectIds()) {
    FrameId last_okf_id =
        full_object_trajectories_refined.at(object_id).maxFrame();

    StatusLandmarkVector points_in_L =
        accessor_->getLocalDynamicLandmarkEstimates(object_id);

    if (points_in_L.empty()) {
      VLOG(20) << "No points for j=" << object_id << ": skipping logging!";
      continue;
    }

    std::string path = dyno::getOutputFilePath(
        "refined_object_map_k" + std::to_string(last_okf_id) + "_j" +
        std::to_string(object_id) + ".pcd");
    VLOG(10) << "Writing object map of size " << points_in_L.size() << " - "
             << path;
    saveAsPointCloud(points_in_L, path);
  }
}

void PoseChangeVIFrontend::logRealTimeObjectClouds(const ObjectIds& objects,
                                                   FrameId frame_id) const {
  for (ObjectId object_id : objects) {
    StatusLandmarkVector points_in_L;
    object_motion_solver_->getObjectStructureinL(object_id, points_in_L);

    if (points_in_L.empty()) {
      VLOG(20) << "No points for j=" << object_id << ": skipping logging!";
      continue;
    }

    std::string path =
        dyno::getOutputFilePath("doo_object_map_k" + std::to_string(frame_id) +
                                "_j" + std::to_string(object_id) + ".pcd");
    VLOG(10) << "Writing object map of size " << points_in_L.size() << " - "
             << path;
    saveAsPointCloud(points_in_L, path);
  }
}

bool PoseChangeVIFrontend::checkAndConsumeUpdate(FrameId frame_id_k) {
  if (has_backend_update_.exchange(false)) {
    utils::ChronoTimingStats timer(this->moduleName() + ".consume_update");
    // intermediate way of updating all poses
    dyno::FastSet<FrameId> frames_in_pgo;
    dyno::FastSet<FrameId> frames_propogated;
    dyno_state_.camera_trajectory = this->refinePerFrameCameraPGO(
        dyno_state_.camera_trajectory, frames_in_pgo, frames_propogated);

    for (const auto& entry : dyno_state_.camera_trajectory) {
      FrameId frame_id = entry.frame_id;
      if (rel_egopose_infos_.exists(frame_id)) {
        rel_egopose_infos_.at(frame_id).frame_j->T_world_camera_ = entry.data;
      }
    }

    // TODO: not updating velocity or bias!
    // TODO: by proxy of updating the frames this should also update the
    // lCKF_frame_
    // TODO: I think this is the most vital one...
    auto maybe_nav_state_km1 = accessor_->getNavState(nav_state_km1_.frame_id);
    if (maybe_nav_state_km1) {
      nav_state_km1_.state = maybe_nav_state_km1.get();
    }

    auto maybe_nav_state_lkf = accessor_->getNavState(nav_state_lkf_.frame_id);
    if (maybe_nav_state_lkf) {
      nav_state_lkf_.state = maybe_nav_state_lkf.get();
    }

    if (formulation_->isImuInitalized()) {
      // try and get the latest IMU bias
      const FrameId frame_with_best_bias = frames_in_pgo.back();
      auto maybe_imu_bias = accessor_->getImuBias(frame_with_best_bias);
      if (maybe_imu_bias) {
        VLOG(10) << "Updating imu bias at k=" << frame_with_best_bias;
        // imu_bias_ = maybe_imu_bias.get();
      }
    }

    return true;
  }

  // no update consumed
  return false;
}

PoseTrajectory PoseChangeVIFrontend::refinePerFrameCameraPGO(
    const PoseTrajectory& camera_trajectory,
    dyno::FastSet<FrameId>& frames_in_pgo,
    dyno::FastSet<FrameId>& frames_propogated) const {
  // opimized camera trajectory only containing keyframes
  const PoseTrajectory& camera_trajectory_kf = accessor_->getCameraTrajectory();
  auto noise_models = formulation_->noiseModels();

  gtsam::Values values;
  gtsam::NonlinearFactorGraph graph;

  FrameId max_frame = camera_trajectory_kf.maxFrame();

  gtsam::SharedNoiseModel relative_noise_model =
      gtsam::noiseModel::Isotropic::Sigma(6u, 0.8);

  PoseTrajectory optimized_camera_trajectory =
      camera_trajectory.range({}, max_frame);
  for (const auto& entry : optimized_camera_trajectory) {
    const FrameId frame_id = entry.frame_id;
    const gtsam::Pose3 X_Wk = entry.data;

    CHECK(rel_egopose_infos_.exists(frame_id));
    const auto& relative_ego_motion = rel_egopose_infos_.at(frame_id);
    CHECK_EQ(relative_ego_motion.j_id, frame_id);
    const gtsam::Key key_j = CameraPoseSymbol(frame_id);
    const gtsam::Key key_i = CameraPoseSymbol(relative_ego_motion.i_id);

    values.insert(key_j, X_Wk);

    if (camera_trajectory_kf.exists(frame_id)) {
      gtsam::Pose3 X_W_k_refined = camera_trajectory_kf.at(frame_id);
      graph.addPrior<gtsam::Pose3>(key_j, X_W_k_refined,
                                   noise_models.initial_pose_prior);
    }
    // add relative motion constraint
    // // TODO: use pim
    // graph.push_back(boost::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
    //     CameraPoseSymbol(relative_ego_motion.lkf_id), key,
    //     relative_ego_motion.T_lkf_j, noise_models.odometry_noise));
    // TODO: use pim
    graph.push_back(boost::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        key_i, key_j, relative_ego_motion.T_i_j, relative_noise_model));
  }

  // graph.print("Camera PGO ", DynosamKeyFormatter);

  using GNOptimizer = dyno::NonlinearOptimizer<gtsam::GaussNewtonOptimizer>;
  GNOptimizer solver(graph, values);

  NonlinearOptimizerSummary summary;
  NonlinearOptimizerOptions options;

  LOG(INFO) << "Beginning Camera PGO";
  gtsam::Values optimised_values;
  CHECK(solver.solve(optimised_values, options, &summary));

  LOG(INFO) << "Initial error: " << summary.initial_error << " final error "
            << summary.final_error << " time[s] "
            << summary.cumulative_time_in_seconds
            << " #iterations= " << summary.numIterations();

  for (const auto& entry : optimized_camera_trajectory) {
    const FrameId frame_id = entry.frame_id;
    const Timestamp timestamp = entry.timestamp;
    gtsam::Key key = CameraPoseSymbol(frame_id);
    gtsam::Pose3 X_W_k_refined = optimised_values.at<gtsam::Pose3>(key);

    optimized_camera_trajectory.update(frame_id, X_W_k_refined);
    frames_in_pgo.insert(frame_id);
  }

  // for the unoptimised camera poses (ie. max optimised frame -> current frame)
  // propogate from last optimised frame using visual odometry
  gtsam::Pose3 X_W = optimized_camera_trajectory.at(max_frame);
  auto it = camera_trajectory.upperBound(max_frame);

  FrameId from_frame = max_frame;
  for (; it != camera_trajectory.end(); ++it) {
    const auto frame_id = it->frame_id;
    const auto timestamp = it->timestamp;
    LOG(INFO) << " Doing VO propogation to " << it->frame_id;

    CHECK(rel_egopose_infos_.exists(frame_id));
    const auto& relative_ego_motion = rel_egopose_infos_.at(frame_id);
    // sanity check that our frame bookkeeping is correct
    CHECK_EQ(from_frame, relative_ego_motion.i_id);

    // propogate with visual odometry
    X_W = X_W * relative_ego_motion.T_i_j;
    optimized_camera_trajectory.insert(frame_id, timestamp, X_W);

    from_frame = relative_ego_motion.j_id;
    frames_propogated.insert(frame_id);
  }

  CHECK_EQ(optimized_camera_trajectory.size(), camera_trajectory.size());

  return optimized_camera_trajectory;
}

}  // namespace dyno
