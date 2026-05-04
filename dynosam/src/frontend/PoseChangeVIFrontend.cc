#include "dynosam/frontend/PoseChangeVIFrontend.hpp"

#include <gflags/gflags.h>

#include "dynosam_common/viz/Colour.hpp"

DEFINE_bool(pc_smoother_allow_backend_updates, false,
            "If updates from the backend should be received.");

DEFINE_bool(pc_log_object_kf_structure, false,
            "If the object point cloud should be logged at keyframes");

namespace dyno {

PoseChangeVIFrontend::PoseChangeVIFrontend(
    const DynoParams& params, Camera::Ptr camera,
    HybridFormulationKeyFrame::Ptr formulation,
    ImageDisplayQueue* display_queue,
    const SharedGroundTruth& shared_ground_truth)
    : VIFrontend("pc-frontend", params, camera, display_queue,
                 shared_ground_truth),
      formulation_(CHECK_NOTNULL(formulation)),
      map_(CHECK_NOTNULL(formulation->map())),
      tracking_viz_(params.frontend_params_.image_tracks_vis_params) {
  // TODo
  HybridObjectMotionSolverParams motion_params;
  motion_params.optical_flow_solver_params.use_robust = false;

  SharedGroundTruth ground_truth;
  if (FLAGS_init_object_pose_from_gt) {
    LOG(INFO) << "FLAGS_init_object_pose_from_gt is true. Object motion solver "
                 "will attempt to initalise object poses using provided ground "
                 "truth pose!";
    ground_truth = shared_ground_truth_;
  }

  object_motion_solver_ = std::make_unique<HybridObjectMotionSolver>(
      motion_params, camera_->getParams(), ground_truth);
}

PoseChangeVIFrontend::~PoseChangeVIFrontend() { logBestEstimates(); }

void PoseChangeVIFrontend::onBackendUpdateComplete(
    const PoseChangeUpdateComplete& event) {
  // TODO: this is definitely not thread safe
  const FrameId frame_id = event.ending_frame_id;
  const FrameId starting_frame_id = event.starting_frame_id;

  LOG(INFO) << "Recieved backend update for frames " << starting_frame_id
            << " -> " << frame_id;

  if (FLAGS_pc_smoother_allow_backend_updates) {
    LOG(INFO) << "Recieved backend update at frame " << frame_id;
    object_motion_solver_->receiveUpdate(event);
  }

  if (FLAGS_pc_log_object_kf_structure) {
    auto accessor =
        formulation_->derivedAccessor<HybridFormulationKeyFrameAccessor>();

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
          accessor->getLocalDynamicLandmarkEstimates(object_id);

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

  CameraMeasurementStatusVector static_measurements;
  fillMeasurementsFromFeatureIterator(
      &static_measurements, frame_k->usableStaticIterator(), frame_id_k,
      timestamp_k, static_pixel_sigmas_, static_point_sigma_,
      &realtime_output->state.local_static_map);

  // first frame is always KF
  map_->updateObservations(static_measurements);
  map_->setInitialSensorPose(frame_id_k, timestamp_k,
                             Pose3Measurement(identity_pose));
  map_->setCameraKeyFrame(frame_id_k);

  auto pc_input = std::make_shared<SinglePoseChangeInput>();
  pc_input->frame_id = frame_id_k;
  pc_input->timestamp = timestamp_k;
  pc_input->keyframe_info.camera_keyframe = true;

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
  ImageContainer::Ptr image_container = input->image_container_;
  const auto frame_id_k = input->getFrameId();
  const auto timestamp_k = input->getTimestamp();

  ImuFrontend::PimPtr pim = nullptr;
  std::optional<gtsam::NavState> imu_propogated_nav_state_k =
      tryPropogateImu(input, nav_state_lkf_, pim);

  //! Rotation from k-1 to k in k-1
  std::optional<gtsam::Rot3> R_km1_k;
  if (imu_propogated_nav_state_k) {
    CHECK(pim);
    R_km1_k = nav_state_km1_.attitude().inverse() *
              imu_propogated_nav_state_k->attitude();
  }

  Frame::Ptr frame_k = featureTrack(input, R_km1_k);
  Frame::Ptr frame_km1 = tracker_->getPreviousFrame();
  CHECK(frame_km1);

  VLOG(1) << to_string(tracker_->getTrackerInfo());

  bool stereo_matching_result = stereoMatch(frame_k);

  RealtimeOutput::Ptr realtime_output = std::make_shared<RealtimeOutput>();
  realtime_output->state.frame_id = frame_id_k;
  realtime_output->state.timestamp = timestamp_k;

  // when providing the propogated imu state only provide if it was
  // actually filled by a prediction from the IMU - otherwise it will ne
  // nullopt. This tells the function to use a constant motion model from the
  // previous frame ie. T_km1_k_ if tracking fails
  StatusLandmarkVector& static_landmarks_used_vo =
      realtime_output->state.local_static_map;
  TrackingQuality camera_tracking_quality;
  const bool ego_motion_solve = solveAndRefineEgoMotion(
      frame_k, frame_km1, static_landmarks_used_vo, camera_tracking_quality,
      imu_propogated_nav_state_k, R_km1_k);

  // TODO: amagamate this function and the frame->updateDepths for when we are
  // stereo/rgbd
  if (stereo_matching_result) {
    // Need to match aagain after optical flow used to update the keypoints
    // This seems to make a pretty big difference!!
    stereo_matching_result &= stereoMatch(frame_k);
  }

  // if(input->ground_truth_packet) {
  //   frame_k->T_world_camera_ = input->ground_truth_packet->X_world_;
  // }

  // we currently use the frame pose as the nav state - this value can come from
  // either the VO OR the IMU, depending on the result from the
  // solveCameraMotion this is only relevant since we dont solve incremental so
  // the backend is not immediately updating the frontend at which point we can
  // just use the best estimate in the case of the VO, the nav_state velocity
  const gtsam::NavState nav_state_k(frame_k->getPose(),
                                    imu_propogated_nav_state_k
                                        ? imu_propogated_nav_state_k->velocity()
                                        : gtsam::Vector3(0, 0, 0));

  RelEgoPoseInfo rel_egopose;
  rel_egopose.lkf_id = lCKF_frame_->getFrameId();
  rel_egopose.i_id = frame_km1->getFrameId();
  rel_egopose.j_id = frame_id_k;
  rel_egopose.frame_j = frame_k;
  rel_egopose.frontend_nav_state_j = nav_state_k;
  rel_egopose.T_i_j = nav_state_km1_.pose().inverse() * nav_state_k.pose();
  rel_egopose.T_lkf_j = nav_state_lkf_.pose().inverse() * nav_state_k.pose();
  rel_egopose.pim_lk_j = (pim) ? ImuFrontend::copyPim(pim) : nullptr;
  rel_egopose.imu_measurements =
      input->imu_measurements.value_or(ImuMeasurements{});
  // very important to store this
  rel_egopose_infos_.insert2(frame_id_k, rel_egopose);
  nav_state_km1_ = nav_state_k;

  dyno_state_.camera_trajectory.insert(frame_id_k, timestamp_k,
                                       nav_state_k.pose());

  // ObjectPoseChangeInfoMap pose_change_infos;
  ObjectIds objects_with_new_motions;
  ObjectPoseChangeInfoMap kf_pose_change_infos;
  solveObjectMotions(dyno_state_.object_trajectories, objects_with_new_motions,
                     kf_pose_change_infos, frame_k, frame_km1);

  // TODO: slow and rematching all points again!
  //  need to rematch after solving flow with objects
  stereoMatch(frame_k);

  // test noisy on object motions now W is aligned
  for (auto& [object_id, info] : kf_pose_change_infos) {
    gtsam::Pose3& est = info.H_W_KF_k;
    // est = utils::perturbWithNoise(est, 0.03);
  }

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
      &static_measurements, frame_k->usableStaticIterator(), frame_id_k,
      timestamp_k, static_pixel_sigmas_, static_point_sigma_);

  CameraMeasurementStatusVector dynamic_measurements;
  fillMeasurementsFromFeatureIterator(
      &dynamic_measurements, frame_k->usableDynamicIterator(), frame_id_k,
      timestamp_k, dynamic_pixel_sigmas_, dynamic_point_sigma_
      /*&realtime_output->state.dynamic_map*/);

  // fill output dynamic map with current structure
  for (const auto& object_id : objects_with_new_motions) {
    // assume that getObjectStructureinW does not clear the vector
    object_motion_solver_->getObjectStructureinW(
        object_id, realtime_output->state.dynamic_map);
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
  const bool any_object_keyframes = num_object_keyframes > 0;
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

  if (ego_motion_keyframe) {
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
    // collect all dynamic measurements at k
    CameraMeasurementStatusVector dynamic_measurements_kf_k;
    for (const auto& dm : dynamic_measurements) {
      const auto& object_id = dm.objectId();
      if (kf_pose_change_infos.exists(object_id)) {
        dynamic_measurements_kf_k.push_back(dm);
      }
    }
    // TODO: this will fail when we start adding OKF's for LOST objects

    // update map after collecting all measurements for this frame
    map_->updateObservations(dynamic_measurements_kf_k);

    for (const auto& [object_id, info] : kf_pose_change_infos) {
      CHECK(info.isKeyFrame());
      const auto& H_W_KF_k = info.H_W_KF_k;
      const auto frame_id_motion_from = H_W_KF_k.from();
      CHECK_EQ(H_W_KF_k.to(), frame_id_k);

      // add dynamic measurements observed at the from frame
      const RelEgoPoseInfo& rel_egopose_lkf_j =
          rel_egopose_infos_.at(frame_id_motion_from);
      CHECK_EQ(rel_egopose_lkf_j.j_id, frame_id_motion_from);

      // if object is already a keyframe at this frame then assume
      // measurements have already been added
      // jesse: is this correct? Since we never go back and add new features I
      // think this is fine
      if (!map_->isObjectKeyFrame(frame_id_motion_from, object_id)) {
        // add measurements at from frame for object motion
        CameraMeasurementStatusVector dynamic_measurements_kf;
        size_t n = fillMeasurementsFromFeatureIterator(
            &dynamic_measurements_kf,
            rel_egopose_lkf_j.frame_j->usableDynamicIterator(object_id),
            rel_egopose_lkf_j.j_id, rel_egopose_lkf_j.frame_j->getTimestamp(),
            dynamic_pixel_sigmas_, dynamic_point_sigma_);
        // LOG(INFO) << "Adding n=" << n << " dyn object measurements to map at
        // k="
        //           << frame_id_motion_from;

        // update map after collecting all measurements for this frame
        map_->updateObservations(dynamic_measurements_kf);

        // mark object as keyframe for both the from and to (this frame) frames
        // this indicates that a motion variable exists at both frames
        CHECK(map_->setObjectKeyFrame(frame_id_motion_from, object_id));
      }

      // mark object as keyframe in this frame
      //  the measurements for k have already been addded
      CHECK(map_->setObjectKeyFrame(frame_id_k, object_id));

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

  SharedModuleStates* shared_module_states = map_->getSharedModuleStates();
  shared_module_states->current_frontend_frame = frame_id_k;

  // any keyframe triggered
  if (withBackend() && is_any_keyframe) {
    pose_change_backend_sink_(pc_input);
  }

  // fillDebugImagery(realtime_output->debug_imagery, frame_k, frame_km1);
  // set only the debug tracking imagery to avoid also calling the (somewhat
  // depricated) computeTracks function from the tracker
  ViTrackingViz::Data viz_data;
  viz_data.camera_tracking_quality = camera_tracking_quality;
  viz_data.keyframe_info = pc_input->keyframe_info;
  realtime_output->debug_imagery.tracking_image =
      tracking_viz_.vizTracking(*frame_km1, *frame_k, viz_data);

  pushImageToDisplayQueue("Tracks",
                          realtime_output->debug_imagery.tracking_image);

  // if (stereo_matching_result) {
  //   cv::Mat stereo_track;
  //   tracker_->drawStereoMatches(stereo_track, *frame_k);
  //   pushImageToDisplayQueue("Stereo-Matches", stereo_track);
  // }

  logRealTimeOutput(realtime_output);

  return {State::Nominal, realtime_output};
}

bool PoseChangeVIFrontend::solveAndRefineEgoMotion(
    Frame::Ptr frame_k, const Frame::Ptr& frame_km1,
    StatusLandmarkVector& points_W_used, TrackingQuality& tracking_quality,
    std::optional<gtsam::NavState> propogated_nav_state_k,
    std::optional<gtsam::Rot3> R_km1_k) {
  AbsolutePoseCorrespondences m_matches;

  double tracking_quality_cost;
  bool success = formulation_->matchToStaticMap(frame_k, m_matches,
                                                &tracking_quality_cost);

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
  AbsolutePoseCorrespondences correspondences_used;
  if (use_map) {
    VLOG(5) << "Tracking against Map: k=" << frame_k->getFrameId()
            << ": matches=" << m_matches.size()
            << " tracking quality=" << tracking_quality_cost;
    // solve PnP
    pnp_result = pnp_ransac_.solve3d2d(m_matches, R_km1_k);
    correspondences_used = std::move(m_matches);
  } else {
    VLOG(5) << "Tracking aginast Previous frame";
    AbsolutePoseCorrespondences correspondences;
    frame_k->getCorrespondences(correspondences, *frame_km1,
                                KeyPointType::STATIC,
                                frame_k->landmarkWorldKeypointCorrespondance());

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

  // collect points that were used for VO tracking
  // used for frontend display
  points_W_used.reserve(correspondences_used.size());
  for (const auto& corr : correspondences_used) {
    points_W_used.push_back(LandmarkStatus::StaticInGlobal(
        corr.ref_, frame_id_k, timestamp_k, corr.tracklet_id_));
  }

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
      frame_k->T_world_camera_ = nav_state_km1_.pose() * best_relative_motion;
      VLOG(10) << "Number usable features invalid or too few at k= "
               << frame_k->getFrameId()
               << " - using constant velocity model to propogated camera pose!";
    }
    tracking_quality = TrackingQuality::Lost;

    return false;
  } else {
    // update camera pose
    frame_k->T_world_camera_ = pnp_result.best_result;

    const auto& frontend_params = dyno_params_.frontend_params_;
    if (frontend_params.refine_camera_pose_with_joint_of) {
      VLOG(10) << "Refining camera pose with joint optical-flow";

      utils::ChronoTimingStats timer(this->moduleName() +
                                     ".camera_motion.refine");

      // this is actually doing most of the heavy lifting in terms of making the
      // VO smooth so would be nice to refine the flow w.r.t the map
      const auto refinement_result =
          optical_flow_pose_solver_.optimizeAndUpdate(
              frame_km1, frame_k, pnp_result.inliers, pnp_result.best_result);

      // TODO: refresh depth or with stereo (NOT: we refresh depth with stereo
      // outside this function)!!
      //  refresh depth information for each frame
      if (frame_k->imageContainer().hasDepth()) {
        CHECK(frame_k->updateDepths());
      }

      frame_k->T_world_camera_ = refinement_result.best_result.refined_pose;
      tracking_quality =
          use_map ? TrackingQuality::Good : TrackingQuality::Marginal;

      VLOG(15) << "Refined camera pose with optical flow - error before: "
               << refinement_result.error_before.value_or(NaN)
               << " error_after: "
               << refinement_result.error_after.value_or(NaN);
    }
    return true;
  }
}

void PoseChangeVIFrontend::solveObjectMotions(
    MultiObjectTrajectories& trajectories, ObjectIds& object_with_new_motions,
    ObjectPoseChangeInfoMap& infos, Frame::Ptr frame_k, Frame::Ptr frame_km1) {
  MotionEstimateMap estimated_motions;

  constexpr static bool kParallelSolve = true;
  // solved trajectories will have frame-to-frame motion
  object_motion_solver_->solve(frame_k, frame_km1, trajectories,
                               estimated_motions, kParallelSolve);

  object_with_new_motions.reserve(estimated_motions.size());
  for (const auto& [object_id, _] : estimated_motions) {
    object_with_new_motions.push_back(object_id);
  }

  // LOG(INFO) << "Solved motions " <<
  // container_to_string(object_with_new_motions)
  //           << " k=" << frame_k->getFrameId();

  // only keyframes!!
  infos = std::move(object_motion_solver_->poseChangeInfoMap());
}

bool PoseChangeVIFrontend::shouldFrameBeKeyFrame(Frame::Ptr frame_k,
                                                 Frame::Ptr frame_km1) const {
  // TODO: keyframes_ not used anymore?
  // CHECK(keyframes_.exists(lCKF_frame_->getFrameId()));
  // const KeyFrameData& lkf_data = keyframes_.at(lCKF_frame_->getFrameId());
  // const Frame::Ptr lkf_frame = lkf_data.frame;

  // return frame_k->getFrameId() % 10 == 0;

  if (frame_k->getFrameId() < 4) {
    // just starting, so yes, we need this as a new keyframe
    return true;
  }

  // double tracking_quality;
  // AbsolutePoseCorrespondences matches;
  // std::vector<double> repr_errors;

  // formulation_->matchToStaticMap(
  //   lCKF_frame_,
  //   matches,
  //   repr_errors,
  //   &tracking_quality
  // );

  // LOG(INFO) << "Tracking: k=" << frame_k->getFrameId() << ": matches=" <<
  // matches.size()
  //    << " tracking quality=" << tracking_quality << " against CKF=" <<
  //    lCKF_frame_->getFrameId();

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

  const Timestamp lkf_time = lCKF_frame_->getTimestamp();
  const Timestamp k_time = frame_k->getTimestamp();

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
  // this not predicted_nav_state?
  // TOODO: may better get the nav state via the VIOformulation!
  const gtsam::NavState& nav_state_k = rel_lkf_k.frontend_nav_state_j;
  nav_state_lkf_ = nav_state_k;
  lCKF_frame_ = frame_k;
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

  auto accessor =
      formulation_->derivedAccessor<HybridFormulationKeyFrameAccessor>();
  CHECK_NOTNULL(accessor);

  const PoseTrajectory& camera_trajectory = accessor->getCameraTrajectory();
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
        accessor->getLocalDynamicLandmarkEstimates(object_id);

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

ViTrackingViz::ViTrackingViz(const ImageTracksParams& viz_params)
    : viz_params_(viz_params) {}

cv::Mat ViTrackingViz::vizTracking(const Frame& frame_km1, const Frame& frame_k,
                                   const Data& data) {
  const ImageWrapper<ImageType::RGBMono>& img_wrapper =
      frame_k.imageContainer().rgb();
  cv::Mat img_rgb = img_wrapper.toRGB().clone();

  std::string static_tracks_info_string;
  drawStaticTracks(img_rgb, static_tracks_info_string, frame_km1, frame_k,
                   data);

  std::string dynamic_tracks_info_string;
  drawDynamicTracks(img_rgb, dynamic_tracks_info_string, frame_km1, frame_k,
                    data);

  if (viz_params_.showFrameInfo()) {
    std::string info_string =
        static_tracks_info_string + dynamic_tracks_info_string;
    writeFrameInfo(img_rgb, info_string);
  }
  return img_rgb;
}

void ViTrackingViz::drawStaticTracks(cv::Mat& img, std::string& info,
                                     const Frame& frame_km1,
                                     const Frame& frame_k, const Data& data) {
  const bool debug = viz_params_.isDebug();
  const bool show_intermediate_tracking =
      viz_params_.showIntermediateTracking();
  const int static_point_thickness = viz_params_.featureThickness();

  static const cv::Scalar red(Color::red().bgra());
  static const cv::Scalar green(Color::green().bgra());
  static const cv::Scalar blue(Color::blue().bgra());

  size_t num_points_tracked = 0;

  // Add all keypoints in cur_frame with the tracks.
  for (const Feature::Ptr& feature : frame_k.static_features_) {
    const Keypoint& px_cur = feature->keypoint();
    const auto pc_cur = utils::gtsamPointToCv(px_cur);
    if (!feature->usable() &&
        show_intermediate_tracking) {  // Untracked landmarks are red.
      cv::circle(img, pc_cur, static_point_thickness, red, 2, cv::LINE_AA);
    } else {
      const Feature::Ptr& prev_feature =
          frame_km1.static_features_.getByTrackletId(feature->trackletId());
      if (prev_feature) {
        // If feature was in previous frame, display tracked feature with
        // green circle/line:
        cv::circle(img, pc_cur, static_point_thickness, green, 1);

        // draw the optical flow arrow
        const auto pc_prev = utils::gtsamPointToCv(prev_feature->keypoint());
        cv::arrowedLine(img, pc_prev, pc_cur, green, 1);

        num_points_tracked++;

      } else if (debug &&
                 show_intermediate_tracking) {  // New feature tracks are blue.
        cv::circle(img, pc_cur, 6, blue, 1);
      }
    }
  }

  if (data.keyframe_info.camera_keyframe) {
    CKF_count++;
  }

  std::stringstream ss;
  ss << "Frame: " << frame_k.getFrameId() << " ";
  ss << "[VO tracks: " << num_points_tracked << " ";
  ss << "Cam KFs: " << CKF_count << " ";
  ss << "quailty: " << to_string(data.camera_tracking_quality) << "]";

  info = ss.str();
}

void ViTrackingViz::drawDynamicTracks(cv::Mat& img, std::string& info,
                                      const Frame& frame_km1,
                                      const Frame& frame_k, const Data& data) {
  for (const Feature::Ptr& feature :
       frame_k.dynamic_features_.usableIterator()) {
    const auto px_cur = utils::gtsamPointToCv(feature->keypoint());
    const Feature::Ptr& prev_feature =
        frame_km1.dynamic_features_.getByTrackletId(feature->trackletId());
    if (prev_feature) {
      const auto px_prev = utils::gtsamPointToCv(prev_feature->keypoint());
      const cv::Scalar colour = Color::uniqueId(feature->objectId()).bgra();

      // cv::arrowedLine(img, px_prev,px_cur,colour, 1, 8, 0, 0.1);
      // cv::circle(img, utils::gtsamPointToCv(px_cur), 2, colour, -1);

      // draw feature as rectangle
      // 8 pixels size
      constexpr static int size = 6;
      int half = size / 2;
      cv::Point tl(px_cur.x - half, px_cur.y - half);
      cv::Point br(px_cur.x + half, px_cur.y + half);
      cv::rectangle(img, tl, br, colour, 2, -1);
    }
  }

  // mark new object keyframes
  for (const auto& kf_info : data.keyframe_info.object_keyframes) {
    ObjectId object_id = kf_info.object_id;
    if (!OKF_count_.exists(object_id)) {
      OKF_count_[object_id] = 0;
    }
    OKF_count_[object_id]++;
  }

  std::vector<ObjectId> objects_to_print;
  double now = frame_k.getTimestamp();
  for (const auto& object_observation_pair : frame_k.getObjectObservations()) {
    const ObjectId object_id = object_observation_pair.first;
    const cv::Rect& bb = object_observation_pair.second.bounding_box;

    if (bb.empty()) continue;

    auto& state = states_[object_id];
    objects_to_print.push_back(object_id);

    if (state.first_seen_time < 0.0) {
      state.first_seen_time = now;
      state.object_id = object_id;
    }

    state.last_seen_time = now;

    // time-based progress
    double elapsed = now - state.first_seen_time;
    state.appear_progress =
        std::min(static_cast<float>(elapsed / appear_duration_sec_), 1.0f);

    if (viz_params_.drawObjectBoundingBox()) {
      // const cv::Scalar colour = Color::uniqueId(object_id).bgra();
      // const std::string label = "object " + std::to_string(object_id);
      // utils::drawLabeledBoundingBox(img_rgb, label, colour, bb,
      // bbox_thickness);
      drawAnimatedBox(img, bb, state);
    }
  }

  if (viz_params_.drawObjectMask()) {
    const cv::Mat& object_mask = frame_k.imageContainer().objectMotionMask();

    constexpr static float kAlpha = 0.7;
    utils::labelMaskToRGB(object_mask, img, img, kAlpha);
  }

  std::stringstream ss;
  ss << " [Objects (KF): ";

  if (objects_to_print.empty()) {
    ss << "None";
  } else {
    for (size_t i = 0; i < objects_to_print.size(); ++i) {
      ss << objects_to_print[i] << " (" << OKF_count_[objects_to_print[i]]
         << ")";
      if (i != objects_to_print.size() - 1) {
        ss << ", ";  // Add comma between elements
      }
    }
  }
  ss << "]";

  info = ss.str();
}

void ViTrackingViz::writeFrameInfo(cv::Mat& img,
                                   const std::string& info_string) const {
  constexpr static double kFontScale = 0.4;
  constexpr static int kFontFace = cv::FONT_HERSHEY_SIMPLEX;
  constexpr static int kThickness = 1;

  int base_line;
  cv::Size text_size = cv::getTextSize(info_string, kFontFace, kFontScale,
                                       kThickness, &base_line);
  cv::Mat image_text =
      cv::Mat(img.rows + text_size.height + 10, img.cols, img.type());
  img.copyTo(image_text.rowRange(0, img.rows).colRange(0, img.cols));
  image_text.rowRange(img.rows, image_text.rows) =
      cv::Mat::zeros(text_size.height + 10, img.cols, img.type());
  cv::putText(image_text, info_string, cv::Point(5, image_text.rows - 5),
              kFontFace, kFontScale, cv::Scalar(255, 255, 255), kThickness);

  img = image_text;
}

// declare helper function for Viz
void roundedRectangle(cv::Mat& img, const cv::Point& topLeft,
                      const cv::Point& bottomRight, const cv::Scalar& color,
                      int thickness = 2, int cornerRadius = 20,
                      float extensionAlpha = 2);

void ViTrackingViz::drawAnimatedBox(cv::Mat& img, const cv::Rect& bbox,
                                    const TemporalObjectState& state) const {
  const float t = state.appear_progress;

  // --- slower, readable lock ---
  const float lock = 1.0f - std::exp(-3.5f * t);

  // optional: slow near end (magnetic feel)
  const float smooth_lock = lock * lock;

  std::array<cv::Point2f, 4> raw_corners = {
      cv::Point2f(bbox.x, bbox.y), cv::Point2f(bbox.x + bbox.width, bbox.y),
      cv::Point2f(bbox.x, bbox.y + bbox.height),
      cv::Point2f(bbox.x + bbox.width, bbox.y + bbox.height)};

  if (!state.corners_initialized) {
    state.filtered_corners = raw_corners;
    state.corners_initialized = true;
  }

  // ----------------------------
  // EMA smoothing (visual jitter reduction)
  // ----------------------------
  for (int i = 0; i < 4; ++i) {
    state.filtered_corners[i] =
        corner_smoothing_alpha_ * raw_corners[i] +
        (1.0f - corner_smoothing_alpha_) * state.filtered_corners[i];
  }

  const auto& filtered_tl = state.filtered_corners[0];
  const auto& filtered_br = state.filtered_corners[3];

  const cv::Rect filtered_bbox(filtered_tl, filtered_br);
  cv::Point2f filtered_center(filtered_bbox.x + filtered_bbox.width * 0.5f,
                              filtered_bbox.y + filtered_bbox.height * 0.5f);

  // --- corner growth ---
  const float corner_t = std::min(t * 1.8f, 1.0f);
  const int base_len = static_cast<int>(
      std::min(filtered_bbox.width, filtered_bbox.height) * 0.25f);
  const int len = static_cast<int>(base_len * corner_t);
  const float radius = len * 0.4f;

  // ----------------------------
  // far-to-near scale (lock-on feel)
  // ----------------------------
  const float start_scale = 3.0f;
  const float scale = 1.0f + (start_scale - 1.0f) * (1.0f - smooth_lock);

  // ----------------------------
  // flash effect (subtle acquisition cue)
  // ----------------------------
  const float flash = std::exp(-4.0f * t);

  const cv::Scalar base_color = Color::uniqueId(state.object_id).bgra();

  cv::Scalar color(std::min(255.0, base_color[0] + 255.0 * flash),
                   std::min(255.0, base_color[1] + 255.0 * flash),
                   std::min(255.0, base_color[2] + 255.0 * flash),
                   base_color[3]);

  static constexpr int thickness = 5;

  const cv::Point2f dir_tl = filtered_tl - filtered_center;
  const cv::Point2f p_tl = filtered_center + dir_tl * scale;

  const cv::Point2f dir_br = filtered_br - filtered_center;
  const cv::Point2f p_br = filtered_center + dir_br * scale;

  roundedRectangle(img, p_tl, p_br, color, thickness, radius);
}

static inline cv::Point pt(int x, int y) { return cv::Point(x, y); }

// mostly vibe-coded function to draw a target lock with rounded corners
void roundedRectangle(cv::Mat& img, const cv::Point& topLeft,
                      const cv::Point& bottomRight, const cv::Scalar& color,
                      int thickness, int cornerRadius, float extensionAlpha) {
  static constexpr int lineType = cv::LINE_AA;

  int x1 = topLeft.x;
  int y1 = topLeft.y;
  int x2 = bottomRight.x;
  int y2 = bottomRight.y;

  // ------------------------------------------------------------
  // corners (same layout as Python version)
  // p1 - p2
  // |     |
  // p4 - p3
  // ------------------------------------------------------------
  const cv::Point p1(x1, y1);
  const cv::Point p2(x2, y1);
  const cv::Point p3(x2, y2);
  const cv::Point p4(x1, y2);

  const int r = cornerRadius;

  // ------------------------------------------------------------
  // helper length (same as Python intent)
  // ------------------------------------------------------------
  // float line_length = r * extensionAlpha;

  // ============================================================
  // TOP LEFT
  // ============================================================
  cv::line(img, pt(p1.x + r, p1.y), pt(p1.x + 2 * r, p2.y), color, thickness,
           lineType);

  cv::line(img, pt(p1.x, p1.y + r), pt(p1.x, p2.y + 2 * r), color, thickness,
           lineType);

  // ============================================================
  // TOP RIGHT
  // ============================================================
  cv::line(img, pt(p2.x - r, p2.y), pt(p2.x - 2 * r, p1.y), color, thickness,
           lineType);

  cv::line(img, pt(p2.x, p2.y + r), pt(p2.x, p1.y + 2 * r), color, thickness,
           lineType);

  // ============================================================
  // BOTTOM LEFT
  // ============================================================
  cv::line(img, pt(p4.x + r, p4.y), pt(p4.x + 2 * r, p3.y), color, thickness,
           lineType);

  cv::line(img, pt(p4.x, p4.y - r), pt(p4.x, p3.y - 2 * r), color, thickness,
           lineType);

  // ============================================================
  // BOTTOM RIGHT
  // ============================================================
  cv::line(img, pt(p3.x - r, p3.y), pt(p3.x - 2 * r, p4.y), color, thickness,
           lineType);

  cv::line(img, pt(p3.x, p3.y - r), pt(p3.x, p4.y - 2 * r), color, thickness,
           lineType);

  // ============================================================
  // ARCS (same quadrant logic as Python/OpenCV version)
  // ============================================================

  cv::ellipse(img, p1 + cv::Point(r, r), cv::Size(r, r), 0, 180, 270, color,
              thickness, lineType);

  cv::ellipse(img, p2 + cv::Point(-r, r), cv::Size(r, r), 0, 270, 360, color,
              thickness, lineType);

  cv::ellipse(img, p3 + cv::Point(-r, -r), cv::Size(r, r), 0, 0, 90, color,
              thickness, lineType);

  cv::ellipse(img, p4 + cv::Point(r, -r), cv::Size(r, r), 0, 90, 180, color,
              thickness, lineType);
}

}  // namespace dyno
