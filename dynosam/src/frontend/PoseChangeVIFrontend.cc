#include "dynosam/frontend/PoseChangeVIFrontend.hpp"

#include <gflags/gflags.h>

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
      map_(CHECK_NOTNULL(formulation->map())) {
  // TODo
  HybridObjectMotionSolverParams motion_params;

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
  const FrameId frame_id = event.frame_id;

  if (FLAGS_pc_smoother_allow_backend_updates) {
    LOG(INFO) << "Recieved backend update at frame " << frame_id;
    object_motion_solver_->receiveUpdate(formulation_->generateUpdateInfo());
  }

  if (FLAGS_pc_log_object_kf_structure) {
    auto accessor =
        formulation_->derivedAccessor<HybridFormulationKeyFrameAccessor>();

    LOG(INFO) << "Logging estimated object structures...";

    // TODO: later when we use a different map we can check for keyframes etc!!
    auto frame_node_k = map_->getFrame(frame_id);
    CHECK_NOTNULL(frame_node_k);

    // only log for object seen at this frame
    for (ObjectId object_id : frame_node_k->objectSeenIds()) {
      StatusLandmarkVector points_in_L =
          accessor->getLocalDynamicLandmarkEstimates(object_id);

      if (points_in_L.empty()) {
        VLOG(20) << "No points for j=" << object_id << ": skipping logging!";
        continue;
      }

      std::string path = dyno::getOutputFilePath(
          "refined_object_map_k" + std::to_string(frame_id) + "_j" +
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

  auto pc_input = std::make_shared<PoseChangeInput>();
  pc_input->frame_id = frame_id_k;
  pc_input->timestamp = timestamp_k;

  formulation_->addStatesInitalise(pc_input->new_values, pc_input->new_factors,
                                   frame_id_k, timestamp_k, identity_pose,
                                   zero_velocity);

  UpdateObservationParams update_params;
  update_params.enable_debug_info = true;
  update_params.do_backtrack = false;

  PostUpdateData post_update_data(frame_id_k);
  post_update_data.static_update_result =
      formulation_->updateStaticObservations(frame_id_k, pc_input->new_values,
                                             pc_input->new_factors,
                                             update_params);

  logRealTimeOutput(realtime_output);

  SharedModuleStates* shared_module_states = map_->getSharedModuleStates();
  shared_module_states->current_frontend_frame = frame_id_k;

  if (pose_change_backend_sink_) {
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

  VLOG(5) << to_string(tracker_->getTrackerInfo());

  FeaturePtrs stereo_matches_1;
  bool stereo_matching_result =
      tryStereoMatchStaticFeatures(frame_k, image_container, stereo_matches_1);

  RealtimeOutput::Ptr realtime_output = std::make_shared<RealtimeOutput>();
  realtime_output->state.frame_id = frame_id_k;
  realtime_output->state.timestamp = timestamp_k;

  // when providing the propogated imu state only provide if it was
  // actually filled by a prediction from the IMU - otherwise it will ne
  // nullopt. This tells the function to use a constant motion model from the
  // previous frame ie. T_km1_k_ if tracking fails
  StatusLandmarkVector& static_landmarks_used_vo =
      realtime_output->state.local_static_map;
  const bool ego_motion_solve =
      solveAndRefineEgoMotion(frame_k, frame_km1, static_landmarks_used_vo,
                              imu_propogated_nav_state_k, R_km1_k);

  if (stereo_matching_result) {
    // Need to match aagain after optical flow used to update the keypoints
    // This seems to make a pretty big difference!!
    FeaturePtrs stereo_matches_2;
    tryStereoMatchStaticFeatures(frame_k, image_container, stereo_matches_2);
  }

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

  // TODO: may not be used if is_any_keyframe is false
  auto pc_input = std::make_shared<PoseChangeInput>();
  pc_input->frame_id = frame_id_k;
  pc_input->timestamp = timestamp_k;

  UpdateObservationParams update_params;
  update_params.enable_debug_info = true;
  update_params.do_backtrack = false;

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
        fillMeasurementsFromFeatureIterator(
            &dynamic_measurements_kf,
            rel_egopose_lkf_j.frame_j->usableDynamicIterator(object_id),
            rel_egopose_lkf_j.j_id, rel_egopose_lkf_j.frame_j->getTimestamp(),
            dynamic_pixel_sigmas_, dynamic_point_sigma_);

        // update map after collecting all measurements for this frame
        map_->updateObservations(dynamic_measurements_kf);

        // mark object as keyframe for both the from and to (this frame) frames
        // this indicates that a motion variable exists at both frames
        CHECK(map_->setObjectKeyFrame(frame_id_motion_from, object_id));
      }

      // mark object as keyframe in this frame
      //  the measurements for k have already been addded
      CHECK(map_->setObjectKeyFrame(frame_id_k, object_id));

      pc_input->involved_objects.push_back(object_id);
    }
    // add objects to backend with initial motion estimates
    formulation_->addObjects(frame_id_k, kf_pose_change_infos);

    // generate new factors for dynamic objects based on latest measurements
    // and object keyframe states
    post_update_data.dynamic_update_result =
        formulation_->updateDynamicObservations(
            frame_id_k, pc_input->new_values, pc_input->new_factors,
            update_params);
  }

  SharedModuleStates* shared_module_states = map_->getSharedModuleStates();
  shared_module_states->current_frontend_frame = frame_id_k;

  // any keyframe triggered
  if (withBackend() && is_any_keyframe) {
    pose_change_backend_sink_(pc_input);
  }

  fillDebugImagery(realtime_output->debug_imagery, frame_k, frame_km1);

  pushImageToDisplayQueue("Tracks",
                          realtime_output->debug_imagery.tracking_image);

  logRealTimeOutput(realtime_output);

  return {State::Nominal, realtime_output};
}

bool PoseChangeVIFrontend::solveAndRefineEgoMotion(
    Frame::Ptr frame_k, const Frame::Ptr& frame_km1,
    StatusLandmarkVector& points_W_used,
    std::optional<gtsam::NavState> propogated_nav_state_k,
    std::optional<gtsam::Rot3> R_km1_k) {
  AbsolutePoseCorrespondences m_matches;
  double tracking_quality;

  bool success =
      formulation_->matchToStaticMap(frame_k, m_matches, &tracking_quality);

  bool use_map = false;
  if (success) {
    // double median_repr = calculateMedian(repr_errors);
    int num_matches = m_matches.size();

    LOG(INFO) << "Tracking: k=" << frame_k->getFrameId()
              << ": matches=" << m_matches.size()
              << " tracking quality=" << tracking_quality;
    double min_matches = 40;
    // double repr_thresh = 2.0;
    // from OKVIS < 0.3 is marginal and < 0.01 is LOST
    double coverage_thresh = 0.3;

    use_map =
        (num_matches > min_matches) && (tracking_quality > coverage_thresh);
  }

  Pose3SolverResult pnp_result;
  AbsolutePoseCorrespondences correspondences_used;
  if (use_map) {
    LOG(INFO) << "Tracking aginast MAP";
    // solve PnP
    pnp_result = pnp_ransac_.solve3d2d(m_matches, R_km1_k);
    correspondences_used = std::move(m_matches);
  } else {
    LOG(INFO) << "Tracking aginast Previous frame";
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

      frame_k->T_world_camera_ = refinement_result.best_result.refined_pose;

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

  LOG(INFO) << "Solved motions " << container_to_string(object_with_new_motions)
            << " k=" << frame_k->getFrameId();

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
  const double overlap_thresh = 0.55;   // spatial redundancy
  const double disp_thresh = 15.0;      // pixels (tune)
  const double retention_thresh = 0.5;  // optional

  // ===============================
  // 5. Final decision
  // ===============================

  // Case A: not well explained by keyframe → new content
  if (overlap < overlap_thresh) {
    return true;
  }

  // Case B: strong motion → useful geometry
  if (median_disp > disp_thresh) {
    return true;
  }

  // Optional: tracking degrading relative to KF
  if (retention < retention_thresh) {
    return true;
  }

  // Otherwise: redundant frame
  return false;
}

void PoseChangeVIFrontend::handleCameraKeyframe(
    const RelEgoPoseInfo& rel_lkf_k,
    const UpdateObservationParams& update_params,
    PostUpdateData& post_update_data, PoseChangeInput::Ptr pc_input) {
  Frame::Ptr frame_k = rel_lkf_k.frame_j;
  const FrameId frame_id_k = frame_k->getFrameId();
  const Timestamp timestamp_k = frame_k->getTimestamp();

  CHECK_EQ(rel_lkf_k.j_id, frame_id_k);
  CHECK_EQ(rel_lkf_k.lkf_id, lCKF_frame_->getFrameId());
  CHECK_EQ(formulation_->getLastPropogatedFrame(), lCKF_frame_->getFrameId());
  CHECK(map_->isCameraKeyFrame(lCKF_frame_->getFrameId()));

  const gtsam::Pose3& T_lk_k = rel_lkf_k.T_lkf_j;
  ImuFrontend::PimPtr pim = rel_lkf_k.pim_lk_j;

  LOG(INFO) << "New Camera Keyframe (CKF) at k=" << frame_id_k;
  const gtsam::NavState predicted_nav_state = formulation_->addStatesPropogate(
      pc_input->new_values, pc_input->new_factors, frame_id_k, timestamp_k,
      T_lk_k, pim);

  map_->setCameraKeyFrame(frame_id_k);

  // NOTE: this is different from the nav state that is mantained in the
  // frontend so the initial states may be slightly different (only if IMU)
  // NOTE: must be after the updateObs -> these create new frames with the
  // correct attrivutes (ie. timestamp) while setInitialSensorPose
  // creates a new frame id necessary but does not populdate with timestamp!!
  // this is a known bufg!!
  map_->setInitialSensorPose(frame_id_k, timestamp_k,
                             Pose3Measurement(predicted_nav_state.pose()));

  post_update_data.static_update_result =
      formulation_->updateStaticObservations(frame_id_k, pc_input->new_values,
                                             pc_input->new_factors,
                                             update_params);

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

}  // namespace dyno
