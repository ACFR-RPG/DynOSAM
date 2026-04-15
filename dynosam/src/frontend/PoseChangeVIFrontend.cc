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

  gtsam::Pose3 X_W_k_initial = gtsam::Pose3::Identity();

  dyno_state_.camera_trajectory.insert(frame_id_k, timestamp_k, X_W_k_initial);

  lkf_id_ = frame_id_k;

  // no motion as first frame!
  const gtsam::Pose3 T_km1_k = gtsam::Pose3::Identity();
  T_km1_k_ = T_km1_k;
  T_lkf_k_ = T_km1_k;

  RealtimeOutput::Ptr realtime_output = std::make_shared<RealtimeOutput>();
  realtime_output->state.frame_id = frame_id_k;
  realtime_output->state.timestamp = timestamp_k;
  realtime_output->state.camera_trajectory = dyno_state_.camera_trajectory;
  realtime_output->ground_truth = input->ground_truth_packet;

  IntermediateMotion intermediate_motion;
  intermediate_motion.from = lkf_id_;
  intermediate_motion.to = frame_id_k;
  intermediate_motion.timestamp = timestamp_k;
  // NOT setting the nav state!
  intermediate_motion.frame = frame_k;
  intermediate_motion.pim = nullptr;
  intermediate_motion.T_from_to = gtsam::Pose3::Identity();
  intermediate_motions_.insert2(intermediate_motion.to, intermediate_motion);

  // KeyFrameData keyframe_data;
  // keyframe_data.kf_id = frame_id_k;
  // keyframe_data.kf_id_prev = lkf_id_;
  // keyframe_data.frame = frame_k;
  // keyframe_data.camera_keyframe = true;
  // // objects are not added becuase on the first frame they can only ever be a
  // // "from" frame
  // keyframe_data.nav_state = gtsam::NavState();
  // keyframes_.insert2(frame_id_k, keyframe_data);

  CameraMeasurementStatusVector static_measurements;
  fillMeasurementsFromFeatureIterator(
      &static_measurements, frame_k->usableStaticIterator(), frame_id_k,
      timestamp_k, static_pixel_sigmas_, static_point_sigma_,
      &realtime_output->state.local_static_map);

  // // TODO: hack for now to add measurements at first frame!!!!
  // CameraMeasurementStatusVector dynamic_measurements;
  // fillMeasurementsFromFeatureIterator(
  //     &dynamic_measurements, frame_k->usableDynamicIterator(),
  //     frame_id_k, timestamp_k, dynamic_pixel_sigmas_, dynamic_point_sigma_,
  //     &realtime_output->state.dynamic_map);

  // // HACK for now = eventually should add measurments as needed based on
  // // estimated from/to motions!
  // map_->updateObservations(dynamic_measurements);

  // first frame is always KF
  map_->updateObservations(static_measurements);
  map_->setInitialSensorPose(frame_id_k, timestamp_k,
                             Pose3Measurement(X_W_k_initial));
  map_->setCameraKeyFrame(frame_id_k);

  auto pc_input = std::make_shared<PoseChangeInput>();
  pc_input->frame_id = frame_id_k;
  pc_input->timestamp = timestamp_k;

  formulation_->addStatesInitalise(pc_input->new_values, pc_input->new_factors,
                                   frame_id_k, timestamp_k, X_W_k_initial,
                                   gtsam::Vector3(0, 0, 0));

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

  // add initial state/propogate
  // add measurements to map
  // add motion PC info to formulation (ie. what was pre-update)
  // build factors

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

  // when providing the propogated imu state only provide if it was
  // actually filled by a prediction from the IMU - otherwise it will ne
  // nullopt. This tells the function to use a constant motion model from the
  // previous frame ie. T_km1_k_ if tracking fails
  const bool ego_motion_solve =
      solveAndRefineEgoMotion(frame_k, frame_km1, nav_state_km1_, T_km1_k_,
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

  T_km1_k_ = nav_state_km1_.pose().inverse() * nav_state_k.pose();
  // this may be updated later
  T_lkf_k_ = nav_state_lkf_.pose().inverse() * nav_state_k.pose();
  nav_state_km1_ = nav_state_k;

  IntermediateMotion intermediate_motion;
  intermediate_motion.from = lkf_id_;
  intermediate_motion.to = frame_id_k;
  intermediate_motion.timestamp = timestamp_k;
  intermediate_motion.frame = frame_k;
  intermediate_motion.frontend_nav_state = nav_state_k;
  // intermediate_motion.pim = (pim) ? ImuFrontend::copyPim(pim) : nullptr;
  // intermediate_motion.imu_measurements =
  //     input->imu_measurements.value_or(ImuMeasurements{});
  intermediate_motion.T_from_to = T_lkf_k_;
  intermediate_motions_.insert2(intermediate_motion.to, intermediate_motion);

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

  RealtimeOutput::Ptr realtime_output = std::make_shared<RealtimeOutput>();
  realtime_output->state.frame_id = frame_id_k;
  realtime_output->state.timestamp = timestamp_k;
  realtime_output->state.camera_trajectory = dyno_state_.camera_trajectory;
  realtime_output->state.object_trajectories = dyno_state_.object_trajectories;
  realtime_output->ground_truth = input->ground_truth_packet;

  CameraMeasurementStatusVector static_measurements;
  fillMeasurementsFromFeatureIterator(
      &static_measurements, frame_k->usableStaticIterator(), frame_id_k,
      timestamp_k, static_pixel_sigmas_, static_point_sigma_,
      &realtime_output->state.local_static_map);

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
    CHECK_EQ(formulation_->getLastPropogatedFrame(), lkf_id_);
    CHECK(map_->isCameraKeyFrame(lkf_id_));
    const gtsam::NavState predicted_nav_state =
        formulation_->addStatesPropogate(pc_input->new_values,
                                         pc_input->new_factors, frame_id_k,
                                         timestamp_k, T_lkf_k_, pim);

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
    nav_state_lkf_ = nav_state_k;
    lkf_id_ = frame_id_k;
  } else {
    // if not a camera keyframe, update KF node with relative ego motion data
    auto frame_lkf_node = CHECK_NOTNULL(map_->getFrame(lkf_id_));
    // TODO: this relative motion should be consistent with the frontend
    // TODO: use IMU if available!
    frame_lkf_node->addRelativeEgoMotion(T_lkf_k_, frame_id_k);
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
      const auto from_motion_frame_j = H_W_KF_k.from();
      CHECK_EQ(H_W_KF_k.to(), frame_id_k);

      // add dynamic measurements observed at the from frame
      const IntermediateMotion& intermediate_motion_lkf_j =
          intermediate_motions_.at(from_motion_frame_j);
      CHECK_EQ(intermediate_motion_lkf_j.to, from_motion_frame_j);

      LOG(INFO) << "Adding dynamic measurements j=" << object_id << " at "
                << intermediate_motion_lkf_j.to;

      // if object is already a keyframe at this frame then assume
      // measurements have already been added
      // jesse: is this correct? Since we never go back and add new features I
      // think this is fine
      if (!map_->isObjectKeyFrame(from_motion_frame_j, object_id)) {
        // add measurements at from frame for object motion
        CameraMeasurementStatusVector dynamic_measurements_kf;
        auto num_added = fillMeasurementsFromFeatureIterator(
            &dynamic_measurements_kf,
            intermediate_motion_lkf_j.frame->usableDynamicIterator(object_id),
            intermediate_motion_lkf_j.to, intermediate_motion_lkf_j.timestamp,
            dynamic_pixel_sigmas_, dynamic_point_sigma_);

        // update map after collecting all measurements for this frame
        map_->updateObservations(dynamic_measurements_kf);

        // mark object as keyframe for both the from and to (this frame) frames
        // this indicates that a motion variable exists at both frames
        CHECK(map_->setObjectKeyFrame(from_motion_frame_j, object_id));
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
  // CHECK(keyframes_.exists(lkf_id_));
  // const KeyFrameData& lkf_data = keyframes_.at(lkf_id_);
  // const Frame::Ptr lkf_frame = lkf_data.frame;

  // return frame_k->getFrameId() % 10 == 0;

  // first 4 frames must keyframes to help initalise!
  return frame_k->getTrackingInfo()->new_static_detections ||
         frame_k->getFrameId() < 4;
  // FOR NOW!
  // return true;
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
