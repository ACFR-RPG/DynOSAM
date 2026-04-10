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
    auto accessor = formulation_->derivedAccessor<HybridAccessor>();

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

  KeyFrameData keyframe_data;
  keyframe_data.kf_id = frame_id_k;
  keyframe_data.kf_id_prev = lkf_id_;
  keyframe_data.frame = frame_k;
  keyframe_data.camera_keyframe = true;
  // objects are not added becuase on the first frame they can only ever be a
  // "from" frame
  keyframe_data.nav_state = gtsam::NavState();
  keyframes_.insert2(frame_id_k, keyframe_data);

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
  constructVisualFactors(update_params, frame_id_k, pc_input->new_values,
                         pc_input->new_factors, post_update_data);

  logRealTimeOutput(realtime_output);

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
                                    (imu_propogated_nav_state_k)
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
  intermediate_motion.pim = (pim) ? ImuFrontend::copyPim(pim) : nullptr;
  intermediate_motion.imu_measurements =
      input->imu_measurements.value_or(ImuMeasurements{});
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

  // ObjectPoseChangeInfoMap kf_pose_change_infos;
  // const size_t num_object_keyframes =
  //     extractKeyFramedMotions(kf_pose_change_infos, pose_change_infos);
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
  const bool is_keyframe = ego_motion_keyframe || num_object_keyframes > 0;
  if (withBackend() && is_keyframe) {
    LOG(INFO) << "KF selected for k=" << frame_id_k
              << " (ego kf= " << std::boolalpha << ego_motion_keyframe
              << " #object KF " << num_object_keyframes << ")";
    LOG(INFO) << "Last KF= " << lkf_id_;

    // add KF data immediately so that the object keyframe logic knows about
    // this frame
    KeyFrameData keyframe_data;
    keyframe_data.kf_id = frame_id_k;
    keyframe_data.kf_id_prev = lkf_id_;
    keyframe_data.frame = frame_k;
    keyframe_data.camera_keyframe = ego_motion_keyframe;
    keyframe_data.object_keyframes = objects_with_keyframes;
    keyframe_data.nav_state = nav_state_k;
    keyframes_.insert2(frame_id_k, keyframe_data);

    CameraMeasurementStatusVector dynamic_measurements_kf;
    for (const auto& dm : dynamic_measurements) {
      const auto& object_id = dm.objectId();
      if (kf_pose_change_infos.exists(object_id)) {
        dynamic_measurements_kf.push_back(dm);
      }
    }

    auto pc_input = std::make_shared<PoseChangeInput>();
    pc_input->frame_id = frame_id_k;
    pc_input->timestamp = timestamp_k;
    // Ordered set of KF objects ordered by their "from" motion
    // to make Keyframes at the from motion we need to propogate the ego-motion
    // states starting with the earliest
    // TODO: what if multiple objects need adding at the same from frame?
    // rely on gtsam::FastMap being ordered by std::less<>
    gtsam::FastMap<FrameId, ObjectIds> objects_by_earliest_kf;

    for (const auto& [object_id, info] : kf_pose_change_infos) {
      CHECK(info.isKeyFrame());

      const auto& H_W_KF_k = info.H_W_KF_k;
      const auto from_motion_frame_j = H_W_KF_k.from();

      if (!objects_by_earliest_kf.exists(from_motion_frame_j)) {
        objects_by_earliest_kf.insert2(from_motion_frame_j, ObjectIds{});
      }

      // fill object that has a motion from_motion_frame_j -> k
      objects_by_earliest_kf.at(from_motion_frame_j).push_back(object_id);
      // if the from frame is smaller than the current last keyframe we
      // (currently) have no way of fixing as all the factors relating to the
      // last kf will be already in the smoother and we have no way of
      // recovering this!! it is allowed as long as the from frame is not
      // already a KF ie. if the from frame is new it must be greater than the
      // last kf
      if (!keyframes_.exists(from_motion_frame_j)) {
        CHECK_GE(from_motion_frame_j, lkf_id_) << " for j=" << object_id;
      }

      // also fill involved objects in the PoseChangeInput since we loop over
      // the id's here anyway
      pc_input->involved_objects.push_back(object_id);
    }

    for (const auto& [from_motion_frame_j, object_ids] :
         objects_by_earliest_kf) {
      // check that the last kf as considered by the frontend
      // is the same as considered by the formulation
      CHECK_EQ(lkf_id_, formulation_->getLastPropogatedFrame());
      // from motion
      const auto lkf_id_j = from_motion_frame_j;
      LOG(INFO) << info_string(frame_id_k, object_ids)
                << " added with keyframe with from motion " << lkf_id_j;

      {
        // sanity check
        for (const ObjectId& j : object_ids) {
          const ObjectPoseChangeInfo& info = kf_pose_change_infos.at(j);
          const auto& H_W_KF_k = info.H_W_KF_k;
          CHECK_EQ(lkf_id_j, H_W_KF_k.from());
          CHECK_EQ(frame_id_k, H_W_KF_k.to());
        }
      }

      const FrameId original_lkf_id = lkf_id_;
      // if from object motion does not already exists as a keyframe

      // TODO: not sure what happens if we are dealing with multiple objects and
      // there is somehow
      //  keyframes before and after an existing one! This should never happen!
      //  all new KEyframes should be between original lf and kf!
      //  CHECK_GE(lkf_id_j, original_lkf_id);
      CHECK_LE(lkf_id_j, frame_id_k);

      if (!keyframes_.exists(lkf_id_j)) {
        VLOG(20) << info_string(frame_id_k, object_ids)
                 << " has a from motion k=" << lkf_id_j
                 << " that does not exist as a keyframe. Adding!";

        // intermediate motion at desired keyframe
        IntermediateMotion& intermediate_motion_lkf_j =
            intermediate_motions_.at(lkf_id_j);
        CHECK_EQ(intermediate_motion_lkf_j.to, lkf_id_j);
        CHECK_EQ(intermediate_motion_lkf_j.from, original_lkf_id);

        const auto& nav_state_lkf_j =
            intermediate_motion_lkf_j.frontend_nav_state;

        // update relative motions (ie. T_from_to and/or PIM)
        // given we may have added new Keyframes in the past (earlier in this
        // loop) such that the intermediate motions from motion is no longer
        // from the most recent keyframe!
        if (intermediate_motion_lkf_j.from != lkf_id_) {
          // IMU is tricky lets ignore for now!!!
          // intermediate motion at new last keyframe
          // CHECK(intermediate_motions_.exists(lkf_id_));
          const IntermediateMotion& intermediate_motion_lkf =
              intermediate_motions_.at(lkf_id_);
          const auto& nav_state_lkf =
              intermediate_motion_lkf.frontend_nav_state;

          // recalculate relative motion
          intermediate_motion_lkf_j.T_from_to =
              nav_state_lkf.pose().inverse() * nav_state_lkf_j.pose();
          intermediate_motion_lkf_j.from = lkf_id_;

          // auto& pim_lkf_j = intermediate_motion_lkf_j.pim;
          // if(pim_lkf_j) {
          //   imu_frontend_.resetIntegration();

          //   // this should take us intermediate motion from -> to
          //   intermediate_motion_lkf_j.pim =
          //   imu_frontend_.preintegrateImuMeasurements(intermediate_motion_lkf_j.imu_measurements);
          // }
        }

        const KeyFrameData& lkf_data = keyframes_.at(lkf_id_);
        CHECK_EQ(lkf_data.kf_id, lkf_id_);

        ObjectIds object_keyframes;
        // propogate from current last keyframe to the desired keyframe
        gtsam::NavState predicted_nav_state = formulation_->addStatesPropogate(
            pc_input->new_values, pc_input->new_factors,
            intermediate_motion_lkf_j.to, intermediate_motion_lkf_j.timestamp,
            intermediate_motion_lkf_j.T_from_to, intermediate_motion_lkf_j.pim);

        // collect meaasurements
        fillMeasurementsFromFeatureIterator(
            &static_measurements,
            intermediate_motion_lkf_j.frame->usableStaticIterator(),
            intermediate_motion_lkf_j.to, intermediate_motion_lkf_j.timestamp,
            static_pixel_sigmas_, static_point_sigma_);

        // fill dynamic_measurements_kf for all objects present at this frame
        for (const ObjectId& j : object_ids) {
          auto num_added = fillMeasurementsFromFeatureIterator(
              &dynamic_measurements_kf,
              intermediate_motion_lkf_j.frame->usableDynamicIterator(j),
              intermediate_motion_lkf_j.to, intermediate_motion_lkf_j.timestamp,
              dynamic_pixel_sigmas_, dynamic_point_sigma_);

          object_keyframes.push_back(j);

          LOG(INFO) << "object j= " << j << " was not originally present at k="
                    << intermediate_motion_lkf_j.to
                    << " # features added=" << num_added;
        }

        // NOTE: this is different from the nav state that is mantained in the
        // frontend so the initial states may be slightly different (only if
        // IMU)
        map_->setInitialSensorPose(
            intermediate_motion_lkf_j.to, intermediate_motion_lkf_j.timestamp,
            Pose3Measurement(predicted_nav_state.pose()));

        KeyFrameData keyframe_data;
        keyframe_data.kf_id = intermediate_motion_lkf_j.to;
        keyframe_data.kf_id_prev = lkf_id_;
        keyframe_data.camera_keyframe = false;
        keyframe_data.retroactively_made_keyframe = true;
        // actually not sure this is right as these keyframes could be "from"
        keyframe_data.object_keyframes = std::move(object_keyframes);
        keyframe_data.frame = intermediate_motion_lkf_j.frame;
        keyframe_data.nav_state = nav_state_lkf_j;
        keyframes_.insert2(keyframe_data.kf_id, keyframe_data);

        // progressively update internal keyframe related properties
        lkf_id_ = intermediate_motion_lkf_j.to;
        T_lkf_k_ = nav_state_lkf_j.pose().inverse() * nav_state_k.pose();
        nav_state_lkf_ = nav_state_lkf_j;

        // // also must update the pim so that it represents the motion from the
        // new lkf to k pim = intermediate_motion_lkf_j.pim;
      } else {
        // keyframe does exist! I sure hope we dont have to change the
        // ego-motion propogation! how to check this
        KeyFrameData& keyframe_data = keyframes_.at(lkf_id_j);
        CHECK_EQ(keyframe_data.kf_id, lkf_id_j);
        // cannot be a keyframe that was retroactively made
        CHECK_EQ(keyframe_data.retroactively_made_keyframe, false);

        // TODO: should add object to keyframe object data

        // object was not initially present in the keyframe so measurements will
        // not exist at this frame
        for (const ObjectId& j : object_ids) {
          if (!keyframe_data.isObjectKeyFrame(j)) {
            auto num_added = fillMeasurementsFromFeatureIterator(
                &dynamic_measurements_kf,
                keyframe_data.frame->usableDynamicIterator(j),
                keyframe_data.kf_id, keyframe_data.frame->getTimestamp(),
                dynamic_pixel_sigmas_, dynamic_point_sigma_);
            LOG(INFO) << "object j= " << j
                      << " was not originally present at k=" << lkf_id_j
                      << " # features added=" << num_added;

            keyframe_data.object_keyframes.push_back(j);
          }
        }
      }
    }

    // NOTE: this is different from the nav state that is mantained in the
    // frontend
    /// hmmmm errr dont need to do this if we did this in the loop before!!!
    // TODO: otherwise get Forward predicting k=16 t=16 using VO
    CHECK_EQ(formulation_->getLastPropogatedFrame(), lkf_id_);
    const gtsam::NavState predicted_nav_state =
        formulation_->addStatesPropogate(pc_input->new_values,
                                         pc_input->new_factors, frame_id_k,
                                         timestamp_k, T_lkf_k_, pim);

    // updateMapWithMeasurements(frame_k, input, predicted_nav_state.pose());

    map_->updateObservations(static_measurements);
    // this should include new dynamic measurements from any new keyframes we've
    // added
    map_->updateObservations(dynamic_measurements_kf);

    // NOTE: this is different from the nav state that is mantained in the
    // frontend so the initial states may be slightly different (only if IMU)
    // NOTE: must be after the updateObs -> these create new frames with the
    // correct attrivutes (ie. timestamp) while setInitialSensorPose
    // creates a new frame id necessary but does not populdate with timestamp!!
    // this is a known bufg!!
    map_->setInitialSensorPose(frame_id_k, timestamp_k,
                               Pose3Measurement(predicted_nav_state.pose()));

    formulation_->addObjects(frame_id_k, kf_pose_change_infos);

    UpdateObservationParams update_params;
    update_params.enable_debug_info = true;
    update_params.do_backtrack = false;

    PostUpdateData post_update_data(frame_id_k);
    constructVisualFactors(update_params, frame_id_k, pc_input->new_values,
                           pc_input->new_factors, post_update_data);

    if (pose_change_backend_sink_) {
      pose_change_backend_sink_(pc_input);
    }

    imu_frontend_.resetIntegration();
    // this not predicted_nav_state?
    nav_state_lkf_ = nav_state_k;
    lkf_id_ = frame_id_k;
  }

  // add initial state/propogate
  // add measurements to map
  // add motion PC info to formulation (ie. what was pre-update)
  // build factors

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

  constexpr static bool kParallelSolve = false;
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
  CHECK(keyframes_.exists(lkf_id_));
  const KeyFrameData& lkf_data = keyframes_.at(lkf_id_);
  const Frame::Ptr lkf_frame = lkf_data.frame;

  return frame_k->getFrameId() % 10 == 0;
  // FOR NOW!
  // return true;
}

size_t PoseChangeVIFrontend::extractKeyFramedMotions(
    ObjectPoseChangeInfoMap& kf_infos,
    const ObjectPoseChangeInfoMap& all_infos) const {
  for (const auto& [object_id, info] : all_infos) {
    if (info.isKeyFrame()) {
      kf_infos.insert2(object_id, info);
    }
  }
  return kf_infos.size();
}

void PoseChangeVIFrontend::constructVisualFactors(
    const UpdateObservationParams& update_params, FrameId frame_k,
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors,
    PostUpdateData& post_update_data) {
  {
    utils::ChronoTimingStats timer("backend.update_static_obs");
    post_update_data.static_update_result =
        formulation_->updateStaticObservations(frame_k, new_values, new_factors,
                                               update_params);
  }

  {
    // if (!FLAGS_regular_backend_static_only) {
    // LOG(INFO) << "Starting updateDynamicObservations";
    // TODO: better names
    utils::ChronoTimingStats timer("backend.update_dynamic_obs");
    post_update_data.dynamic_update_result =
        formulation_->updateDynamicObservations(frame_k, new_values,
                                                new_factors, update_params);
    // }
  }
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

  auto accessor = formulation_->derivedAccessor<HybridAccessor>();
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
