#include "dynosam/frontend/RegularVIFrontend.hpp"

namespace dyno {

RegularVIFrontend::RegularVIFrontend(
    const DynoParams& params, Camera::Ptr camera,
    ImageDisplayQueue* display_queue,
    const SharedGroundTruth& shared_ground_truth)
    : VIFrontend("regular-frontend", params, camera, display_queue,
                 shared_ground_truth) {
  // TODO: params from config!!!
  ConsecutiveFrameObjectMotionSolverParams motion_params;
  motion_params.refine_motion_with_3d = false;

  if (FLAGS_init_object_pose_from_gt) {
    LOG(INFO) << "FLAGS_init_object_pose_from_gt is true. Object motion solver "
                 "will attempt to initalise object poses using provided ground "
                 "truth pose!";
    object_motion_solver_ =
        std::make_unique<ConsecutiveFrameObjectMotionSolver>(
            motion_params, camera_->getParams(), shared_ground_truth);
  } else {
    object_motion_solver_ =
        std::make_unique<ConsecutiveFrameObjectMotionSolver>(
            motion_params, camera_->getParams());
  }
}

RegularVIFrontend::SpinReturn RegularVIFrontend::boostrapSpin(
    VIFrontendInput::ConstPtr input) {
  Frame::Ptr frame_k = featureTrack(input);
  const auto frame_id_k = input->getFrameId();
  const auto timestamp_k = input->getTimestamp();

  gtsam::Pose3 X_W_k_initial = gtsam::Pose3::Identity();
  dyno_state_.camera_trajectory.insert(frame_id_k, timestamp_k, X_W_k_initial);

  VisionImuPacket::Ptr vision_imu_packet = std::make_shared<VisionImuPacket>();
  vision_imu_packet->frameId(frame_id_k);
  vision_imu_packet->timestamp(timestamp_k);
  vision_imu_packet->groundTruthPacket(input->ground_truth_packet);

  // no motion as first frame!
  const gtsam::Pose3 T_km1_k = gtsam::Pose3::Identity();
  T_km1_k_ = T_km1_k;
  fillOutputPacketWithTracks(vision_imu_packet, *frame_k, X_W_k_initial,
                             T_km1_k_, dyno_state_.object_trajectories);

  if (regular_backend_output_sink_) {
    regular_backend_output_sink_(vision_imu_packet);
  }

  RealtimeOutput::Ptr realtime_output = std::make_shared<RealtimeOutput>();
  realtime_output->state.frame_id = frame_id_k;
  realtime_output->state.timestamp = timestamp_k;
  realtime_output->state.camera_trajectory = dyno_state_.camera_trajectory;
  realtime_output->ground_truth = input->ground_truth_packet;

  logRealTimeOutput(realtime_output);

  return {State::Nominal, realtime_output};
}

RegularVIFrontend::SpinReturn RegularVIFrontend::nominalSpin(
    VIFrontendInput::ConstPtr input) {
  ImageContainer::Ptr image_container = input->image_container_;
  const auto frame_id_k = input->getFrameId();
  const auto timestamp_k = input->getTimestamp();

  ImuFrontend::PimPtr pim = nullptr;
  std::optional<gtsam::NavState> imu_propogated_nav_state_k =
      tryPropogateImu(input, nav_state_km1_, pim);

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
  nav_state_km1_ = nav_state_k;

  dyno_state_.camera_trajectory.insert(frame_id_k, timestamp_k,
                                       nav_state_k.pose());

  constexpr static bool kParallelSolve = true;
  object_motion_solver_->solve(frame_k, frame_km1,
                               dyno_state_.object_trajectories, kParallelSolve);

  // construct output packet for backend
  VisionImuPacket::Ptr vision_imu_packet = std::make_shared<VisionImuPacket>();
  vision_imu_packet->frameId(frame_id_k);
  vision_imu_packet->timestamp(timestamp_k);
  vision_imu_packet->pim(pim);
  vision_imu_packet->groundTruthPacket(input->ground_truth_packet);

  fillOutputPacketWithTracks(vision_imu_packet, *frame_k, nav_state_k.pose(),
                             T_km1_k_, dyno_state_.object_trajectories);

  // we solve every frame so reset the preintegraion
  // this (of course) assumes we get IMU data between each frame
  if (imu_propogated_nav_state_k) {
    imu_frontend_.resetIntegration();
  }

  RealtimeOutput::Ptr realtime_output = std::make_shared<RealtimeOutput>();
  realtime_output->state.frame_id = frame_id_k;
  realtime_output->state.timestamp = timestamp_k;
  realtime_output->state.camera_trajectory = dyno_state_.camera_trajectory;
  realtime_output->state.object_trajectories = dyno_state_.object_trajectories;
  realtime_output->state.local_static_map =
      vision_imu_packet->staticLandmarkMeasurements();
  realtime_output->state.dynamic_map =
      vision_imu_packet->dynamicLandmarkMeasurements();
  realtime_output->ground_truth = input->ground_truth_packet;

  fillDebugImagery(realtime_output->debug_imagery, frame_k, frame_km1);

  pushImageToDisplayQueue("Tracks",
                          realtime_output->debug_imagery.tracking_image);

  // if (FLAGS_set_dense_labelled_cloud) {
  //     VLOG(30) << "Setting dense labelled cloud";
  //     utils::ChronoTimingStats labelled_clout_timer(
  //         this->moduleName() + ".dense_labelled_cloud");
  //     const cv::Mat& board_detection_mask =
  //     tracker_->getBoarderDetectionMask();
  //     realtime_output->dense_labelled_cloud =
  //         frame_k->projectToDenseCloud(&board_detection_mask);

  //     //TODO: remove dense labelled cloud from VIOutput!!
  // }

  logRealTimeOutput(realtime_output);

  if (regular_backend_output_sink_) {
    regular_backend_output_sink_(vision_imu_packet);
  }

  return {State::Nominal, realtime_output};
}

bool RegularVIFrontend::solveAndRefineEgoMotion(
    Frame::Ptr frame_k, const Frame::Ptr& frame_km1,
    const gtsam::NavState& nav_state_km1, const gtsam::Pose3& T_km1_k,
    std::optional<gtsam::NavState> propogated_nav_state_k,
    std::optional<gtsam::Rot3> R_km1_k) {
  utils::ChronoTimingStats timer(this->moduleName() + ".camera_motion");
  const auto& frontend_params = dyno_params_.frontend_params_;

  if (!frontend_params.use_ego_motion_pnp) {
    LOG(WARNING) << "Frontend param use_ego_motion_pnp set to false but only "
                    "PnP implemented";
  }

  AbsolutePoseCorrespondences correspondences;
  frame_k->getCorrespondences(correspondences, *frame_km1, KeyPointType::STATIC,
                              frame_k->landmarkWorldKeypointCorrespondance());

  // solve PnP
  Pose3SolverResult pnp_result =
      pnp_ransac_.solve3d2d(correspondences, R_km1_k);

  // sanity check
  const TrackletIds tracklets = frame_k->static_features_.collectTracklets();
  // tracklets shoudl be more (or same as) correspondances as there will be new
  // points untracked
  CHECK_GE(tracklets.size(),
           pnp_result.inliers.size() + pnp_result.outliers.size());
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
      frame_k->T_world_camera_ = nav_state_km1.pose() * T_km1_k;
      VLOG(10) << "Number usable features invalid or too few at k= "
               << frame_k->getFrameId()
               << " - using constant velocity model to propogated camera pose!";
    }

    // TODO: should almost definitely do this in future, but right now we use
    // measurements to construct a framenode in the backend so if there are no
    // measurements we get a frame_node null.... for now... make hack and set
    // all ages of inliers to 1!!! since we need n measurements in the backend
    // this will ensure that they dont get added to the
    //  optimisation problem but will get added to the map...
    // for (const auto& inlier : result.inliers) {
    // frame_k->static_features_.getByTrackletId(inlier)->age(1u);
    // }
    return false;
  } else {
    // update camera pose
    frame_k->T_world_camera_ = pnp_result.best_result;

    if (frontend_params.refine_camera_pose_with_joint_of) {
      VLOG(10) << "Refining camera pose with joint optical-flow";

      utils::ChronoTimingStats timer(this->moduleName() +
                                     ".camera_motion.refine");

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

void RegularVIFrontend::fillOutputPacketWithTracks(
    VisionImuPacket::Ptr vision_imu_packet, const Frame& frame,
    const gtsam::Pose3 X_W_k, const gtsam::Pose3& T_k_1_k,
    const MultiObjectTrajectories& object_trajectories) const {
  CHECK(vision_imu_packet);

  // assumes vision_imu_packet wil get set with the same values!
  const auto frame_id = frame.getFrameId();
  const auto timestamp = frame.getTimestamp();

  VisionImuPacket::CameraTracks camera_tracks;
  auto* static_measurements = &camera_tracks.measurements;
  fillMeasurementsFromFeatureIterator(
      static_measurements, frame.usableStaticIterator(), frame_id, timestamp,
      static_pixel_sigmas_, static_point_sigma_);

  camera_tracks.X_W_k = X_W_k;
  camera_tracks.T_k_1_k = T_k_1_k;
  vision_imu_packet->cameraTracks(camera_tracks);

  // First collect all dynamic measurements then split them by object
  // This is a bit silly
  CameraMeasurementStatusVector dynamic_measurements;
  fillMeasurementsFromFeatureIterator(
      &dynamic_measurements, frame.usableDynamicIterator(), frame_id, timestamp,
      dynamic_pixel_sigmas_, dynamic_point_sigma_);

  VisionImuPacket::ObjectTrackMap object_tracks;

  const auto object_estimates_k = object_trajectories.entriesAtFrame(frame_id);
  LOG(INFO) << "Object estimates at " << frame_id
            << " size= " << object_estimates_k.size();
  for (const auto& [object_id, object_estimate] : object_estimates_k) {
    const auto& L_W_k = object_estimate.data.pose;
    const auto& H_W_km1_k = object_estimate.data.motion;

    CHECK_EQ(H_W_km1_k.from(), frame_id - 1u);
    CHECK_EQ(H_W_km1_k.to(), frame_id);

    VisionImuPacket::ObjectTracks object_track;
    object_track.H_W_k_1_k = H_W_km1_k;
    object_track.L_W_k = L_W_k;
    object_tracks.insert2(object_id, object_track);
  }

  for (const auto& dm : dynamic_measurements) {
    const auto& object_id = dm.objectId();
    // throw out features detected on objects where the tracking failed
    if (object_tracks.exists(object_id)) {
      VisionImuPacket::ObjectTracks& object_track = object_tracks.at(object_id);
      object_track.measurements.push_back(dm);
    }
  }
  vision_imu_packet->objectTracks(object_tracks);
}

}  // namespace dyno
