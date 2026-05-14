#include "dynosam/frontend/solvers/HybridObjectMotionSolver.hpp"

#include <gflags/gflags.h>

#include "dynosam/frontend/vision/FeatureTrackerBase.hpp"  // just for tracklet mananger
#include "dynosam_common/PointCloudProcess.hpp"

DEFINE_int32(hybrid_motion_solver, 0,
             "Which solver to use. 0: EIF, 1: Smart Smoother, 2: Full "
             "Smoother, 3: PnP Only");

namespace dyno {

bool isWellTracked(
    const std::optional<ObjectTrackingStatus>& maybe_tracking_Status) {
  if (!maybe_tracking_Status) {
    return false;
  }
  return *maybe_tracking_Status == ObjectTrackingStatus::WellTracked;
}

// A class that just uses PnP to solve the motion but looks like a solver object
// so it integrates in with the HybridObjectMotionSmoother
class PnPOnlySolver : public HybridObjectMotionSolverImpl {
 private:
  gtsam::Pose3 L_KF_;
  // Frame Id for the reference KF
  FrameId frame_id_KF_;
  // Timestamp for the KeyMotion
  Timestamp timestamp_KF_;
  //! Frame id used for last update
  FrameId frame_id_;
  //! Timestamp used for the last update
  Timestamp timestamp_;

  FrameId frame_id_km1_;
  //! Camera pose at the reference KF
  gtsam::Pose3 X_W_KF_;

  PoseWithMotionTrajectory trajectory_;

  // Accumulated motion
  gtsam::Pose3 H_W_KF_k_;

  // latest frame-to-frame motion
  gtsam::Pose3 H_W_km1_k_;

 protected:
  PnPOnlySolver(ObjectId object_id, Camera::Ptr camera)
      : HybridObjectMotionSolverImpl(object_id, camera) {}

  void setTrajectory(const PoseWithMotionTrajectory& past_trajectory) override {
    trajectory_ = past_trajectory;
  }

 public:
  DYNO_POINTER_TYPEDEFS(PnPOnlySolver)

  static PnPOnlySolver::Ptr CreateWithInitialMotion(
      const ObjectId object_id, const gtsam::Pose3& L_KF_k, Frame::Ptr frame_k,
      const TrackletIds& tracklets) {
    auto smoother = std::shared_ptr<PnPOnlySolver>(
        new PnPOnlySolver(object_id, frame_k->getCamera()));
    smoother->resetWithNewKeyedMotion(L_KF_k, frame_k, tracklets);
    return smoother;
  }

  bool update(const gtsam::Pose3& H_w_km1_k_predict, Frame::Ptr frame,
              const TrackletIds& tracklets) override {
    H_W_km1_k_ = H_w_km1_k_predict;

    // update accumulated motion
    H_W_KF_k_ = H_W_km1_k_ * H_W_KF_k_;
    gtsam::Pose3 L_W_k = H_W_KF_k_ * L_KF_;

    frame_id_km1_ = frame_id_;
    frame_id_ = frame->getFrameId();
    timestamp_ = frame->getTimestamp();

    // frameToFrameMotionReference will be valid after H_W_km1_k_, frame_id_km1_
    // and frame_id_ are set
    trajectory_.insert(frame_id_, timestamp_,
                       {L_W_k, frameToFrameMotionReference()});
    return true;
  }

  bool resetWithNewKeyedMotion(const gtsam::Pose3& L_KF, Frame::Ptr frame,
                               const TrackletIds& tracklets) override {
    frame_id_ = frame->getFrameId();
    timestamp_ = frame->getTimestamp();
    frame_id_KF_ = frame->getFrameId();
    timestamp_KF_ = frame->getTimestamp();
    frame_id_km1_ = frame->getFrameId();

    L_KF_ = L_KF;
    H_W_KF_k_ = gtsam::Pose3::Identity();
    X_W_KF_ = frame->getPose();
    return true;
  }

  PoseWithMotionTrajectory trajectory() const override { return trajectory_; }

  PoseWithMotionTrajectory localTrajectory() const override {
    return trajectory_.range(keyFrameId());
  }

  gtsam::Pose3 keyFrameMotion() const override { return H_W_KF_k_; }

  Motion3ReferenceFrame frameToFrameMotionReference() const override {
    return Motion3ReferenceFrame(H_W_km1_k_, Motion3ReferenceFrame::Style::F2F,
                                 ReferenceFrame::GLOBAL, frame_id_km1_,
                                 frame_id_);
  }

  gtsam::Pose3 keyFramePose() const override { return L_KF_; }
  gtsam::Pose3 keyFrameCameraPose() const override { return X_W_KF_; }

  FrameId keyFrameId() const override { return frame_id_KF_; }
  FrameId frameId() const override { return frame_id_; }
  Timestamp timestamp() const override { return timestamp_; }

  gtsam::FastMap<TrackletId, gtsam::Point3> getObjectPoints() const override {
    return {};
  }
};

HybridObjectMotionSolver::HybridObjectMotionSolver(
    const HybridObjectMotionSolverParams& params,
    const CameraParams& camera_params,
    const SharedGroundTruth& shared_ground_truth)
    : params_(params),
      pnp_ransac_solver_(params.pnp_ransac_params, camera_params),
      optical_flow_pose_solver_(params.optical_flow_solver_params),
      shared_ground_truth_(shared_ground_truth) {
  VLOG(10) << "HybridObjectMotionSolver initalised with ground truth "
           << std::boolalpha << shared_ground_truth_.valid();
}

HybridObjectMotionSolver::~HybridObjectMotionSolver() {}

void HybridObjectMotionSolver::solve(Frame::Ptr frame_k, Frame::Ptr frame_km1,
                                     MultiObjectTrajectories& trajectories_out,
                                     MotionEstimateMap& motion_estimate_out,
                                     bool parallel_solve) {
  // Handle lost objects: objects in filters_ but not in current frame's
  // object_observations_
  // the objects in of object_observations_ should correspond with
  // the objects that had a successful motion estimation in the previous frame
  // and not just the set of objects that were observed.
  // The base ObjectMotionSolver::solve function should erase the observations
  // with with failed solves!
  const FrameId frame_id_k = frame_k->getFrameId();

  std::set<ObjectId> current_objects;
  for (const auto& [obj_id, _] : frame_k->getObjectObservations()) {
    current_objects.insert(obj_id);
  }

  // clear before solvers loop as we might add to pose change info for any lost
  // objects
  pose_change_info_.clear();

  keyframe_debug_image_ = frame_k->imageContainer().rgb().clone();

  for (const auto& [obj_id, solver] : solvers_) {
    if (current_objects.find(obj_id) == current_objects.end()) {
      // collect status data before marking as lost
      std::optional<ObjectTrackingStatus> maybe_tracking_state =
          object_statuses_.getStatus(obj_id);

      auto smoother =
          std::dynamic_pointer_cast<HybridObjectMotionSmoother>(solver);
      if (smoother && isWellTracked(maybe_tracking_state)) {
        // Cannot be lost immediately after making only one keyframe
        if (smoother->numKeyframes() > 1 &&
            smoother->numFramesSinceKeyframe() > 2) {
          LOG(INFO) << "Making RKF for object j=" << obj_id << ". Reason: LOST";
          // mark as post before adding pose change info so the resulting
          // PC-info object has the correct tracking status
          // object_statuses_.setStatus(obj_id, frame_id_k,
          // ObjectTrackingStatus::Lost);
          // appendPoseChangeInfo(obj_id,ObjectKeyFrameStatus::RegularKeyFrame);
        }
      }

      markObjectAsLost(obj_id, frame_k->getFrameId());
      LOG(INFO) << "Object " << obj_id << " marked as Lost at frame "
                << frame_k->getFrameId();
    }
  }

  // Call base solve
  ObjectMotionSolver::solve(frame_k, frame_km1, trajectories_out,
                            motion_estimate_out, parallel_solve);
  // for each object seen update the object status cache
  latest_object_statuses_.clear();
  for (ObjectId object_id : current_objects) {
    std::optional<ObjectTrackingStatus> maybe_status =
        object_statuses_.getStatus(object_id);

    // sanity check
    //  if object has a motion estimate it must be well tracked
    if (motion_estimate_out.exists(object_id)) {
      CHECK(maybe_status);
      CHECK_EQ(maybe_status.value(), ObjectTrackingStatus::WellTracked);
    }

    if (maybe_status) {
      latest_object_statuses_.insert2(object_id, maybe_status.value());
    }
  }
}

////////// THIS ONE IS GOOOD!!!???????/////////////////
bool HybridObjectMotionSolver::solveImpl(
    Frame::Ptr frame_k, Frame::Ptr frame_km1, ObjectId object_id,
    Motion3ReferenceFrame& motion_estimate) {
  const FrameId frame_id_k = frame_k->getFrameId();
  // Initialize or update tracking status
  bool is_new = !solverExists(object_id);
  bool is_resampled = std::find(frame_k->retracked_objects_.begin(),
                                frame_k->retracked_objects_.end(),
                                object_id) != frame_k->retracked_objects_.end();

  std::optional<ObjectTrackingStatus> maybe_previous_tracking_state =
      object_statuses_.getStatus(object_id);

  // get the corresponding feature pairs
  AbsolutePoseCorrespondences dynamic_correspondences;
  bool corr_result = frame_k->getDynamicCorrespondences(
      dynamic_correspondences, *frame_km1, object_id,
      frame_k->landmarkWorldKeypointCorrespondance());

  const size_t& n_matches = dynamic_correspondences.size();

  TrackletIds all_tracklets;
  std::transform(dynamic_correspondences.begin(), dynamic_correspondences.end(),
                 std::back_inserter(all_tracklets),
                 [](const AbsolutePoseCorrespondence& corres) {
                   return corres.tracklet_id_;
                 });
  CHECK_EQ(all_tracklets.size(), n_matches);

  utils::ChronoTimingStats update_timer("hybrid_motion_solver.solve_impl", 50);
  Pose3SolverResult geometric_result =
      pnp_ransac_solver_.solve3d2d(dynamic_correspondences);

  TrackletIds inlier_tracklets = geometric_result.inliers;
  const TrackletIds& outlier_tracklets = geometric_result.outliers;
  frame_k->dynamic_features_.markOutliers(outlier_tracklets);

  if (is_resampled) {
    LOG(INFO) << "Resampled " << info_string(frame_id_k, object_id)
              << " with matches n=" << n_matches
              << " inliers= " << inlier_tracklets.size();
  }

  // TODO: for now - this will break on small objects like dynopets!
  //  just for testing on real!
  //  originally 4!
  // TODO: new param 'min_dynamic_pnp_inliers'
  if (inlier_tracklets.size() < 10 ||
      geometric_result.status != TrackingStatus::VALID) {
    LOG(WARNING) << "Could not make initial frame for object " << object_id
                 << " as not enough inlier tracks!";
    object_statuses_.setStatus(object_id, frame_id_k,
                               ObjectTrackingStatus::PoorlyTracked);
    return false;
  }

  // To get here we must be in a well tracked state
  object_statuses_.setStatus(object_id, frame_id_k,
                             ObjectTrackingStatus::WellTracked);

  bool object_retracked = false;
  if (maybe_previous_tracking_state) {
    LOG(INFO) << "Previous tracking status "
              << to_string(maybe_previous_tracking_state.value());
    if (maybe_previous_tracking_state.value() ==
            ObjectTrackingStatus::PoorlyTracked ||
        maybe_previous_tracking_state.value() == ObjectTrackingStatus::Lost) {
      LOG(INFO) << "Previous tracking status "
                << to_string(maybe_previous_tracking_state.value())
                << " setting to retracked";
      object_retracked = true;
    }
  } else {
    CHECK(is_new);
  }

  const gtsam::Pose3 X_W_k = frame_k->getPose();
  const gtsam::Pose3 G_W = geometric_result.best_result;

  gtsam::Pose3 G_W_inv = G_W.inverse();

  if (true) {
    auto refinement_result = optical_flow_pose_solver_.optimizeAndUpdate(
        frame_km1, frame_k, inlier_tracklets, G_W);
    // still need to take the inverse as we get the inverse of G out
    // update G_W_inv
    G_W_inv = refinement_result.best_result.refined_pose.inverse();
    // inliers should be a subset of the original refined inlier tracks
    inlier_tracklets = refinement_result.inliers;

    // afrwards run ransac again
    // get the corresponding feature pairs
    AbsolutePoseCorrespondences dynamic_correspondences;
    bool corr_result = frame_k->getDynamicCorrespondences(
        dynamic_correspondences, *frame_km1, object_id,
        frame_k->landmarkWorldKeypointCorrespondance());

    const size_t& n_matches = dynamic_correspondences.size();

    TrackletIds all_tracklets;
    std::transform(dynamic_correspondences.begin(),
                   dynamic_correspondences.end(),
                   std::back_inserter(all_tracklets),
                   [](const AbsolutePoseCorrespondence& corres) {
                     return corres.tracklet_id_;
                   });
    CHECK_EQ(all_tracklets.size(), n_matches);

    utils::ChronoTimingStats update_timer("hybrid_motion_solver.solve_impl",
                                          50);
    geometric_result = pnp_ransac_solver_.solve3d2d(dynamic_correspondences);

    TrackletIds inlier_tracklets = geometric_result.inliers;
    const TrackletIds& outlier_tracklets = geometric_result.outliers;
    frame_k->dynamic_features_.markOutliers(outlier_tracklets);

    G_W_inv = geometric_result.best_result.inverse();
  }

  const gtsam::Pose3 H_W_km1_k_pnp = X_W_k * G_W_inv;

  ObjectKeyFrameStatus keyframe_status{ObjectKeyFrameStatus::NonKeyFrame};
  PoseInitalisationMethod pose_init_method{
      PoseInitalisationMethod::NonKeyFrame};

  bool requires_new_keyframe = false;
  if (is_new) {
    createAndInsertFilter(object_id, frame_km1, inlier_tracklets);
    // keyframe_status = ObjectKeyFrameStatus::AnchorKeyFrame;
    // requires_new_keyframe = true;
  } else if (object_retracked) {
    auto solver = threadSafeFilterAccess(object_id);
    LOG(WARNING) << "Object retracked: " << info_string(frame_id_k, object_id);

    // HACK FOR NOW: to ensure we dont have tracklets across poor poses (ie
    // non-well tracked) just relabal all tracklets in km1 and k
    auto& tracklet_manager = TrackletIdManager::instance();
    TrackletIds new_inlier_tracklets;
    new_inlier_tracklets.reserve(inlier_tracklets.size());
    for (TrackletId old_tracklet : inlier_tracklets) {
      Feature::Ptr feature_km1 = frame_km1->at(old_tracklet);
      Feature::Ptr feature_k = frame_k->at(old_tracklet);

      frame_km1->dynamic_features_.remove(old_tracklet);
      frame_k->dynamic_features_.remove(old_tracklet);

      auto new_tracklet_id = tracklet_manager.getAndIncrementTrackletId();
      feature_km1->trackletId(new_tracklet_id);
      feature_k->trackletId(new_tracklet_id);

      frame_km1->dynamic_features_.add(feature_km1);
      frame_k->dynamic_features_.add(feature_k);

      new_inlier_tracklets.push_back(new_tracklet_id);
    }
    inlier_tracklets = new_inlier_tracklets;

    auto new_KF_pose =
        constructObjectPose(object_id, frame_km1, inlier_tracklets);
    solver->resetWithNewKeyedMotion(new_KF_pose, frame_km1, inlier_tracklets);

    const std::lock_guard<std::mutex> lock(num_kfs_per_object_mutex_);
    num_kfs_per_object_.at(object_id) = 0;
  } else {
    // auto solver = threadSafeFilterAccess(object_id);
  }

  auto solver = threadSafeFilterAccess(object_id);
  CHECK_NOTNULL(solver);
  const bool solver_okay =
      solver->update(H_W_km1_k_pnp, frame_k, inlier_tracklets);
  auto update_time_ms = update_timer.stop();

  if (!solver_okay) {
    LOG(WARNING) << "Solver failed " << info_string(frame_id_k, object_id);
    object_statuses_.setStatus(object_id, frame_id_k,
                               ObjectTrackingStatus::PoorlyTracked);
    return false;
  }

  const auto H_W_km1_k = solver->frameToFrameMotionReference();
  motion_estimate = H_W_km1_k;

  // now see if needs new keyframe
  // important to not make new keyframe if object-retracked as we would have
  // just made a new one!
  if (maybe_previous_tracking_state &&
      maybe_previous_tracking_state.value() != ObjectTrackingStatus::New &&
      !object_retracked) {
    auto smoother =
        std::dynamic_pointer_cast<HybridObjectMotionSmoother>(solver);
    if (smoother) {
      // for OMD
      if (smoother /*&& previous_tracking_state != ObjectTrackingStatus::New*/) {
        utils::ChronoTimingStats timer("object_motion_solver.is_keyframe");
        // TODO: all logic around anchor keyframe/if reset/or juust new kf
        // should be made here!
        requires_new_keyframe =
            smoother->shouldBeKeyframe(frame_k, &keyframe_debug_image_);
        // LOG(INFO) << "object j=" << object_id << " TRACKING Q " << quality;

        // if(quality < 0.3) {
        //   requires_new_keyframe = true;
        // }
      }
    }

    if (requires_new_keyframe) {
      CHECK_EQ(solver->frameId(), frame_id_k)
          << "j=" << object_id << " k=" << solver->frameId();
      keyframe_status = ObjectKeyFrameStatus::RegularKeyFrame;

      const std::lock_guard<std::mutex> lock(num_kfs_per_object_mutex_);
      const int num_kf = num_kfs_per_object_.at(object_id);
      // ie. is first keyframe
      if (num_kf == 0) {
        LOG(INFO) << "j=" << object_id << " made anchor frame as is first KF";
        keyframe_status = ObjectKeyFrameStatus::AnchorKeyFrame;
      }
      // initalise with previous track
      pose_init_method = PoseInitalisationMethod::Previous;
    }
  }

  // double repr_error = solver->reprojectionError(frame_k, inlier_tracklets);
  // LOG(INFO) << "j=" << object_id << " repr error: " << repr_error;

  // always add motion at k not k-1?
  if (keyframe_status != ObjectKeyFrameStatus::NonKeyFrame) {
    CHECK(pose_init_method != PoseInitalisationMethod::NonKeyFrame);
    /// mmmm if we keyframe at this frame
    // then the estimated motion is from km-1 to k
    // which is NOT what we want to estimate
    // we want KF to k, (where k-1 is the new keyframe?)
    const ObjectPoseChangeInfo& info =
        appendPoseChangeInfo(object_id, keyframe_status);
    CHECK_EQ(info.H_W_KF_k.to(), info.frame_id);
    CHECK_EQ(info.H_W_KF_k.to(), frame_id_k);
    CHECK(info.isKeyFrame());

    LOG(INFO) << "Making hybrid info for j=" << object_id << " with "
              << "motion KF: " << info.H_W_KF_k.from()
              << " to: " << info.H_W_KF_k.to()
              << " with kf status: " << info.keyframe_status;

    auto smoother =
        std::dynamic_pointer_cast<HybridObjectMotionSmoother>(solver);
    if (smoother) {
      auto repr_error = smoother->reprojectionError(frame_k);
      LOG(INFO) << info_string(frame_id_k, object_id)
                << " repr error: " << repr_error;
      if (repr_error > 10) {
        CHECK_EQ(pose_init_method, PoseInitalisationMethod::Previous);
        solver->resetWithNewKeyedMotion(solver->pose(), frame_k,
                                        inlier_tracklets);
      } else {
        solver->setNewKeyframe(frame_k);
      }
    } else {
      solver->setNewKeyframe(frame_k);
    }

    const std::lock_guard<std::mutex> l(num_kfs_per_object_mutex_);
    num_kfs_per_object_.at(object_id)++;
  }

  return true;
}

ObjectPoseChangeInfo& HybridObjectMotionSolver::appendPoseChangeInfo(
    ObjectId object_id, ObjectKeyFrameStatus keyframe_status) {
  auto solver = threadSafeFilterAccess(object_id);
  CHECK_NOTNULL(solver);

  ObjectPoseChangeInfo info;
  info.frame_id = solver->frameId();
  info.H_W_KF_k = solver->keyFrameMotionReference();
  info.H_Lkf_k = solver->relativeTransform();
  info.L_W_KF = solver->keyFramePose();
  info.L_W_k = solver->pose();
  info.X_W_KF = solver->keyFrameCameraPose();
  info.keyframe_status = keyframe_status;

  // should either be well-tracked or lost
  std::optional<ObjectTrackingStatus> maybe_tracking_state =
      object_statuses_.getStatus(object_id);
  CHECK(maybe_tracking_state);
  info.tracking_status = *maybe_tracking_state;

  CHECK(getObjectStructureinL(object_id, info.initial_object_points));

  const std::lock_guard<std::mutex> lock(pose_change_info_mutex_);
  pose_change_info_.insert2(object_id, info);

  return pose_change_info_.at(object_id);
}

void HybridObjectMotionSolver::markObjectAsLost(ObjectId object_id,
                                                FrameId frame_id) {
  object_statuses_.setStatus(object_id, frame_id, ObjectTrackingStatus::Lost);

  {
    const std::lock_guard<std::mutex> lock(num_kfs_per_object_mutex_);
    num_kfs_per_object_[object_id] = 0;
  }

  {
    const std::lock_guard<std::mutex> lock(solvers_mutex_);
    // set past trajectory before erasing
    past_trajectories_[object_id] = solvers_.at(object_id)->trajectory();

    solvers_.erase(object_id);
  }
}

HybridObjectMotionSolverImpl::Ptr
HybridObjectMotionSolver::threadSafeFilterAccess(ObjectId object_id) const {
  const std::lock_guard<std::mutex> lock(solvers_mutex_);
  if (!solvers_.exists(object_id)) {
    return nullptr;
  }

  return solvers_.at(object_id);
}

bool HybridObjectMotionSolver::getObjectStructureinL(
    ObjectId object_id, StatusLandmarkVector& object_points) const {
  if (!solverExists(object_id)) {
    return false;
  }
  auto filter = threadSafeFilterAccess(object_id);

  const auto fixed_points = filter->getObjectPoints();

  object_points.reserve(object_points.size() + fixed_points.size());
  for (const auto& [tracklet_id, m_L] : fixed_points) {
    object_points.push_back(LandmarkStatus::Dynamic(
        // currently no covariance!
        Point3Measurement(m_L), LandmarkStatus::MeaninglessFrame, NaN,
        tracklet_id, object_id, ReferenceFrame::OBJECT));
  }

  return true;
}

bool HybridObjectMotionSolver::getObjectStructureinW(
    ObjectId object_id, StatusLandmarkVector& object_points) const {
  if (!solverExists(object_id)) {
    return false;
  }
  auto filter = threadSafeFilterAccess(object_id);

  const auto fixed_points = filter->getObjectPoints();
  const auto frame_id_k = filter->frameId();
  const auto timestamp = filter->timestamp();
  const auto L_W_k = filter->pose();

  object_points.reserve(object_points.size() + fixed_points.size());
  for (const auto& [tracklet_id, m_L] : fixed_points) {
    const auto m_W_k = L_W_k * m_L;
    object_points.push_back(LandmarkStatus::Dynamic(
        // currently no covariance!
        Point3Measurement(m_W_k), frame_id_k, timestamp, tracklet_id, object_id,
        ReferenceFrame::GLOBAL));
  }

  return true;
}

void HybridObjectMotionSolver::receiveUpdate(
    const PoseChangeUpdateComplete& update_info) {
  LOG(INFO) << "Recieved point update!";

  const std::lock_guard<std::mutex> lock(solvers_mutex_);
  // apply to all solver and let the solver decide if it has an internal update
  for (auto [_, solver] : solvers_) {
    solver->receiveUpdate(update_info);
  }
}

gtsam::Pose3 HybridObjectMotionSolver::constructObjectPose(
    const ObjectId object_id, const Frame::Ptr frame,
    const TrackletIds& tracklets) const {
  // always try ground truth first such that if it is provided we assume that
  // we want to initalise with ground truth
  // ground truth is only given if the shared ground truth is valid
  auto gt_pose = objectPoseFromGroundTruth(object_id, frame);
  if (gt_pose) {
    return gt_pose.value();
  }

  return objectPoseFromCentroid(object_id, frame, tracklets);
}

std::optional<gtsam::Pose3> HybridObjectMotionSolver::objectPoseFromGroundTruth(
    ObjectId object_id, const Frame::Ptr frame) const {
  // is provided ground truth is valid we assume there should be some ground
  // truth for this object
  std::optional<GroundTruthPacketMap> ground_truth =
      shared_ground_truth_.access();

  if (!ground_truth) {
    return {};
  }

  const FrameId frame_id = frame->getFrameId();

  if (ground_truth->exists(frame_id)) {
    const GroundTruthInputPacket& packet = ground_truth->at(frame_id);

    ObjectPoseGT object_ground_truth;
    if (packet.getObject(object_id, object_ground_truth)) {
      return object_ground_truth.L_world_;
    }
  }

  return {};
}

gtsam::Pose3 HybridObjectMotionSolver::objectPoseFromCentroid(
    const ObjectId object_id, const Frame::Ptr frame,
    const TrackletIds& tracklets) const {
  // important to initliase with zero values (otherwise nan's!)
  gtsam::Point3 object_position(0, 0, 0);
  size_t count = 0;
  for (TrackletId tracklet : tracklets) {
    const Feature::Ptr feature = frame->at(tracklet);
    CHECK_NOTNULL(feature);
    CHECK_EQ(feature->objectId(), object_id);

    gtsam::Point3 lmk = frame->backProjectToCamera(feature->trackletId());
    object_position += lmk;

    count++;
  }

  // TODO: filtering?
  object_position /= count;
  object_position = frame->getPose() * object_position;
  return gtsam::Pose3(gtsam::Rot3::Identity(), object_position);
}

HybridObjectMotionSolverImpl::Ptr
HybridObjectMotionSolver::createAndInsertFilter(ObjectId object_id,
                                                Frame::Ptr frame,
                                                const TrackletIds& tracklets) {
  gtsam::Pose3 keyframe_pose = constructObjectPose(object_id, frame, tracklets);

  HybridObjectMotionSolverImpl::Ptr solver = nullptr;
  if (FLAGS_hybrid_motion_solver == 0) {
    gtsam::Matrix33 R = gtsam::Matrix33::Identity() * 1.0;
    // Initial State Covariance P (6x6)
    gtsam::Matrix66 P = gtsam::Matrix66::Identity() * 0.3;
    // // Process Model noise (6x6)
    // gtsam::Matrix66 Q = gtsam::Matrix66::Identity() * 0.2;
    gtsam::Vector6 q_diag;
    q_diag << 1e-2, 1e-4, 1e-4,  // Rotation noise (std approx 0.01 rad)
        1e-3, 1e-3, 1e-3;        // Translation noise (std approx 0.03 m)
    gtsam::Matrix66 Q = q_diag.asDiagonal();

    constexpr static double kHuberKFilter = 0.05;
    solver = std::make_shared<FullHybridObjectMotionSRIF>(
        object_id, gtsam::Pose3::Identity(), keyframe_pose, frame->getFrameId(),
        frame->getTimestamp(), P, Q, R, frame->getCamera(), kHuberKFilter);
  } else if (FLAGS_hybrid_motion_solver == 1) {
    // run as smoother with smart factors
    solver = HybridObjectMotionSmoother::CreateWithInitialMotion<
        HybridObjectMotionSmartSmoother>(object_id, 15, keyframe_pose, frame,
                                         tracklets);
  } else if (FLAGS_hybrid_motion_solver == 2) {
    // run as full smoother
    solver = HybridObjectMotionSmoother::CreateWithInitialMotion<
        HybridObjectMotionFullSmoother>(object_id, 40, keyframe_pose, frame,
                                        tracklets);
  } else if (FLAGS_hybrid_motion_solver == 3) {
    solver = PnPOnlySolver::CreateWithInitialMotion(object_id, keyframe_pose,
                                                    frame, tracklets);
  } else if (FLAGS_hybrid_motion_solver == 4) {
    solver = HybridObjectMotionSmoother::CreateWithInitialMotion<
        HybridObjectMotionOnlySmoother>(object_id, 10, keyframe_pose, frame,
                                        tracklets);
  }
  CHECK_NOTNULL(solver);

  {
    const std::lock_guard<std::mutex> lock(solvers_mutex_);

    if (past_trajectories_.exists(object_id)) {
      // if we've seen this object before, set its full trajectory
      // this is mostly so that we can recover its full trajectory
      // in updateTrajectories
      solver->setTrajectory(past_trajectories_.at(object_id));
    }

    solvers_.insert2(object_id, solver);
  }

  {
    const std::lock_guard<std::mutex> lock(num_kfs_per_object_mutex_);
    num_kfs_per_object_.insert2(object_id, 0);
  }

  LOG(INFO) << "Created new filter for object " << object_id << " at frame "
            << frame->getFrameId();

  return solver;
}

void HybridObjectMotionSolver::updateTrajectories(
    MultiObjectTrajectories& object_trajectories,
    const MotionEstimateMap& motion_estimates, Frame::Ptr /*frame_k*/,
    Frame::Ptr /*frame_km1*/) {
  for (const auto& [object_id, _] : motion_estimates) {
    CHECK(solvers_.exists(object_id));

    PoseWithMotionTrajectory trajectory = solvers_.at(object_id)->trajectory();
    object_trajectories.insert2(object_id, trajectory);
  }
}

}  // namespace dyno
