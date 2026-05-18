#include "dynosam/frontend/solvers/ConsecutiveFrameObjectMotionSolver.hpp"

namespace dyno {

ConsecutiveFrameObjectMotionSolver::ConsecutiveFrameObjectMotionSolver(
    const ConsecutiveFrameObjectMotionSolverParams& params,
    const CameraParams& camera_params,
    const SharedGroundTruth& shared_ground_truth)
    : params_(params),
      pnp_ransac_solver_(params.pnp_ransac_params, camera_params),
      optical_flow_pose_solver_(params.optical_flow_solver_params),
      motion_only_refinement_solver_(params.motion_only_refinement_params),
      shared_ground_truth_(shared_ground_truth) {}

bool ConsecutiveFrameObjectMotionSolver::solveImpl(
    Frame::Ptr frame_k, Frame::Ptr frame_km1, ObjectId object_id,
    Motion3ReferenceFrame& motion_estimate) {
  utils::ChronoTimingStats timer("consecutive_motion_solver.solve_impl");

  AbsolutePoseCorrespondences dynamic_correspondences;
  frame_k->getDynamicCorrespondences(
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
  const Pose3SolverResult geometric_result =
      pnp_ransac_solver_.solve3d2d(dynamic_correspondences);

  if (geometric_result.status == TrackingStatus::VALID) {
    TrackletIds refined_inlier_tracklets = geometric_result.inliers;

    CHECK_EQ(geometric_result.inliers.size() + geometric_result.outliers.size(),
             n_matches);

    // debug only (just checking that the inlier/outliers we get from the
    // geometric rejection match the original one)
    TrackletIds extracted_all_tracklets = refined_inlier_tracklets;
    extracted_all_tracklets.insert(extracted_all_tracklets.end(),
                                   geometric_result.outliers.begin(),
                                   geometric_result.outliers.end());
    CHECK_EQ(all_tracklets.size(), extracted_all_tracklets.size());

    gtsam::Pose3 G_w = geometric_result.best_result.inverse();
    if (params_.refine_motion_with_joint_of) {
      // Use the original result as the input to the refine joint optical flow
      // function the result.best_result variable is actually equivalent to
      // ^wG^{-1} and we want to solve something in the form e(T, flow) =
      // [u,v]_{k-1} + {k-1}_flow_k - pi(T^{-1}^wm_{k-1}) so T must take the
      // point from k-1 in the world frame to the local frame at k-1 ^wG^{-1} =
      //^wX_k \: {k-1}^wH_k (which takes does this) but the error term uses the
      // inverse of T hence we must parse in the inverse of G
      auto flow_opt_result = optical_flow_pose_solver_.optimizeAndUpdate(
          frame_km1, frame_k, refined_inlier_tracklets,
          geometric_result.best_result);
      // still need to take the inverse as we get the inverse of G out
      G_w = flow_opt_result.best_result.refined_pose.inverse();
      // inliers should be a subset of the original refined inlier tracks
      refined_inlier_tracklets = flow_opt_result.inliers;
    }
    // still need to take the inverse as we get the inverse of G out
    const gtsam::Pose3 X_W_k = frame_k->getPose();
    gtsam::Pose3 H_W_km1_k = X_W_k * G_w;

    if (params_.refine_motion_with_3d) {
      VLOG(10) << "Refining object motion pose with 3D refinement";
      auto motion_refinement_result =
          motion_only_refinement_solver_.optimizeAndUpdate(
              frame_km1, frame_k, refined_inlier_tracklets, object_id,
              H_W_km1_k);

      // should be further subset
      refined_inlier_tracklets = motion_refinement_result.inliers;
      H_W_km1_k = motion_refinement_result.best_result;
    }

    motion_estimate = Motion3ReferenceFrame(
        H_W_km1_k, Motion3ReferenceFrame::Style::F2F, ReferenceFrame::GLOBAL,
        frame_km1->getFrameId(), frame_k->getFrameId());

    TrackletIds final_outliers;
    determineOutlierIds(refined_inlier_tracklets, all_tracklets,
                        final_outliers);

    frame_k->dynamic_features_.markOutliers(final_outliers);
    // sanity check that we have accounted for all initial matches
    CHECK_EQ(refined_inlier_tracklets.size() + final_outliers.size(),
             n_matches);

    if (refined_inlier_tracklets.size() < 5u) {
      return false;
    }

    return true;
  }

  return false;
}

void ConsecutiveFrameObjectMotionSolver::updateTrajectories(
    MultiObjectTrajectories& object_trajectories,
    const MotionEstimateMap& motion_estimates, Frame::Ptr frame_k,
    Frame::Ptr frame_km1) {
  gtsam::Point3Vector object_centroids_km1, object_centroids_k;

  struct Predicate {
    const ObjectId object_id;
    const Frame::Ptr frame;

    Predicate(ObjectId j, const Frame::Ptr& f) : object_id(j), frame(f) {}

    bool operator()(const Feature::Ptr& f) const {
      return Feature::IsUsable(f) && f->objectId() == object_id &&
             frame->exists(f->trackletId()) &&
             frame->isFeatureUsable(f->trackletId());
    }
  };
  using Iterator = ConstFeatureIterator<Predicate>;

  for (const auto& [object_id, motion_estimate] : motion_estimates) {
    Iterator object_points_iterator(frame_km1->dynamic_features_,
                                    Predicate(object_id, frame_k));

    gtsam::Point3 centroid_km1(0, 0, 0);
    gtsam::Point3 centroid_k(0, 0, 0);
    size_t count = 0;
    for (const auto& feature : object_points_iterator) {
      centroid_km1 += frame_km1->backProjectToCamera(feature->trackletId());
      centroid_k = frame_k->backProjectToCamera(feature->trackletId());

      count++;
    }

    centroid_km1 /= count;
    centroid_k /= count;

    centroid_km1 = frame_km1->getPose() * centroid_km1;
    centroid_k = frame_k->getPose() * centroid_k;

    object_centroids_km1.push_back(centroid_km1);
    object_centroids_k.push_back(centroid_k);
  }

  // ground truth object poses will be used if available from the shared ground
  // truth if you wish to enforce non-ground truth behaviour pass an empty
  // shared_ground_truth in the constructor
  dyno::propogateObjectTrajectory(
      object_trajectories_, motion_estimates, object_centroids_km1,
      object_centroids_k, frame_k->getFrameId(), frame_k->getTimestamp(),
      frame_km1->getTimestamp(), shared_ground_truth_.access());

  object_trajectories = object_trajectories_;
}

}  // namespace dyno
