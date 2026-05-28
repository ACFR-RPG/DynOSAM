#include "dynosam/frontend/solvers/PnPRansac.hpp"

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/TranslationOnlySacProblem.hpp>

#include "dynosam_common/utils/TimingStats.hpp"

namespace dyno {

using AbsolutePoseProblem =
    opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem;
using AbsolutePoseAdaptor = opengv::absolute_pose::CentralAbsoluteAdapter;

// Mono (2d2d) using 5-point ransac
using RelativePoseProblem =
    opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem;

// Mono (2d2d, with given rotation) MonoTranslationOnly:
// TranslationOnlySacProblem 2-point ransac
using RelativePoseProblemGivenRot =
    opengv::sac_problems::relative_pose::TranslationOnlySacProblem;
using RelativePoseAdaptor = opengv::relative_pose::CentralRelativeAdapter;

PnPRansacSolver::PnPRansacSolver(const PnPRansacSolverParams& pnp_ransac_params,
                                 const CameraParams& camera_params)
    : pnp_ransac_params_(pnp_ransac_params),
      camera_params_(camera_params),
      K_inv_(camera_params.getCameraMatrixEigen().inverse()) {
  // pre-compute thresholds
  const double reprojection_error = pnp_ransac_params_.ransac_threshold_pnp;
  const double avg_focal_length =
      0.5 * static_cast<double>(camera_params_.fx() + camera_params_.fy());
  ransac_threshold_3d2d_ =
      1.0 - std::cos(std::atan(std::sqrt(2.0) * reprojection_error /
                               avg_focal_length));
}

Pose3SolverResult PnPRansacSolver::solve2d2d(
    const KeypointKeypointCorrespondences& correspondences,
    std::optional<gtsam::Rot3> R_curr_ref) {
  utils::ChronoTimingStats timer("pnp_ransac.solve2d2d");
  Pose3SolverResult result;
  const size_t& n_matches = correspondences.size();

  if (n_matches < 5u) {
    result.status = TrackingStatus::FEW_MATCHES;
    return result;
  }

  TrackletIds tracklets;
  tracklets.reserve(n_matches);
  // NOTE: currently without distortion! the correspondences should be made into
  // bearing vector elsewhere!
  BearingVectors ref_bearing_vectors, cur_bearing_vectors;
  ref_bearing_vectors.reserve(n_matches);
  cur_bearing_vectors.reserve(n_matches);
  for (size_t i = 0u; i < n_matches; i++) {
    const auto& corres = correspondences.at(i);
    const Keypoint& ref_kp = corres.ref_;
    const Keypoint& cur_kp = corres.cur_;

    gtsam::Vector3 ref_versor =
        (K_inv_ * gtsam::Vector3(ref_kp(0), ref_kp(1), 1.0));
    gtsam::Vector3 cur_versor =
        (K_inv_ * gtsam::Vector3(cur_kp(0), cur_kp(1), 1.0));

    ref_versor = ref_versor.normalized();
    cur_versor = cur_versor.normalized();

    ref_bearing_vectors.push_back(ref_versor);
    cur_bearing_vectors.push_back(cur_versor);

    tracklets.push_back(corres.tracklet_id_);
  }

  RelativePoseAdaptor adapter(ref_bearing_vectors, cur_bearing_vectors);

  const bool use_2point_mono =
      pnp_ransac_params_.ransac_use_2point_mono && R_curr_ref;
  if (use_2point_mono) {
    adapter.setR12((*R_curr_ref).matrix());
  }

  gtsam::Pose3 best_result;
  std::vector<int> ransac_inliers;
  bool success = false;
  if (use_2point_mono) {
    success = runRansac<RelativePoseProblemGivenRot>(
        std::make_shared<RelativePoseProblemGivenRot>(
            adapter, pnp_ransac_params_.ransac_randomize),
        pnp_ransac_params_.ransac_threshold_mono,
        pnp_ransac_params_.ransac_iterations,
        pnp_ransac_params_.ransac_probability,
        pnp_ransac_params_.optimize_2d2d_pose_from_inliers, best_result,
        ransac_inliers);
  } else {
    success = runRansac<RelativePoseProblem>(
        std::make_shared<RelativePoseProblem>(
            adapter, RelativePoseProblem::NISTER,
            pnp_ransac_params_.ransac_randomize),
        pnp_ransac_params_.ransac_threshold_mono,
        pnp_ransac_params_.ransac_iterations,
        pnp_ransac_params_.ransac_probability,
        pnp_ransac_params_.optimize_2d2d_pose_from_inliers, best_result,
        ransac_inliers);
  }

  if (!success) {
    result.status = TrackingStatus::INVALID;
  } else {
    constructTrackletInliers(result.inliers, result.outliers, correspondences,
                             ransac_inliers, tracklets);
    // NOTE: 2-point always returns the identity rotation, hence we have to
    // substitute it:
    if (use_2point_mono) {
      CHECK(R_curr_ref->equals(best_result.rotation()));
    }
    result.status = TrackingStatus::VALID;
    result.best_result = best_result;
  }

  return result;
}

Pose3SolverResult PnPRansacSolver::solve3d2d(
    const LandmarkKeypointCorrespondences& correspondences,
    std::optional<gtsam::Rot3> R_curr_ref) {
  utils::ChronoTimingStats timer("pnp_ransac.solve3d2d");

  Pose3SolverResult result;
  const size_t& n_matches = correspondences.size();

  if (n_matches < 5u) {
    result.status = TrackingStatus::FEW_MATCHES;
    VLOG(40) << "3D2D tracking failed as there are to few matches" << n_matches;
    return result;
  }

  TrackletIds tracklets, inliers, outliers;
  tracklets.reserve(n_matches);
  // NOTE: currently without distortion! the correspondences should be made into
  // bearing vector elsewhere!
  BearingVectors bearing_vectors;
  bearing_vectors.reserve(n_matches);

  Landmarks points;
  points.reserve(n_matches);
  for (size_t i = 0u; i < n_matches; i++) {
    const LandmarkKeypointCorrespondence& corres = correspondences.at(i);
    const Keypoint& kp = corres.cur_;
    // make Bearing vector
    gtsam::Vector3 versor = (K_inv_ * gtsam::Vector3(kp(0), kp(1), 1.0));
    versor = versor.normalized();
    bearing_vectors.push_back(versor);

    points.push_back(corres.ref_);
    tracklets.push_back(corres.tracklet_id_);
  }

  VLOG(100) << "Collected " << tracklets.size() << " initial correspondances";

  AbsolutePoseAdaptor adapter(bearing_vectors, points);

  if (R_curr_ref) {
    adapter.setR(R_curr_ref->matrix());
  }

  gtsam::Pose3 best_result;
  std::vector<int> ransac_inliers;

  bool success = runRansac<AbsolutePoseProblem>(
      std::make_shared<AbsolutePoseProblem>(adapter,
                                            AbsolutePoseProblem::KNEIP),
      ransac_threshold_3d2d_, pnp_ransac_params_.ransac_iterations,
      pnp_ransac_params_.ransac_probability,
      pnp_ransac_params_.optimize_3d2d_pose_from_inliers, best_result,
      ransac_inliers);

  constructTrackletInliers(result.inliers, result.outliers, correspondences,
                           ransac_inliers, tracklets);

  if (success) {
    if (result.inliers.size() < 5u) {
      result.status = TrackingStatus::FEW_MATCHES;
    } else {
      result.status = TrackingStatus::VALID;
      result.best_result = best_result;
    }

  } else {
    result.status = TrackingStatus::INVALID;
  }

  return result;
}

}  // namespace dyno
