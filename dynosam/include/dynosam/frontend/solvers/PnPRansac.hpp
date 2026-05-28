#pragma once

#include <Eigen/Core>
#include <opengv/sac/Ransac.hpp>

#include "dynosam/frontend/Frontend-Definitions.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam/frontend/vision/Vision-Definitions.hpp"
#include "dynosam/frontend/vision/VisionTools.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/GtsamUtils.hpp"
#include "dynosam_sensors/CameraParams.hpp"

namespace dyno {

struct RansacProblemParams {
  double threshold = 1.0;
  double ransac_iterations = 500;
  double ransac_probability = 0.995;
  bool do_nonlinear_optimization = false;
};

template <class SampleConsensusProblem>
bool runRansac(
    std::shared_ptr<SampleConsensusProblem> sample_consensus_problem_ptr,
    const double& threshold, const int& max_iterations,
    const double& probability, const bool& do_nonlinear_optimization,
    gtsam::Pose3& best_pose, std::vector<int>& inliers) {
  CHECK(sample_consensus_problem_ptr);
  inliers.clear();

  //! Create ransac
  opengv::sac::Ransac<SampleConsensusProblem> ransac(max_iterations, threshold,
                                                     probability);

  //! Setup ransac
  ransac.sac_model_ = sample_consensus_problem_ptr;

  //! Run ransac
  bool success = ransac.computeModel(0);

  if (success) {
    if (ransac.iterations_ >= max_iterations && ransac.inliers_.empty()) {
      success = false;
      best_pose = gtsam::Pose3();
      inliers = {};
    } else {
      best_pose = utils::openGvTfToGtsamPose3(ransac.model_coefficients_);
      inliers = ransac.inliers_;

      if (do_nonlinear_optimization) {
        opengv::transformation_t optimized_pose;
        sample_consensus_problem_ptr->optimizeModelCoefficients(
            inliers, ransac.model_coefficients_, optimized_pose);
        best_pose = Eigen::MatrixXd(optimized_pose);
      }
    }
  } else {
    CHECK(ransac.inliers_.empty());
    best_pose = gtsam::Pose3();
    inliers.clear();
  }

  return success;
}

template <class SampleConsensusProblem>
bool runRansac(
    std::shared_ptr<SampleConsensusProblem> sample_consensus_problem_ptr,
    const RansacProblemParams& params, gtsam::Pose3& best_pose,
    std::vector<int>& inliers) {
  return runRansac<SampleConsensusProblem>(
      sample_consensus_problem_ptr, params.threshold, params.ransac_iterations,
      params.ransac_probability, params.do_nonlinear_optimization, best_pose,
      inliers);
}

template <typename T>
struct SolverResult {
  T best_result;
  TrackletIds inliers;
  TrackletIds outliers;
  TrackingStatus status;

  std::optional<double> error_before{};
  std::optional<double> error_after{};
};

using Pose3SolverResult = SolverResult<gtsam::Pose3>;

// TODO: config!
struct PnPRansacSolverParams {
  bool ransac_randomize = true;

  //! Mono (2d2d) related params
  // if mono pipeline is used AND an additional inertial sensor is provided
  // (e.g IMU) then 2d point ransac will be used to estimate the camera pose
  bool ransac_use_2point_mono = false;
  // https://github.com/laurentkneip/opengv/issues/121
  double ransac_threshold_mono =
      2.0 * (1.0 - cos(atan(sqrt(2.0) * 0.5 / 800.0)));
  bool optimize_2d2d_pose_from_inliers = false;

  //! equivalent to reprojection error in pixels
  double ransac_threshold_pnp = 1.0;
  //! Use 3D-2D tracking to remove outliers
  bool optimize_3d2d_pose_from_inliers = false;

  //! 3D-3D options
  double ransac_threshold_stereo = 0.001;
  //! Use 3D-3D tracking to remove outliers
  bool optimize_3d3d_pose_from_inliers = false;

  //! Generic ransac params
  double ransac_iterations = 500;
  double ransac_probability = 0.995;
};

class PnPRansacSolver {
 public:
  PnPRansacSolver(const PnPRansacSolverParams& pnp_ransac_params,
                  const CameraParams& camera_params);

  Pose3SolverResult solve2d2d(
      const KeypointKeypointCorrespondences& correspondences,
      std::optional<gtsam::Rot3> R_curr_ref = {});

  Pose3SolverResult solve3d2d(
      const LandmarkKeypointCorrespondences& correspondences,
      std::optional<gtsam::Rot3> R_curr_ref = {});

 protected:
  /**
   * @brief Helper function to build a set of Tracklet inliers and outliers
   * from the original set of tracklet ids using the RANSAC inlier set provided
   * by OpenGV
   *
   * @tparam Ref
   * @tparam Curr
   * @param inliers
   * @param outliers
   * @param correspondences
   * @param ransac_inliers
   * @param tracklets
   */
  template <typename Ref, typename Curr>
  static void constructTrackletInliers(
      TrackletIds& inliers, TrackletIds& outliers,
      const GenericCorrespondences<Ref, Curr>& correspondences,
      const std::vector<int>& ransac_inliers, const TrackletIds& tracklets) {
    CHECK_EQ(correspondences.size(), tracklets.size());
    CHECK(ransac_inliers.size() <= correspondences.size());

    // pre-allocate
    inliers.reserve(ransac_inliers.size());
    outliers.reserve(correspondences.size() - ransac_inliers.size());

    for (int inlier_idx : ransac_inliers) {
      const auto& corres = correspondences.at(inlier_idx);
      inliers.push_back(corres.tracklet_id_);
    }

    determineOutlierIds(inliers, tracklets, outliers);
    CHECK_EQ((inliers.size() + outliers.size()), tracklets.size());
    CHECK_EQ(inliers.size(), ransac_inliers.size());
  }

 protected:
  const PnPRansacSolverParams pnp_ransac_params_;
  const CameraParams camera_params_;
  //! Inverse of camera calibration matrix (cached)
  const gtsam::Matrix33 K_inv_;

  //! Pre-computed threshold used for the 3d2d solve
  double ransac_threshold_3d2d_;
};

}  // namespace dyno
