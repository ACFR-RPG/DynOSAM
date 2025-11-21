/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a
 copy
 *   of this software and associated documentation files (the "Software"), to
 deal
 *   in the Software without restriction, including without limitation the
 rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in
 all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE
 *   SOFTWARE.
 */
#pragma once

#include <glog/logging.h>
#include <gtsam/geometry/StereoPoint2.h>
#include <gtsam/linear/LossFunctions.h>
#include <gtsam/nonlinear/ISAM2Params.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/point_cloud/PointCloudAdapter.hpp>
#include <opengv/point_cloud/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/sac_problems/point_cloud/PointCloudSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/TranslationOnlySacProblem.hpp>
#include <optional>

#include "dynosam/backend/BackendDefinitions.hpp"  //for formulation hooks
#include "dynosam/frontend/Frontend-Definitions.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam/frontend/vision/Vision-Definitions.hpp"
#include "dynosam/frontend/vision/VisionTools.hpp"
#include "dynosam_common/Types.hpp"

// PnP (3d2d)
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

// Stereo (3d3d)
// Arun's problem (3-point ransac)
using Problem3d3d = opengv::sac_problems::point_cloud::PointCloudSacProblem;
using Adapter3d3d = opengv::point_cloud::PointCloudAdapter;

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
    gtsam::Pose3& best_pose, std::vector<int>& inliers);

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

class EssentialDecompositionResult;  // forward declare

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
using Motion3SolverResult = SolverResult<Motion3ReferenceFrame>;

/**
 * @brief Joinly refines optical flow with with the given pose
 * using the error term:
 * e = [u,v]_{k_1} + f_{k-1, k} - \pi(X^{-1} \: m_k)
 * where f is flow, [u,v]_{k-1} is the observed keypoint at k-1, X is the pose
 * and m_k is the back-projected keypoint at k.
 *
 * The parsed tracklets are the set of correspondances with which to build the
 * optimisation problem and the refined inliers will be a subset of these
 * tracklets. THe number of refined flows should be the number of refined
 * inliers and be a 1-to-1 match
 *
 */
class OpticalFlowAndPoseOptimizer {
 public:
  struct Params {
    double flow_sigma{10.0};
    double flow_prior_sigma{3.33};
    double k_huber{0.001};
    bool outlier_reject{true};
    // When true, this indicates that the optical flow images go from k to k+1
    // (rather than k-1 to k, when false) this left over from some original
    // implementations. This param is used when updated the frames after
    // optimization
    bool flow_is_future{true};
  };

  struct ResultType {
    gtsam::Pose3 refined_pose;
    gtsam::Point2Vector refined_flows;
    ObjectId object_id;
  };
  using Result = SolverResult<ResultType>;

  OpticalFlowAndPoseOptimizer(const Params& params) : params_(params) {}

  /**
   * @brief Builds the factor-graph problem using the set of specificed
   * correspondences (tracklets) in frame k-1 and k and the initial pose.
   *
   * The optimisation joinly refines optical flow with with the given pose
   * using the error term:
   * e = [u,v]_{k_1} + f_{k-1, k} - \pi(X^{-1} \: m_k)
   * where f is flow, [u,v]_{k-1} is the observed keypoint at k-1, X is the pose
   * and m_k is the back-projected keypoint at k.
   *
   * The parsed tracklets are the set of correspondances with which to build the
   * optimisation problem and the refined inliers will be a subset of these
   * tracklets. THe number of refined flows should be the number of refined
   * inliers and be a 1-to-1 match
   *
   * This is agnostic to if the problem is solving for a motion or a pose so the
   * user must make sure the initial pose is in the right form.
   *
   * @tparam CALIBRATION
   * @param frame_k_1
   * @param frame_k
   * @param tracklets
   * @param initial_pose
   * @return Result
   */
  template <typename CALIBRATION>
  Result optimize(const Frame::Ptr frame_k_1, const Frame::Ptr frame_k,
                  const TrackletIds& tracklets,
                  const gtsam::Pose3& initial_pose) const;

  /**
   * @brief Builds the factor-graph problem using the set of specificed
   * correspondences (tracklets) in frame k-1 and k and the initial pose. Unlike
   * the optimize only version this also update the features within the frames
   * as outliers after optimisation. It will also update the feature data (depth
   * keypoint etc...) with the refined flows.
   *
   * It will NOT update the frame with the result pose as this could be any
   * pose.
   *
   * @tparam CALIBRATION
   * @param frame_k_1
   * @param frame_k
   * @param tracklets
   * @param initial_pose
   * @return Result
   */
  template <typename CALIBRATION>
  Result optimizeAndUpdate(Frame::Ptr frame_k_1, Frame::Ptr frame_k,
                           const TrackletIds& tracklets,
                           const gtsam::Pose3& initial_pose) const;

 private:
  void updateFrameOutliersWithResult(const Result& result, Frame::Ptr frame_k_1,
                                     Frame::Ptr frame_k) const;

 private:
  Params params_;
};

/**
 * @brief Jointly refined the motion of an object using the 3D-motion-residual.
 *
 */
class MotionOnlyRefinementOptimizer {
 public:
  struct Params {
    double landmark_motion_sigma{0.001};
    double projection_sigma{2.0};
    double k_huber{0.0001};
    bool outlier_reject{true};
  };

  MotionOnlyRefinementOptimizer(const Params& params) : params_(params) {}
  enum RefinementSolver { ProjectionError, PointError };

  template <typename CALIBRATION>
  Pose3SolverResult optimize(
      const Frame::Ptr frame_k_1, const Frame::Ptr frame_k,
      const TrackletIds& tracklets, const ObjectId object_id,
      const gtsam::Pose3& initial_motion,
      const RefinementSolver& solver = RefinementSolver::ProjectionError) const;

  template <typename CALIBRATION>
  Pose3SolverResult optimizeAndUpdate(
      Frame::Ptr frame_k_1, Frame::Ptr frame_k, const TrackletIds& tracklets,
      const ObjectId object_id, const gtsam::Pose3& initial_motion,
      const RefinementSolver& solver = RefinementSolver::ProjectionError) const;

 private:
  Params params_;
};

// TODO: eventually when we have a map, should we look up these values from
// there (the optimized versions, not the tracked ones?)
class EgoMotionSolver {
 public:
  struct Params {
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

  EgoMotionSolver(const Params& params, const CameraParams& camera_params);
  virtual ~EgoMotionSolver() = default;

  /**
   * @brief Runs 2d-2d PnP with optional Rotation (ie. from IMU)
   *
   * @param frame_k_1
   * @param frame_k
   * @param R_curr_ref Should rotate from ref -> curr
   * @return Pose3SolverResult
   */
  Pose3SolverResult geometricOutlierRejection2d2d(
      Frame::Ptr frame_k_1, Frame::Ptr frame_k,
      std::optional<gtsam::Rot3> R_curr_ref = {});

  /**
   * @brief Runs 3d-2d PnP with optional Rotation (i.e from IMU)
   *
   * @param frame_k_1
   * @param frame_k
   * @param R_curr_ref
   * @return Pose3SolverResult
   */
  Pose3SolverResult geometricOutlierRejection3d2d(
      Frame::Ptr frame_k_1, Frame::Ptr frame_k,
      std::optional<gtsam::Rot3> R_curr_ref = {});

  Pose3SolverResult geometricOutlierRejection3d2d(
      const AbsolutePoseCorrespondences& correspondences,
      std::optional<gtsam::Rot3> R_curr_ref = {});

  Pose3SolverResult geometricOutlierRejection3d3d(
      Frame::Ptr frame_k_1, Frame::Ptr frame_k,
      std::optional<gtsam::Rot3> R_curr_ref = {});

  Pose3SolverResult geometricOutlierRejection3d3d(
      const PointCloudCorrespondences& correspondences,
      std::optional<gtsam::Rot3> R_curr_ref = {});

 protected:
  template <typename Ref, typename Curr>
  void constructTrackletInliers(
      TrackletIds& inliers, TrackletIds& outliers,
      const GenericCorrespondences<Ref, Curr>& correspondences,
      const std::vector<int>& ransac_inliers, const TrackletIds tracklets) {
    CHECK_EQ(correspondences.size(), tracklets.size());
    CHECK(ransac_inliers.size() <= correspondences.size());
    for (int inlier_idx : ransac_inliers) {
      const auto& corres = correspondences.at(inlier_idx);
      inliers.push_back(corres.tracklet_id_);
    }

    determineOutlierIds(inliers, tracklets, outliers);
    CHECK_EQ((inliers.size() + outliers.size()), tracklets.size());
    CHECK_EQ(inliers.size(), ransac_inliers.size());
  }

 protected:
  const Params params_;
  const CameraParams camera_params_;
};

class ObjectMotionSolver {
 public:
  DYNO_POINTER_TYPEDEFS(ObjectMotionSolver)

  ObjectMotionSolver() = default;
  virtual ~ObjectMotionSolver() = default;

  using Result = std::pair<ObjectMotionMap, ObjectPoseMap>;

  virtual Result solve(Frame::Ptr frame_k, Frame::Ptr frame_k_1) = 0;

 protected:
};

class ObjectMotionSovlerF2F : public ObjectMotionSolver,
                              protected EgoMotionSolver {
 public:
  DYNO_POINTER_TYPEDEFS(ObjectMotionSovlerF2F)

  //! Result from solve including the object motions and poses
  using ObjectMotionSolver::Result;

  struct Params : public EgoMotionSolver::Params {
    bool refine_motion_with_joint_of = true;
    bool refine_motion_with_3d = true;

    //! Hook to get the ground truth packet. Used when collecting the object
    //! poses (on conditional) to ensure the first pose matches with the gt when
    //! evaluation
    FormulationHooks::GroundTruthPacketsRequest ground_truth_packets_request;

    OpticalFlowAndPoseOptimizer::Params joint_of_params =
        OpticalFlowAndPoseOptimizer::Params();
    MotionOnlyRefinementOptimizer::Params object_motion_refinement_params =
        MotionOnlyRefinementOptimizer::Params();
  };

  ObjectMotionSovlerF2F(const Params& params,
                        const CameraParams& camera_params);

  Result solve(Frame::Ptr frame_k, Frame::Ptr frame_k_1) override;

  Motion3SolverResult geometricOutlierRejection3d2d(
      Frame::Ptr frame_k_1, Frame::Ptr frame_k, const gtsam::Pose3& T_world_k,
      ObjectId object_id);

  const ObjectMotionSovlerF2F::Params& objectMotionParams() const {
    return object_motion_params;
  }

 protected:
  virtual bool solveImpl(Frame::Ptr frame_k, Frame::Ptr frame_k_1,
                         ObjectId object_id,
                         MotionEstimateMap& motion_estimates);

 private:
  const ObjectPoseMap& updatePoses(MotionEstimateMap& motion_estimates,
                                   Frame::Ptr frame_k, Frame::Ptr frame_k_1);

  const ObjectMotionMap& updateMotions(MotionEstimateMap& motion_estimates,
                                       Frame::Ptr frame_k,
                                       Frame::Ptr frame_k_1);

 private:
  //! All object poses (from k to K) and updated by updatePoses at each
  //! iteration of sovle
  ObjectPoseMap object_poses_;
  //! All object motions (from k to K) and updated by updatedMotions at each
  //! iteration of sovle
  ObjectMotionMap object_motions_;

 protected:
  const ObjectMotionSovlerF2F::Params object_motion_params;
};

namespace testing {
using namespace Eigen;
using namespace std;
using namespace gtsam;

// Define common matrix types for EKF
using StateCovariance = gtsam::Matrix66;        // P (6x6)
using PerturbationVector = gtsam::Vector6;      // delta_x (6x1)
using MeasurementCovariance = gtsam::Matrix22;  // R (2x2)

using MeasurementCovarianceStereo = gtsam::Matrix33;

// // Define common matrix types for EKF
// using StateCovariance = Matrix6d;        // P (6x6)
// using PerturbationVector = Vector6d;     // delta_w (6x1)
// using MeasurementCovariance = Matrix2d;  // R (2x2)

class ExtendedKalmanFilterGTSAM {
 private:
  Pose3 H_w_;
  StateCovariance P_;            // State Covariance Matrix
  Cal3_S2::shared_ptr K_gtsam_;  // GTSAM Camera Intrinsic (shared pointer)
  MeasurementCovariance
      R_;  // Individual Measurement Noise Covariance (assumed constant)
  StateCovariance Q_;  // Process Noise Covariance (for prediction step)

 public:
  // Constructor initializes state, covariance, and GTSAM camera model
  ExtendedKalmanFilterGTSAM(const Pose3& initial_pose,
                            const StateCovariance& initial_P,
                            const Matrix3d& K_eigen,
                            const MeasurementCovariance& R)
      : H_w_(initial_pose), P_(initial_P), R_(R) {
    // Extract intrinsic parameters from Eigen matrix K
    double fx = K_eigen(0, 0);
    double fy = K_eigen(1, 1);
    double s = K_eigen(0, 1);  // Skew is usually zero
    double u0 = K_eigen(0, 2);
    double v0 = K_eigen(1, 2);

    // Create GTSAM intrinsic object
    K_gtsam_ = boost::make_shared<Cal3_S2>(fx, fy, s, u0, v0);

    // For simplicity, initialize process noise Q
    Q_ = StateCovariance::Identity() * 1e-4;
  }

  // EKF Prediction Step (Trivial motion model)
  void predict() {
    // P_k = P_{k-1} + Q
    // H_w_ = H_w_;
    P_ = P_ + Q_;
    // H_w remains unchanged
    cout << "Prediction Step Complete. Covariance inflated by Q." << endl;
  }

  void update(const vector<Point3>& P_w_list, const vector<Point2>& z_obs_list,
              const gtsam::Pose3& X_W_k);

  const Pose3& getPose() const { return H_w_; }
  const StateCovariance& getCovariance() const { return P_; }
};

}  // namespace testing

/**
 * @brief Implements a Square Root Information Filter (SRIF) for 6-DoF pose
 * estimation.
 * * Instead of propagating a state (W) and covariance (P), this filter
 * propagates:
 * 1. R_info_ (R): An 6x6 upper-triangular matrix, the Cholesky factor of the
 * information matrix (Lambda = R^T * R).
 * 2. d_info_ (d): A 6x1 vector, where R^T * d = y (the information vector).
 * 3. W_linearization_point_: The nominal state (Pose3) around which the filter
 * is linearized.
 *
 * The state is recovered as a perturbation (delta_w) from this linearization
 * point by solving R * delta_w = d.
 */
class SquareRootInfoFilterGTSAM {
 private:
  // --- SRIF State Variables ---
  gtsam::Matrix66
      R_info_;  // R (6x6) - Upper triangular Cholesky factor of Info Matrix
  gtsam::Vector6 d_info_;  // d (6x1) - Transformed information vector
  gtsam::Pose3 H_linearization_point_;  // Nominal state (linearization point)

  // --- System Parameters ---
  gtsam::Cal3_S2::shared_ptr K_gtsam_;  // GTSAM Camera Intrinsic
  gtsam::Matrix22 R_noise_;             // 2x2 Measurement Noise
  gtsam::Matrix66 Q_;  // Process Noise Covariance (for prediction step)

 public:
  SquareRootInfoFilterGTSAM(const gtsam::Pose3& initial_state_H,
                            const gtsam::Matrix66& initial_P,
                            const gtsam::Cal3_S2::shared_ptr& K,
                            const gtsam::Matrix22& R);

  /**
   * @brief Recovers the state perturbation delta_w by solving R * delta_w = d.
   */
  gtsam::Vector6 getStatePerturbation() const;

  const gtsam::Pose3& getCurrentLinearization() const;

  /**
   * @brief Recovers the full state pose W by applying the perturbation
   * to the linearization point.
   */
  gtsam::Pose3 getStatePoseW() const;

  /**
   * @brief Recovers the state covariance P by inverting the information matrix.
   * @note This is a slow operation (O(N^3)) and should only be called
   * for inspection, not inside the filter loop.
   */
  gtsam::Matrix66 getCovariance() const;

  /**
   * @brief Recovers the information matrix Lambda = R^T * R.
   */
  gtsam::Matrix66 getInformationMatrix() const;

  /**
   * @brief EKF Prediction Step (Trivial motion model for W)
   * @note Prediction is the hard/slow part of an Information Filter.
   * This implementation is a "hack" that converts to covariance,
   * adds noise, and converts back. A "pure" SRIF predict is complex.
   */
  void predict();
  /**
   * @brief SRIF Update Step using Iteratively Reweighted Least Squares (IRLS)
   * with QR decomposition to achieve robustness.
   */
  void update(const std::vector<gtsam::Point3>& P_w_list,
              const std::vector<gtsam::Point2>& z_obs_list,
              const gtsam::Pose3& X_W_k, const int num_irls_iterations = 1);
};

/**
 * @brief Hybrid Object motion Square-Root Information Filter
 *
 */
class HybridObjectMotionSRIF {
 public:
  struct Result {
    double error{0.0};
    double reweighted_error{0.0};
    // gtsam::Pose3 H_W_e_k;
    // gtsam::Pose3 H_W_km1_k;
  };

 private:
  // --- SRIF State Variables ---
  gtsam::Pose3 H_linearization_point_;  // Nominal state (linearization point)
  const gtsam::Matrix66 Q_;  // Process Noise Covariance (for prediction step)
  const gtsam::Matrix33 R_noise_;  // 3x3 Measurement Noise
  //! Cached R_noise inverse
  const gtsam::Matrix33 R_inv_;
  const gtsam::Matrix66 initial_P_;

  gtsam::Pose3 L_e_;
  // Frame Id for the reference frame e
  FrameId frame_id_e_;

  gtsam::Matrix66
      R_info_;  // R (6x6) - Upper triangular Cholesky factor of Info Matrix
  gtsam::Vector6 d_info_;  // d (6x1) - Transformed information vector

  // --- System Parameters ---
  std::shared_ptr<RGBDCamera> rgbd_camera_;
  gtsam::Cal3_S2Stereo::shared_ptr stereo_calibration_;

  //! Points in L (current linearization)
  gtsam::FastMap<TrackletId, gtsam::Point3> m_linearized_;

  //! should be from e to k-1. Currently set in predict
  gtsam::Pose3 previous_H_;
  double huber_k_{1.23};

  constexpr static int StateDim = gtsam::traits<gtsam::Pose3>::dimension;
  constexpr static int ZDim = gtsam::traits<gtsam::StereoPoint2>::dimension;

 public:
  HybridObjectMotionSRIF(const gtsam::Pose3& initial_state_H,
                         const gtsam::Pose3& L_e, const FrameId& frame_id_e,
                         const gtsam::Matrix66& initial_P,
                         const gtsam::Matrix66& Q, const gtsam::Matrix33& R,
                         Camera::Ptr camera, double huber_k = 1.23);

  /**
   * @brief Recovers the state perturbation delta_w by solving R * delta_w = d.
   */
  gtsam::Vector6 getStatePerturbation() const;

  const gtsam::Pose3& getCurrentLinearization() const;

  /**
   * @brief Recovers the full state pose W by applying the perturbation
   * to the linearization point.
   */
  gtsam::Pose3 getStatePoseW() const;

  /**
   * @brief Recovers the state covariance P by inverting the information matrix.
   * @note This is a slow operation (O(N^3)) and should only be called
   * for inspection, not inside the filter loop.
   */
  gtsam::Matrix66 getCovariance() const;

  /**
   * @brief Recovers the information matrix Lambda = R^T * R.
   */
  gtsam::Matrix66 getInformationMatrix() const;

  /**
   * @brief EKF Prediction Step (Trivial motion model for W)
   * @note Prediction is the hard/slow part of an Information Filter.
   * This implementation is a "hack" that converts to covariance,
   * adds noise, and converts back. A "pure" SRIF predict is complex.
   */
  void predict(const gtsam::Pose3& H_W_km1_k);
  /**
   * @brief SRIF Update Step using Iteratively Reweighted Least Squares (IRLS)
   * with QR decomposition to achieve robustness.
   */
  Result update(Frame::Ptr frame, const TrackletIds& tracklets,
                const int num_irls_iterations = 1);

 private:
  /**
   * @brief Resets information d_info_ and R_info.
   * d_inifo is set to zero and R_info is constructed from the initial
   * covariance P. L_e_ is updated with new value and previous_H_ reset to
   * identity
   *
   * @param L_e
   * @param frame_id_e
   */
  void resetState(const gtsam::Pose3& L_e, FrameId frame_id_e);
};

class ObjectMotionSolverFilter : public ObjectMotionSovlerF2F {
 public:
  //! Result from solve including the object motions and poses
  using ObjectMotionSovlerF2F::Params;
  using ObjectMotionSovlerF2F::Result;

  using ObjectMotionSovlerF2F::EgoMotionSolver;

  ObjectMotionSolverFilter(const Params& params,
                           const CameraParams& camera_params);

 protected:
  bool solveImpl(Frame::Ptr frame_k, Frame::Ptr frame_k_1, ObjectId object_id,
                 MotionEstimateMap& motion_estimates) override;

  // gtsam::FastMap<ObjectId,
  // std::shared_ptr<testing::ExtendedKalmanFilterGTSAM>>
  //     filters_;
  gtsam::FastMap<ObjectId, std::shared_ptr<HybridObjectMotionSRIF>> filters_;
};

void declare_config(OpticalFlowAndPoseOptimizer::Params& config);
void declare_config(MotionOnlyRefinementOptimizer::Params& config);

void declare_config(EgoMotionSolver::Params& config);
void declare_config(ObjectMotionSovlerF2F::Params& config);

}  // namespace dyno

#include "dynosam/frontend/vision/MotionSolver-inl.hpp"
