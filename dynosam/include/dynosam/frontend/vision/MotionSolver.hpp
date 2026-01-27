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
#include "dynosam/frontend/VisionImuOutputPacket.hpp"
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

  Result solve(Frame::Ptr frame_k, Frame::Ptr frame_k_1);

 protected:
  virtual bool solveImpl(Frame::Ptr frame_k, Frame::Ptr frame_k_1,
                         ObjectId object_id,
                         MotionEstimateMap& motion_estimates) = 0;

  virtual void updatePoses(ObjectPoseMap& object_poses,
                           const MotionEstimateMap& motion_estimates,
                           Frame::Ptr frame_k, Frame::Ptr frame_k_1) = 0;

  virtual void updateMotions(ObjectMotionMap& object_motions,
                             const MotionEstimateMap& motion_estimates,
                             Frame::Ptr frame_k, Frame::Ptr frame_k_1) = 0;
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

  Motion3SolverResult geometricOutlierRejection3d2d(
      Frame::Ptr frame_k_1, Frame::Ptr frame_k, const gtsam::Pose3& T_world_k,
      ObjectId object_id);

  const ObjectMotionSovlerF2F::Params& objectMotionParams() const {
    return object_motion_params;
  }

 protected:
  virtual bool solveImpl(Frame::Ptr frame_k, Frame::Ptr frame_k_1,
                         ObjectId object_id,
                         MotionEstimateMap& motion_estimates) override;

 private:
  void updatePoses(ObjectPoseMap& object_poses,
                   const MotionEstimateMap& motion_estimates,
                   Frame::Ptr frame_k, Frame::Ptr frame_k_1) override;

  void updateMotions(ObjectMotionMap& object_motions,
                     const MotionEstimateMap& motion_estimates,
                     Frame::Ptr frame_k, Frame::Ptr frame_k_1) override;

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

  // FOR TESTING!
 public:
  // --- SRIF State Variables ---
  gtsam::Pose3 H_linearization_point_;  // Nominal state (linearization point)
  const gtsam::Matrix66 Q_;  // Process Noise Covariance (for prediction step)
  const gtsam::Matrix33 R_noise_;  // 3x3 Measurement Noise
  //! Cached R_noise inverse
  const gtsam::Matrix33 R_inv_;
  const gtsam::Matrix66 initial_P_;

  gtsam::Pose3 L_e_;
  // Frame Id for the reference KF
  FrameId frame_id_e_;
  //! Last camera pose used within predict
  gtsam::Pose3 X_K_;
  //! Frame id used for last update
  FrameId frame_id_;

  gtsam::Matrix66
      R_info_;  // R (6x6) - Upper triangular Cholesky factor of Info Matrix
  gtsam::Vector6 d_info_;  // d (6x1) - Transformed information vector

  // --- System Parameters ---
  std::shared_ptr<RGBDCamera> rgbd_camera_;
  gtsam::Cal3_S2Stereo::shared_ptr stereo_calibration_;

  //! Points in L (current linearization)
  gtsam::FastMap<TrackletId, gtsam::Point3> m_linearized_;

  ObjectTrackingStatus motion_track_status_;

  //! should be from e to k-1. Currently set in predict
  gtsam::Pose3 previous_H_;
  double huber_k_{1.23};

  constexpr static int StateDim = gtsam::traits<gtsam::Pose3>::dimension;
  constexpr static int ZDim = gtsam::traits<gtsam::StereoPoint2>::dimension;

 public:
  bool needs_resetting_from_last_frame{false};
  bool was_reset_this_update{false};

 public:
  HybridObjectMotionSRIF(const gtsam::Pose3& initial_state_H,
                         const gtsam::Pose3& L_e, const FrameId& frame_id_e,
                         const gtsam::Matrix66& initial_P,
                         const gtsam::Matrix66& Q, const gtsam::Matrix33& R,
                         Camera::Ptr camera, double huber_k = 1.23);

  inline const gtsam::Pose3& getKeyFramePose() const { return L_e_; }
  inline const gtsam::Pose3& lastCameraPose() const { return X_K_; }
  inline FrameId getKeyFrameId() const { return frame_id_e_; }
  inline ObjectTrackingStatus getMotionTrackingStatus() const {
    return motion_track_status_;
  }
  inline FrameId getFrameId() const { return frame_id_; }

  // could also mean the object was new?
  inline bool resetThisUpdate() const { return was_reset_this_update; }

  inline const gtsam::FastMap<TrackletId, gtsam::Point3>&
  getCurrentLinearizedPoints() const {
    return m_linearized_;
  }

  gtsam::Pose3 getPose() const {
    return getKeyFramedMotion() * getKeyFramePose();
  }

  void predictAndUpdate(const gtsam::Pose3& H_w_km1_k_predict, Frame::Ptr frame,
                        const TrackletIds& tracklets,
                        const int num_irls_iterations = 1);

  /**
   * @brief Recovers the state perturbation delta_w by solving R * delta_w = d.
   */
  gtsam::Vector6 getStatePerturbation() const;

  // this is H_W_e_k
  const gtsam::Pose3& getCurrentLinearization() const;

  // this is H_W_e_k
  // calculate best estimate!!
  gtsam::Pose3 getKeyFramedMotion() const;

  Motion3ReferenceFrame getKeyFramedMotionReference() const;

  /**
   * @brief Recovers the full state pose W by applying the perturbation
   * to the linearization point.
   *
   * LIES: thie is H_W_km1_k
   */
  gtsam::Pose3 getF2FMotion() const;

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
   * @brief Resets information d_info_ and R_info.
   * d_inifo is set to zero and R_info is constructed from the initial
   * covariance P. L_e_ is updated with new value and previous_H_ reset to
   * identity
   *
   * @param L_e
   * @param frame_id_e
   */
  void resetState(const gtsam::Pose3& L_e, FrameId frame_id_e);

 private:
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
};

class ObjectMotionSolverFilter : public ObjectMotionSolver,
                                 protected EgoMotionSolver {
 public:
  //! Result from solve including the object motions and poses
  using ObjectMotionSolver::Result;

  struct Params : public EgoMotionSolver::Params {
    // TODO: filter params!!
  };

  ObjectMotionSolverFilter(const Params& params,
                           const CameraParams& camera_params);

  void fillHybridInfo(ObjectId object_id,
                      VisionImuPacket::ObjectTracks& object_track);
  void markObjectAsLost(ObjectId object_id) { filters_.erase(object_id); }

 protected:
  bool solveImpl(Frame::Ptr frame_k, Frame::Ptr frame_k_1, ObjectId object_id,
                 MotionEstimateMap& motion_estimates) override;

  void updatePoses(ObjectPoseMap& object_poses,
                   const MotionEstimateMap& motion_estimates,
                   Frame::Ptr frame_k, Frame::Ptr frame_k_1) override;

  void updateMotions(ObjectMotionMap& object_motions,
                     const MotionEstimateMap& motion_estimates,
                     Frame::Ptr frame_k, Frame::Ptr frame_k_1) override;

 private:
  const Params filter_params_;
  //! All object poses (from k to K) and updated by updatePoses at each
  //! iteration of sovled
  ObjectPoseMap object_poses_;
  //! All object motions (from k to K) and updated by updatedMotions at each
  //! iteration of sovled
  ObjectMotionMap object_motions_;

 public:  // TODO: for testing!
  gtsam::FastMap<ObjectId, std::shared_ptr<HybridObjectMotionSRIF>> filters_;
};

void declare_config(OpticalFlowAndPoseOptimizer::Params& config);
void declare_config(MotionOnlyRefinementOptimizer::Params& config);

void declare_config(EgoMotionSolver::Params& config);
void declare_config(ObjectMotionSovlerF2F::Params& config);

}  // namespace dyno

#include "dynosam/frontend/vision/MotionSolver-inl.hpp"
