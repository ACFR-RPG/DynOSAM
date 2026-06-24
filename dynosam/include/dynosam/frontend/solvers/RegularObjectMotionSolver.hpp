#pragma once

#include <config_utilities/config_utilities.h>

#include "dynosam/frontend/solvers/MotionOnlyRefinementSolver.hpp"
#include "dynosam/frontend/solvers/ObjectMotionSolver.hpp"
#include "dynosam/frontend/solvers/OpticalFlowAndPoseSolver.hpp"
#include "dynosam/frontend/solvers/PnPRansac.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam_common/GroundTruthPacket.hpp"
#include "dynosam_sensors/Camera.hpp"

namespace dyno {

struct RegularObjectMotionSolverParams {
  PnPRansacSolverParams pnp_ransac_params;
  OpticalFlowAndPoseSolverParams optical_flow_solver_params;
  MotionOnlyRefinementSolverParams motion_only_refinement_params;

  bool refine_with_flow{true};
  bool refine_with_3d{false};
};

void declare_config(RegularObjectMotionSolverParams& config);

/**
 * @brief Regular object motion solver as described in the T-RO DynoSAM paper
 * (ie. morris2025dynosam) PnP+Ransac + Refinement with Optical Flow +
 * Refinement via Rigid-Body refinement.
 *
 * The latter two are optional and can be configurated via config.
 *
 */
class RegularObjectMotionSolver : public ObjectMotionSolver {
 public:
  DYNO_POINTER_TYPEDEFS(RegularObjectMotionSolver)

  // if shared ground truth contains no ground truth the objects will be
  // initalised with centroid
  RegularObjectMotionSolver(const RegularObjectMotionSolverParams& params,
                            const CameraParams& camera_params,
                            const DepthUpdater& depth_updater,
                            const SharedGroundTruth& shared_ground_truth = {});

 private:
  bool solveImpl(Frame::Ptr frame_k, Frame::Ptr frame_km1, ObjectId object_id,
                 Motion3ReferenceFrame& motion_estimate) override;

  void updateTrajectories(MultiObjectTrajectories& object_trajectories,
                          const MotionEstimateMap& motion_estimates,
                          Frame::Ptr frame_k, Frame::Ptr frame_km1) override;

 private:
  RegularObjectMotionSolverParams params_;
  PnPRansacSolver pnp_ransac_solver_;
  OpticalFlowAndPoseSolver<Camera::CalibrationType> optical_flow_pose_solver_;
  MotionOnlyRefinementSolver<Camera::CalibrationType>
      motion_only_refinement_solver_;

  const SharedGroundTruth shared_ground_truth_;

  //! Stored object trajectories
  MultiObjectTrajectories object_trajectories_;
};

}  // namespace dyno
