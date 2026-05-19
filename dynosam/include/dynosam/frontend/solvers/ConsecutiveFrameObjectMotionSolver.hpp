#pragma once

#include "dynosam/frontend/solvers/MotionOnlyRefinementSolver.hpp"
#include "dynosam/frontend/solvers/ObjectMotionSolver.hpp"
#include "dynosam/frontend/solvers/OpticalFlowAndPoseSolver.hpp"
#include "dynosam/frontend/solvers/PnPRansac.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam_common/GroundTruthPacket.hpp"
#include "dynosam_cv/Camera.hpp"

namespace dyno {

struct ConsecutiveFrameObjectMotionSolverParams {
  PnPRansacSolverParams pnp_ransac_params;
  OpticalFlowAndPoseSolverParams optical_flow_solver_params;
  MotionOnlyRefinementSolverParams motion_only_refinement_params;

  bool refine_motion_with_joint_of{true};
  bool refine_motion_with_3d{false};
};

class ConsecutiveFrameObjectMotionSolver : public ObjectMotionSolver {
 public:
  DYNO_POINTER_TYPEDEFS(ConsecutiveFrameObjectMotionSolver)

  // if shared ground truth contains no ground truth the objects will be
  // initalised with centroid
  ConsecutiveFrameObjectMotionSolver(
      const ConsecutiveFrameObjectMotionSolverParams& params,
      const CameraParams& camera_params, const DepthUpdater& depth_updater,
      const SharedGroundTruth& shared_ground_truth = {});

 private:
  bool solveImpl(Frame::Ptr frame_k, Frame::Ptr frame_km1, ObjectId object_id,
                 Motion3ReferenceFrame& motion_estimate) override;

  void updateTrajectories(MultiObjectTrajectories& object_trajectories,
                          const MotionEstimateMap& motion_estimates,
                          Frame::Ptr frame_k, Frame::Ptr frame_km1) override;

 private:
  ConsecutiveFrameObjectMotionSolverParams params_;
  PnPRansacSolver pnp_ransac_solver_;
  OpticalFlowAndPoseSolver<Camera::CalibrationType> optical_flow_pose_solver_;
  MotionOnlyRefinementSolver<Camera::CalibrationType>
      motion_only_refinement_solver_;

  const SharedGroundTruth shared_ground_truth_;

  //! Stored object trajectories
  MultiObjectTrajectories object_trajectories_;
};

}  // namespace dyno
