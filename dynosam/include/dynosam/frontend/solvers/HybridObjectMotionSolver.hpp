#pragma once

#include "dynosam/frontend/Frontend-Definitions.hpp"
#include "dynosam/frontend/solvers/HybridObjectMotionSRIF.hpp"
#include "dynosam/frontend/solvers/HybridObjectMotionSmoother.hpp"
#include "dynosam/frontend/solvers/HybridObjectMotionSolver-Impl.hpp"
#include "dynosam/frontend/solvers/ObjectMotionSolver.hpp"
#include "dynosam/frontend/solvers/OpticalFlowAndPoseSolver.hpp"
#include "dynosam/frontend/solvers/PnPRansac.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam_common/GroundTruthPacket.hpp"
#include "dynosam_common/MotionKeyFrame.hpp"
#include "dynosam_cv/Camera.hpp"

namespace dyno {

struct HybridObjectMotionSolverParams {
  PnPRansacSolverParams pnp_ransac_params;
  OpticalFlowAndPoseSolverParams optical_flow_solver_params;
};

class HybridObjectMotionSolver : public ObjectMotionSolver {
 public:
  DYNO_POINTER_TYPEDEFS(HybridObjectMotionSolver)

  HybridObjectMotionSolver(const HybridObjectMotionSolverParams& params,
                           const CameraParams& camera_params,
                           const SharedGroundTruth& shared_ground_truth = {});

  ~HybridObjectMotionSolver();

  void solve(Frame::Ptr frame_k, Frame::Ptr frame_km1,
             MultiObjectTrajectories& trajectories_out,
             MotionEstimateMap& motion_estimate_out,
             bool parallel_solve = true) override;

  bool getObjectStructureinL(ObjectId object_id,
                             StatusLandmarkVector& object_points) const;
  bool getObjectStructureinW(ObjectId object_id,
                             StatusLandmarkVector& object_points) const;

  const auto& getFilters() const { return solvers_; }

  const ObjectPoseChangeInfoMap& poseChangeInfoMap() const {
    return pose_change_info_;
  }

  void receiveUpdate(const PoseChangeUpdateComplete& update_info);

 protected:
  bool solveImpl(Frame::Ptr frame_k, Frame::Ptr frame_km1, ObjectId object_id,
                 Motion3ReferenceFrame& motion_estimate) override;

  void updateTrajectories(MultiObjectTrajectories& object_trajectories,
                          const MotionEstimateMap& motion_estimates,
                          Frame::Ptr frame_k, Frame::Ptr frame_km1) override;

 private:
  // may be from centroid or gronud truth depending on availablility
  gtsam::Pose3 constructObjectPose(const ObjectId object_id,
                                   const Frame::Ptr frame,
                                   const TrackletIds& tracklets) const;

  std::optional<gtsam::Pose3> objectPoseFromGroundTruth(
      ObjectId object_id, const Frame::Ptr frame) const;

  gtsam::Pose3 objectPoseFromCentroid(const ObjectId object_id,
                                      const Frame::Ptr frame,
                                      const TrackletIds& tracklets) const;

  HybridObjectMotionSolverImpl::Ptr createAndInsertFilter(
      ObjectId object_id, Frame::Ptr frame, const TrackletIds& tracklets);

  void markObjectAsLost(ObjectId object_id, FrameId frame_id);

  bool solverExists(ObjectId object_id) const {
    const std::lock_guard<std::mutex> lock(solvers_mutex_);
    return solvers_.exists(object_id);
  }

  HybridObjectMotionSolverImpl::Ptr threadSafeFilterAccess(
      ObjectId object_id) const;

  bool threadSafeGetNumKeyframes(ObjectId object_id, int& num_keyframes) const {
    const std::lock_guard<std::mutex> lock(num_kfs_per_object_mutex_);
    if (!num_kfs_per_object_.exists(object_id)) {
      return false;
    }

    num_keyframes = num_kfs_per_object_.at(object_id);
    return true;
  }

  ObjectPoseChangeInfo& appendPoseChangeInfo(
      ObjectId object_id, ObjectKeyFrameStatus keyframe_status);

 private:
  HybridObjectMotionSolverParams params_;
  PnPRansacSolver pnp_ransac_solver_;
  OpticalFlowAndPoseSolver<Camera::CalibrationType> optical_flow_pose_solver_;
  const SharedGroundTruth shared_ground_truth_;

  MultiObjectTrajectories object_trajectories_;

  gtsam::FastMap<ObjectId, HybridObjectMotionSolverImpl::Ptr> solvers_;
  // Info from the last frame. ONly stores change info with keyframes
  gtsam::FastMap<ObjectId, ObjectPoseChangeInfo> pose_change_info_;

  enum PoseInitalisationMethod { NonKeyFrame, Centroid, Previous };

 private:
  struct TrackingStatusHistory {
    std::vector<std::pair<FrameId, ObjectTrackingStatus>> statuses;
    ObjectTrackingStatus currentStatus() const {
      return statuses.back().second;
    }
    void add(FrameId frame_id, ObjectTrackingStatus status) {
      statuses.push_back(std::make_pair(frame_id, status));
    }
  };

  struct ObjectTrackingStatuses {
    mutable std::mutex mutex;
    gtsam::FastMap<ObjectId, TrackingStatusHistory> statuses;

    bool exists(ObjectId object_id) const {
      const std::lock_guard<std::mutex> lock(mutex);
      return statuses.exists(object_id);
    }
    void setStatus(ObjectId object_id, FrameId frame_id,
                   ObjectTrackingStatus status) {
      const std::lock_guard<std::mutex> lock(mutex);
      if (!statuses.exists(object_id)) {
        statuses[object_id] = TrackingStatusHistory{};
      }
      statuses.at(object_id).add(frame_id, status);
    }

    std::optional<ObjectTrackingStatus> getStatus(ObjectId object_id) const {
      std::optional<ObjectTrackingStatus> status;
      const std::lock_guard<std::mutex> lock(mutex);
      if (statuses.exists(object_id)) {
        status.emplace(statuses.at(object_id).currentStatus());
      }
      return status;
    }
  };

  ObjectTrackingStatuses object_statuses_;
  gtsam::FastMap<ObjectId, PoseWithMotionTrajectory> past_trajectories_;
  gtsam::FastMap<ObjectId, int> num_kfs_per_object_;
  mutable std::mutex num_kfs_per_object_mutex_;
  mutable std::mutex solvers_mutex_;
  mutable std::mutex pose_change_info_mutex_;
};

}  // namespace dyno

// void declare_config(OpticalFlowAndPoseOptimizer::Params& config);
// void declare_config(MotionOnlyRefinementOptimizer::Params& config);

// void declare_config(EgoMotionSolver::Params& config);
// void declare_config(ConsecutiveFrameObjectMotionSolver::Params& config);
