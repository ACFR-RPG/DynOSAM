#pragma once

#include "dynosam/backend/PoseChangeBackendModule.hpp"
#include "dynosam/formulations/KeyFrameHybridEstimator.hpp"
#include "dynosam/frontend/VIFrontend.hpp"
#include "dynosam/frontend/solvers/HybridObjectMotionSolver.hpp"
#include "dynosam_cv/RGBDCamera.hpp"

namespace dyno {

using PoseChangeBackendSink =
    std::function<void(const PoseChangeInput::ConstPtr&)>;

class PoseChangeVIFrontend : public VIFrontend {
 public:
  DYNO_POINTER_TYPEDEFS(PoseChangeVIFrontend)
  PoseChangeVIFrontend(const DynoParams& params, Camera::Ptr camera,
                       HybridFormulationKeyFrame::Ptr formulation,
                       ImageDisplayQueue* display_queue = nullptr,
                       const SharedGroundTruth& shared_ground_truth = {});

  ~PoseChangeVIFrontend();

  /** Add sink to send PC data to the backend */
  void addPoseChangeOutputSink(const PoseChangeBackendSink& func) {
    pose_change_backend_sink_ = func;
  };

  /** Callback triggered when the backend has finished a single update */
  void onBackendUpdateComplete(const PoseChangeUpdateComplete& event);

 private:
  SpinReturn boostrapSpin(VIFrontendInput::ConstPtr input) override;
  SpinReturn nominalSpin(VIFrontendInput::ConstPtr input) override;

  bool withBackend() const {
    //! Use existance of backend sink as proxy logicc for "use backend"
    return (bool)pose_change_backend_sink_;
  }

  void solveObjectMotions(MultiObjectTrajectories& trajectories,
                          ObjectIds& object_with_new_motions,
                          ObjectPoseChangeInfoMap& infos, Frame::Ptr frame_k,
                          Frame::Ptr frame_km1);

  bool shouldFrameBeKeyFrame(Frame::Ptr frame_k, Frame::Ptr frame_km1) const;

  void logBestEstimates() const;
  void logRealTimeObjectClouds(const ObjectIds& objects,
                               FrameId frame_id) const;

  struct IntermediateMotion {
    //! Should be from a Keyframe
    FrameId from;
    //! To the current frame
    FrameId to;
    //! Timestamp at the current (ie. to) frame
    Timestamp timestamp;

    //! Frame ptr at current frame (ie. to)
    Frame::Ptr frame;
    //! Nav state at current frame (ie. to)
    gtsam::NavState frontend_nav_state;

    ImuFrontend::PimPtr pim;
    //! Should exist only if PIM is non null
    ImuMeasurements imu_measurements;
    gtsam::Pose3 T_from_to;
  };

  bool solveAndRefineEgoMotion(
      Frame::Ptr frame_k, const Frame::Ptr& frame_km1,
      AbsolutePoseCorrespondences& map_matches,
      std::optional<gtsam::NavState> propogated_nav_state_k = std::nullopt,
      std::optional<gtsam::Rot3> R_km1_k = std::nullopt);

 private:
  HybridFormulationKeyFrame::Ptr formulation_;
  KeyFrameMap::Ptr map_;
  HybridObjectMotionSolver::UniquePtr object_motion_solver_;

  gtsam::NavState nav_state_km1_;
  gtsam::NavState nav_state_lkf_;
  //! The relative camera pose (T_k_1_k) from the previous frame
  //! this is used as a constant velocity model when VO tracking fails and the
  //! IMU is not available!
  gtsam::Pose3 T_km1_k_;
  gtsam::Pose3 T_lkf_k_;

  //! Last camera keyframe
  Frame::Ptr lCKF_frame_;

  //! Current trajectories. Copied to the DynoState output.
  //! Only contains trajectories for objects observed at the latest frame
  DynoStateTrajectories dyno_state_;
  MultiObjectTrajectories full_object_trajectories_;

  PoseChangeBackendSink pose_change_backend_sink_;

  // Mapping of intermediate relative motions. Stored by to frame.
  gtsam::FastMap<FrameId, IntermediateMotion> intermediate_motions_;
  // gtsam::FastMap<FrameId, KeyFrameData> keyframes_;
};

}  // namespace dyno
