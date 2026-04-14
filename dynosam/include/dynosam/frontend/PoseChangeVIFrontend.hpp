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

  // struct KeyFrameData {
  //   FrameId kf_id;
  //   FrameId kf_id_prev;
  //   Frame::Ptr frame;
  //   gtsam::NavState nav_state;

  //   //! If camera keyframe logic was true for this frame
  //   bool camera_keyframe{false};

  //   bool retroactively_made_keyframe{false};

  //   //! Signifcies which objects had motion variables added at this frame
  //   (with
  //   //! the kf_id being the "to" frame of each motion)
  //   ObjectIds object_keyframes;

  //   bool isObjectKeyframe() const { return !object_keyframes.empty(); }

  //   bool isObjectKeyFrame(const ObjectId object_id) const {
  //     return std::find(object_keyframes.begin(), object_keyframes.end(),
  //                      object_id) != object_keyframes.end();
  //   }
  // };

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

  //! Last keyframe id
  FrameId lkf_id_;

  //! Current trajectories. Copied to the DynoState output.
  //! Only contains trajectories for objects observed at the latest frame
  DynoStateTrajectories dyno_state_;
  MultiObjectTrajectories full_object_trajectories_;

  PoseChangeBackendSink pose_change_backend_sink_;

  // Mapping of intermediate relative motions. Stored by to frame.
  gtsam::FastMap<FrameId, IntermediateMotion> intermediate_motions_;
  // gtsam::FastMap<FrameId, KeyFrameData> keyframes_;

  gtsam::Values refined_backend_states_;
  std::atomic_bool has_updated_backend_values_{false};
  std::mutex backend_update_mutex_;
};

}  // namespace dyno
