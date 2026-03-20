#pragma once

#include "dynosam/frontend/VIFrontend.hpp"
#include "dynosam/frontend/VisionImuOutputPacket.hpp"
#include "dynosam/frontend/solvers/ConsecutiveFrameObjectMotionSolver.hpp"
#include "dynosam_cv/RGBDCamera.hpp"

namespace dyno {

// maybe a better name is VIModule or something? Depends on what we mean by
// regular!!!
using RegularBackendSink =
    std::function<void(const VisionImuPacket::ConstPtr&)>;

// TODO: include sinks (callback) to backend with VisionImuOutput
class RegularVIFrontend : public VIFrontend {
 public:
  DYNO_POINTER_TYPEDEFS(RegularVIFrontend)
  RegularVIFrontend(const DynoParams& params, Camera::Ptr camera,
                    ImageDisplayQueue* display_queue = nullptr,
                    const SharedGroundTruth& shared_ground_truth = {});

  void addVIOutputSink(const RegularBackendSink& func) {
    regular_backend_output_sink_ = func;
  };

 private:
  SpinReturn boostrapSpin(VIFrontendInput::ConstPtr input) override;
  SpinReturn nominalSpin(VIFrontendInput::ConstPtr input) override;

  void fillOutputPacketWithTracks(
      VisionImuPacket::Ptr vision_imu_packet, const Frame& frame,
      const gtsam::Pose3 X_W_k, const gtsam::Pose3& T_k_1_k,
      const MultiObjectTrajectories& object_trajectories) const;

 private:
  ConsecutiveFrameObjectMotionSolver::UniquePtr object_motion_solver_;

  gtsam::NavState nav_state_km1_;
  //! The relative camera pose (T_k_1_k) from the previous frame
  //! this is used as a constant velocity model when VO tracking fails and the
  //! IMU is not available!
  gtsam::Pose3 T_km1_k_;

  //! Current trajectories. Copied to the DynoState output
  DynoStateTrajectories dyno_state_;

  RegularBackendSink regular_backend_output_sink_;
};

}  // namespace dyno
