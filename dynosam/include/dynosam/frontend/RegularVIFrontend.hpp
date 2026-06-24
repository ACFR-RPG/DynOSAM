#pragma once

#include "dynosam/frontend/VIFrontend.hpp"
#include "dynosam/frontend/VisionImuOutputPacket.hpp"
#include "dynosam/frontend/solvers/RegularObjectMotionSolver.hpp"
#include "dynosam_sensors/RGBDCamera.hpp"

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

  /**
   * @brief Solve the visual odometry (k-1 to k) using PnP + Refinement with
   * Optical Flow
   *
   * The input T_km1_k should be the best known relative camera motion (usually
   * the relative motion from the previous frame) to act as constant motion
   * model if the tracking fails.
   *
   * @param frame_k Frame::Ptr current frame at k
   * @param frame_km1 Frame::Ptr previous frame at k-1
   * @param nav_state_km1 const gtsam::NavState& previous nav state at k-1
   * @param T_km1_k const gtsam::Pose3& relative camera motion to act as a
   * constant velocity model.
   * @param propogated_nav_state_k std::optional<gtsam::NavState> nav state at k
   * as propogated by the IMU
   * @param R_km1_k std::optional<gtsam::Rot3> relative camera rotation from k-1
   * to k
   * @return true
   * @return false
   */
  bool solveAndRefineEgoMotion(
      Frame::Ptr frame_k, const Frame::Ptr& frame_km1,
      const gtsam::NavState& nav_state_km1, const gtsam::Pose3& T_km1_k,
      std::optional<gtsam::NavState> propogated_nav_state_k = std::nullopt,
      std::optional<gtsam::Rot3> R_km1_k = std::nullopt);

  void fillOutputPacketWithTracks(
      VisionImuPacket::Ptr vision_imu_packet, const Frame& frame,
      const gtsam::Pose3 X_W_k, const gtsam::Pose3& T_k_1_k,
      const MultiObjectTrajectories& object_trajectories) const;

 private:
  RegularObjectMotionSolver::UniquePtr object_motion_solver_;

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
