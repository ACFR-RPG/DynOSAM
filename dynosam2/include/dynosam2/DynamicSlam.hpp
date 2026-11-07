#pragma once

#include "dynosam2/common/DynamicSlamInterface.hpp"
#include "dynosam2/common/Parameters.hpp"
#include "dynosam2/estimator/SlamBackend.hpp"
#include "dynosam2/frontend/Frontend.hpp"
#include "dynosam2/frontend/FrontendViz.hpp"
#include "dynosam_common/Trajectories.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_processing/ThreadSafeQueue.hpp"
#include "dynosam_sensors/ImuFrontend.hpp"
#include "dynosam_sensors/SensorRig.hpp"

namespace dyno2 {

// Eventually put the estimator here as well
class DynamicSlam : public DynamicSlamInterface {
 public:
  DYNO_POINTER_TYPEDEFS(DynamicSlam)

  DynamicSlam(const PipelineParameters& parameters,
              CanonicalSensorRig::Ptr sensor_rig);

  ~DynamicSlam();

  const SlamBackend& getEstimator() const { return estimator_; }

 protected:
  StateEstimate::Ptr spinOnce(dyno::VIFrontendInput::ConstPtr input) override;

 private:
  /// @brief Initialise pose from IMU measurements (assuming non-accelerating).
  /// @param imu_measurements
  /// @param T_WC
  /// @return
  bool initPoseImu(const dyno::ImuMeasurements& imu_measurements,
                   gtsam::Pose3& T_WC) const;

  void shutdownThreads();

  /// @brief Performs FrontendViz processing in the background
  void visualisationLoop();

 private:
  PipelineParameters parameters_;
  CanonicalSensorRig::Ptr sensor_rig_;
  dyno::Camera::Ptr camera_;
  //! Set even when not in IMU mode
  dyno::CombinedPIMParamPtr imu_params_;

  Frontend frontend_;
  SlamBackend estimator_;

  bool first_frame_{true};

  /// @brief Viz responsible for matching/tracking drawing
  FrontendViz frontend_viz_;
  ThreadsafeQueue<FrontendViz::Data::Ptr> frontend_viz_data_queue_;
  /// @brief Thread running the frontend viz thread
  std::thread frontend_viz_thread_;

  std::atomic_bool is_shutdown_{false};
};

}  // namespace dyno2
