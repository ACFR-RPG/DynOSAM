#pragma once

#include "dynosam2/common/DynamicSlamInterface.hpp"
#include "dynosam2/common/Parameters.hpp"
#include "dynosam2/estimator/SlamBackend.hpp"
#include "dynosam2/frontend/Frontend.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_sensors/SensorRig.hpp"

namespace dyno2 {

// Eventually put the estimator here as well
class DynamicSlam : public DynamicSlamInterface {
 public:
  DYNO_POINTER_TYPEDEFS(DynamicSlam)

  DynamicSlam(const PipelineParameters& parameters,
              CanonicalSensorRig::Ptr sensor_rig);

  ~DynamicSlam();

 protected:
  StateEstimate::Ptr spinOnce(dyno::VIFrontendInput::ConstPtr input) override;

 private:
  PipelineParameters parameters_;
  CanonicalSensorRig::Ptr sensor_rig_;

  Frontend frontend_;
  SlamBackend estimator_;
};

}  // namespace dyno2
