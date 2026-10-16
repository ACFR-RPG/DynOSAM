#pragma once

#include "dynosam2/common/DynamicSlamInterface.hpp"
#include "dynosam2/common/Parameters.hpp"
#include "dynosam2/frontend/Frontend.hpp"
#include "dynosam_common/Types.hpp"

namespace dyno2 {

// Eventually put the estimator here as well
class DynamicSlam : public DynamicSlamInterface {
 public:
  DYNO_POINTER_TYPEDEFS(DynamicSlam)

  DynamicSlam(const PipelineParameters& parameters);

  ~DynamicSlam();

 protected:
  StateEstimate::Ptr spinOnce(dyno::VIFrontendInput::ConstPtr input) override;
};

}  // namespace dyno2
