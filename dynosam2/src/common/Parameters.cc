#include "dynosam2/common/Parameters.hpp"

#include <config_utilities/config_utilities.h>

namespace dyno2 {

void declare_config(EstimationParameters& config) {
  using namespace config;
  name("EstimationParameters");
}
void declare_config(PipelineParameters& config) {
  using namespace config;
  name("PipelineParameters");

  field(config.estimator, "estimator_parameters");
}

}  // namespace dyno2
