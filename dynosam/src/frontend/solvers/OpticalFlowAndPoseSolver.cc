#include "dynosam/frontend/solvers/OpticalFlowAndPoseSolver.hpp"

namespace dyno {

void declare_config(OpticalFlowAndPoseSolverParams& config) {
  using namespace config;

  name("OpticalFlowAndPoseSolverParams");
  field(config.flow_sigma, "flow_sigma");
  field(config.flow_prior_sigma, "flow_prior_sigma");
  field(config.k_huber, "k_huber");
  field(config.outlier_reject, "outlier_reject");
  field(config.use_robust, "use_robust");
}

}  // namespace dyno
