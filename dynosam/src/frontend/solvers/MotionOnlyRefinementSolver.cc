#include "dynosam/frontend/solvers/MotionOnlyRefinementSolver.hpp"

namespace dyno {

void declare_config(MotionOnlyRefinementSolverParams& config) {
  using namespace config;

  name("MotionOnlyRefinementSolverParams");
  field(config.landmark_motion_sigma, "landmark_motion_sigma");
  field(config.projection_sigma, "projection_sigma");
  field(config.k_huber, "k_huber");
  field(config.outlier_reject, "outlier_reject");
}

}  // namespace dyno
