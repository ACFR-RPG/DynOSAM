#include "dynosam2/estimator/BaGraph.hpp"

#include <gtsam/nonlinear/DoglegOptimizer.h>

// namespace dyno2 {

// bool BaOptInterface::optimize(gtsam::Values& optimised_values, const
// gtsam::Values& values, const gtsam::NonlinearFactorGraph& graph) {
// optimised_values = values;

//   dyno::NonlinearOptimizerSummary summary;
//   // Dog leg is just way faster for will defined problems
//   gtsam::DoglegParams opt_params;
//   opt_params.setMaxIterations(2);

//   dyno::NonlinearOptimizer<gtsam::DoglegOptimizer> solver(graph, values,
//   opt_params); if(!solver.solve(optimised_values, options_, &summary)) {
//     return false;
//   }

//   VLOG(10) << "Initial error: " << summary.initial_error << " final error "
//            << summary.final_error << " time[s] "
//            << summary.cumulative_time_in_seconds
//            << " #iterations= " << summary.numIterations();
//   return true;

// }

// }
