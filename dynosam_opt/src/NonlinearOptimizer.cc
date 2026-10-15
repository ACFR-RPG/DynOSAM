#include "dynosam_opt/NonlinearOptimizer.hpp"

#include <unordered_map>

#include "dynosam_common/Types.hpp"

namespace dyno {

template <>
std::string to_string(const TerminationType& termination_type) {
  static const std::unordered_map<TerminationType, std::string> map = {
      {TerminationType::CONVERGENCE, "CONVERGENCE"},
      {TerminationType::NO_CONVERGENCE, "NO_CONVERGENCE"},
      {TerminationType::FAILURE, "FAILURE"},
      {TerminationType::USER_SUCCESS, "USER_SUCCESS"},
      {TerminationType::USER_FAILURE, "USER_FAILURE"}};
  return map.at(termination_type);
}

TimeBudgetOptimizationCallback::TimeBudgetOptimizationCallback(
    Timestamp time_limit, int minimum_iterations)
    : time_limit_(time_limit), minimum_iterations_(minimum_iterations) {
  checkAndThrow(time_limit_ > 0.0,
                "Time limit in TimeBudgetOptimizationCallback must be > 0.0");
}

CallbackReturnType TimeBudgetOptimizationCallback::operator()(
    const IterationSummary& summary) {
  // assume next iteration takes the same time as current iteration
  if (summary.iterations >= minimum_iterations_ &&
      summary.cumulative_time_in_seconds + summary.iteration_time_in_seconds >
          time_limit_) {
    return CallbackReturnType::USER_SOLVER_TERMINATE_SUCCESSFULLY;
  }
  return CallbackReturnType::USER_SOLVER_CONTINUE;
}

void TimeBudgetOptimizationCallback::setTimeLimit(double time_limit) {
  time_limit_ = time_limit;
}

void TimeBudgetOptimizationCallback::setMinimumIterations(
    int minimum_iterations) {
  minimum_iterations_ = minimum_iterations;
}

}  // namespace dyno
