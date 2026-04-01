#include "dynosam_common/Types.hpp"
#include "dynosam_opt/NonlinearOptimizer.hpp"

// Implemented from
// https://github.dev/ethz-mrl/okvis2/blob/main/okvis_ceres/include/okvis/ceres/CeresIterationCallback.hpp
namespace dyno {

class TimeBudgetOptimizationCallback : public IterationCallback {
 public:
  DYNO_POINTER_TYPEDEFS(TimeBudgetOptimizationCallback)
  TimeBudgetOptimizationCallback(Timestamp time_limit, int minimum_iterations);

  CallbackReturnType operator()(const IterationSummary& summary) override;

  void setTimeLimit(Timestamp time_limit);
  void setMinimumIterations(int minimum_iterations);

 private:
  //! Limit in seconds
  Timestamp time_limit_;
  int minimum_iterations_;
};

}  // namespace dyno
