#pragma once

#include <gtsam/nonlinear/NonlinearOptimizer.h>

#include "dynosam_common/Exceptions.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/Timing.hpp"
#include "dynosam_opt/Symbols.hpp"

namespace dyno {

struct IterationSummary {
  // Time (in seconds) spent inside the minimizer loop in the current
  // iteration.
  double iteration_time_in_seconds = 0.0;

  // Time (in seconds) since the user called optimize().
  double cumulative_time_in_seconds = 0.0;

  //! Number of iterations made by the solver
  int iterations = 0;

  //! Whittened error after update
  double error = 0;

  //! Cost change delta (curent - previous)
  double cost_change = 0;
};

enum CallbackReturnType {
  // Continue solving to next iteration.
  USER_SOLVER_CONTINUE,
  USER_SOLVER_ABORT,
  USER_SOLVER_TERMINATE_SUCCESSFULLY
};

enum TerminationType {
  CONVERGENCE,

  // The solver ran for maximum number of iterations or maximum amount
  // of time specified by the user, but none of the convergence
  // criterion specified by the user were met.
  NO_CONVERGENCE,
  // The minimizer terminated because of an error.  The user's
  // parameter blocks will not be updated.
  FAILURE,

  // Using an IterationCallback object, user code can control the
  // minimizer. The following enums indicate that the user code was
  // responsible for termination.
  //
  // Minimizer terminated successfully because a user
  // IterationCallback returned USER_SOLVER_TERMINATE_SUCCESSFULLY.
  //
  USER_SUCCESS,

  // Minimizer terminated because because a user IterationCallback
  // returned USER_SOLVER_ABORT.
  USER_FAILURE
};

template <>
inline std::string to_string(const TerminationType& termination_type) {
  std::string str = "";
  switch (termination_type) {
    case TerminationType::CONVERGENCE: {
      str = "CONVERGENCE";
      break;
    }
    case TerminationType::NO_CONVERGENCE: {
      str = "NO_CONVERGENCE";
      break;
    }
    case TerminationType::FAILURE: {
      str = "FAILURE";
      break;
    }
    case TerminationType::USER_SUCCESS: {
      str = "USER_SUCCESS";
      break;
    }
    case TerminationType::USER_FAILURE: {
      str = "USER_FAILURE";
      break;
    }
  }
  return str;
}

class IterationCallback {
 public:
  virtual ~IterationCallback() = default;
  virtual CallbackReturnType operator()(const IterationSummary& summary) = 0;
};

struct NonlinearOptimizerOptions {
  bool throw_exception_if_failure = false;

  std::vector<IterationCallback*> callbacks;
};

struct NonlinearOptimizerSummary {
  TerminationType termination_type = TerminationType::FAILURE;

  // Reason why the solver terminated.
  std::string message = "dyno::NonlinearOptimizer was not called.";

  // IterationSummary for each minimizer iteration in order.
  std::vector<IterationSummary> iterations;

  size_t numIterations() const { return iterations.size(); }

  double initial_error = 0;
  double final_error = 0;

  // Time (in seconds) since the user called optimize().
  double cumulative_time_in_seconds = 0.0;

  inline bool isSolutionUsable() const {
    return (termination_type == TerminationType::CONVERGENCE ||
            termination_type == TerminationType::NO_CONVERGENCE ||
            termination_type == TerminationType::USER_SUCCESS);
  }
};

struct NonlinearOptimizerException : public DynosamException {
  NonlinearOptimizerException(const std::string& what)
      : DynosamException(what) {}
};

template <typename SOLVER>
class NonlinearOptimizer : public SOLVER {
 public:
  static_assert(std::is_base_of<gtsam::NonlinearOptimizer, SOLVER>::value,
                "SOLVER must inherit from gtsam::NonlinearOptimizer");

  using Solver = SOLVER;

  template <typename... Args>
  NonlinearOptimizer(Args&&... args) : SOLVER(std::forward<Args>(args)...) {}

  // I think breaks if called multiple times?
  bool solve(gtsam::Values& estimate, const NonlinearOptimizerOptions& options,
             NonlinearOptimizerSummary* summary) {
    CHECK_NOTNULL(summary);
    defaultOptimizeImpl(options, summary);

    estimate = this->template values();
    const bool result = summary->isSolutionUsable();

    if (!result) {
      if (options.throw_exception_if_failure) {
        throw NonlinearOptimizerException(
            "NonlinearOptimizer::solve failed with message " +
            summary->message);
      }
    }

    return result;
  }

  bool solve(gtsam::Values& estimate,
             const NonlinearOptimizerOptions& options) {
    NonlinearOptimizerSummary summary;
    return solve(estimate, options, &summary);
  }

  bool solve(gtsam::Values& estimate, NonlinearOptimizerSummary* summary) {
    NonlinearOptimizerOptions options;
    return solve(estimate, options, summary);
  }

  bool solve(gtsam::Values& estimate) {
    NonlinearOptimizerOptions options;
    NonlinearOptimizerSummary summary;
    return solve(estimate, options, &summary);
  }

  void saveGraph(
      const std::string& file_name,
      const gtsam::KeyFormatter& keyFormatter = DynosamKeyFormatter) {
    const auto& factors = this->graph();
    factors.saveGraph(file_name, keyFormatter);
  }

 private:
  // Make deafult optimize safe function
  using Solver::optimizeSafely;

  virtual const gtsam::Values& optimize() override {
    NonlinearOptimizerSummary summary;
    NonlinearOptimizerOptions options;
    defaultOptimizeImpl(options, &summary);
    return this->template values();
  };

  void defaultOptimizeImpl(const NonlinearOptimizerOptions& options,
                           NonlinearOptimizerSummary* summary) {
    const auto optimize_start_tic = utils::Timer::tic();

    const gtsam::NonlinearOptimizerParams& params = this->_params();
    double currentError = this->template error();
    summary->initial_error = currentError;

    // check if we're already close enough
    if (currentError <= params.errorTol) {
      summary->termination_type = TerminationType::CONVERGENCE;
      summary->message = "Solver initialised within error tolerance";
      return;
    }
    // Iterative loop
    double newError = currentError;  // used to avoid repeated calls to error()
    int iterations = 0;
    do {
      // Do next iteration
      currentError = newError;
      const auto iterate_tic = utils::Timer::tic();
      this->template iterate();
      const auto iterate_toc = utils::Timer::toc(iterate_tic);
      const auto cumulative_iterate_toc = utils::Timer::toc(optimize_start_tic);
      // Update newError for either printouts or conditional-end checks:
      newError = this->template error();
      iterations = this->template iterations();

      IterationSummary iteration_summary;
      iteration_summary.iteration_time_in_seconds =
          utils::Timer::toSeconds(iterate_toc);
      iteration_summary.cumulative_time_in_seconds =
          utils::Timer::toSeconds(cumulative_iterate_toc);
      iteration_summary.iterations = iterations;
      iteration_summary.error = newError;
      iteration_summary.cost_change = newError - currentError;

      summary->iterations.push_back(iteration_summary);

      if (!runCallbacks(options, iteration_summary, summary)) {
        break;
      }
    } while (
        !shouldTerminate(iterations, currentError, newError, params, summary));

    summary->final_error = newError;

    const auto cumulative_iterate_toc = utils::Timer::toc(optimize_start_tic);
    summary->cumulative_time_in_seconds =
        utils::Timer::toSeconds(cumulative_iterate_toc);
  }

  bool shouldTerminate(int iterations, double currentError, double newError,
                       const gtsam::NonlinearOptimizerParams& params,
                       NonlinearOptimizerSummary* summary) {
    const bool exceeded_max_iterations = iterations >= params.maxIterations;
    const bool has_converged = gtsam::checkConvergence(
        params.relativeErrorTol, params.absoluteErrorTol, params.errorTol,
        currentError, newError, params.verbosity);
    const bool has_infinite_error = std::isinf(currentError);

    if (has_converged) {
      summary->termination_type = TerminationType::CONVERGENCE;
      summary->message = "Solver converged successfully";
    } else if (has_infinite_error) {
      summary->termination_type = TerminationType::FAILURE;
      summary->message = "Solver terminated with INF error";
    } else if (exceeded_max_iterations) {
      summary->termination_type = TerminationType::NO_CONVERGENCE;
      summary->message = "Solver terminated with max iterations";
    }

    const bool should_terminate =
        exceeded_max_iterations || has_infinite_error || has_converged;
    return should_terminate;
  }

  static bool runCallbacks(const NonlinearOptimizerOptions& options,
                           const IterationSummary& iteration_summary,
                           NonlinearOptimizerSummary* summary) {
    CallbackReturnType status = CallbackReturnType::USER_SOLVER_CONTINUE;
    int i = 0;
    while (status == CallbackReturnType::USER_SOLVER_CONTINUE &&
           i < options.callbacks.size()) {
      status = (*options.callbacks[i])(iteration_summary);
      ++i;
    }

    switch (status) {
      case CallbackReturnType::USER_SOLVER_CONTINUE:
        return true;
      case CallbackReturnType::USER_SOLVER_TERMINATE_SUCCESSFULLY:
        summary->termination_type = TerminationType::USER_SUCCESS;
        summary->message =
            "User callback returned USER_SOLVER_TERMINATE_SUCCESSFULLY.";
        return false;
      case CallbackReturnType::USER_SOLVER_ABORT:
        summary->termination_type = TerminationType::USER_FAILURE;
        summary->message = "User callback returned USER_SOLVER_ABORT.";
        return false;
      default:
        LOG(FATAL) << "Unknown type of user callback status";
    }
    return false;
  }
};

/**
 * @brief Provides an interface for gtsam's NonlinearOptimizer classes where
 * the linear system (Ax=b, as represented by a gtsam::GaussianFactorGraph)
 * is solved using dense decomposition rather than sparse Cholesky/QR.
 *
 * @tparam SOLVER Derived Nonlinear solver, must inherit from
 * gtsam::NonlinearOptimizer.
 */
template <typename SOLVER>
class DenseNonlinearSolver : public SOLVER {
 public:
  static_assert(std::is_base_of<gtsam::NonlinearOptimizer, SOLVER>::value,
                "SOLVER must inherit from gtsam::NonlinearOptimizer");

  using Solver = SOLVER;

  template <typename... Args>
  DenseNonlinearSolver(Args&&... args) : SOLVER(std::forward<Args>(args)...) {}

  /** Overwrite the linear solve Ax=b via dense composition (rather than the
   * sparse solve as used by default)*/
  gtsam::VectorValues solve(
      const gtsam::GaussianFactorGraph& gfg,
      const gtsam::NonlinearOptimizerParams&) const override {
    // internal linear solve using Eigen's dense LLT solve
    return gfg.optimizeDensely();
  }
};

}  // namespace dyno
