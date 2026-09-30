/*
 *   Copyright (c) 2025 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a
 copy
 *   of this software and associated documentation files (the "Software"), to
 deal
 *   in the Software without restriction, including without limitation the
 rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in
 all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE
 *   SOFTWARE.
 */

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/NonlinearOptimizerParams.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>

#include <dynosam_opt/IncrementalOptimization.hpp>

#include "dynosam/factors/Pose3FlowProjectionFactor.h"
#include "internal/math.hpp"

using namespace dyno;

TEST(IncrementalOptInterface, testBasic) {
  gtsam::ISAM2 smoother;
  IncrementalInterface<gtsam::ISAM2> ii(&smoother);
}

using Camera = gtsam::PinholeCamera<gtsam::Cal3_S2>;
using FlowProjectionFactor = Pose3FlowProjectionFactor2<gtsam::Cal3_S2>;

using Clock = std::chrono::steady_clock;

inline double elapsedMs(const Clock::time_point& start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

struct SolverTiming {
  double linearize_ms = 0.0;
  double factor_eval_ms = 0.0;
  double matrix_ops_ms = 0.0;
  double local_solve_ms = 0.0;
  double pose_solve_ms = 0.0;
  double flow_recovery_ms = 0.0;
  double pose_update_ms = 0.0;

  void add(const SolverTiming& other) {
    linearize_ms += other.linearize_ms;
    factor_eval_ms += other.factor_eval_ms;
    matrix_ops_ms += other.matrix_ops_ms;
    local_solve_ms += other.local_solve_ms;
    pose_solve_ms += other.pose_solve_ms;
    flow_recovery_ms += other.flow_recovery_ms;
    pose_update_ms += other.pose_update_ms;
  }

  void divide(double value) {
    linearize_ms /= value;
    factor_eval_ms /= value;
    matrix_ops_ms /= value;
    local_solve_ms /= value;
    pose_solve_ms /= value;
    flow_recovery_ms /= value;
    pose_update_ms /= value;
  }
};

inline void printTiming(const SolverTiming& timing, size_t num_features,
                        size_t num_iterations_per_run, size_t num_runs) {
  const double features_per_run = static_cast<double>(num_features) *
                                  static_cast<double>(num_iterations_per_run);

  const auto perFeatureIteration = [features_per_run](double ms) {
    return ms / features_per_run * 1000.0;
  };

  // IMPORTANT:
  //
  // linearize_ms is the complete linearisation + Schur stage.
  // factor_eval_ms, matrix_ops_ms and local_solve_ms are sub-components
  // of linearize_ms and therefore must NOT be added to the total.
  //
  const double total_ms = timing.linearize_ms + timing.pose_solve_ms +
                          timing.flow_recovery_ms + timing.pose_update_ms;

  std::cout << std::fixed << std::setprecision(3);

  std::cout << "\n";
  std::cout << "FastFlowPoseOptimizer timing\n";
  std::cout << "============================\n";
  std::cout << "Runs:                 " << num_runs << '\n';
  std::cout << "Features:             " << num_features << '\n';
  std::cout << "Iterations / run:     " << num_iterations_per_run << '\n';
  std::cout << "\n";

  std::cout << "Average timing per optimize() call\n\n";

  std::cout << std::left << std::setw(30) << "Stage" << std::right
            << std::setw(14) << "[ms]" << std::setw(24)
            << "[us / feature / iter]" << '\n';

  std::cout
      << "---------------------------------------------------------------\n";

  const auto printRow = [&](const char* name, double value) {
    std::cout << std::left << std::setw(30) << name << std::right
              << std::setw(14) << value << std::setw(24)
              << perFeatureIteration(value) << '\n';
  };

  printRow("Linearisation + Schur", timing.linearize_ms);

  printRow("  Factor evaluation", timing.factor_eval_ms);

  printRow("  Matrix operations", timing.matrix_ops_ms);

  printRow("  2x2 local solves", timing.local_solve_ms);

  printRow("Pose 6x6 solve", timing.pose_solve_ms);

  printRow("Flow recovery", timing.flow_recovery_ms);

  printRow("Pose update", timing.pose_update_ms);

  std::cout
      << "---------------------------------------------------------------\n";

  printRow("TOTAL", total_ms);

  std::cout << '\n';
}

class FastFlowPoseOptimizer {
 public:
  using Camera = gtsam::PinholeCamera<gtsam::Cal3_S2>;
  using FlowProjectionFactor =
      Pose3FlowProjectionFactor2<Camera::CalibrationType>;

  using Matrix66 = Eigen::Matrix<double, 6, 6>;
  using Matrix62 = Eigen::Matrix<double, 6, 2>;
  using Matrix26 = Eigen::Matrix<double, 2, 6>;
  using Matrix22 = Eigen::Matrix<double, 2, 2>;
  using Vector6 = Eigen::Matrix<double, 6, 1>;
  using Vector2 = Eigen::Vector2d;

  enum class RobustLoss { None, Huber };

  struct Measurement {
    gtsam::Point2 ref_kp;
    gtsam::Point3 landmark;
    gtsam::Point2 measured_flow;
    gtsam::Point2 flow;
  };

  struct Params {
    size_t max_iterations = 2;

    // Projection factor noise.
    double flow_sigma = 1.0;

    // Prior on the flow variable.
    double flow_prior_sigma = 1.0;

    // Robust loss applied only to the projection factor.
    RobustLoss robust_loss = RobustLoss::None;

    // Huber threshold in whitened residual units.
    // This matches GTSAM's Huber parameter k.
    double huber_k = 1.345;
  };

  struct IterationStats {
    double avg_projection_error = 0.0;
    // GTSAM-style unnormalized nonlinear error:
    // sum 0.5 * ||whitened residual||^2
    double regular_error = 0.0;

    // Error after applying the robust loss.
    double robust_error = 0.0;

    // // Useful diagnostics.
    // double max_whitened_error = 0.0;
    // double pose_update_norm = 0.0;
    // double max_flow_update_norm = 0.0;
  };

  static void printIterationStats(const IterationStats& stats,
                                  std::size_t iteration,
                                  std::ostream& os = std::cout) {
    os << std::fixed << std::setprecision(6) << "Iteration " << iteration
       << '\n'
       << "  Projection error       : " << stats.avg_projection_error << '\n'
       << "  Regular error          : " << stats.regular_error << '\n'
       << "  Robust error           : " << stats.robust_error << '\n';
    // << "  Max whitened error     : " << stats.max_whitened_error << '\n'
    // << "  Pose update norm       : " << stats.pose_update_norm << '\n'
    // << "  Max flow update norm   : " << stats.max_flow_update_norm << '\n';
  }

  struct Result {
    gtsam::Pose3 pose;
    std::vector<gtsam::Point2> flows;
    size_t iterations = 0;
    SolverTiming timing;
  };

  FastFlowPoseOptimizer(const Camera::CalibrationType& calibration,
                        Params params)
      : calibration_(calibration), params_(params) {
    validateParams();

    switch (params_.robust_loss) {
      case RobustLoss::None:
        loss_ = gtsam::noiseModel::mEstimator::Null::Create();
        break;

      case RobustLoss::Huber:
        loss_ = gtsam::noiseModel::mEstimator::Huber::Create(params_.huber_k);
        break;
        // return huberWeight(whitened_error.norm(), params_.huber_k);
    }
  }

  Result optimize(const gtsam::Pose3& initial_pose,
                  const std::vector<Measurement>& input_measurements) const {
    gtsam::Pose3 pose = initial_pose;

    // We need to update the flow variables, so copy the input once.
    std::vector<Measurement> measurements = input_measurements;

    SolverTiming timing;

    const double flow_information =
        1.0 / (params_.flow_sigma * params_.flow_sigma);

    const double prior_information =
        1.0 / (params_.flow_prior_sigma * params_.flow_prior_sigma);

    const double flow_sqrt_information = std::sqrt(flow_information);
    const double prior_sqrt_information = std::sqrt(prior_information);

    struct LinearizedMeasurement {
      Matrix62 Hxf;
      Matrix22 Hff;
      Vector2 bf;
    };

    std::vector<LinearizedMeasurement> linearized(measurements.size());

    size_t iterations = 0;
    double current_error = 0.0;
    IterationStats stats;
    do {
      stats = IterationStats();

      const auto linearize_start = Clock::now();

      Matrix66 H = Matrix66::Zero();
      Vector6 b = Vector6::Zero();

      for (size_t i = 0; i < measurements.size(); ++i) {
        Measurement& measurement = measurements[i];

        Eigen::Matrix<double, 2, 2> Jf;
        Eigen::Matrix<double, 2, 6> Jx;

        // --------------------------------------------------------------
        // Evaluate projection factor.
        // --------------------------------------------------------------

        const auto factor_start = Clock::now();

        const gtsam::Point2 projection_error =
            evaluateProjection(measurement, pose, Jf, Jx);
        stats.avg_projection_error += projection_error.norm();

        timing.factor_eval_ms += elapsedMs(factor_start);

        // --------------------------------------------------------------
        // Whiten projection factor.
        // --------------------------------------------------------------

        const auto matrix_start = Clock::now();

        // whitened error
        gtsam::Vector2 whitened_error =
            projection_error * flow_sqrt_information;
        stats.regular_error += 0.5 * whitened_error.squaredNorm();

        // whitened Jacobians
        Jf *= flow_sqrt_information;
        Jx *= flow_sqrt_information;

        // --------------------------------------------------------------
        // Robust reweighting.
        //
        // This matches GTSAM's Block robust weighting:
        //
        //   w = HuberWeight(||r||)
        //
        // followed by:
        //
        //   J <- sqrt(w) J
        //   r <- sqrt(w) r
        //
        // where r is already whitened by sigma.
        // --------------------------------------------------------------

        const double whitened_distance = whitened_error.norm();
        const double robust_weight = loss_->weight(whitened_distance);
        const double sqrt_weight = std::sqrt(robust_weight);

        stats.robust_error += loss_->loss(whitened_distance);

        // const double robust_weight_manual = projectionWeight(whitened_error);
        // const double sqrt_weight_manual = std::sqrt(robust_weight_manual);

        if (robust_weight != 1.0) {
          Jf *= sqrt_weight;
          Jx *= sqrt_weight;
          whitened_error *= sqrt_weight;
        }
        // Jf *= sqrt_weight;
        // Jx *= sqrt_weight;
        // whitened_error *= sqrt_weight;

        // --------------------------------------------------------------
        // Projection Hessian and gradient contributions.
        // --------------------------------------------------------------
        const Matrix62 Jx_T = Jx.transpose();
        const Matrix22 Jf_T = Jf.transpose();

        const Matrix66 Hxx = Jx_T * Jx;

        const Matrix62 Hxf = Jx_T * Jf;

        Matrix22 Hff = Jf_T * Jf;

        const Vector6 bx = Jx_T * whitened_error;

        Vector2 bf = Jf_T * whitened_error;

        // --------------------------------------------------------------
        // Flow prior.
        //
        // The prior is intentionally NOT robustified.
        // --------------------------------------------------------------

        const Vector2 prior_r = (measurement.flow - measurement.measured_flow)
                                    .template cast<double>();

        // Hessian: Jᵀ Λ J, with J = I.
        Hff.noalias() += prior_information * Matrix22::Identity();

        // Gradient: Jᵀ Λ r, with J = I.
        const Vector2 prior_gradient = prior_information * prior_r;

        bf.noalias() += prior_gradient;

        // Error: 0.5 rᵀ Λ r = 0.5 ||Λ½ r||².
        const Vector2 whitened_prior_r = prior_sqrt_information * prior_r;

        const double prior_error = 0.5 * whitened_prior_r.squaredNorm();

        stats.regular_error += prior_error;
        stats.robust_error += prior_error;

        timing.matrix_ops_ms += elapsedMs(matrix_start);

        // --------------------------------------------------------------
        // Save local system for flow recovery.
        // --------------------------------------------------------------

        linearized[i].Hxf = Hxf;
        linearized[i].Hff = Hff;
        linearized[i].bf = bf;

        // --------------------------------------------------------------
        // Schur complement.
        // --------------------------------------------------------------

        const auto local_solve_start = Clock::now();

        // Factor Hff only once.
        const Eigen::LDLT<Matrix22> Hff_ldlt(Hff);

        const Matrix26 Hff_inv_Hfx = Hff_ldlt.solve(Hxf.transpose());

        const Vector2 Hff_inv_bf = Hff_ldlt.solve(bf);

        H.noalias() += Hxx - Hxf * Hff_inv_Hfx;

        b.noalias() += bx - Hxf * Hff_inv_bf;

        timing.local_solve_ms += elapsedMs(local_solve_start);
      }

      if (measurements.size() > 0) {
        stats.avg_projection_error /= (double)measurements.size();
      } else {
        stats.avg_projection_error = 0.0;
      }

      printIterationStats(stats, iterations);
      double error_change = current_error - stats.regular_error;
      current_error = stats.regular_error;

      timing.linearize_ms += elapsedMs(linearize_start);

      // ---------------------------------------------------------------
      // Solve reduced pose system.
      // ---------------------------------------------------------------

      const auto pose_solve_start = Clock::now();

      const Vector6 dx = H.ldlt().solve(-b);

      timing.pose_solve_ms += elapsedMs(pose_solve_start);

      // ---------------------------------------------------------------
      // Recover flow updates from exactly the same linearization.
      // ---------------------------------------------------------------

      const auto flow_start = Clock::now();

      for (size_t i = 0; i < measurements.size(); ++i) {
        const LinearizedMeasurement& lin = linearized[i];

        const Eigen::LDLT<Matrix22> Hff_ldlt(lin.Hff);

        const Vector2 df = -Hff_ldlt.solve(lin.bf + lin.Hxf.transpose() * dx);

        measurements[i].flow += gtsam::Point2(df.x(), df.y());
      }

      timing.flow_recovery_ms += elapsedMs(flow_start);

      // ---------------------------------------------------------------
      // Update pose.
      // ---------------------------------------------------------------

      const auto pose_update_start = Clock::now();

      pose = pose.retract(gtsam::Vector6(dx.data()));

      timing.pose_update_ms += elapsedMs(pose_update_start);
      ++iterations;
    } while (iterations < params_.max_iterations);

    std::vector<gtsam::Point2> flows;
    flows.reserve(measurements.size());

    for (const Measurement& measurement : measurements) {
      flows.push_back(measurement.flow);
    }

    return {pose, std::move(flows), iterations, timing};
  }

  static double huberWeight(const double norm, const double k) {
    if (norm <= k) {
      return 1.0;
    }

    return k / norm;
  }

 private:
  double projectionWeight(const Vector2& whitened_error) const {
    switch (params_.robust_loss) {
      case RobustLoss::None:
        return 1.0;

      case RobustLoss::Huber:
        return huberWeight(whitened_error.norm(), params_.huber_k);
    }

    // Should be unreachable.
    throw std::logic_error("Unknown robust loss");
  }

  gtsam::Point2 evaluateProjection(const Measurement& measurement,
                                   const gtsam::Pose3& pose,
                                   Eigen::Matrix<double, 2, 2>& Jf,
                                   Eigen::Matrix<double, 2, 6>& Jx) const {
    // // Unit noise is intentional here. The optimizer performs
    // // whitening explicitly so that robust weighting can be applied
    // // to the whitened residual.
    // const gtsam::SharedNoiseModel unit_noise =
    //     gtsam::noiseModel::Isotropic::Sigma(2, 1.0);

    // FlowProjectionFactor factor(gtsam::Symbol('f', 0), gtsam::Symbol('X', 0),
    //                             measurement.ref_kp, measurement.landmark,
    //                             calibration_, unit_noise);

    // gtsam::Matrix H1;
    // gtsam::Matrix H2;

    // const gtsam::Vector2 error =
    //     factor.evaluateError(measurement.flow, pose, H1, H2);

    // Jf = H1;
    // Jx = H2;

    // probably dont need to recompute this every time
    Pose3FlowProjectionResidual2 residual(measurement.ref_kp,
                                          measurement.landmark, calibration_);

    gtsam::Matrix H1;
    gtsam::Matrix H2;

    gtsam::Vector2 error = residual(measurement.flow, pose, H1, H2);
    Jf = H1;
    Jx = H2;

    return error;

    // return gtsam::Point2(error(0), error(1));
  }

  void validateParams() const {
    if (params_.max_iterations == 0) {
      throw std::invalid_argument("max_iterations must be greater than zero");
    }

    if (!(params_.flow_sigma > 0.0)) {
      throw std::invalid_argument("flow_sigma must be greater than zero");
    }

    if (!(params_.flow_prior_sigma > 0.0)) {
      throw std::invalid_argument("flow_prior_sigma must be greater than zero");
    }

    if (params_.robust_loss == RobustLoss::Huber && !(params_.huber_k > 0.0)) {
      throw std::invalid_argument("huber_k must be greater than zero");
    }
  }

  using Clock = std::chrono::steady_clock;

  static double elapsedMs(const Clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start)
        .count();
  }

  const Camera::CalibrationType& calibration_;
  Params params_;
  gtsam::noiseModel::mEstimator::Base::shared_ptr loss_;
};

struct FlowRefinementProblem {
  gtsam::NonlinearFactorGraph graph;
  gtsam::Values values;
  gtsam::Ordering ordering;

  // Useful for validating / profiling the generated problem.
  gtsam::Pose3 true_pose;
  gtsam::Pose3 initial_pose;

  std::vector<FastFlowPoseOptimizer::Measurement> measurements;

  size_t num_features = 0;

  gtsam::Cal3_S2 calibration;
};

FlowRefinementProblem makeFlowRefinementProblem(size_t num_features,
                                                double flow_sigma = 1.0,
                                                double flow_prior_sigma = 1.0,
                                                bool use_robust = false,
                                                double huber_k = 1.345) {
  FlowRefinementProblem problem;
  problem.num_features = num_features;

  // --------------------------------------------------------------------------
  // Camera
  // --------------------------------------------------------------------------

  // Representative 640x480 pinhole camera.
  //
  // fx/fy ~ 500 px is deliberately fairly ordinary for a calibrated RGB
  // camera rather than using an unrealistic huge focal length.
  const gtsam::Cal3_S2 calibration(500.0,   // fx
                                   500.0,   // fy
                                   0.0,     // s
                                   320.0,   // cx
                                   240.0);  // cy
  problem.calibration = calibration;

  // --------------------------------------------------------------------------
  // Ground-truth camera pose
  // --------------------------------------------------------------------------

  const gtsam::Pose3 true_pose(gtsam::Rot3::RzRyRx(0.0, 0.0, 0.0),
                               gtsam::Point3(0.0, 0.0, 0.0));

  // Initial pose is deliberately slightly wrong.
  //
  // This makes the optimisation actually do useful work instead of starting
  // exactly at the optimum.
  const gtsam::Pose3 initial_pose(gtsam::Rot3::RzRyRx(0.005,   // roll
                                                      -0.008,  // pitch
                                                      0.01),   // yaw
                                  gtsam::Point3(0.02, -0.015, 0.03));

  problem.true_pose = true_pose;
  problem.initial_pose = initial_pose;

  // --------------------------------------------------------------------------
  // Noise
  // --------------------------------------------------------------------------

  gtsam::SharedNoiseModel flow_noise =
      gtsam::noiseModel::Isotropic::Sigma(2u, flow_sigma);

  const gtsam::SharedNoiseModel flow_prior_noise =
      gtsam::noiseModel::Isotropic::Sigma(2u, flow_prior_sigma);

  if (use_robust) {
    flow_noise = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(huber_k), flow_noise);
  }

  // --------------------------------------------------------------------------
  // Generate realistic 3D points
  // --------------------------------------------------------------------------

  std::mt19937 rng(42);

  // Keep points comfortably inside the image.
  std::uniform_real_distribution<double> x_dist(-2.0, 2.0);
  std::uniform_real_distribution<double> y_dist(-1.4, 1.4);

  // Realistic scene depth.
  std::uniform_real_distribution<double> z_dist(4.0, 12.0);

  // Small optical-flow measurement noise.
  std::normal_distribution<double> flow_noise_dist(0.0, flow_sigma);

  const gtsam::Symbol X_sym('X', 0);

  // --------------------------------------------------------------------------
  // Generate factors
  // --------------------------------------------------------------------------

  for (size_t i = 0; i < num_features; ++i) {
    const gtsam::Key flow_key = gtsam::Symbol('f', i);

    // 3D landmark in the reference camera frame.
    const gtsam::Point3 landmark(x_dist(rng), y_dist(rng), z_dist(rng));

    // Reference keypoint.
    //
    // The reference pose is identity, so projection is simply the landmark
    // projected into the reference camera.
    const Camera reference_camera(true_pose, calibration);

    const gtsam::Point2 ref_kp = reference_camera.project(landmark);

    // Reject points which are outside the image.
    //
    // This keeps the synthetic data representative of your actual frontend.
    if (ref_kp.x() < 10.0 || ref_kp.x() > 630.0 || ref_kp.y() < 10.0 ||
        ref_kp.y() > 470.0) {
      --i;
      continue;
    }

    // Project the same landmark from the true camera pose.
    const Camera current_camera(true_pose, calibration);

    const gtsam::Point2 true_kp = current_camera.project(landmark);

    const gtsam::Point2 true_flow = true_kp - ref_kp;

    // Simulate the measured optical flow.
    const gtsam::Point2 measured_flow(true_flow.x() + flow_noise_dist(rng),
                                      true_flow.y() + flow_noise_dist(rng));

    // ------------------------------------------------------------------------
    // Flow projection factor
    // ------------------------------------------------------------------------

    problem.graph.add(boost::make_shared<FlowProjectionFactor>(
        flow_key, X_sym, ref_kp, landmark, calibration, flow_noise));

    problem.measurements.push_back(
        {ref_kp, landmark, measured_flow, measured_flow});

    // ------------------------------------------------------------------------
    // Flow prior
    // ------------------------------------------------------------------------

    problem.graph.addPrior<gtsam::Point2>(flow_key, measured_flow,
                                          flow_prior_noise);

    // ------------------------------------------------------------------------
    // Initial values
    // ------------------------------------------------------------------------

    problem.values.insert(flow_key, measured_flow);

    problem.ordering += flow_key;
  }

  // Pose comes last, matching your current explicit ordering.
  problem.values.insert(X_sym, initial_pose);

  problem.ordering += X_sym;

  return problem;
}

using Clock = std::chrono::steady_clock;

struct TimingResult {
  double mean_ms;
  double min_ms;
  double max_ms;
};

template <typename Fn>
TimingResult benchmark(Fn&& fn, size_t warmup_iterations = 5,
                       size_t iterations = 50) {
  // Warm up caches, Eigen paths, allocator behaviour, etc.
  for (size_t i = 0; i < warmup_iterations; ++i) {
    fn();
  }

  std::vector<double> times;
  times.reserve(iterations);

  for (size_t i = 0; i < iterations; ++i) {
    const auto start = Clock::now();

    fn();

    const auto end = Clock::now();

    times.push_back(
        std::chrono::duration<double, std::milli>(end - start).count());
  }

  const double mean = std::accumulate(times.begin(), times.end(), 0.0) /
                      static_cast<double>(times.size());

  const auto [min_it, max_it] = std::minmax_element(times.begin(), times.end());

  return {
      mean,
      *min_it,
      *max_it,
  };
}

void benchmarkCustomSolver(const FastFlowPoseOptimizer::Params& params,
                           const FlowRefinementProblem& problem,
                           size_t warmup_iterations = 5,
                           size_t benchmark_iterations = 50) {
  FastFlowPoseOptimizer optimizer(problem.calibration, params);

  // ------------------------------------------------------------------------
  // Warmup
  //
  // These runs are deliberately NOT included in the reported timing.
  // ------------------------------------------------------------------------

  for (size_t i = 0; i < warmup_iterations; ++i) {
    auto result =
        optimizer.optimize(problem.initial_pose, problem.measurements);

    // Prevent the compiler from considering the result unused.
    volatile double pose_value = result.pose.translation().x();

    (void)pose_value;
  }

  // ------------------------------------------------------------------------
  // Benchmark
  // ------------------------------------------------------------------------

  SolverTiming total_timing;

  for (size_t i = 0; i < benchmark_iterations; ++i) {
    auto result =
        optimizer.optimize(problem.initial_pose, problem.measurements);

    total_timing.add(result.timing);

    // Prevent optimisation away.
    volatile double pose_value = result.pose.translation().x();

    (void)pose_value;
  }

  // ------------------------------------------------------------------------
  // Convert accumulated timing into average timing per optimize() call.
  // ------------------------------------------------------------------------

  total_timing.divide(static_cast<double>(benchmark_iterations));

  printTiming(total_timing, problem.num_features, params.max_iterations,
              benchmark_iterations);
}

TEST(GtsamFlowRefinementBenchmark, BuildProblems) {
  for (const size_t num_features : {50, 100, 200, 500, 1000, 2000, 5000}) {
    auto problem = makeFlowRefinementProblem(num_features);

    ASSERT_EQ(problem.values.size(), num_features + 1);

    ASSERT_EQ(problem.ordering.size(), num_features + 1);

    std::cout << "features=" << num_features
              << " variables=" << problem.values.size()
              << " factors=" << problem.graph.size() << '\n';
  }
}

TEST(GtsamFlowRefinementBenchmark, CustomVsGtsam) {
  constexpr size_t kRuns = 50;

  const auto problem = makeFlowRefinementProblem(1000, 1.0, 1.0, false);

  for (const size_t max_iterations : {4u, 10u}) {
    // ======================================================================
    // GN
    // ======================================================================

    {
      gtsam::GaussNewtonParams params;
      params.setMaxIterations(max_iterations);
      params.setOrdering(problem.ordering);

      const auto timing = benchmark(
          [&]() {
            gtsam::GaussNewtonOptimizer optimizer(problem.graph, problem.values,
                                                  params);

            const gtsam::Values result = optimizer.optimize();

            ASSERT_EQ(result.size(), problem.values.size());
          },
          5, kRuns);

      std::cout << "GN  "
                << "N=" << problem.num_features << " iter=" << max_iterations
                << " mean=" << timing.mean_ms << " ms\n";
    }

    // ======================================================================
    // LM
    // ======================================================================

    {
      gtsam::LevenbergMarquardtParams params;
      params.setMaxIterations(max_iterations);
      params.setOrdering(problem.ordering);

      const auto timing = benchmark(
          [&]() {
            gtsam::LevenbergMarquardtOptimizer optimizer(
                problem.graph, problem.values, params);

            const gtsam::Values result = optimizer.optimize();

            ASSERT_EQ(result.size(), problem.values.size());
          },
          5, kRuns);

      std::cout << "LM  "
                << "N=" << problem.num_features << " iter=" << max_iterations
                << " mean=" << timing.mean_ms << " ms\n";
    }

    // ======================================================================
    // Custom
    // ======================================================================

    {
      FastFlowPoseOptimizer::Params params;
      params.max_iterations = max_iterations;
      params.flow_sigma = 1.0;
      params.flow_prior_sigma = 1.0;

      //   benchmarkCustomSolver(params, problem);

      FastFlowPoseOptimizer optimizer(problem.calibration, params);

      const auto timing = benchmark(
          [&]() {
            const auto result =
                optimizer.optimize(problem.initial_pose, problem.measurements);

            ASSERT_EQ(result.flows.size(), problem.num_features);
          },
          5, kRuns);

      std::cout << "CUSTOM "
                << "N=" << problem.num_features << " iter=" << max_iterations
                << " mean=" << timing.mean_ms << " ms\n";
    }
  }
}

void benchmarkEigenSmallMatrices(size_t iterations = 1000000) {
  using Matrix26 = Eigen::Matrix<double, 2, 6>;
  using Matrix62 = Eigen::Matrix<double, 6, 2>;
  using Matrix66 = Eigen::Matrix<double, 6, 6>;
  using Matrix22 = Eigen::Matrix<double, 2, 2>;
  using Vector2 = Eigen::Vector2d;
  using Vector6 = Eigen::Matrix<double, 6, 1>;

  Matrix26 Jx = Matrix26::Random();

  Matrix22 Jf = Matrix22::Random();

  Vector2 r = Vector2::Random();

  volatile double sink = 0.0;

  // ------------------------------------------------------------
  // Jx^T Jx
  // ------------------------------------------------------------

  auto start = Clock::now();

  for (size_t i = 0; i < iterations; ++i) {
    Matrix66 Hxx = Jx.transpose() * Jx;

    sink += Hxx(0, 0);
  }

  std::cout << "Jx^T Jx: " << elapsedMs(start) << " ms\n";

  // ------------------------------------------------------------
  // Jx^T Jf
  // ------------------------------------------------------------

  start = Clock::now();

  for (size_t i = 0; i < iterations; ++i) {
    Matrix62 Hxf = Jx.transpose() * Jf;

    sink += Hxf(0, 0);
  }

  std::cout << "Jx^T Jf: " << elapsedMs(start) << " ms\n";

  // ------------------------------------------------------------
  // Jf^T Jf
  // ------------------------------------------------------------

  start = Clock::now();

  for (size_t i = 0; i < iterations; ++i) {
    Matrix22 Hff = Jf.transpose() * Jf;

    sink += Hff(0, 0);
  }

  std::cout << "Jf^T Jf: " << elapsedMs(start) << " ms\n";

  // ------------------------------------------------------------
  // 2x2 LDLT solve
  // ------------------------------------------------------------

  start = Clock::now();

  for (size_t i = 0; i < iterations; ++i) {
    Matrix22 H = Jf.transpose() * Jf;

    Vector2 b = r;

    Vector2 x = H.ldlt().solve(b);

    sink += x(0);
  }

  std::cout << "2x2 LDLT solve: " << elapsedMs(start) << " ms\n";

  std::cout << "sink: " << sink << '\n';
}

void benchmarkExplicitSmallMatrices(size_t iterations = 1000000) {
  Matrix26 Jx = Matrix26::Random();
  Matrix22 Jf = Matrix22::Random();
  Vector2 r = Vector2::Random();

  volatile double sink = 0.0;

  auto start = Clock::now();

  for (size_t i = 0; i < iterations; ++i) {
    Matrix66 H = outerProduct6x2(Jx);
    sink += H(0, 0);
  }

  std::cout << "Explicit Jx^T Jx: " << elapsedMs(start) << " ms\n";

  start = Clock::now();

  for (size_t i = 0; i < iterations; ++i) {
    Matrix62 H = crossProduct6x2(Jx, Jf);
    sink += H(0, 0);
  }

  std::cout << "Explicit Jx^T Jf: " << elapsedMs(start) << " ms\n";

  start = Clock::now();

  for (size_t i = 0; i < iterations; ++i) {
    Matrix22 H = outerProduct2x2(Jf);
    sink += H(0, 0);
  }

  std::cout << "Explicit Jf^T Jf: " << elapsedMs(start) << " ms\n";

  start = Clock::now();

  for (size_t i = 0; i < iterations; ++i) {
    Vector2 x = solve2x2(Jf, r);
    sink += x(0);
  }

  std::cout << "Explicit 2x2 solve: " << elapsedMs(start) << " ms\n";

  std::cout << "sink: " << sink << '\n';
}

TEST(GtsamFlowRefinementBenchmark, eigenSmallMatices) {
  auto iter = 100000;
  benchmarkEigenSmallMatrices(iter);
  benchmarkExplicitSmallMatrices(iter);
}

TEST(GtsamFlowRefinement, CustomSolverMatchesGtsam) {
  constexpr size_t N = 100;

  const auto problem = makeFlowRefinementProblem(N, 1.0, 1.0, false);

  constexpr size_t kIterations = 2;

  // --------------------------------------------------------------------------
  // GTSAM
  // --------------------------------------------------------------------------

  gtsam::GaussNewtonParams params;
  params.setMaxIterations(kIterations);
  params.setOrdering(problem.ordering);

  gtsam::GaussNewtonOptimizer gtsam_optimizer(problem.graph, problem.values,
                                              params);

  const gtsam::Values gtsam_result = gtsam_optimizer.optimize();

  const gtsam::Pose3 gtsam_pose =
      gtsam_result.at<gtsam::Pose3>(gtsam::Symbol('X', 0));

  // --------------------------------------------------------------------------
  // Custom
  // --------------------------------------------------------------------------

  FastFlowPoseOptimizer::Params custom_params;
  custom_params.max_iterations = kIterations;
  custom_params.flow_sigma = 1.0;
  custom_params.flow_prior_sigma = 1.0;

  FastFlowPoseOptimizer custom_optimizer(problem.calibration, custom_params);

  const auto custom_result =
      custom_optimizer.optimize(problem.initial_pose, problem.measurements);

  // --------------------------------------------------------------------------
  // Compare pose.
  // --------------------------------------------------------------------------

  const gtsam::Vector6 pose_error =
      gtsam_pose.localCoordinates(custom_result.pose);

  EXPECT_LT(pose_error.norm(), 1e-6);

  // --------------------------------------------------------------------------
  // Compare flows.
  // --------------------------------------------------------------------------

  for (size_t i = 0; i < N; ++i) {
    const gtsam::Point2 gtsam_flow =
        gtsam_result.at<gtsam::Point2>(gtsam::Symbol('f', i));

    const gtsam::Point2 custom_flow = custom_result.flows.at(i);

    EXPECT_LT((gtsam_flow - custom_flow).norm(), 1e-6);
  }
}

TEST(FastFlowPoseOptimizer, HuberWeight) {
  constexpr double k = 1.345;

  EXPECT_DOUBLE_EQ(FastFlowPoseOptimizer::huberWeight(0.0, k), 1.0);

  EXPECT_DOUBLE_EQ(FastFlowPoseOptimizer::huberWeight(1.0, k), 1.0);

  EXPECT_DOUBLE_EQ(FastFlowPoseOptimizer::huberWeight(k, k), 1.0);

  EXPECT_NEAR(FastFlowPoseOptimizer::huberWeight(2.0 * k, k), 0.5, 1e-12);

  EXPECT_NEAR(FastFlowPoseOptimizer::huberWeight(10.0, k), k / 10.0, 1e-12);
}

struct GtsamResult {
  gtsam::Pose3 pose;
  std::vector<gtsam::Point2> flows;
};

GtsamResult optimizeWithGtsam(
    const Camera::CalibrationType& calibration,
    const FastFlowPoseOptimizer::Params& params,
    const gtsam::Pose3& initial_pose,
    const std::vector<FastFlowPoseOptimizer::Measurement>& measurements) {
  gtsam::NonlinearFactorGraph graph;
  gtsam::Values values;
  gtsam::Ordering ordering;

  const gtsam::Symbol X_sym('X', 0);

  gtsam::SharedNoiseModel flow_noise =
      gtsam::noiseModel::Isotropic::Sigma(2, params.flow_sigma);

  if (params.robust_loss == FastFlowPoseOptimizer::RobustLoss::Huber) {
    flow_noise = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(params.huber_k),
        flow_noise);
  }

  const gtsam::SharedNoiseModel prior_noise =
      gtsam::noiseModel::Isotropic::Sigma(2, params.flow_prior_sigma);

  for (size_t i = 0; i < measurements.size(); ++i) {
    const auto& measurement = measurements[i];

    const gtsam::Symbol flow_sym('f', static_cast<uint64_t>(i));

    graph.emplace_shared<FlowProjectionFactor>(
        flow_sym, X_sym, measurement.ref_kp, measurement.landmark, calibration,
        flow_noise);

    graph.addPrior<gtsam::Point2>(flow_sym, measurement.measured_flow,
                                  prior_noise);

    values.insert(flow_sym, measurement.flow);

    ordering += flow_sym;
  }

  values.insert(X_sym, initial_pose);
  ordering += X_sym;

  gtsam::GaussNewtonParams gn_params;

  gn_params.setMaxIterations(static_cast<int>(params.max_iterations));

  gn_params.setOrdering(ordering);

  // We want to compare a fixed number of GN iterations.
  gn_params.relativeErrorTol = 0.0;
  gn_params.absoluteErrorTol = 0.0;

  gtsam::GaussNewtonOptimizer optimizer(graph, values, gn_params);

  const gtsam::Values result = optimizer.optimize();

  GtsamResult output;
  output.pose = result.at<gtsam::Pose3>(X_sym);

  output.flows.reserve(measurements.size());

  for (size_t i = 0; i < measurements.size(); ++i) {
    const gtsam::Symbol flow_sym('f', static_cast<uint64_t>(i));

    output.flows.push_back(result.at<gtsam::Point2>(flow_sym));
  }

  return output;
}

TEST(FastFlowPoseOptimizer, MatchesGtsamGaussNewton) {
  FastFlowPoseOptimizer::Params params;
  params.max_iterations = 2;
  params.flow_sigma = 1.0;
  params.flow_prior_sigma = 1.0;
  params.robust_loss = FastFlowPoseOptimizer::RobustLoss::None;

  std::vector<FastFlowPoseOptimizer::Measurement> measurements;

  // Use your existing problem generator here.
  auto problem = makeFlowRefinementProblem(400);
  measurements = problem.measurements;

  const gtsam::Pose3 initial_pose = problem.initial_pose;

  FastFlowPoseOptimizer fast(problem.calibration, params);

  const auto fast_result = fast.optimize(initial_pose, measurements);

  const auto gtsam_result = optimizeWithGtsam(problem.calibration, params,
                                              initial_pose, measurements);

  EXPECT_TRUE(fast_result.pose.equals(gtsam_result.pose, 1e-8));

  ASSERT_EQ(fast_result.flows.size(), gtsam_result.flows.size());

  for (size_t i = 0; i < fast_result.flows.size(); ++i) {
    EXPECT_TRUE(fast_result.flows[i].isApprox(gtsam_result.flows[i], 1e-8))
        << "Flow " << i;
  }
}

TEST(FastFlowPoseOptimizer, RobustHuberMatchesGtsam) {
  FastFlowPoseOptimizer::Params params;
  params.max_iterations = 2;
  params.flow_sigma = 1.0;
  params.flow_prior_sigma = 1.0;
  params.robust_loss = FastFlowPoseOptimizer::RobustLoss::Huber;
  params.huber_k = 1.345;

  auto problem = makeFlowRefinementProblem(400);
  auto measurements = problem.measurements;

  // Inject a large projection outlier.
  ASSERT_GT(measurements.size(), 10u);

  measurements[3].flow.x() += 25.0;
  measurements[3].flow.y() -= 15.0;

  measurements[7].flow.x() -= 20.0;
  measurements[7].flow.y() += 10.0;

  const gtsam::Pose3 initial_pose = problem.initial_pose;

  FastFlowPoseOptimizer fast(problem.calibration, params);

  const auto fast_result = fast.optimize(initial_pose, measurements);

  const auto gtsam_result = optimizeWithGtsam(problem.calibration, params,
                                              initial_pose, measurements);

  EXPECT_TRUE(fast_result.pose.equals(gtsam_result.pose, 1e-7));

  ASSERT_EQ(fast_result.flows.size(), gtsam_result.flows.size());

  for (size_t i = 0; i < fast_result.flows.size(); ++i) {
    EXPECT_TRUE(fast_result.flows[i].isApprox(gtsam_result.flows[i], 1e-7))
        << "Flow " << i << "\nFast:  " << fast_result.flows[i]
        << "\nGTSAM: " << gtsam_result.flows[i];
  }
}

TEST(FastFlowPoseOptimizer, HuberDownweightsLargeProjectionResidual) {
  auto problem = makeFlowRefinementProblem(400);
  auto measurements = problem.measurements;
  ASSERT_GT(measurements.size(), 0u);

  // Create a substantial outlier.
  measurements[0].flow.x() += 50.0;
  measurements[0].flow.y() += 50.0;

  const gtsam::Pose3 initial_pose = problem.initial_pose;

  FastFlowPoseOptimizer::Params normal_params;
  normal_params.max_iterations = 2;

  FastFlowPoseOptimizer::Params robust_params = normal_params;

  robust_params.robust_loss = FastFlowPoseOptimizer::RobustLoss::Huber;

  robust_params.huber_k = 1.345;

  FastFlowPoseOptimizer normal(problem.calibration, normal_params);

  FastFlowPoseOptimizer robust(problem.calibration, robust_params);

  const auto normal_result = normal.optimize(initial_pose, measurements);

  const auto robust_result = robust.optimize(initial_pose, measurements);

  EXPECT_FALSE(normal_result.pose.equals(robust_result.pose, 1e-10));
}
