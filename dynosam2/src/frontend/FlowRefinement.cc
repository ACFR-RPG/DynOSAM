#include "dynosam2/frontend/FlowRefinement.hpp"

#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearOptimizer.h>

#include "dynosam2/frontend/Utils.hpp"
#include "dynosam_common/utils/TimingStats.hpp"
#include "dynosam_opt/factors/Pose3FlowProjectionFactor.h"

namespace dyno2 {

struct FlowRefinement::ImplOptimizer {
  const FlowRefinementParams params_;

  using Calibration = Camera::CalibrationType;
  const Calibration& calibration_;

  gtsam::noiseModel::mEstimator::Base::shared_ptr loss_;

  double flow_information_;
  double prior_information_;
  double flow_sqrt_information_;
  double prior_sqrt_information_;

  struct Inputs {
    gtsam::Point2Vector ref_kps;
    gtsam::Point3Vector ref_lmks;
    //! doubles as the measured flow from optical flow AND the initial value for
    //! the estimated flow
    gtsam::Point2Vector measured_flows;
  };

  // TODO: and timing!
  struct TerminationCriteria {
    size_t max_iterations = 10;
    double relative_error_tol{1e-5};
    double absolute_error_tol{1e-5};
    double error_tol{0.0};

    bool shouldTerminate(size_t iterations, double current_error,
                         double new_error) const {
      const bool exceeded_max_iterations = iterations >= max_iterations;
      const bool has_converged = gtsam::checkConvergence(
          relative_error_tol, absolute_error_tol, error_tol, current_error,
          new_error, gtsam::NonlinearOptimizerParams::Verbosity::SILENT);
      const bool has_infinite_error = std::isinf(current_error);

      const bool should_terminate =
          exceeded_max_iterations || has_infinite_error || has_converged;
      return should_terminate;
    }
  };

  struct Result {
    gtsam::Pose3 pose;
    gtsam::Point2Vector flows;
    // TODO: inliers!

    double error_before{0.0};
    double error_after{0.0};

    double total_time_ms{0.0};
    size_t iterations{0};
  };

  struct IterationStats {
    double avg_projection_error = 0.0;
    // GTSAM-style unnormalized nonlinear error:
    // sum 0.5 * ||whitened residual||^2
    double regular_error = 0.0;

    // Error after applying the robust loss.
    double robust_error = 0.0;

    double error_change = 0.0;
  };

  ImplOptimizer(const FlowRefinementParams& params,
                const Calibration& calibration)
      : params_(params), calibration_(calibration) {
    // validateParams
    // setup loss
    flow_information_ = 1.0 / (params_.flow_sigma * params_.flow_sigma);
    prior_information_ =
        1.0 / (params_.flow_prior_sigma * params_.flow_prior_sigma);

    flow_sqrt_information_ = std::sqrt(flow_information_);
    prior_sqrt_information_ = std::sqrt(prior_information_);

    if (params_.use_robust) {
      loss_ = gtsam::noiseModel::mEstimator::Huber::Create(params_.k_huber);
    } else {
      loss_ = gtsam::noiseModel::mEstimator::Null::Create();
    }
  }

  ~ImplOptimizer() = default;

  Result optimize(const gtsam::Pose3& initial_pose, const Inputs& inputs,
                  const TerminationCriteria& criteria) const {
    CHECK_EQ(inputs.ref_kps.size(), inputs.ref_lmks.size());
    CHECK_EQ(inputs.ref_kps.size(), inputs.measured_flows.size());

    const size_t num_measurements = inputs.ref_kps.size();
    auto tic = dyno::utils::Timer::tic();

    Result result;
    result.pose = initial_pose;
    result.flows = inputs.measured_flows;

    gtsam::Point2Vector& refined_flows = result.flows;
    gtsam::Pose3& refined_pose = result.pose;

    std::vector<Pose3FlowProjectionResidual2> factors;
    factors.reserve(num_measurements);

    double current_error = 0.0;
    // build factors and compute initial error
    // there is some tiny overhead as we recompute the whitened
    // error here and then again during the iterations
    for (size_t i = 0; i < num_measurements; i++) {
      Pose3FlowProjectionResidual2 residual(inputs.ref_kps[i],
                                            inputs.ref_lmks[i], calibration_);
      gtsam::Vector2 unwhitened_error =
          residual(refined_flows[i], refined_pose);
      gtsam::Vector2 whitened_error = unwhitened_error * flow_sqrt_information_;
      current_error += 0.5 * whitened_error.squaredNorm();

      factors.push_back(std::move(residual));
    }

    struct LinearizedMeasurement {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      gtsam::Matrix62 Hxf;
      gtsam::Matrix22 Hff;
      gtsam::Vector2 bf;
    };
    AlignedVector<LinearizedMeasurement> linearized(num_measurements);

    double new_error = current_error;
    size_t iterations = 0;
    std::vector<IterationStats> stats_per_iterations;
    do {
      stats_per_iterations.emplace_back();
      IterationStats& stats = stats_per_iterations.back();
      current_error = new_error;

      gtsam::Matrix66 H = gtsam::Matrix66::Zero();
      gtsam::Vector6 b = gtsam::Vector6::Zero();

      // temporary variables
      gtsam::Matrix H1;
      gtsam::Matrix H2;

      for (size_t i = 0; i < num_measurements; ++i) {
        Eigen::Matrix<double, 2, 2> Jf;
        Eigen::Matrix<double, 2, 6> Jx;

        const gtsam::Vector2& measured_flow = inputs.measured_flows.at(i);
        const gtsam::Vector2& esimated_flow = refined_flows.at(i);

        const auto& residual = factors.at(i);
        gtsam::Vector2 projection_error =
            residual(esimated_flow, refined_pose, H1, H2);
        Jf = H1;
        Jx = H2;

        stats.avg_projection_error += projection_error.norm();

        // whitened error
        gtsam::Vector2 whitened_error =
            projection_error * flow_sqrt_information_;
        // NOTE: this should actually sum to the current error since it is
        // recalculated from the same linerization point at the end of the last
        // iteration
        // TODO: we could actually cache the residuals then as we compute them
        // once at the end of each iteration and then again at the start!
        stats.regular_error += 0.5 * whitened_error.squaredNorm();

        // whitened Jacobians
        Jf *= flow_sqrt_information_;
        Jx *= flow_sqrt_information_;

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

        if (robust_weight != 1.0) {
          Jf *= sqrt_weight;
          Jx *= sqrt_weight;
          whitened_error *= sqrt_weight;
        }

        // --------------------------------------------------------------
        // Projection Hessian and gradient contributions.
        // --------------------------------------------------------------
        const gtsam::Matrix62 Jx_T = Jx.transpose();
        const gtsam::Matrix22 Jf_T = Jf.transpose();

        const gtsam::Matrix66 Hxx = Jx_T * Jx;
        const gtsam::Matrix62 Hxf = Jx_T * Jf;
        gtsam::Matrix22 Hff = Jf_T * Jf;

        const gtsam::Vector6 bx = Jx_T * whitened_error;
        gtsam::Vector2 bf = Jf_T * whitened_error;

        // Flow prior.  The prior is intentionally NOT robustified.
        const gtsam::Vector2 prior_r =
            (esimated_flow - measured_flow).template cast<double>();

        // Hessian: Jᵀ Λ J, with J = I.
        Hff.noalias() += prior_information_ * gtsam::Matrix22::Identity();

        // Gradient: Jᵀ Λ r, with J = I.
        const gtsam::Vector2 prior_gradient = prior_information_ * prior_r;
        bf.noalias() += prior_gradient;

        // Error: 0.5 rᵀ Λ r = 0.5 ||Λ½ r||².
        const gtsam::Vector2 whitened_prior_r =
            prior_sqrt_information_ * prior_r;

        const double prior_error = 0.5 * whitened_prior_r.squaredNorm();

        stats.regular_error += prior_error;
        stats.robust_error += prior_error;

        linearized[i].Hxf = Hxf;
        linearized[i].Hff = Hff;
        linearized[i].bf = bf;

        // Factor Hff only once.
        const Eigen::LDLT<gtsam::Matrix22> Hff_ldlt(Hff);
        const gtsam::Matrix26 Hff_inv_Hfx = Hff_ldlt.solve(Hxf.transpose());
        const gtsam::Vector2 Hff_inv_bf = Hff_ldlt.solve(bf);

        H.noalias() += Hxx - Hxf * Hff_inv_Hfx;
        b.noalias() += bx - Hxf * Hff_inv_bf;
      }  // finish measurement loop

      if (num_measurements > 0) {
        stats.avg_projection_error /= (double)num_measurements;
      } else {
        stats.avg_projection_error = 0.0;
      }

      // printIterationStats(stats, iterations);

      // solve reduced system
      const gtsam::Vector6 dx = H.ldlt().solve(-b);
      // update pose
      refined_pose = refined_pose.retract(dx);

      new_error = 0.0;
      // update flows and recompute error term
      for (size_t i = 0; i < num_measurements; ++i) {
        const LinearizedMeasurement& lin = linearized[i];

        const Eigen::LDLT<gtsam::Matrix22> Hff_ldlt(lin.Hff);

        const gtsam::Vector2 df =
            -Hff_ldlt.solve(lin.bf + lin.Hxf.transpose() * dx);

        refined_flows.at(i) += df;

        const auto& residual = factors.at(i);
        // TODO: probably could cache then and then use it for the start of the
        // next iteration!
        gtsam::Vector2 unwhitened_error =
            residual(refined_flows.at(i), refined_pose);
        gtsam::Vector2 whitened_error =
            unwhitened_error * flow_sqrt_information_;
        new_error += 0.5 * whitened_error.squaredNorm();
      }

      ++iterations;
      stats.error_change = new_error - current_error;

    } while (!criteria.shouldTerminate(iterations, current_error, new_error));

    auto toc = utils::Timer::toc(tic);
    double optimize_time_ms = utils::Timer::toUnits<std::milli>(toc);

    result.error_before = stats_per_iterations.front().regular_error;
    result.error_after = stats_per_iterations.back().regular_error;
    result.total_time_ms = optimize_time_ms;
    result.iterations = iterations;

    return result;
  }

  Result optimize(const gtsam::Pose3& initial_pose,
                  const Inputs& inputs) const {
    return this->optimize(initial_pose, inputs, TerminationCriteria{});
  }
};

FlowRefinement::FlowRefinement(const Camera::Ptr camera,
                               const ImageContainer& images,
                               FrameAbsoluteAdaptor& adaptor,
                               const FlowRefinementParams& params)
    : images_(images),
      adaptor_(adaptor),
      impl_(std::make_unique<ImplOptimizer>(params,
                                            *camera->getGtsamCalibration())) {}

FlowRefinement::~FlowRefinement() = default;

void FlowRefinement::refine(const gtsam::Pose3& pose_in,
                            gtsam::Pose3& pose_out) {
  dyno::utils::ChronoTimingStats timer("flow_refine.refine");

  ImplOptimizer::Inputs impl_inputs;

  // takes a few milliseconds to build...
  const size_t num_matches = adaptor_.getNumberCorrespondences();

  // store reference keypoints to avoid lookup
  gtsam::Point2Vector ref_kps;
  ref_kps.reserve(num_matches);

  std::vector<Index> matching_index;
  matching_index.reserve(num_matches);

  std::vector<uchar*> inlier_ptrs;
  inlier_ptrs.reserve(num_matches);

  ObjectIds object_ids;
  object_ids.reserve(num_matches);

  for (size_t i = 0; i < num_matches; i++) {
    // if we recompute this should not be needed!
    uchar* inlier_ptr = adaptor_.inlierPtr(i);
    // dereference and check if valid
    if (!(*inlier_ptr)) {
      continue;
    }

    const gtsam::Point3& ref_lmk = adaptor_.landmark(i);
    // assuming the previous frame is the reference frame!
    const gtsam::Point2& ref_kp = adaptor_.keypointPrev(i);
    const ObjectId object_id = adaptor_.objectId(i);

    const gtsam::Point2& curr_kp = adaptor_.keypoint(i);
    const gtsam::Point2 initial_flow = curr_kp - ref_kp;

    matching_index.push_back(i);
    ref_kps.push_back(ref_kp);
    inlier_ptrs.push_back(inlier_ptr);
    object_ids.push_back(object_id);

    impl_inputs.ref_kps.push_back(ref_kp);
    impl_inputs.ref_lmks.push_back(ref_lmk);
    impl_inputs.measured_flows.push_back(initial_flow);
  }

  ImplOptimizer::TerminationCriteria criteria;
  criteria.max_iterations = 10;

  dyno::utils::ChronoTimingStats solve_t("flow_refine.shur_solve", 7);
  const auto result = impl_->optimize(pose_in, impl_inputs, criteria);
  solve_t.stop();

  CHECK_EQ(matching_index.size(), result.flows.size());
  const cv::Mat& object_masks = images_.objectMotionMask();
  const int rows = object_masks.rows;
  const int cols = object_masks.cols;

  cv::Point2f pt;
  for (size_t i = 0; i < matching_index.size(); i++) {
    const gtsam::Point2 refined_keypoint = result.flows[i] + ref_kps[i];

    const Index index = matching_index[i];
    // update both the gtsam representation and the opencv representation in
    // features!
    adaptor_.keypoint(index, refined_keypoint);

    // get updated keypoint
    adaptor_.getCvKeypoint(index, pt);

    // ensure updated keypoint is still within the object mask
    const ObjectId object_id = object_ids[i];
    if (!checkBoundsAndLabel(pt, object_masks, object_id, rows, cols)) {
      *(inlier_ptrs[i]) = 0;
    }
  }

  pose_out = result.pose;

  // VLOG(10) << "Initial error: " << result.error_before << " final error "
  //          << result.error_after << " time[ms] "
  //          << result.total_time_ms
  //          << " #iterations= " << result.iterations;
}

}  // namespace dyno2
