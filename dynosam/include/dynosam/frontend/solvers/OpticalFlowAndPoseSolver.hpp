#pragma once

#include <config_utilities/config_utilities.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include "dynosam/factors/Pose3FlowProjectionFactor.h"
#include "dynosam/frontend/solvers/PnPRansac.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/TimingStats.hpp"
#include "dynosam_opt/FactorGraphTools.hpp"
#include "dynosam_opt/NonlinearOptimizer.hpp"
#include "dynosam_opt/Symbols.hpp"
#include "dynosam_opt/TimeBudgetOptimizationCallback.hpp"

namespace dyno {

struct OpticalFlowAndPoseSolverParams {
  double flow_sigma{10.0};
  double flow_prior_sigma{3.33};
  double k_huber{0.001};
  bool outlier_reject{false};
  bool use_robust{true};
  // When true, this indicates that the optical flow images go from k to k+1
  // (rather than k-1 to k, when false) this left over from some original
  // implementations. This param is used when updated the frames after
  // optimization
  bool flow_is_future{true};
};

void declare_config(OpticalFlowAndPoseSolverParams& config);

struct OpticalFlowAndPoseResult {
  gtsam::Pose3 refined_pose;
  gtsam::Point2Vector refined_flows;
  ObjectId object_id;
};
using OpticalFlowAndPoseSolverResult = SolverResult<OpticalFlowAndPoseResult>;

/**
 * @brief Joinly refines optical flow with with the given pose
 * using the error term:
 * e = [u,v]_{k_1} + f_{k-1, k} - \pi(X^{-1} \: m_k)
 * where f is flow, [u,v]_{k-1} is the observed keypoint at k-1, X is the pose
 * and m_k is the back-projected keypoint at k.
 *
 * The parsed tracklets are the set of correspondances with which to build the
 * optimisation problem and the refined inliers will be a subset of these
 * tracklets. THe number of refined flows should be the number of refined
 * inliers and be a 1-to-1 match
 * @tparam CALIBRATION
 *
 */
template <typename CALIBRATION>
class OpticalFlowAndPoseSolver {
 public:
  using Calibration = CALIBRATION;
  using Pose3FlowProjectionFactorCalib = Pose3FlowProjectionFactor<Calibration>;

  OpticalFlowAndPoseSolver(const OpticalFlowAndPoseSolverParams& params,
                           const DepthUpdater& depth_updater)
      : params_(params), depth_updater_(depth_updater) {
    flow_prior_noise_ =
        gtsam::noiseModel::Isotropic::Sigma(2u, params_.flow_prior_sigma);
    flow_noise_ = gtsam::noiseModel::Isotropic::Sigma(2u, params_.flow_sigma);

    if (params_.use_robust) {
      flow_noise_ = gtsam::noiseModel::Robust::Create(
          gtsam::noiseModel::mEstimator::Huber::Create(params_.k_huber),
          flow_noise_);
    }
  }

  void enforceRealtime(bool flag = true) { try_enforce_realtime_ = flag; }

  /**
   * @brief Builds the factor-graph problem using the set of specificed
   * correspondences (tracklets) in frame k-1 and k and the initial pose.
   *
   * The optimisation joinly refines optical flow with with the given pose
   * using the error term:
   * e = [u,v]_{k_1} + f_{k-1, k} - \pi(X^{-1} \: m_k)
   * where f is flow, [u,v]_{k-1} is the observed keypoint at k-1, X is the pose
   * and m_k is the back-projected keypoint at k.
   *
   * The parsed tracklets are the set of correspondances with which to build the
   * optimisation problem and the refined inliers will be a subset of these
   * tracklets. THe number of refined flows should be the number of refined
   * inliers and be a 1-to-1 match
   *
   * This is agnostic to if the problem is solving for a motion or a pose so the
   * user must make sure the initial pose is in the right form.
   *
   * TODO: update comments (solving_frame)
   *
   * @param frame_k_1
   * @param frame_k
   * @param tracklets
   * @param initial_pose
   * @return Result
   */
  OpticalFlowAndPoseSolverResult optimize(
      const Frame::Ptr frame_k_1, const Frame::Ptr frame_k,
      const TrackletIds& tracklets, const gtsam::Pose3& initial_pose,
      const ReferenceFrame& solving_frame) const {
    utils::ChronoTimingStats timer("of_pose_solver.optimize");

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values values;

    gtsam::Pose3 T = gtsam::Pose3::Identity();
    if (solving_frame == ReferenceFrame::GLOBAL) {
      T = frame_k_1->getPose();
    }
    const gtsam::Symbol pose_key('X', 0);

    gtsam::Ordering ordering;

    // lambda to construct gtsam symbol from a tracklet
    auto flowSymbol = [](TrackletId tracklet) -> gtsam::Symbol {
      return gtsam::symbol_shorthand::F(static_cast<uint64_t>(tracklet));
    };
    auto gtsam_calibration = boost::make_shared<Calibration>(
        frame_k_1->getFrameCamera().calibration());

    // sanity check to ensure all tracklets are on the same object
    CHECK(!tracklets.empty());
    const ObjectId expected_object_id =
        frame_k_1->at(tracklets.front())->objectId();

    for (TrackletId tracklet_id : tracklets) {
      Feature::Ptr feature_k_1 = frame_k_1->at(tracklet_id);
      Feature::Ptr feature_k = frame_k->at(tracklet_id);

      CHECK_NOTNULL(feature_k_1);
      CHECK_NOTNULL(feature_k);

      CHECK_EQ(feature_k_1->objectId(), expected_object_id);
      CHECK_EQ(feature_k->objectId(), expected_object_id);

      CHECK(feature_k_1->hasDepth());
      CHECK(feature_k->hasDepth())
          << " with object id " << feature_k->objectId() << " and is valid "
          << feature_k->usable() << " and age " << feature_k->age()
          << " previous age " << feature_k_1->age() << " kp "
          << feature_k->keypoint();

      const Keypoint kp_k_1 = feature_k_1->keypoint();
      const Depth depth_k_1 = feature_k_1->depth();

      const gtsam::Point2 flow = feature_k_1->measuredFlow();
      CHECK(gtsam::equal(kp_k_1 + flow, feature_k->keypoint()))
          << gtsam::Point2(kp_k_1 + flow) << " " << feature_k->keypoint()
          << " object id " << feature_k->objectId() << " flow " << flow;

      gtsam::Symbol flow_symbol(flowSymbol(tracklet_id));
      auto flow_factor = boost::make_shared<Pose3FlowProjectionFactorCalib>(
          flow_symbol, pose_key, kp_k_1, depth_k_1, T, *gtsam_calibration,
          flow_noise_);
      graph.add(flow_factor);

      // add prior factor on each flow
      graph.addPrior<gtsam::Point2>(flow_symbol, flow, flow_prior_noise_);

      values.insert(flow_symbol, flow);
      ordering += flow_symbol;
    }

    // pose at frame k
    // put pose last and use custom ordering!
    values.insert(pose_key, initial_pose);
    ordering += pose_key;

    OpticalFlowAndPoseSolverResult result;
    result.best_result.object_id = expected_object_id;

    if (!try_enforce_realtime_) {
      result.error_before = graph.error(values);
    }

    std::unordered_set<gtsam::Key> outlier_flows;
    // graph we will mutate by removing outlier factors
    gtsam::NonlinearFactorGraph mutable_graph = graph;

    gtsam::Values optimised_values = solveGN(values, mutable_graph, ordering);

    gtsam::FactorIndices outlier_factors;
    // if we have outliers, enter iteration loop
    // this can also be slow to only do if we are not enforcing realtime
    if (!try_enforce_realtime_ && params_.outlier_reject) {
      utils::ChronoTimingStats timer("of_pose_solver.outlier_reject", 10);

      outlier_factors = factor_graph_tools::determineFactorOutliers<
          Pose3FlowProjectionFactorCalib>(mutable_graph, optimised_values);

      if (outlier_factors.size() > 0) {
        for (size_t itr = 0; itr < 4; itr++) {
          // currently removing factors from graph makes them nullptr
          gtsam::NonlinearFactorGraph mutable_graph_with_null = mutable_graph;
          for (auto outlier_idx : outlier_factors) {
            auto factor = mutable_graph_with_null.at(outlier_idx);
            gtsam::Symbol flow_symbol = factor->keys()[0];
            CHECK_EQ(flow_symbol.chr(), 'f');

            outlier_flows.insert(flow_symbol.key());
            mutable_graph_with_null.remove(outlier_idx);
          }
          // now iterate over graph and add factors that are not null to ensure
          // all factors are ok
          mutable_graph.resize(0);
          for (size_t i = 0; i < mutable_graph_with_null.size(); i++) {
            auto factor = mutable_graph_with_null.at(i);
            if (factor) {
              mutable_graph.add(factor);
            }
          }

          optimised_values.update(pose_key, initial_pose);
          // do we use values or optimised values here?
          // optimised_values = gtsam::LevenbergMarquardtOptimizer(
          //                        mutable_graph, optimised_values, opt_params)
          //                        .optimize();
          // error_after = mutable_graph.error(optimised_values);
          // post_errors.push_back(error_after);

          outlier_factors = factor_graph_tools::determineFactorOutliers<
              Pose3FlowProjectionFactorCalib>(mutable_graph, optimised_values);

          if (outlier_factors.size() == 0) {
            break;
          }
        }
      }
    }

    if (!try_enforce_realtime_) {
      result.error_after = mutable_graph.error(optimised_values);
    }
    // recover values
    result.best_result.refined_pose =
        optimised_values.at<gtsam::Pose3>(pose_key);
    // for each outlier edge, update the set of inliers
    for (TrackletId tracklet_id : tracklets) {
      const gtsam::Symbol flow_sym = flowSymbol(tracklet_id);
      const gtsam::Key flow_key = flow_sym.key();

      if (!outlier_flows.empty() &&
          outlier_flows.find(flow_key) != outlier_flows.end()) {
        // still need to update the result as this is used to actually update
        // the outlier tracklets in the updateFrameOutliersWithResult
        result.outliers.push_back(tracklet_id);
      } else {
        gtsam::Point2 refined_flow =
            optimised_values.at<gtsam::Point2>(flow_sym);
        result.best_result.refined_flows.push_back(refined_flow);
        result.inliers.push_back(tracklet_id);
      }
    }

    CHECK_EQ(tracklets.size(), result.inliers.size() + result.outliers.size());
    return result;
  }

  /**
   * @brief Builds the factor-graph problem using the set of specificed
   * correspondences (tracklets) in frame k-1 and k and the initial pose. Unlike
   * the optimize only version this also update the features within the frames
   * as outliers after optimisation. It will also update the feature data (depth
   * keypoint etc...) with the refined flows.
   *
   * It will NOT update the frame with the result pose as this could be any
   * pose.
   *
   * TODO: update comments (solving_frame)
   *
   * @param frame_k_1
   * @param frame_k
   * @param tracklets
   * @param initial_pose
   * @return Result
   */
  OpticalFlowAndPoseSolverResult optimizeAndUpdate(
      Frame::Ptr frame_k_1, Frame::Ptr frame_k, const TrackletIds& tracklets,
      const gtsam::Pose3& initial_pose,
      const ReferenceFrame& solving_frame = ReferenceFrame::GLOBAL) const {
    auto result =
        optimize(frame_k_1, frame_k, tracklets, initial_pose, solving_frame);
    updateFrameOutliersWithResult(result, frame_k_1, frame_k);
    return result;
  }

 private:
  void updateFrameOutliersWithResult(
      const OpticalFlowAndPoseSolverResult& result, Frame::Ptr frame_k_1,
      Frame::Ptr frame_k) const {
    utils::ChronoTimingStats timer("of_pose_solver.update_frame", 10);
    const auto& image_container_k = frame_k->image_container_;
    const cv::Mat& motion_mask = image_container_k.objectMotionMask();

    auto camera = frame_k->camera_;
    const auto& refined_inliers = result.inliers;
    const auto& refined_flows = result.best_result.refined_flows;

    // outliers from the result. We will update this vector with new outliers
    auto refined_outliers = result.outliers;

    TrackletIds inlier_tracklet_ids;
    for (size_t i = 0; i < refined_inliers.size(); i++) {
      TrackletId tracklet_id = refined_inliers.at(i);
      gtsam::Point2 refined_flow = refined_flows.at(i);

      Feature::Ptr feature_k_1 = frame_k_1->at(tracklet_id);
      Feature::Ptr feature_k = frame_k->at(tracklet_id);

      CHECK_EQ(feature_k->objectId(), result.best_result.object_id);

      const Keypoint kp_k_1 = feature_k_1->keypoint();
      Keypoint refined_keypoint = kp_k_1 + refined_flow;

      // check boundaries?
      if (!camera->isKeypointContained(refined_keypoint)) {
        refined_outliers.push_back(tracklet_id);
        continue;
      }

      ObjectId predicted_label =
          functional_keypoint::at<ObjectId>(refined_keypoint, motion_mask);
      if (predicted_label != result.best_result.object_id) {
        refined_outliers.push_back(tracklet_id);
        // TODO: other fields of the feature does not get updated?
        // Inconsistencies as measured flow, predicted kp etc are no longer
        // correct!!?
        continue;
      }

      // update current keypoint
      feature_k->keypoint(refined_keypoint);
      // update refined frlow
      feature_k_1->measuredFlow(refined_flow);
      // update refined predicted keypoint
      feature_k_1->predictedKeypoint(refined_keypoint);

      inlier_tracklet_ids.push_back(tracklet_id);

      // Logic is a bit convoluted and dependant on other things
      // if we have optical flow (from k to k+1 due to historical reaseons)
      // then use this to set the predicted keypoint
      // if we are tracking using the provided flow then this MUST get set
      // otherwise tracking will fail as tracking with provided optical flow
      // uses the predicted keypoint to get the next keypoint!
      if (image_container_k.hasOpticalFlow()) {
        const cv::Mat& flow_image = image_container_k.opticalFlow();
        const int x = functional_keypoint::u(refined_keypoint);
        const int y = functional_keypoint::v(refined_keypoint);
        double flow_xe = static_cast<double>(flow_image.at<cv::Vec2f>(y, x)[0]);
        double flow_ye = static_cast<double>(flow_image.at<cv::Vec2f>(y, x)[1]);

        OpticalFlow new_measured_flow(flow_xe, flow_ye);
        feature_k->measuredFlow(new_measured_flow);
        // TODO: check predicted flow is within image
        Keypoint predicted_kp = Feature::CalculatePredictedKeypoint(
            refined_keypoint, new_measured_flow);
        feature_k->predictedKeypoint(predicted_kp);
      } else {
        // we dont have a predicted flow but we assume we are using KLT tracking
        //  in this case we dont have a predicted keypoint.
        //  just use the refined flow as the measured flow
        feature_k->measuredFlow(refined_flow);
        Keypoint predicted_kp =
            Feature::CalculatePredictedKeypoint(refined_keypoint, refined_flow);
        feature_k->predictedKeypoint(predicted_kp);
      }
    }

    // update tracks
    for (const auto outlier_tracklet : refined_outliers) {
      Feature::Ptr feature_k_1 = frame_k_1->at(outlier_tracklet);
      Feature::Ptr feature_k = frame_k->at(outlier_tracklet);

      CHECK(feature_k_1->usable());
      CHECK(feature_k->usable());

      feature_k->markOutlier();
      feature_k_1->markOutlier();
    }

    // finally update the depths of each feature corresponding to the new
    // keypoint location
    depth_updater_.update(frame_k, inlier_tracklet_ids);
  }

 private:
  gtsam::Values solveLM(const gtsam::Values& values,
                        const gtsam::NonlinearFactorGraph& graph,
                        const gtsam::Ordering& ordering) const {
    gtsam::LevenbergMarquardtParams opt_params;
    // for speed
    opt_params.setMaxIterations(10);
    // this is basically a set of prior looking factors on a pose so we know we
    // need to eliminate the pose last to avoid fill in therefore we use our own
    // custom ordering that has the pose last
    opt_params.setOrdering(ordering);
    if (VLOG_IS_ON(200)) {
      opt_params.verbosity = gtsam::NonlinearOptimizerParams::Verbosity::ERROR;
    }

    using BaseSolver = gtsam::LevenbergMarquardtOptimizer;
    using DenseSolver = DenseNonlinearSolver<BaseSolver>;

    dyno::NonlinearOptimizer<DenseSolver> solver(graph, values, opt_params);

    NonlinearOptimizerSummary summary;
    NonlinearOptimizerOptions options;

    gtsam::Values optimised_values = values;
    {
      utils::ChronoTimingStats timer("of_pose_solver.LM_solve", 7);
      CHECK(solver.solve(optimised_values, options, &summary));
    }

    VLOG(10) << "Initial error: " << summary.initial_error << " final error "
             << summary.final_error << " time[s] "
             << summary.cumulative_time_in_seconds
             << " #iterations= " << summary.numIterations();

    return optimised_values;
  }

  gtsam::Values solveGN(const gtsam::Values& values,
                        const gtsam::NonlinearFactorGraph& graph,
                        const gtsam::Ordering& ordering) const {
    gtsam::GaussNewtonParams opt_params;

    int max_iterations = try_enforce_realtime_ ? 2 : 6;
    opt_params.setMaxIterations(max_iterations);
    // this is basically a set of prior looking factors on a pose so we know we
    // need to eliminate the pose last to avoid fill in therefore we use our own
    // custom ordering that has the pose last
    opt_params.setOrdering(ordering);
    if (VLOG_IS_ON(200)) {
      opt_params.verbosity = gtsam::NonlinearOptimizerParams::Verbosity::ERROR;
    }

    static CsvWriterToFile timing_logger(
        getOutputFilePath("of_flow_timing_compare.csv"),
        CsvHeader("type", "num_values", "num_factors", "timing[ms]"));

    auto solveDense =
        [&opt_params](
            const gtsam::Values& values,
            const gtsam::NonlinearFactorGraph& graph) -> gtsam::Values {
      using BaseSolver = gtsam::GaussNewtonOptimizer;
      using DenseSolver = DenseNonlinearSolver<BaseSolver>;

      dyno::NonlinearOptimizer<DenseSolver> solver(graph, values, opt_params);
      NonlinearOptimizerSummary summary;
      NonlinearOptimizerOptions options;

      gtsam::Values optimised_values = values;
      utils::ChronoTimingStats timer("of_pose_solver.GN_solve", 7);
      CHECK(solver.solve(optimised_values, options, &summary));

      auto timing_ms = timer.stop();

      timing_logger << "dense" << values.size() << graph.size() << timing_ms;
      return optimised_values;
    };

    auto solveSparse =
        [&opt_params](
            const gtsam::Values& values,
            const gtsam::NonlinearFactorGraph& graph) -> gtsam::Values {
      using BaseSolver = gtsam::GaussNewtonOptimizer;

      dyno::NonlinearOptimizer<BaseSolver> solver(graph, values, opt_params);
      NonlinearOptimizerSummary summary;
      NonlinearOptimizerOptions options;

      gtsam::Values optimised_values = values;
      utils::ChronoTimingStats timer("of_pose_solver.GN_solve", 7);
      CHECK(solver.solve(optimised_values, options, &summary));

      auto timing_ms = timer.stop();

      timing_logger << "sparse" << values.size() << graph.size() << timing_ms;
      return optimised_values;
    };

    auto optimised_values = solveSparse(values, graph);

    return optimised_values;
  }

 private:
  OpticalFlowAndPoseSolverParams params_;
  DepthUpdater depth_updater_;

  //! Robust noise model for the flow
  gtsam::SharedNoiseModel flow_noise_;
  //! Prior noise model for the flow
  gtsam::SharedNoiseModel flow_prior_noise_;

  bool try_enforce_realtime_{false};

  //! Optimization params and configuration
  TimeBudgetOptimizationCallback::UniquePtr opt_callback_{nullptr};
  size_t max_iterations = 100;
};

}  // namespace dyno
