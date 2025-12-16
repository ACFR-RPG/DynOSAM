/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
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
#pragma once

#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/factors/LandmarkMotionTernaryFactor.hpp"
#include "dynosam/factors/Pose3FlowProjectionFactor.h"
#include "dynosam/frontend/vision/MotionSolver.hpp"
#include "dynosam_common/utils/TimingStats.hpp"
#include "dynosam_opt/FactorGraphTools.hpp"

namespace dyno {

template <class SampleConsensusProblem>
bool runRansac(
    std::shared_ptr<SampleConsensusProblem> sample_consensus_problem_ptr,
    const double& threshold, const int& max_iterations,
    const double& probability, const bool& do_nonlinear_optimization,
    gtsam::Pose3& best_pose, std::vector<int>& inliers) {
  CHECK(sample_consensus_problem_ptr);
  inliers.clear();

  //! Create ransac
  opengv::sac::Ransac<SampleConsensusProblem> ransac(max_iterations, threshold,
                                                     probability);

  //! Setup ransac
  ransac.sac_model_ = sample_consensus_problem_ptr;

  //! Run ransac
  bool success = ransac.computeModel(0);

  if (success) {
    if (ransac.iterations_ >= max_iterations && ransac.inliers_.empty()) {
      success = false;
      best_pose = gtsam::Pose3();
      inliers = {};
    } else {
      best_pose = utils::openGvTfToGtsamPose3(ransac.model_coefficients_);
      inliers = ransac.inliers_;

      if (do_nonlinear_optimization) {
        opengv::transformation_t optimized_pose;
        sample_consensus_problem_ptr->optimizeModelCoefficients(
            inliers, ransac.model_coefficients_, optimized_pose);
        best_pose = Eigen::MatrixXd(optimized_pose);
      }
    }
  } else {
    CHECK(ransac.inliers_.empty());
    best_pose = gtsam::Pose3();
    inliers.clear();
  }

  return success;
}

template <typename CALIBRATION>
OpticalFlowAndPoseOptimizer::Result OpticalFlowAndPoseOptimizer::optimize(
    const Frame::Ptr frame_k_1, const Frame::Ptr frame_k,
    const TrackletIds& tracklets, const gtsam::Pose3& initial_pose) const {
  utils::ChronoTimingStats timer("of_motion_solver.optimize");

  using Calibration = CALIBRATION;
  using Pose3FlowProjectionFactorCalib = Pose3FlowProjectionFactor<Calibration>;

  gtsam::NonlinearFactorGraph graph;
  gtsam::Values values;

  gtsam::SharedNoiseModel flow_noise =
      gtsam::noiseModel::Isotropic::Sigma(2u, params_.flow_sigma);

  gtsam::SharedNoiseModel flow_prior_noise =
      gtsam::noiseModel::Isotropic::Sigma(2u, params_.flow_prior_sigma);

  // robust noise model!
  flow_noise = gtsam::noiseModel::Robust::Create(
      gtsam::noiseModel::mEstimator::Huber::Create(params_.k_huber),
      flow_noise);

  const gtsam::Pose3 T_world_k_1 = frame_k_1->getPose();
  const gtsam::Symbol pose_key('X', 0);

  gtsam::Ordering ordering;

  // lambda to construct gtsam symbol from a tracklet
  auto flowSymbol = [](TrackletId tracklet) -> gtsam::Symbol {
    return gtsam::symbol_shorthand::F(static_cast<uint64_t>(tracklet));
  };
  auto gtsam_calibration = boost::make_shared<Calibration>(
      frame_k_1->getFrameCamera().calibration());

  // sanity check to ensure all tracklets are on the same object
  std::unordered_set<ObjectId> object_id;

  for (TrackletId tracklet_id : tracklets) {
    Feature::Ptr feature_k_1 = frame_k_1->at(tracklet_id);
    Feature::Ptr feature_k = frame_k->at(tracklet_id);

    object_id.insert(feature_k_1->objectId());
    object_id.insert(feature_k->objectId());

    CHECK_NOTNULL(feature_k_1);
    CHECK_NOTNULL(feature_k);

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
        flow_symbol, pose_key, kp_k_1, depth_k_1, T_world_k_1,
        *gtsam_calibration, flow_noise);
    graph.add(flow_factor);

    // add prior factor on each flow
    graph.addPrior<gtsam::Point2>(flow_symbol, flow, flow_prior_noise);

    values.insert(flow_symbol, flow);
    ordering += flow_symbol;
  }

  // pose at frame k
  // put pose last and use custom ordering!
  values.insert(pose_key, initial_pose);
  ordering += pose_key;

  // check we only have one label
  CHECK_EQ(object_id.size(), 1u);
  Result result;
  result.best_result.object_id = *object_id.begin();

  const double error_before = graph.error(values);
  std::unordered_set<gtsam::Key> outlier_flows;
  // graph we will mutate by removing outlier factors
  gtsam::NonlinearFactorGraph mutable_graph = graph;
  gtsam::Values optimised_values = values;
  // number of all variables
  utils::StatsCollector("of_motion_solver.joint_of_pose_num_vars_all")
      .AddSample(optimised_values.size());

  gtsam::LevenbergMarquardtParams opt_params;
  // for speed
  opt_params.setMaxIterations(10);
  // this is basically a set of prior looking factors on a pose so we know we
  // need to eliminate the pose last to avoid fill in therefore we use our own
  // custom ordering that has the pose last
  // opt_params.setOrdering(ordering);
  if (VLOG_IS_ON(200))
    opt_params.verbosity = gtsam::NonlinearOptimizerParams::Verbosity::ERROR;

  {
    utils::ChronoTimingStats timer("of_motion_solver.LM_solve");
    optimised_values = gtsam::LevenbergMarquardtOptimizer(
                           mutable_graph, optimised_values, opt_params)
                           .optimize();
  }

  gtsam::FactorIndices outlier_factors;
  // if we have outliers, enter iteration loop
  if (params_.outlier_reject) {
    utils::ChronoTimingStats timer("of_motion_solver.outlier_reject");

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
        optimised_values = gtsam::LevenbergMarquardtOptimizer(
                               mutable_graph, optimised_values, opt_params)
                               .optimize();
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

  // size_t initial_size = graph.size();
  // size_t inlier_size = mutable_graph.size();
  const double error_after = mutable_graph.error(optimised_values);

  // recover values
  result.best_result.refined_pose = optimised_values.at<gtsam::Pose3>(pose_key);
  result.error_before = error_before;
  result.error_after = error_after;

  // for each outlier edge, update the set of inliers
  for (TrackletId tracklet_id : tracklets) {
    const gtsam::Symbol flow_sym = flowSymbol(tracklet_id);
    const gtsam::Key flow_key = flow_sym.key();

    if (outlier_flows.find(flow_key) != outlier_flows.end()) {
      // still need to update the result as this is used to actually update the
      // outlier tracklets in the updateFrameOutliersWithResult
      result.outliers.push_back(tracklet_id);
    } else {
      gtsam::Point2 refined_flow = optimised_values.at<gtsam::Point2>(flow_sym);
      result.best_result.refined_flows.push_back(refined_flow);
      result.inliers.push_back(tracklet_id);
    }
  }
  // number of variables after outlier removal
  utils::StatsCollector("of_motion_solver.joint_of_pose_num_vars_inliers")
      .AddSample(result.inliers.size());

  CHECK_EQ(tracklets.size(), result.inliers.size() + result.outliers.size());
  return result;
}

template <typename CALIBRATION>
OpticalFlowAndPoseOptimizer::Result
OpticalFlowAndPoseOptimizer::optimizeAndUpdate(
    Frame::Ptr frame_k_1, Frame::Ptr frame_k, const TrackletIds& tracklets,
    const gtsam::Pose3& initial_pose) const {
  auto result =
      this->optimize<CALIBRATION>(frame_k_1, frame_k, tracklets, initial_pose);
  updateFrameOutliersWithResult(result, frame_k_1, frame_k);
  return result;
}

template <typename CALIBRATION>
Pose3SolverResult MotionOnlyRefinementOptimizer::optimize(
    const Frame::Ptr frame_k_1, const Frame::Ptr frame_k,
    const TrackletIds& tracklets, const ObjectId object_id,
    const gtsam::Pose3& initial_motion, const RefinementSolver& solver) const {
  gtsam::NonlinearFactorGraph graph;
  gtsam::Values values;

  // noise models are chosen arbitrarily :)
  gtsam::SharedNoiseModel landmark_motion_noise =
      gtsam::noiseModel::Isotropic::Sigma(3u, params_.landmark_motion_sigma);

  // TODO: this will break if we use 3d point error
  gtsam::SharedNoiseModel projection_noise =
      gtsam::noiseModel::Isotropic::Sigma(2u, params_.projection_sigma);

  // make robust (I mean, why not?)
  landmark_motion_noise = gtsam::noiseModel::Robust::Create(
      gtsam::noiseModel::mEstimator::Huber::Create(params_.k_huber),
      landmark_motion_noise);
  projection_noise = gtsam::noiseModel::Robust::Create(
      gtsam::noiseModel::mEstimator::Huber::Create(params_.k_huber),
      projection_noise);

  const gtsam::Key pose_k_1_key = CameraPoseSymbol(frame_k_1->getFrameId());
  const gtsam::Key pose_k_key = CameraPoseSymbol(frame_k->getFrameId());
  const gtsam::Key object_motion_key =
      ObjectMotionSymbol(object_id, frame_k->getFrameId());

  values.insert(pose_k_1_key, frame_k_1->getPose());
  values.insert(pose_k_key, frame_k->getPose());
  values.insert(object_motion_key, initial_motion);

  auto gtsam_calibration = boost::make_shared<CALIBRATION>(
      frame_k_1->getFrameCamera().calibration());

  auto pose_prior = gtsam::noiseModel::Isotropic::Sigma(6u, 0.00001);
  graph.addPrior(pose_k_1_key, frame_k_1->getPose(), pose_prior);
  graph.addPrior(pose_k_key, frame_k->getPose(), pose_prior);

  utils::ChronoTimingStats timer("motion_solver.object_nlo_refinement");

  for (TrackletId tracklet_id : tracklets) {
    Feature::Ptr feature_k_1 = frame_k_1->at(tracklet_id);
    Feature::Ptr feature_k = frame_k->at(tracklet_id);

    CHECK_NOTNULL(feature_k_1);
    CHECK_NOTNULL(feature_k);

    if (!feature_k_1->usable() || !feature_k->usable()) {
      continue;
    }

    CHECK(feature_k_1->hasDepth());
    CHECK(feature_k->hasDepth());

    const Keypoint kp_k_1 = feature_k_1->keypoint();
    const Keypoint kp_k = feature_k->keypoint();

    const gtsam::Point3 lmk_k_1_world =
        frame_k_1->backProjectToWorld(tracklet_id);
    const gtsam::Point3 lmk_k_world = frame_k->backProjectToWorld(tracklet_id);

    const gtsam::Point3 lmk_k_1_local =
        frame_k_1->backProjectToCamera(tracklet_id);
    const gtsam::Point3 lmk_k_local = frame_k->backProjectToCamera(tracklet_id);

    const gtsam::Key lmk_k_1_key =
        DynamicLandmarkSymbol(frame_k_1->getFrameId(), tracklet_id);
    const gtsam::Key lmk_k_key =
        DynamicLandmarkSymbol(frame_k->getFrameId(), tracklet_id);

    // add initial for points
    values.insert(lmk_k_1_key, lmk_k_1_world);
    values.insert(lmk_k_key, lmk_k_world);

    if (solver == RefinementSolver::PointError) {
      LOG(FATAL) << "RefinementSolver::PointError not implemented";
      // graph.emplace_shared<PoseToPointFactor>(
      //     pose_k_1_key, //pose key at previous frames
      //     lmk_k_1_key,
      //     lmk_k_1_local,
      //     projection_noise
      // );

      // graph.emplace_shared<PoseToPointFactor>(
      //         pose_k_key, //pose key at current frames
      //         lmk_k_key,
      //         lmk_k_local,
      //         projection_noise
      // );
    } else if (solver == RefinementSolver::ProjectionError) {
      graph.emplace_shared<GenericProjectionFactor>(
          kp_k_1, projection_noise, pose_k_1_key, lmk_k_1_key,
          gtsam_calibration, false, false);

      graph.emplace_shared<GenericProjectionFactor>(
          kp_k, projection_noise, pose_k_key, lmk_k_key, gtsam_calibration,
          false, false);
    }

    graph.emplace_shared<LandmarkMotionTernaryFactor>(
        lmk_k_1_key, lmk_k_key, object_motion_key, landmark_motion_noise);
  }

  double error_before = graph.error(values);
  std::vector<double> post_errors;

  gtsam::NonlinearFactorGraph mutable_graph = graph;
  gtsam::Values optimised_values = values;

  utils::StatsCollector("motion_solver.object_nlo_refinement_num_vars_all")
      .AddSample(optimised_values.size());

  gtsam::LevenbergMarquardtParams opt_params;
  opt_params.setMaxIterations(5);
  if (VLOG_IS_ON(200))
    opt_params.verbosity = gtsam::NonlinearOptimizerParams::Verbosity::ERROR;

  optimised_values = gtsam::LevenbergMarquardtOptimizer(
                         mutable_graph, optimised_values, opt_params)
                         .optimize();
  double error_after = mutable_graph.error(optimised_values);
  // post_errors.push_back(error_after);

  gtsam::FactorIndices outlier_factors =
      factor_graph_tools::determineFactorOutliers<LandmarkMotionTernaryFactor>(
          mutable_graph, optimised_values);

  std::set<TrackletId> outlier_tracks;
  // if we have outliers, enter iteration loop
  if (outlier_factors.size() > 0u && params_.outlier_reject) {
    for (size_t itr = 0; itr < 4; itr++) {
      // currently removing factors from graph makes them nullptr
      gtsam::NonlinearFactorGraph mutable_graph_with_null = mutable_graph;
      for (auto outlier_idx : outlier_factors) {
        auto factor = mutable_graph_with_null.at(outlier_idx);
        DynamicPointSymbol point_symbol = factor->keys()[0];
        outlier_tracks.insert(point_symbol.trackletId());
        mutable_graph_with_null.remove(outlier_idx);
      }
      // now iterate over graph and add factors that are not null to ensure all
      // factors are ok
      mutable_graph.resize(0);
      for (size_t i = 0; i < mutable_graph_with_null.size(); i++) {
        auto factor = mutable_graph_with_null.at(i);
        if (factor) {
          mutable_graph.add(factor);
        }
      }

      values.insert(object_motion_key, initial_motion);
      // do we use values or optimised values here?
      optimised_values = gtsam::LevenbergMarquardtOptimizer(
                             mutable_graph, optimised_values, opt_params)
                             .optimize();
      error_after = mutable_graph.error(optimised_values);
      post_errors.push_back(error_after);

      outlier_factors = factor_graph_tools::determineFactorOutliers<
          LandmarkMotionTernaryFactor>(mutable_graph, optimised_values);

      if (outlier_factors.size() == 0) {
        break;
      }
    }
  }

  const size_t initial_size = graph.size();
  const size_t inlier_size = mutable_graph.size();
  error_after = mutable_graph.error(optimised_values);

  Pose3SolverResult result;
  result.best_result = optimised_values.at<gtsam::Pose3>(object_motion_key);
  result.outliers = TrackletIds(outlier_tracks.begin(), outlier_tracks.end());
  result.error_before = error_before;
  result.error_after = error_after;

  // naming is confusing, but we already have the outliers but we can use the
  // same function to find the inliers
  determineOutlierIds(result.outliers, tracklets, result.inliers);

  for (auto outlier_tracklet : outlier_tracks) {
    Feature::Ptr feature_k_1 = frame_k_1->at(outlier_tracklet);
    Feature::Ptr feature_k = frame_k->at(outlier_tracklet);

    CHECK(feature_k_1->usable());
    CHECK(feature_k->usable());

    feature_k->markOutlier();
    feature_k_1->markOutlier();
  }

  // number of variables after outlier removal
  utils::StatsCollector("motion_solver.object_nlo_refinement_num_vars_inliers")
      .AddSample(result.inliers.size());

  // LOG(INFO) << "Object Motion refinement - error before: "
  //     << error_before << " error after: " << error_after
  //     << " with initial size " << initial_size << " inlier size " <<
  //     inlier_size;

  return result;
}

template <typename CALIBRATION>
Pose3SolverResult MotionOnlyRefinementOptimizer::optimizeAndUpdate(
    const Frame::Ptr frame_k_1, const Frame::Ptr frame_k,
    const TrackletIds& tracklets, const ObjectId object_id,
    const gtsam::Pose3& initial_motion, const RefinementSolver& solver) const {
  /// currently no outlier rejection
  auto result = this->optimize<CALIBRATION>(frame_k_1, frame_k, tracklets,
                                            object_id, initial_motion, solver);

  // TODO: update frame with outliers
  return result;
}

}  // namespace dyno
