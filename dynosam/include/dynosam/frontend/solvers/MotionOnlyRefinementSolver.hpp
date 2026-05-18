#pragma once

#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/ProjectionFactor.h>

#include "dynosam/factors/LandmarkMotionTernaryFactor.hpp"
#include "dynosam/frontend/solvers/PnPRansac.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/TimingStats.hpp"
#include "dynosam_opt/FactorGraphTools.hpp"
#include "dynosam_opt/Symbols.hpp"

namespace dyno {

struct MotionOnlyRefinementSolverParams {
  double landmark_motion_sigma{0.001};
  double projection_sigma{2.0};
  double k_huber{0.0001};
  bool outlier_reject{true};
};

/**
 * @brief Jointly refined the motion of an object using the 3D-motion-motion
 * and 2D reprojection errors.
 *
 */
template <typename CALIBRATION>
class MotionOnlyRefinementSolver {
 public:
  using Calibtration = CALIBRATION;
  using ProjectionFactor =
      gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, CALIBRATION>;

  MotionOnlyRefinementSolver(const MotionOnlyRefinementSolverParams& params)
      : params_(params) {
    projection_noise_ =
        gtsam::noiseModel::Isotropic::Sigma(2u, params_.projection_sigma);
    projection_noise_ = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(params_.k_huber),
        projection_noise_);
    pose_prior_noise_ = gtsam::noiseModel::Isotropic::Sigma(6u, 0.00001);

    motion_model_noise_ =
        gtsam::noiseModel::Isotropic::Sigma(3u, params_.landmark_motion_sigma);
    motion_model_noise_ = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(params_.k_huber),
        motion_model_noise_);
  }

  Pose3SolverResult optimize(const Frame::Ptr frame_k_1,
                             const Frame::Ptr frame_k,
                             const TrackletIds& tracklets,
                             const ObjectId object_id,
                             const gtsam::Pose3& initial_motion) const {
    utils::ChronoTimingStats timer("motion_only_solver.optimize");

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values values;

    const gtsam::Key pose_k_1_key = CameraPoseSymbol(frame_k_1->getFrameId());
    const gtsam::Key pose_k_key = CameraPoseSymbol(frame_k->getFrameId());
    const gtsam::Key object_motion_key =
        ObjectMotionSymbol(object_id, frame_k->getFrameId());

    values.insert(pose_k_1_key, frame_k_1->getPose());
    values.insert(pose_k_key, frame_k->getPose());
    values.insert(object_motion_key, initial_motion);

    auto gtsam_calibration = boost::make_shared<CALIBRATION>(
        frame_k_1->getFrameCamera().calibration());

    graph.addPrior(pose_k_1_key, frame_k_1->getPose(), pose_prior_noise_);
    graph.addPrior(pose_k_key, frame_k->getPose(), pose_prior_noise_);

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
      const gtsam::Point3 lmk_k_world =
          frame_k->backProjectToWorld(tracklet_id);

      const gtsam::Key lmk_k_1_key =
          DynamicLandmarkSymbol(frame_k_1->getFrameId(), tracklet_id);
      const gtsam::Key lmk_k_key =
          DynamicLandmarkSymbol(frame_k->getFrameId(), tracklet_id);

      // add initial for points
      values.insert(lmk_k_1_key, lmk_k_1_world);
      values.insert(lmk_k_key, lmk_k_world);

      graph.emplace_shared<ProjectionFactor>(kp_k_1, projection_noise_,
                                             pose_k_1_key, lmk_k_1_key,
                                             gtsam_calibration, false, false);

      graph.emplace_shared<ProjectionFactor>(kp_k, projection_noise_,
                                             pose_k_key, lmk_k_key,
                                             gtsam_calibration, false, false);

      graph.emplace_shared<LandmarkMotionTernaryFactor>(
          lmk_k_1_key, lmk_k_key, object_motion_key, motion_model_noise_);
    }

    double error_before = graph.error(values);
    std::vector<double> post_errors;

    gtsam::NonlinearFactorGraph mutable_graph = graph;
    gtsam::Values optimised_values = values;

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
        factor_graph_tools::determineFactorOutliers<
            LandmarkMotionTernaryFactor>(mutable_graph, optimised_values);

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
        // now iterate over graph and add factors that are not null to ensure
        // all factors are ok
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

    return result;
  }

  Pose3SolverResult optimizeAndUpdate(
      Frame::Ptr frame_k_1, Frame::Ptr frame_k, const TrackletIds& tracklets,
      const ObjectId object_id, const gtsam::Pose3& initial_motion) const {
    /// currently no outlier rejection
    // TODO: update frame with outliers

    return this->optimize(frame_k_1, frame_k, tracklets, object_id,
                          initial_motion);
  }

 private:
  const MotionOnlyRefinementSolverParams params_;

  //! Cached 3d noise on rigid-body motion model
  gtsam::SharedNoiseModel motion_model_noise_;
  //! Cached robust 2d pixel noise model
  gtsam::SharedNoiseModel projection_noise_;
  //! Cached noise model for pose
  gtsam::SharedNoiseModel pose_prior_noise_;
};

}  // namespace dyno
