/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam/frontend/vision/MotionSolver.hpp"

#include <config_utilities/config_utilities.h>
#include <glog/logging.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>  //for now? //TODO: clean
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <tbb/tbb.h>

#include <eigen3/Eigen/Dense>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include <opengv/types.hpp>

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/factors/HybridFormulationFactors.hpp"
#include "dynosam/factors/LandmarkMotionTernaryFactor.hpp"
#include "dynosam/factors/Pose3FlowProjectionFactor.h"
#include "dynosam/frontend/Frontend-Definitions.hpp"
#include "dynosam/frontend/vision/VisionTools.hpp"
#include "dynosam_common/DynamicObjects.hpp"
#include "dynosam_common/Flags.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/Accumulator.hpp"
#include "dynosam_common/utils/GtsamUtils.hpp"
#include "dynosam_common/utils/Numerical.hpp"
#include "dynosam_common/utils/TimingStats.hpp"
#include "dynosam_cv/RGBDCamera.hpp"
#include "dynosam_opt/FactorGraphTools.hpp"  //TODO: clean

// GTSAM Includes
// FOR TESTING!
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Pose3.h>
// #include <gtsam/base/FullLinearSolver.h>

#include "dynosam/factors/HybridFormulationFactors.hpp"

namespace dyno {

void declare_config(OpticalFlowAndPoseOptimizer::Params& config) {
  using namespace config;

  name("OpticalFlowAndPoseOptimizerParams");
  field(config.flow_sigma, "flow_sigma");
  field(config.flow_prior_sigma, "flow_prior_sigma");
  field(config.k_huber, "k_huber");
  field(config.outlier_reject, "outlier_reject");
  field(config.flow_is_future, "flow_is_future");
}

void declare_config(MotionOnlyRefinementOptimizer::Params& config) {
  using namespace config;

  name("MotionOnlyRefinementOptimizerParams");
  field(config.landmark_motion_sigma, "landmark_motion_sigma");
  field(config.projection_sigma, "projection_sigma");
  field(config.k_huber, "k_huber");
  field(config.outlier_reject, "outlier_reject");
}

void declare_config(EgoMotionSolver::Params& config) {
  using namespace config;

  name("EgoMotionSolver::Params");
  field(config.ransac_randomize, "ransac_randomize");
  field(config.ransac_use_2point_mono, "ransac_use_2point_mono");
  field(config.optimize_2d2d_pose_from_inliers,
        "optimize_2d2d_pose_from_inliers");
  field(config.ransac_threshold_pnp, "ransac_threshold_pnp");
  field(config.optimize_3d2d_pose_from_inliers,
        "optimize_3d2d_pose_from_inliers");
  field(config.ransac_threshold_stereo, "ransac_threshold_stereo");
  field(config.optimize_3d3d_pose_from_inliers,
        "optimize_3d3d_pose_from_inliers");
  field(config.ransac_iterations, "ransac_iterations");
  field(config.ransac_probability, "ransac_probability");
}
void declare_config(ObjectMotionSovlerF2F::Params& config) {
  using namespace config;
  name("ObjectMotionSovlerF2F::Params");

  base<EgoMotionSolver::Params>(config);
  field(config.refine_motion_with_joint_of, "refine_motion_with_joint_of");
  field(config.refine_motion_with_3d, "refine_motion_with_3d");
  field(config.joint_of_params, "joint_optical_flow");
  field(config.object_motion_refinement_params, "object_motion_3d_refinement");
}

EgoMotionSolver::EgoMotionSolver(const Params& params,
                                 const CameraParams& camera_params)
    : params_(params), camera_params_(camera_params) {}

Pose3SolverResult EgoMotionSolver::geometricOutlierRejection2d2d(
    Frame::Ptr frame_k_1, Frame::Ptr frame_k,
    std::optional<gtsam::Rot3> R_curr_ref) {
  // get correspondences
  RelativePoseCorrespondences correspondences;
  // this does not create proper bearing vectors (at leas tnot for 3d-2d pnp
  // solve) bearing vectors are also not undistorted atm!!
  {
    utils::ChronoTimingStats track_dynamic_timer("mono_frame_correspondences");
    frame_k->getCorrespondences(correspondences, *frame_k_1,
                                KeyPointType::STATIC,
                                frame_k->imageKeypointCorrespondance());
  }

  Pose3SolverResult result;

  const size_t& n_matches = correspondences.size();

  if (n_matches < 5u) {
    result.status = TrackingStatus::FEW_MATCHES;
    return result;
  }

  gtsam::Matrix K = camera_params_.getCameraMatrixEigen();
  K = K.inverse();

  TrackletIds tracklets;
  // NOTE: currently without distortion! the correspondences should be made into
  // bearing vector elsewhere!
  BearingVectors ref_bearing_vectors, cur_bearing_vectors;
  for (size_t i = 0u; i < n_matches; i++) {
    const auto& corres = correspondences.at(i);
    const Keypoint& ref_kp = corres.ref_;
    const Keypoint& cur_kp = corres.cur_;

    gtsam::Vector3 ref_versor = (K * gtsam::Vector3(ref_kp(0), ref_kp(1), 1.0));
    gtsam::Vector3 cur_versor = (K * gtsam::Vector3(cur_kp(0), cur_kp(1), 1.0));

    ref_versor = ref_versor.normalized();
    cur_versor = cur_versor.normalized();

    ref_bearing_vectors.push_back(ref_versor);
    cur_bearing_vectors.push_back(cur_versor);

    tracklets.push_back(corres.tracklet_id_);
  }

  RelativePoseAdaptor adapter(ref_bearing_vectors, cur_bearing_vectors);

  const bool use_2point_mono = params_.ransac_use_2point_mono && R_curr_ref;
  if (use_2point_mono) {
    adapter.setR12((*R_curr_ref).matrix());
  }

  gtsam::Pose3 best_result;
  std::vector<int> ransac_inliers;
  bool success = false;
  if (use_2point_mono) {
    success = runRansac<RelativePoseProblemGivenRot>(
        std::make_shared<RelativePoseProblemGivenRot>(adapter,
                                                      params_.ransac_randomize),
        params_.ransac_threshold_mono, params_.ransac_iterations,
        params_.ransac_probability, params_.optimize_2d2d_pose_from_inliers,
        best_result, ransac_inliers);
  } else {
    success = runRansac<RelativePoseProblem>(
        std::make_shared<RelativePoseProblem>(
            adapter, RelativePoseProblem::NISTER, params_.ransac_randomize),
        params_.ransac_threshold_mono, params_.ransac_iterations,
        params_.ransac_probability, params_.optimize_2d2d_pose_from_inliers,
        best_result, ransac_inliers);
  }

  if (!success) {
    result.status = TrackingStatus::INVALID;
  } else {
    constructTrackletInliers(result.inliers, result.outliers, correspondences,
                             ransac_inliers, tracklets);
    // NOTE: 2-point always returns the identity rotation, hence we have to
    // substitute it:
    if (use_2point_mono) {
      CHECK(R_curr_ref->equals(best_result.rotation()));
    }
    result.status = TrackingStatus::VALID;
    result.best_result = best_result;
  }

  return result;
}

Pose3SolverResult EgoMotionSolver::geometricOutlierRejection3d2d(
    Frame::Ptr frame_k_1, Frame::Ptr frame_k,
    std::optional<gtsam::Rot3> R_curr_ref) {
  AbsolutePoseCorrespondences correspondences;
  // this does not create proper bearing vectors (at leas tnot for 3d-2d pnp
  // solve) bearing vectors are also not undistorted atm!!
  // TODO: change to use landmarkWorldProjectedBearingCorrespondance and then
  // change motion solver to take already projected bearing vectors
  {
    utils::ChronoTimingStats timer("motion_solver.solve_3d2d.correspondances");
    frame_k->getCorrespondences(correspondences, *frame_k_1,
                                KeyPointType::STATIC,
                                frame_k->landmarkWorldKeypointCorrespondance());
  }

  return geometricOutlierRejection3d2d(correspondences, R_curr_ref);
}

Pose3SolverResult EgoMotionSolver::geometricOutlierRejection3d2d(
    const AbsolutePoseCorrespondences& correspondences,
    std::optional<gtsam::Rot3> R_curr_ref) {
  utils::ChronoTimingStats timer("motion_solver.solve_3d2d");
  Pose3SolverResult result;
  const size_t& n_matches = correspondences.size();

  if (n_matches < 5u) {
    result.status = TrackingStatus::FEW_MATCHES;
    VLOG(5) << "3D2D tracking failed as there are to few matches" << n_matches;
    return result;
  }

  gtsam::Matrix K = camera_params_.getCameraMatrixEigen();
  K = K.inverse();

  TrackletIds tracklets, inliers, outliers;
  // NOTE: currently without distortion! the correspondences should be made into
  // bearing vector elsewhere!
  BearingVectors bearing_vectors;
  Landmarks points;
  for (size_t i = 0u; i < n_matches; i++) {
    const AbsolutePoseCorrespondence& corres = correspondences.at(i);
    const Keypoint& kp = corres.cur_;
    // make Bearing vector
    gtsam::Vector3 versor = (K * gtsam::Vector3(kp(0), kp(1), 1.0));
    versor = versor.normalized();
    bearing_vectors.push_back(versor);

    points.push_back(corres.ref_);
    tracklets.push_back(corres.tracklet_id_);
  }

  VLOG(20) << "Collected " << tracklets.size() << " initial correspondances";

  const double reprojection_error = params_.ransac_threshold_pnp;
  const double avg_focal_length =
      0.5 * static_cast<double>(camera_params_.fx() + camera_params_.fy());
  const double threshold =
      1.0 - std::cos(std::atan(std::sqrt(2.0) * reprojection_error /
                               avg_focal_length));

  AbsolutePoseAdaptor adapter(bearing_vectors, points);

  if (R_curr_ref) {
    adapter.setR(R_curr_ref->matrix());
  }

  gtsam::Pose3 best_result;
  std::vector<int> ransac_inliers;

  bool success;
  {
    utils::ChronoTimingStats timer("motion_solver.solve_3d2d.ransac");
    success = runRansac<AbsolutePoseProblem>(
        std::make_shared<AbsolutePoseProblem>(adapter,
                                              AbsolutePoseProblem::KNEIP),
        threshold, params_.ransac_iterations, params_.ransac_probability,
        params_.optimize_3d2d_pose_from_inliers, best_result, ransac_inliers);
  }

  constructTrackletInliers(result.inliers, result.outliers, correspondences,
                           ransac_inliers, tracklets);

  if (success) {
    if (result.inliers.size() < 5u) {
      result.status = TrackingStatus::FEW_MATCHES;
    } else {
      result.status = TrackingStatus::VALID;
      result.best_result = best_result;
    }

  } else {
    result.status = TrackingStatus::INVALID;
  }

  return result;
}

void OpticalFlowAndPoseOptimizer::updateFrameOutliersWithResult(
    const Result& result, Frame::Ptr frame_k_1, Frame::Ptr frame_k) const {
  utils::ChronoTimingStats timer("of_motion_solver.update_frame");
  const auto& image_container_k = frame_k->image_container_;
  const cv::Mat& motion_mask = image_container_k.objectMotionMask();

  auto camera = frame_k->camera_;
  const auto& refined_inliers = result.inliers;
  const auto& refined_flows = result.best_result.refined_flows;

  // outliers from the result. We will update this vector with new outliers
  auto refined_outliers = result.outliers;

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
      // TODO: other fields of the feature does not get updated? Inconsistencies
      // as measured flow, predicted kp etc are no longer correct!!?
      continue;
    }

    // update current keypoint
    feature_k->keypoint(refined_keypoint);
    // update refined frlow
    feature_k_1->measuredFlow(refined_flow);
    // update refined predicted keypoint
    feature_k_1->predictedKeypoint(refined_keypoint);

    // Logic is a bit convoluted and dependant on other things
    // if we have optical flow (from k to k+1 due to historical reaseons)
    // then use this to set the predicted keypoint
    // if we are tracking using the provided flow then this MUST get set
    // otherwise tracking will fail as tracking with provided optical flow uses
    // the predicted keypoint to get the next keypoint!
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
  for (const auto& outlier_tracklet : refined_outliers) {
    Feature::Ptr feature_k_1 = frame_k_1->at(outlier_tracklet);
    Feature::Ptr feature_k = frame_k->at(outlier_tracklet);

    CHECK(feature_k_1->usable());
    CHECK(feature_k->usable());

    feature_k->markOutlier();
    feature_k_1->markOutlier();
  }

  // refresh depth information for each frame
  CHECK(frame_k->updateDepths());
}

Pose3SolverResult EgoMotionSolver::geometricOutlierRejection3d3d(
    Frame::Ptr frame_k_1, Frame::Ptr frame_k,
    std::optional<gtsam::Rot3> R_curr_ref) {
  PointCloudCorrespondences correspondences;
  {
    utils::ChronoTimingStats("pc_correspondences");
    frame_k->getCorrespondences(
        correspondences, *frame_k_1, KeyPointType::STATIC,
        frame_k->landmarkWorldPointCloudCorrespondance());
  }

  return geometricOutlierRejection3d3d(correspondences, R_curr_ref);
}

Pose3SolverResult EgoMotionSolver::geometricOutlierRejection3d3d(
    const PointCloudCorrespondences& correspondences,
    std::optional<gtsam::Rot3> R_curr_ref) {
  const size_t& n_matches = correspondences.size();

  Pose3SolverResult result;
  if (n_matches < 5) {
    result.status = TrackingStatus::FEW_MATCHES;
    return result;
  }

  TrackletIds tracklets;
  BearingVectors ref_bearing_vectors, cur_bearing_vectors;

  for (size_t i = 0u; i < n_matches; i++) {
    const auto& corres = correspondences.at(i);
    const Landmark& ref_lmk = corres.ref_;
    const Landmark& cur_lmk = corres.cur_;
    ref_bearing_vectors.push_back(ref_lmk);
    cur_bearing_vectors.push_back(cur_lmk);

    tracklets.push_back(corres.tracklet_id_);
  }

  //! Setup adapter.
  Adapter3d3d adapter(ref_bearing_vectors, cur_bearing_vectors);

  if (R_curr_ref) {
    adapter.setR12((*R_curr_ref).matrix());
  }

  gtsam::Pose3 best_result;
  std::vector<int> ransac_inliers;

  bool success = runRansac<Problem3d3d>(
      std::make_shared<Problem3d3d>(adapter, params_.ransac_randomize),
      params_.ransac_threshold_stereo, params_.ransac_iterations,
      params_.ransac_probability, params_.optimize_3d3d_pose_from_inliers,
      best_result, ransac_inliers);

  if (success) {
    constructTrackletInliers(result.inliers, result.outliers, correspondences,
                             ransac_inliers, tracklets);

    result.status = TrackingStatus::VALID;
    result.best_result = best_result;
  } else {
    result.status = TrackingStatus::INVALID;
  }

  return result;
}

ObjectMotionSolver::Result ObjectMotionSolver::solve(Frame::Ptr frame_k,
                                                     Frame::Ptr frame_k_1) {
  ObjectIds failed_object_tracks;
  MotionEstimateMap motion_estimates;

  // if only 1 object, no point parallelising
  if (motion_estimates.size() <= 1) {
    for (const auto& [object_id, observations] :
         frame_k->object_observations_) {
      if (!solveImpl(frame_k, frame_k_1, object_id, motion_estimates)) {
        VLOG(5) << "Could not solve motion for object " << object_id
                << " from frame " << frame_k_1->getFrameId() << " -> "
                << frame_k->getFrameId();
        failed_object_tracks.push_back(object_id);
      }
    }
  } else {
    std::mutex mutex;
    // paralleilise the process of each function call.
    tbb::parallel_for_each(
        frame_k->object_observations_.begin(),
        frame_k->object_observations_.end(),
        [&](const std::pair<ObjectId, SingleDetectionResult>& pair) {
          const auto object_id = pair.first;
          if (!solveImpl(frame_k, frame_k_1, object_id, motion_estimates)) {
            VLOG(5) << "Could not solve motion for object " << object_id
                    << " from frame " << frame_k_1->getFrameId() << " -> "
                    << frame_k->getFrameId();

            std::lock_guard<std::mutex> lk(mutex);
            failed_object_tracks.push_back(object_id);
          }
        });
  }

  // remove objects from the object observations list
  // does not remove the features etc but stops the object being propogated to
  // the backend as we loop over the object observations in the constructOutput
  // function
  for (auto object_id : failed_object_tracks) {
    frame_k->object_observations_.erase(object_id);
  }

  ObjectPoseMap poses;
  ObjectMotionMap motions;

  updateMotions(motions, motion_estimates, frame_k, frame_k_1);
  updatePoses(poses, motion_estimates, frame_k, frame_k_1);
  return {motions, poses};
}

ObjectMotionSovlerF2F::ObjectMotionSovlerF2F(
    const ObjectMotionSovlerF2F::Params& params,
    const CameraParams& camera_params)
    : EgoMotionSolver(static_cast<const EgoMotionSolver::Params&>(params),
                      camera_params),
      object_motion_params(params) {}

void ObjectMotionSovlerF2F::updatePoses(
    ObjectPoseMap& object_poses, const MotionEstimateMap& motion_estimates,
    Frame::Ptr frame_k, Frame::Ptr frame_k_1) {
  gtsam::Point3Vector object_centroids_k_1, object_centroids_k;

  for (const auto& [object_id, motion_estimate] : motion_estimates) {
    auto object_points = FeatureFilterIterator(
        const_cast<FeatureContainer&>(frame_k_1->dynamic_features_),
        [object_id, &frame_k](const Feature::Ptr& f) -> bool {
          return Feature::IsUsable(f) && f->objectId() == object_id &&
                 frame_k->exists(f->trackletId()) &&
                 frame_k->isFeatureUsable(f->trackletId());
        });

    gtsam::Point3 centroid_k_1(0, 0, 0);
    gtsam::Point3 centroid_k(0, 0, 0);
    size_t count = 0;
    for (const auto& feature : object_points) {
      gtsam::Point3 lmk_k_1 =
          frame_k_1->backProjectToCamera(feature->trackletId());
      centroid_k_1 += lmk_k_1;

      gtsam::Point3 lmk_k = frame_k->backProjectToCamera(feature->trackletId());
      centroid_k += lmk_k;

      count++;
    }

    centroid_k_1 /= count;
    centroid_k /= count;

    centroid_k_1 = frame_k_1->getPose() * centroid_k_1;
    centroid_k = frame_k->getPose() * centroid_k;

    object_centroids_k_1.push_back(centroid_k_1);
    object_centroids_k.push_back(centroid_k);
  }

  if (FLAGS_init_object_pose_from_gt) {
    CHECK(object_motion_params.ground_truth_packets_request)
        << "FLAGS_init_object_pose_from_gt is true but no ground truth packets "
           "hook is set!";

    const auto ground_truth_packets =
        object_motion_params.ground_truth_packets_request();
    LOG_IF(WARNING, !ground_truth_packets.has_value())
        << "FLAGS_init_object_pose_from_gt but no ground truth provided! "
           "Object poses will be initalised using centroid!";

    dyno::propogateObjectPoses(object_poses_, motion_estimates,
                               object_centroids_k_1, object_centroids_k,
                               frame_k->getFrameId(), ground_truth_packets);
  } else {
    dyno::propogateObjectPoses(object_poses_, motion_estimates,
                               object_centroids_k_1, object_centroids_k,
                               frame_k->getFrameId());
  }

  object_poses = object_poses_;
}

void ObjectMotionSovlerF2F::updateMotions(
    ObjectMotionMap& object_motions, const MotionEstimateMap& motion_estimates,
    Frame::Ptr frame_k, Frame::Ptr) {
  const FrameId frame_id_k = frame_k->getFrameId();
  for (const auto& [object_id, motion_reference_frame] : motion_estimates) {
    object_motions_.insert22(object_id, frame_id_k, motion_reference_frame);
  }
  object_motions = object_motions_;
}

bool ObjectMotionSovlerF2F::solveImpl(Frame::Ptr frame_k, Frame::Ptr frame_k_1,
                                      ObjectId object_id,
                                      MotionEstimateMap& motion_estimates) {
  Motion3SolverResult result = geometricOutlierRejection3d2d(
      frame_k_1, frame_k, frame_k->getPose(), object_id);

  frame_k->dynamic_features_.markOutliers(result.outliers);

  VLOG(15) << " object motion estimate " << object_id << " at frame "
           << frame_k->frame_id_
           << (result.status == TrackingStatus::VALID ? " success "
                                                      : " failure ")
           << ":\n"
           << "- Tracking Status: " << to_string(result.status) << '\n'
           << "- Total Correspondences: "
           << result.inliers.size() + result.outliers.size() << '\n'
           << "\t- # inliers: " << result.inliers.size() << '\n'
           << "\t- # outliers: " << result.outliers.size() << '\n';

  // if valid, remove outliers and add to motion estimation
  if (result.status == TrackingStatus::VALID) {
    motion_estimates.insert({object_id, result.best_result});
    return true;
  } else {
    return false;
  }
}

Motion3SolverResult ObjectMotionSovlerF2F::geometricOutlierRejection3d2d(
    Frame::Ptr frame_k_1, Frame::Ptr frame_k, const gtsam::Pose3& T_world_k,
    ObjectId object_id) {
  utils::ChronoTimingStats timer("motion_solver.object_solve3d2d");
  AbsolutePoseCorrespondences dynamic_correspondences;
  // get the corresponding feature pairs
  bool corr_result = frame_k->getDynamicCorrespondences(
      dynamic_correspondences, *frame_k_1, object_id,
      frame_k->landmarkWorldKeypointCorrespondance());

  const size_t& n_matches = dynamic_correspondences.size();

  TrackletIds all_tracklets;
  std::transform(dynamic_correspondences.begin(), dynamic_correspondences.end(),
                 std::back_inserter(all_tracklets),
                 [](const AbsolutePoseCorrespondence& corres) {
                   return corres.tracklet_id_;
                 });
  CHECK_EQ(all_tracklets.size(), n_matches);

  Pose3SolverResult geometric_result =
      EgoMotionSolver::geometricOutlierRejection3d2d(dynamic_correspondences);
  Pose3SolverResult pose_result = geometric_result;

  Motion3SolverResult motion_result;
  motion_result.status = pose_result.status;

  if (pose_result.status == TrackingStatus::VALID) {
    TrackletIds refined_inlier_tracklets = pose_result.inliers;

    {
      CHECK_EQ(pose_result.inliers.size() + pose_result.outliers.size(),
               n_matches);

      // debug only (just checking that the inlier/outliers we get from the
      // geometric rejection match the original one)
      TrackletIds extracted_all_tracklets = refined_inlier_tracklets;
      extracted_all_tracklets.insert(extracted_all_tracklets.end(),
                                     pose_result.outliers.begin(),
                                     pose_result.outliers.end());
      CHECK_EQ(all_tracklets.size(), extracted_all_tracklets.size());
    }

    gtsam::Pose3 G_w = pose_result.best_result.inverse();
    if (object_motion_params.refine_motion_with_joint_of) {
      OpticalFlowAndPoseOptimizer flow_optimizer(
          object_motion_params.joint_of_params);
      // Use the original result as the input to the refine joint optical flow
      // function the result.best_result variable is actually equivalent to
      // ^wG^{-1} and we want to solve something in the form e(T, flow) =
      // [u,v]_{k-1} + {k-1}_flow_k - pi(T^{-1}^wm_{k-1}) so T must take the
      // point from k-1 in the world frame to the local frame at k-1 ^wG^{-1} =
      //^wX_k \: {k-1}^wH_k (which takes does this) but the error term uses the
      // inverse of T hence we must parse in the inverse of G
      auto flow_opt_result = flow_optimizer.optimizeAndUpdate<CalibrationType>(
          frame_k_1, frame_k, refined_inlier_tracklets,
          pose_result.best_result);
      // still need to take the inverse as we get the inverse of G out
      G_w = flow_opt_result.best_result.refined_pose.inverse();
      // inliers should be a subset of the original refined inlier tracks
      refined_inlier_tracklets = flow_opt_result.inliers;

      VLOG(10) << "Refined object " << object_id
               << "pose with optical flow - error before: "
               << flow_opt_result.error_before.value_or(NaN)
               << " error_after: " << flow_opt_result.error_after.value_or(NaN);
    }
    // still need to take the inverse as we get the inverse of G out
    gtsam::Pose3 H_w = T_world_k * G_w;

    if (object_motion_params.refine_motion_with_3d) {
      VLOG(10) << "Refining object motion pose with 3D refinement";
      MotionOnlyRefinementOptimizer motion_refinement_graph(
          object_motion_params.object_motion_refinement_params);
      auto motion_refinement_result =
          motion_refinement_graph.optimizeAndUpdate<CalibrationType>(
              frame_k_1, frame_k, refined_inlier_tracklets, object_id, H_w);

      // should be further subset
      refined_inlier_tracklets = motion_refinement_result.inliers;
      H_w = motion_refinement_result.best_result;
    }

    motion_result.status = pose_result.status;
    motion_result.best_result = Motion3ReferenceFrame(
        H_w, Motion3ReferenceFrame::Style::F2F, ReferenceFrame::GLOBAL,
        frame_k_1->getFrameId(), frame_k->getFrameId());
    motion_result.inliers = refined_inlier_tracklets;
    determineOutlierIds(motion_result.inliers, all_tracklets,
                        motion_result.outliers);

    // sanity check that we have accounted for all initial matches
    CHECK_EQ(motion_result.inliers.size() + motion_result.outliers.size(),
             n_matches);
  }

  // needed when running things like TartanAir S where we get vew few point on
  // the object...
  // if (motion_result.inliers.size() < 30) {
  //   motion_result.status = TrackingStatus::FEW_MATCHES;
  // }

  return motion_result;
}

///////////////////////////////////////////////////////////////////////////

HybridObjectMotionSRIF::HybridObjectMotionSRIF(
    const gtsam::Pose3& initial_state_H, const gtsam::Pose3& L_e,
    const FrameId& frame_id_e, const gtsam::Matrix66& initial_P,
    const gtsam::Matrix66& Q, const gtsam::Matrix33& R, Camera::Ptr camera,
    double huber_k)
    : H_linearization_point_(initial_state_H),
      Q_(Q),
      R_noise_(R),
      R_inv_(R.inverse()),
      initial_P_(initial_P),
      huber_k_(huber_k),
      motion_track_status_(ObjectTrackingStatus::New) {
  // set camera
  rgbd_camera_ = CHECK_NOTNULL(camera)->safeGetRGBDCamera();
  CHECK(rgbd_camera_);
  stereo_calibration_ = rgbd_camera_->getFakeStereoCalib();

  resetState(L_e, frame_id_e);
}

void HybridObjectMotionSRIF::predictAndUpdate(
    const gtsam::Pose3& H_w_km1_k_predict, Frame::Ptr frame,
    const TrackletIds& tracklets, const int num_irls_iterations) {
  utils::ChronoTimingStats timer("motion_solver.ekf_predict&update");
  predict(H_w_km1_k_predict);

  const FrameId frame_id_before_update = frame_id_;
  const ObjectTrackingStatus motion_track_status_before = motion_track_status_;
  update(frame, tracklets, num_irls_iterations);
  const FrameId frame_id_after_update = frame_id_;
}

/**
 * @brief Recovers the state perturbation delta_w by solving R * delta_w = d.
 */
gtsam::Vector6 HybridObjectMotionSRIF::getStatePerturbation() const {
  // R is upper triangular, so this is a fast back-substitution
  return R_info_.triangularView<Eigen::Upper>().solve(d_info_);
}

const gtsam::Pose3& HybridObjectMotionSRIF::getCurrentLinearization() const {
  return H_linearization_point_;
}

gtsam::Pose3 HybridObjectMotionSRIF::getKeyFramedMotion() const {
  return H_linearization_point_.retract(getStatePerturbation());
}

Motion3ReferenceFrame HybridObjectMotionSRIF::getKeyFramedMotionReference()
    const {
  return Motion3ReferenceFrame(
      getKeyFramedMotion(), Motion3ReferenceFrame::Style::KF,
      ReferenceFrame::GLOBAL, getKeyFrameId(), getFrameId());
}

gtsam::Pose3 HybridObjectMotionSRIF::getF2FMotion() const {
  gtsam::Pose3 H_W_e_k = H_linearization_point_.retract(getStatePerturbation());
  gtsam::Pose3 H_W_e_km1 = previous_H_;
  return H_W_e_k * H_W_e_km1.inverse();
}

gtsam::Matrix66 HybridObjectMotionSRIF::getCovariance() const {
  gtsam::Matrix66 Lambda = R_info_.transpose() * R_info_;
  return Lambda.inverse();
}

gtsam::Matrix66 HybridObjectMotionSRIF::getInformationMatrix() const {
  return R_info_.transpose() * R_info_;
}

void HybridObjectMotionSRIF::predict(const gtsam::Pose3&) {
  // 1. Get current mean state and covariance
  gtsam::Vector6 delta_w = getStatePerturbation();
  gtsam::Pose3 H_current_mean = H_linearization_point_.retract(delta_w);
  gtsam::Matrix66 P_current = getCovariance();  // Slow step

  // call before updating the previous_H_
  // previous motion -> assume constant!
  gtsam::Pose3 H_W_km1_k = getF2FMotion();

  // gtsam::Pose3 L_km1 = previous_H_ * L_e_;
  // gtsam::Pose3 L_k = H_current_mean * L_e_;

  // //calculate motion relative to object
  // gtsam::Pose3 local_motion = L_km1.inverse() * L_k;
  // // forward predict pose using constant velocity model
  // gtsam::Pose3 predicted_pose = L_k * local_motion;
  // // convert pose back to motion
  // gtsam::Pose3 predicted_motion = predicted_pose * L_e_.inverse();

  previous_H_ = H_current_mean;

  // 2. Perform EKF prediction (add process noise)
  // P_k = P_{k-1} + Q
  gtsam::Matrix66 P_predicted = P_current + Q_;

  // 3. Re-linearize: Set new linearization point to the current mean
  H_linearization_point_ = H_W_km1_k * H_current_mean;
  // predict forward with constant velocity model
  //  H_linearization_point_ = predicted_motion;

  // 4. Recalculate R and d based on new P and new linearization point
  // The new perturbation is 0 relative to the new linearization point.
  d_info_ = gtsam::Vector6::Zero();
  gtsam::Matrix66 Lambda_predicted = P_predicted.inverse();
  R_info_ =
      Lambda_predicted.llt().matrixU();  // R_info_ = L^T where L*L^T = Lambda
}

HybridObjectMotionSRIF::Result HybridObjectMotionSRIF::update(
    Frame::Ptr frame, const TrackletIds& tracklets,
    const int num_irls_iterations) {
  const size_t num_measurements = tracklets.size();
  const size_t total_rows = num_measurements * ZDim;

  const gtsam::Pose3& X_W_k = frame->getPose();
  const gtsam::Pose3 X_W_k_inv = X_W_k.inverse();
  X_K_ = X_W_k;
  frame_id_ = frame->getFrameId();

  // 1. Calculate Jacobians (H) and Linearized Residuals (y_lin)
  // These are calculated ONCE at the linearization point and are fixed
  // for all IRLS iterations.
  gtsam::Matrix H_stacked = gtsam::Matrix ::Zero(total_rows, StateDim);
  gtsam::Vector y_linearized = gtsam::Vector::Zero(total_rows);

  const gtsam::Pose3 A = X_W_k_inv * H_linearization_point_;
  // G will project the point from the object frame into the camera frame
  const gtsam::Pose3 G_w = A * L_e_;
  const gtsam::Matrix6 J_correction = -A.AdjointMap();

  gtsam::StereoCamera gtsam_stereo_camera(G_w.inverse(), stereo_calibration_);
  // whitened error
  gtsam::Vector initial_error = gtsam::Vector::Zero(num_measurements);
  gtsam::Vector re_weighted_error = gtsam::Vector::Zero(num_measurements);

  size_t num_new_features = 0u, num_tracked_features = 0u;

  for (size_t i = 0; i < num_measurements; ++i) {
    const TrackletId& tracklet_id = tracklets.at(i);
    const Feature::Ptr feature = frame->at(tracklet_id);
    CHECK(feature);

    if (!m_linearized_.exists(tracklet_id)) {
      // this is initalising from a predicted motion (bad!) Should use previous
      // linearization (ie k-1)
      const gtsam::Point3 m_X_k = frame->backProjectToCamera(tracklet_id);
      Landmark m_init = HybridObjectMotion::projectToObject3(
          X_W_k, H_linearization_point_, L_e_, m_X_k);
      m_linearized_.insert2(tracklet_id, m_init);
      // VLOG(10) << "Initalising new point i=" << tracklet_id << " Le " <<
      // L_e_;
      num_new_features++;
    } else {
      num_tracked_features++;
    }

    gtsam::Point3 m_L = m_linearized_.at(tracklet_id);

    const auto [stereo_keypoint_status, stereo_measurement] =
        rgbd_camera_->getStereo(feature);
    if (!stereo_keypoint_status) {
      continue;
    }

    const auto& z_obs = stereo_measurement;

    // const Point2& z_obs = feature->keypoint();
    gtsam::Matrix36 J_pi;  // 2x6
    gtsam::StereoPoint2 z_pred;
    try {
      // Project using the *linearization point*
      // z_pred = camera.project(m_L, J_pi);
      z_pred = gtsam_stereo_camera.project2(m_L, J_pi);
    } catch (const gtsam::StereoCheiralityException& e) {
      LOG(WARNING) << "Warning: Point " << i << " behind camera. Skipping.";
      continue;  // Skip this measurement
    }

    // Calculate linearized residual y_lin = z - h(x_nom)
    gtsam::Vector3 y_i_lin = (z_obs - z_pred).vector();
    Eigen::Matrix<double, ZDim, StateDim> H_ekf_i = J_pi * J_correction;

    size_t row_idx = i * ZDim;

    H_stacked.block<ZDim, StateDim>(row_idx, 0) = H_ekf_i;
    y_linearized.segment<ZDim>(row_idx) = y_i_lin;

    double error_sq = y_i_lin.transpose() * R_inv_ * y_i_lin;
    double error = std::sqrt(error_sq);
    initial_error(i) = error;
  }

  VLOG(30) << "Feature stats. New features: " << num_new_features << "/"
           << num_measurements << " Tracked features " << num_tracked_features
           << "/" << num_measurements;

  // 2. Store the prior information
  gtsam::Matrix6 R_info_prior = R_info_;
  gtsam::Vector6 d_info_prior = d_info_;

  // 3. --- Start Iteratively Reweighted Least Squares (IRLS) Loop ---
  // cout << "\n--- SRIF Robust Update (IRLS) Started ---" << endl;
  for (int iter = 0; iter < num_irls_iterations; ++iter) {
    // 3a. Get current state estimate (from previous iteration)
    const gtsam::Vector6 delta_w =
        R_info_.triangularView<Eigen::Upper>().solve(d_info_);
    const gtsam::Pose3 H_current_mean = H_linearization_point_.retract(delta_w);
    // Need to validate this:
    //  We intentionally do not recalculate the Jacobian block (H_stacked).
    //  to solve for the single best perturbation (delta_w) relative to the
    //  single linearization point (W_linearization_point_) that we had at the
    //  start of the update.
    const gtsam::Pose3 A = X_W_k_inv * H_current_mean;
    const gtsam::Pose3 G_w = A * L_e_;
    gtsam::StereoCamera gtsam_stereo_camera_current(G_w.inverse(),
                                                    stereo_calibration_);

    gtsam::Vector weights = gtsam::Vector::Ones(num_measurements);

    for (size_t i = 0; i < num_measurements; ++i) {
      const TrackletId& tracklet_id = tracklets.at(i);
      const Feature::Ptr feature = frame->at(tracklet_id);
      CHECK(feature);

      const auto [stereo_keypoint_status, stereo_measurement] =
          rgbd_camera_->getStereo(feature);
      if (!stereo_keypoint_status) {
        weights(i) = 0.0;  // Skip point
        continue;
      }

      auto z_obs = stereo_measurement;

      gtsam::Point3 m_L = m_linearized_.at(tracklet_id);

      gtsam::StereoPoint2 z_pred_current;
      try {
        z_pred_current = gtsam_stereo_camera_current.project(m_L);
      } catch (const gtsam::StereoCheiralityException& e) {
        LOG(WARNING) << "Warning: Point " << i << " behind camera. Skipping.";
        weights(i) = 0.0;  // Skip point
        continue;
      }

      // Calculate non-linear residual
      gtsam::Vector3 y_nonlinear = (z_obs - z_pred_current).vector();

      // //CHECK we cannot do this with gtsam's noise models (we can...)
      // // Calculate Mahalanobis-like distance (whitened error)
      double error_sq = y_nonlinear.transpose() * R_inv_ * y_nonlinear;
      double error = std::sqrt(error_sq);

      re_weighted_error(i) = error;

      // Calculate Huber weight w(e) = min(1, delta / |e|)
      weights(i) = (error <= huber_k_) ? 1.0 : huber_k_ / error;
      if (/*weights(i) < 0.99 &&*/ iter == num_irls_iterations - 1) {
        VLOG(50) << "  [Meas " << i << "] Final Weight: " << weights(i)
                 << " (Error: " << error << ")";
      }
    }

    // 3c. Construct the giant matrix A_qr for this iteration
    gtsam::Matrix A_qr =
        gtsam::Matrix::Zero(total_rows + StateDim, StateDim + 1);

    for (size_t i = 0; i < num_measurements; ++i) {
      double w_i = weights(i);
      if (w_i < 1e-6) continue;  // Skip 0-weighted points

      // R_robust = R / w_i  (Note: w is |e|^-1, so R_robust = R * |e|/delta)
      // R_robust_inv = R_inv * w_i
      // We need W_i such that W_i^T * W_i = R_robust_inv
      // W_i = sqrt(w_i) * R_inv_sqrt

      gtsam::Matrix33 R_robust_i = R_noise_ / w_i;
      gtsam::Matrix33 L_robust_i = R_robust_i.llt().matrixL();
      gtsam::Matrix33 R_robust_inv_sqrt = L_robust_i.inverse();

      size_t row_idx = i * ZDim;

      A_qr.block<ZDim, StateDim>(row_idx, 0) =
          R_robust_inv_sqrt * H_stacked.block<3, StateDim>(row_idx, 0);
      A_qr.block<ZDim, 1>(row_idx, StateDim) =
          R_robust_inv_sqrt * y_linearized.segment<3>(row_idx);
    }

    // 3d. Fill the prior part of the QR matrix
    // This is *always* the original prior
    A_qr.block<StateDim, StateDim>(total_rows, 0) = R_info_prior;
    A_qr.block<StateDim, 1>(total_rows, StateDim) = d_info_prior;

    // 3e. Perform QR decomposition
    Eigen::HouseholderQR<gtsam::Matrix> qr(A_qr);
    gtsam::Matrix R_full = qr.matrixQR().triangularView<Eigen::Upper>();

    // 3f. Extract the new R_info and d_info for the *next* iteration
    R_info_ = R_full.block<StateDim, StateDim>(0, 0);
    d_info_ = R_full.block<StateDim, 1>(0, StateDim);
  }

  // (Optional) Enforce R is upper-triangular
  R_info_ = R_info_.triangularView<Eigen::Upper>();

  Result result;
  result.error = initial_error.norm();
  result.reweighted_error = re_weighted_error.norm();
  return result;

  // LOG(INFO) << "--- SRIF Robust Update Complete --- initial error "
  //           << initial_error.norm()
  //           << " re-weighted error: " << re_weighted_error.norm();
}

void HybridObjectMotionSRIF::resetState(const gtsam::Pose3& L_e,
                                        FrameId frame_id_e) {
  // 2. Initialize SRIF State (R, d) from EKF State (W, P)
  // W_linearization_point_ is set to initial_state_W
  // The initial perturbation is 0, so the initial d_info_ is 0.
  d_info_ = gtsam::Vector6::Zero();

  // Calculate initial Information Matrix Lambda = P^-1
  gtsam::Matrix66 Lambda_initial = initial_P_.inverse();

  // Calculate R_info_ from Cholesky decomposition: Lambda = R^T * R
  // We use LLT (L*L^T) and take the upper-triangular factor of L^T.
  // Or, more robustly, U^T*U from a U^T*U decomposition.
  // Eigen's LLT computes L (lower) such that A = L * L^T.
  // We need U (upper) such that A = U^T * U.
  // A.llt().matrixU() gives U where A = L*L^T (U is L^T). No.
  // A.llt().matrixL() gives L where A = L*L^T.
  // Let's use Eigen's LDLT and reconstruct.
  // A more direct way:
  R_info_ = Lambda_initial.llt().matrixU();  // L^T

  // will this give us discontinuinities betweek keyframes?
  //  reset other variables
  previous_H_ = gtsam::Pose3::Identity();
  L_e_ = L_e;
  frame_id_e_ = frame_id_e;
  frame_id_ = frame_id_e;
  X_K_ = gtsam::Pose3::Identity();

  m_linearized_.clear();

  // should always be identity since the deviation from L_e = I when frame = e
  // the initial H should not be parsed into the constructor by this logic as
  // well!
  H_linearization_point_ = gtsam::Pose3::Identity();
}

ObjectMotionSolverFilter::ObjectMotionSolverFilter(
    const ObjectMotionSolverFilter::Params& params,
    const CameraParams& camera_params)
    : EgoMotionSolver(static_cast<const EgoMotionSolver::Params&>(params),
                      camera_params),
      filter_params_(params) {}

ObjectMotionSolver::Result ObjectMotionSolverFilter::solve(
    Frame::Ptr frame_k, Frame::Ptr frame_k_1) {
  // Handle lost objects: objects in filters_ but not in current frame's
  // object_observations_
  std::set<ObjectId> current_objects;
  for (const auto& [obj_id, _] : frame_k->object_observations_) {
    current_objects.insert(obj_id);
  }
  for (const auto& [obj_id, _] : filters_) {
    if (current_objects.find(obj_id) == current_objects.end()) {
      object_statuses_[obj_id] = ObjectTrackingStatus::Lost;
      filters_.erase(obj_id);
      LOG(INFO) << "Object " << obj_id << " marked as Lost at frame "
                << frame_k->getFrameId();
    }
    // TODO: no re-tracked logic!!

    object_keyframe_statuses_[obj_id] = ObjectKeyFrameStatus::NonKeyFrame;
  }

  // Call base solve
  return ObjectMotionSolver::solve(frame_k, frame_k_1);
}

bool ObjectMotionSolverFilter::solveImpl(Frame::Ptr frame_k,
                                         Frame::Ptr frame_k_1,
                                         ObjectId object_id,
                                         MotionEstimateMap& motion_estimates) {
  // Initialize or update tracking status
  bool is_new = !filters_.exists(object_id);
  bool is_resampled = std::find(frame_k->retracked_objects_.begin(),
                                frame_k->retracked_objects_.end(),
                                object_id) != frame_k->retracked_objects_.end();

  const ObjectTrackingStatus& previous_tracking_state =
      object_statuses_[object_id];

  // if (is_new) {
  //   object_statuses_[object_id] = ObjectTrackingStatus::New;
  //   stable_frame_counts_[object_id] = 0;
  //   LOG(INFO) << "Object " << object_id << " initialized as New at frame " <<
  //   frame_k->getFrameId();
  //   // dont update keyframe status to anchor yet - only do so if object
  //   successfully created
  // } else if (is_retracked) {

  //   //if re-tracked and previous state was well tracked - set keyframe status
  //   to regular if(previous_tracking_state ==
  //   ObjectTrackingStatus::WellTracked){
  //     LOG(INFO) << "Object " << object_id << " re-tracked at frame " <<
  //     frame_k->getFrameId() << ", but WellTracked maintained, set to
  //     RegularKeyFrame"; object_keyframe_statuses_[object_id] =
  //     ObjectKeyFrameStatus::RegularKeyFrame;
  //   }
  //   else {
  //     //keep object as poorly tracked if was poorly tracked before
  //     object_statuses_[object_id] = ObjectTrackingStatus::PoorlyTracked;
  //     stable_frame_counts_[object_id] = 0;
  //     LOG(INFO) << "Object " << object_id << " re-tracked at frame " <<
  //     frame_k->getFrameId() << ", set to PoorlyTracked";

  //   }
  // }

  AbsolutePoseCorrespondences dynamic_correspondences;
  // get the corresponding feature pairs
  bool corr_result = frame_k->getDynamicCorrespondences(
      dynamic_correspondences, *frame_k_1, object_id,
      frame_k->landmarkWorldKeypointCorrespondance());

  // FeaturePairs correspondences;
  // frame_k->getDynamicCorrespondences(correspondences, *frame_k_1, object_id);

  const size_t& n_matches = dynamic_correspondences.size();

  cv::Mat rgb = frame_k->image_container_.rgb();
  cv::Mat viz;
  rgb.copyTo(viz);

  // TrackletIds all_tracklets(n_matches);

  TrackletIds all_tracklets;
  std::transform(dynamic_correspondences.begin(), dynamic_correspondences.end(),
                 std::back_inserter(all_tracklets),
                 [](const AbsolutePoseCorrespondence& corres) {
                   return corres.tracklet_id_;
                 });
  CHECK_EQ(all_tracklets.size(), n_matches);

  Pose3SolverResult geometric_result =
      EgoMotionSolver::geometricOutlierRejection3d2d(dynamic_correspondences);

  const size_t num_inliers =
      geometric_result.inliers.size();  // after outlier rejection

  if (is_new) {
    object_statuses_[object_id] = ObjectTrackingStatus::New;
    stable_frame_counts_[object_id] = 0;
    LOG(INFO) << "Object " << object_id << " initialized as New at frame "
              << frame_k->getFrameId();
    // dont update keyframe status to anchor yet - only do so if object
    // successfully created
  } else if (is_resampled) {
    if (num_inliers > 4) {
      object_statuses_[object_id] = ObjectTrackingStatus::WellTracked;
      // stable_frame_counts_[object_id] = 0;
      LOG(INFO) << "Object " << object_id
                << "resampled & WellTracked  at frame " << frame_k->getFrameId()
                << ", set to RegularKeyFrame";
      object_keyframe_statuses_[object_id] =
          ObjectKeyFrameStatus::RegularKeyFrame;
    } else {
      // keep object as poorly tracked if was poorly tracked before
      object_statuses_[object_id] = ObjectTrackingStatus::WellTracked;
      object_keyframe_statuses_[object_id] =
          ObjectKeyFrameStatus::AnchorKeyFrame;
      LOG(INFO) << "Object " << object_id
                << "resampled & WellTracked  at frame " << frame_k->getFrameId()
                << ", set to AnchorKeyFrame";
    }
  }

  auto construct_initial_frame =
      [](const Frame::Ptr frame, const TrackletIds& tracklets) -> gtsam::Pose3 {
    // L_e
    gtsam::Point3 initial_object_frame(
        0, 0, 0);  // important to initliase with zero values (otherwise nan's!)
    size_t count = 0;
    for (TrackletId tracklet : tracklets) {
      const Feature::Ptr feature = frame->at(tracklet);
      CHECK_NOTNULL(feature);

      gtsam::Point3 lmk = frame->backProjectToCamera(feature->trackletId());
      initial_object_frame += lmk;

      count++;
    }

    initial_object_frame /= count;
    initial_object_frame = frame->getPose() * initial_object_frame;
    return gtsam::Pose3(gtsam::Rot3::Identity(), initial_object_frame);
  };

  bool new_or_reset_object = false;
  bool filter_needs_reset = false;
  // bool new_object = false;
  // bool object_reset = false;
  if (!is_new) {
    auto filter = filters_.at(object_id);
    // filter->was_reset_this_update = false;
    // if (object_resampled) {
    //   LOG(INFO) << object_id
    //             << " retracked - resetting filter k=" <<
    //             frame_k->getFrameId();
    //   // with current logic will trigger a keyframe at k
    //   filter->was_reset_this_update = true;
    //   filter->needs_resetting_from_last_frame = true;
    // } else if (filter->needs_resetting_from_last_frame) {
    //   LOG(INFO) << object_id << " needs retting from last frame! current k="
    //             << frame_k->getFrameId();
    //   filter->needs_resetting_from_last_frame = false;

    //   // this is the pose as the last frame (ie k-1) which will serve as the
    //   new
    //   // keyframe pose
    //   gtsam::Pose3 pose = filter->getPose();
    //   // assert that the last filter update was at k-1 and therefore the pose
    //   // should be L_w_km1
    //   CHECK_EQ(filter->getFrameId(), frame_k_1->getFrameId());
    //   filter->resetState(pose, frame_k_1->getFrameId());
    //   new_or_reset_object = true;
    // }
    // handle reset first so the object actual gets reset even if it also needs
    // resampling
    if (filter->needs_resetting_from_last_frame) {
      LOG(INFO) << object_id << " needs retting from last frame! current k="
                << frame_k->getFrameId();
      filter->needs_resetting_from_last_frame = false;

      // this is the pose as the last frame (ie k-1) which will serve as the new
      // keyframe pose
      gtsam::Pose3 pose = filter->getPose();
      // assert that the last filter update was at k-1 and therefore the pose
      // should be L_w_km1
      CHECK_EQ(filter->getFrameId(), frame_k_1->getFrameId());
      filter->resetState(pose, frame_k_1->getFrameId());
      new_or_reset_object = true;
      // stable_frame_counts_[object_id] = 0;
      LOG(INFO) << "Object " << object_id << " reset, stable count reset to 0";
    }

    if (is_resampled) {
      LOG(INFO) << object_id
                << " retracked - resetting filter k=" << frame_k->getFrameId();

      // with current logic will trigger a keyframe at k
      // filter->was_reset_this_update = true;
      filter->needs_resetting_from_last_frame = true;
      LOG(INFO) << "Object " << object_id << " resampled, marked for reset";
    }
  }
  // becuuse we erase it if object new in previous!
  // not sure this is the best way to handle reappearing objects!
  if (is_new) {
    new_or_reset_object = true;
    gtsam::Matrix33 R = gtsam::Matrix33::Identity() * 1.0;
    // Initial State Covariance P (6x6)
    gtsam::Matrix66 P = gtsam::Matrix66::Identity() * 0.3;
    // Process Model noise (6x6)
    gtsam::Matrix66 Q = gtsam::Matrix66::Identity() * 0.2;

    if (geometric_result.inliers.size() < 4) {
      LOG(WARNING) << "Could not make initial frame for object " << object_id
                   << " as not enough inlier tracks!";
      // TODO: set tracking state to POOR!
      //  stable_frame_counts_[object_id] = 0;
      object_statuses_[object_id] = ObjectTrackingStatus::PoorlyTracked;
      // LOG(INFO) << "Object " << object_id << " in PoorlyTracked state at
      // frame " << frame_k->getFrameId() << " with stable count " <<
      // stable_frame_counts_[object_id];
      return false;
    }

    gtsam::Pose3 keyframe_pose =
        construct_initial_frame(frame_k_1, geometric_result.inliers);

    constexpr static double kHuberKFilter = 0.05;
    filters_.insert2(object_id, std::make_shared<HybridObjectMotionSRIF>(
                                    gtsam::Pose3::Identity(), keyframe_pose,
                                    frame_k_1->getFrameId(), P, Q, R,
                                    frame_k_1->getCamera(), kHuberKFilter));
    // tell frontend to make a new kf
    // filters_.at(object_id)->was_reset_this_update = true;

    object_keyframe_statuses_[object_id] = ObjectKeyFrameStatus::AnchorKeyFrame;

    LOG(INFO) << "Created new filter for object " << object_id << " at frame "
              << frame_k->getFrameId();
  }
  auto filter = filters_.at(object_id).get();

  // update and predict should be one step so that if we dont have enough points
  // NOTE: this logic seemed pretty important to ensure the estimate was good!!!
  // we dont predict?
  if (new_or_reset_object) {
    // if new or set must perform two predict and update steps?
    // //this must get set
    // filter->was_reset_this_update = true;
    filter->predictAndUpdate(gtsam::Pose3::Identity(), frame_k_1,
                             geometric_result.inliers, 2);
  } else {
    // filter->was_reset_this_update = false;
  }

  // on the first frame we want to init everything (ie. points)
  // so that they align with the keyframe but dont need to update anyting
  //  as the motion should be identity!!!
  gtsam::Pose3 G_w_inv_pnp = geometric_result.best_result.inverse();
  gtsam::Pose3 H_w_km1_k_pnp = frame_k->getPose() * G_w_inv_pnp;

  filter->predictAndUpdate(H_w_km1_k_pnp, frame_k, geometric_result.inliers, 2);

  bool return_result = false;
  if (geometric_result.status == TrackingStatus::VALID) {
    TrackletIds refined_inlier_tracklets = geometric_result.inliers;
    // if (homography_result) {
    // gtsam::Pose3 G_w = geometric_result.best_result.inverse();
    // gtsam::Pose3 H_w = filter->getPose();
    gtsam::Pose3 H_w_filter = filter->getF2FMotion();
    gtsam::Pose3 G_w_filter_inv =
        (frame_k->getPose().inverse() * H_w_filter).inverse();

    gtsam::Pose3 H_w = H_w_filter;

    // camera at frame_k->getPose()
    auto gtsam_camera = frame_k->getFrameCamera();

    Motion3SolverResult motion_result;
    motion_result.status = geometric_result.status;
    motion_result.inliers = geometric_result.inliers;
    motion_result.outliers = geometric_result.outliers;
    // motion_result.inliers = refined_inlier_tracklets;
    // determineOutlierIds(motion_result.inliers, all_tracklets,
    //                     motion_result.outliers);

    motion_result.best_result = Motion3ReferenceFrame(
        H_w, Motion3ReferenceFrame::Style::F2F, ReferenceFrame::GLOBAL,
        frame_k_1->getFrameId(), frame_k->getFrameId());

    frame_k->dynamic_features_.markOutliers(motion_result.outliers);
    motion_estimates.insert({object_id, motion_result.best_result});

    // Update stable frames and status
    // stable_frame_counts_[object_id]++;
    // if (stable_frame_counts_[object_id] >= 2 && object_statuses_[object_id]
    // != ObjectTrackingStatus::Tracked) {
    //   object_statuses_[object_id] = ObjectTrackingStatus::Tracked;
    //   LOG(INFO) << "Object " << object_id << " promoted to Tracked at frame "
    //   << frame_k->getFrameId() << " with stable count " <<
    //   stable_frame_counts_[object_id];
    // }

    // MUST be >= 2 so we only need two good frames in a row to be well tracked
    // if more, need to be careful when the object filter "reset" so that we
    // dont send a motion which an updated reference frame before telling the
    // backend to also update its reference frame!
    //  stable_frame_counts_[object_id]++;
    //  if (stable_frame_counts_[object_id] >= 2) {
    //    if (object_statuses_[object_id] != ObjectTrackingStatus::WellTracked)
    //    {
    //      object_statuses_[object_id] = ObjectTrackingStatus::WellTracked;
    //      LOG(INFO) << "Object " << object_id << " promoted to WellTracked at
    //      frame " << frame_k->getFrameId() << " with stable count " <<
    //      stable_frame_counts_[object_id];
    //    }
    //  } else {
    //    //not new and not poorly tracked already
    //    if (object_statuses_[object_id] != ObjectTrackingStatus::New &&
    //    object_statuses_[object_id] != ObjectTrackingStatus::PoorlyTracked) {
    //      object_statuses_[object_id] = ObjectTrackingStatus::PoorlyTracked;
    //      LOG(INFO) << "Object " << object_id << " in PoorlyTracked state at
    //      frame " << frame_k->getFrameId() << " with stable count " <<
    //      stable_frame_counts_[object_id];
    //    }
    //  }

    return_result = true;
  } else {
    // TODO: not sure lost is the right logic here!
    object_statuses_[object_id] = ObjectTrackingStatus::Lost;
    // stable_frame_counts_[object_id] = 0;
    LOG(INFO) << "Object " << object_id << " set to Lost at frame "
              << frame_k->getFrameId();
    // so tha
    filters_.erase(object_id);
    return_result = false;
    ;
  }

  LOG(INFO) << "Object " << object_id
            << " final status: " << to_string(object_statuses_[object_id])
            << ", stable frames: " << stable_frame_counts_[object_id]
            << " with keyframe status: "
            << to_string(object_keyframe_statuses_[object_id]);
  return return_result;
}

void ObjectMotionSolverFilter::fillHybridInfo(
    ObjectId object_id, VisionImuPacket::ObjectTracks& object_track) {
  CHECK(filters_.exists(object_id));
  auto filter = filters_.at(object_id);

  CHECK(object_keyframe_statuses_.exists(object_id));
  auto object_kf_status = object_keyframe_statuses_.at(object_id);

  CHECK(object_statuses_.exists(object_id));
  auto object_motion_track_status = object_statuses_.at(object_id);

  VisionImuPacket::ObjectTracks::HybridInfo hybrid_info;
  hybrid_info.H_W_KF_k = filter->getKeyFramedMotionReference();
  hybrid_info.L_W_KF = filter->getKeyFramePose();
  hybrid_info.L_W_k = filter->getPose();
  // hybrid_info.was_reset = filter->resetThisUpdate();

  object_track.regular_keyframe = false;
  object_track.anchor_keyframe = false;

  object_track.motion_track_status = object_motion_track_status;

  if (object_track.motion_track_status == ObjectTrackingStatus::New ||
      object_track.motion_track_status == ObjectTrackingStatus::WellTracked) {
    if (object_kf_status == ObjectKeyFrameStatus::AnchorKeyFrame) {
      object_track.regular_keyframe = true;
      object_track.anchor_keyframe = true;
    } else if (object_kf_status == ObjectKeyFrameStatus::RegularKeyFrame) {
      object_track.regular_keyframe = true;
    }
  }

  LOG(INFO) << "Making hybrid info for j=" << object_id << " with "
            << "motion KF: " << hybrid_info.H_W_KF_k.from()
            << " to: " << hybrid_info.H_W_KF_k.to()
            << " track status: " << to_string(object_motion_track_status)
            << " with regular kf " << std::boolalpha
            << object_track.regular_keyframe << " anchor kf "
            << object_track.anchor_keyframe;

  object_track.hybrid_info = hybrid_info;
}

void ObjectMotionSolverFilter::updatePoses(
    ObjectPoseMap& object_poses, const MotionEstimateMap& motion_estimates,
    Frame::Ptr frame_k, Frame::Ptr /*frame_k_1*/) {
  const FrameId frame_id_k = frame_k->getFrameId();
  for (const auto& [object_id, _] : motion_estimates) {
    CHECK(filters_.exists(object_id));

    auto filter = filters_.at(object_id);
    gtsam::Pose3 L_k_j = filter->getPose();
    object_poses_.insert22(object_id, frame_id_k, L_k_j);
  }
  object_poses = object_poses_;
}

void ObjectMotionSolverFilter::updateMotions(
    ObjectMotionMap& object_motions, const MotionEstimateMap& motion_estimates,
    Frame::Ptr frame_k, Frame::Ptr frame_k_1) {
  // same as ObjectMotionSovlerF2F
  const FrameId frame_id_k = frame_k->getFrameId();
  for (const auto& [object_id, motion_reference_frame] : motion_estimates) {
    object_motions_.insert22(object_id, frame_id_k, motion_reference_frame);
  }
  object_motions = object_motions_;
}

}  // namespace dyno
