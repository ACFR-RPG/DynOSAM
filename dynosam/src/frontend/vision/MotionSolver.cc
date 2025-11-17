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
#include "dynosam/factors/LandmarkMotionTernaryFactor.hpp"
#include "dynosam/factors/Pose3FlowProjectionFactor.h"
#include "dynosam/frontend/vision/VisionTools.hpp"
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
    utils::TimingStatsCollector track_dynamic_timer(
        "mono_frame_correspondences");
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
    utils::TimingStatsCollector timer(
        "motion_solver.solve_3d2d.correspondances");
    frame_k->getCorrespondences(correspondences, *frame_k_1,
                                KeyPointType::STATIC,
                                frame_k->landmarkWorldKeypointCorrespondance());
  }

  return geometricOutlierRejection3d2d(correspondences, R_curr_ref);
}

Pose3SolverResult EgoMotionSolver::geometricOutlierRejection3d2d(
    const AbsolutePoseCorrespondences& correspondences,
    std::optional<gtsam::Rot3> R_curr_ref) {
  utils::TimingStatsCollector timer("motion_solver.solve_3d2d");
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
    utils::TimingStatsCollector timer("motion_solver.solve_3d2d.ransac");
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
  utils::TimingStatsCollector timer("of_motion_solver.update_frame");
  // //original flow image that goes from k to k+1 (gross, im sorry!)
  // TODO: use flow_is_future param
  // const cv::Mat& flow_image = frame_k->image_container_.opticalFlow();
  const cv::Mat& motion_mask = frame_k->image_container_.objectMotionMask();

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

    // // we now have to update the prediced keypoint using the original flow!!
    // // TODO: code copied from feature tracker
    // const int x = functional_keypoint::u(refined_keypoint);
    // const int y = functional_keypoint::v(refined_keypoint);
    // double flow_xe = static_cast<double>(flow_image.at<cv::Vec2f>(y, x)[0]);
    // double flow_ye = static_cast<double>(flow_image.at<cv::Vec2f>(y, x)[1]);
    // // the measured flow after the origin has been updated
    // OpticalFlow new_measured_flow(flow_xe, flow_ye);
    // feature_k->measuredFlow(new_measured_flow);
    // // TODO: check predicted flow is within image
    // Keypoint predicted_kp = Feature::CalculatePredictedKeypoint(
    //     refined_keypoint, new_measured_flow);
    // feature_k->predictedKeypoint(predicted_kp);
    // feature_k->keypoint(refined_keypoint);

    // // we now have to update the prediced keypoint using the original flow!!
    // // TODO: code copied from feature tracker
    // const int x = functional_keypoint::u(refined_keypoint);
    // const int y = functional_keypoint::v(refined_keypoint);
    // double flow_xe = static_cast<double>(flow_image.at<cv::Vec2f>(y, x)[0]);
    // double flow_ye = static_cast<double>(flow_image.at<cv::Vec2f>(y, x)[1]);
    // the measured flow after the origin has been updated
    feature_k->measuredFlow(refined_flow);
    // TODO: check predicted flow is within image
    Keypoint predicted_kp =
        Feature::CalculatePredictedKeypoint(refined_keypoint, refined_flow);
    feature_k->predictedKeypoint(predicted_kp);
    feature_k->keypoint(refined_keypoint);

    // feature_k_1->predictedKeypoint(refined_keypoint);
    // feature_k_1->measuredFlow(refined_flow);
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
    utils::TimingStatsCollector("pc_correspondences");
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

ObjectMotionSovlerF2F::ObjectMotionSovlerF2F(
    const ObjectMotionSovlerF2F::Params& params,
    const CameraParams& camera_params)
    : EgoMotionSolver(static_cast<const EgoMotionSolver::Params&>(params),
                      camera_params),
      object_motion_params(params) {}

ObjectMotionSovlerF2F::Result ObjectMotionSovlerF2F::solve(
    Frame::Ptr frame_k, Frame::Ptr frame_k_1) {
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
  auto motions = updateMotions(motion_estimates, frame_k, frame_k_1);
  auto poses = updatePoses(motion_estimates, frame_k, frame_k_1);
  return std::make_pair(motions, poses);
}

const ObjectPoseMap& ObjectMotionSovlerF2F::updatePoses(
    MotionEstimateMap& motion_estimates, Frame::Ptr frame_k,
    Frame::Ptr frame_k_1) {
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

  return object_poses_;
}

const ObjectMotionMap& ObjectMotionSovlerF2F::updateMotions(
    MotionEstimateMap& motion_estimates, Frame::Ptr frame_k, Frame::Ptr) {
  const FrameId frame_id_k = frame_k->getFrameId();
  for (const auto& [object_id, motion_reference_frame] : motion_estimates) {
    object_motions_.insert22(object_id, frame_id_k, motion_reference_frame);
  }
  return object_motions_;
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
  utils::TimingStatsCollector timer("motion_solver.object_solve3d2d");
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

namespace testing {

using namespace cv;
using namespace std;

// Compute isotropic normalization transform for a set of 2D points
static Mat computeNormalizationTransform(const vector<Point2f>& pts) {
  int n = (int)pts.size();
  Point2f centroid(0, 0);
  for (const auto& p : pts) centroid += p;
  centroid.x /= n;
  centroid.y /= n;

  double meanDist = 0.0;
  for (const auto& p : pts) {
    double dx = p.x - centroid.x;
    double dy = p.y - centroid.y;
    meanDist += sqrt(dx * dx + dy * dy);
  }
  meanDist /= n;

  double scale = (meanDist > 0) ? (sqrt(2.0) / meanDist) : 1.0;

  Mat T = Mat::eye(3, 3, CV_64F);
  T.at<double>(0, 0) = scale;
  T.at<double>(1, 1) = scale;
  T.at<double>(0, 2) = -scale * centroid.x;
  T.at<double>(1, 2) = -scale * centroid.y;
  return T;
}

// Normalized DLT homography estimation: src -> dst (both length >= 4)
Mat estimateHomographyDLT(const vector<Point2f>& src,
                          const vector<Point2f>& dst) {
  CV_Assert(src.size() >= 4 && src.size() == dst.size());
  int n = (int)src.size();

  Mat T1 = computeNormalizationTransform(src);
  Mat T2 = computeNormalizationTransform(dst);

  // Normalize points
  vector<Point2f> nsrc(n), ndst(n);
  for (int i = 0; i < n; i++) {
    Mat p = (Mat_<double>(3, 1) << src[i].x, src[i].y, 1.0);
    Mat pn = T1 * p;
    nsrc[i] = Point2f((float)(pn.at<double>(0, 0) / pn.at<double>(2, 0)),
                      (float)(pn.at<double>(1, 0) / pn.at<double>(2, 0)));

    Mat q = (Mat_<double>(3, 1) << dst[i].x, dst[i].y, 1.0);
    Mat qn = T2 * q;
    ndst[i] = Point2f((float)(qn.at<double>(0, 0) / qn.at<double>(2, 0)),
                      (float)(qn.at<double>(1, 0) / qn.at<double>(2, 0)));
  }

  // Build A matrix (2n x 9)
  Mat A = Mat::zeros(n * 2, 9, CV_64F);
  for (int i = 0; i < n; i++) {
    double x = nsrc[i].x;
    double y = nsrc[i].y;
    double u = ndst[i].x;
    double v = ndst[i].y;
    A.at<double>(2 * i, 0) = -x;
    A.at<double>(2 * i, 1) = -y;
    A.at<double>(2 * i, 2) = -1;
    A.at<double>(2 * i, 6) = x * u;
    A.at<double>(2 * i, 7) = y * u;
    A.at<double>(2 * i, 8) = u;

    A.at<double>(2 * i + 1, 3) = -x;
    A.at<double>(2 * i + 1, 4) = -y;
    A.at<double>(2 * i + 1, 5) = -1;
    A.at<double>(2 * i + 1, 6) = x * v;
    A.at<double>(2 * i + 1, 7) = y * v;
    A.at<double>(2 * i + 1, 8) = v;
  }

  // Solve Ah = 0 via SVD
  Mat w, u, vt;
  SVD::compute(A, w, u, vt, SVD::MODIFY_A);
  Mat h =
      vt.row(8).reshape(0, 3);  // last row of V^T -> smallest singular value

  // Denormalize: H = inv(T2) * h * T1
  Mat H = Mat::zeros(3, 3, CV_64F);
  Mat Hn;
  h.convertTo(Hn, CV_64F);
  H = T2.inv() * Hn * T1;

  // Normalize so H(2,2)=1
  if (fabs(H.at<double>(2, 2)) > 1e-12) H /= H.at<double>(2, 2);
  return H;
}

// Decompose homography to possible poses and choose best using cheirality
// (points in front)
bool recoverPoseFromHomography(const Mat& H_in, const Mat& K,
                               const vector<Point3f>& planePts3D,
                               const vector<Point2f>& imagePts, Mat& bestR,
                               Mat& bestT) {
  CV_Assert(H_in.size() == Size(3, 3));
  Mat H;
  H_in.convertTo(H, CV_64F);

  // OpenCV expects normalized homography such that H = K * [r1 r2 t]
  vector<Mat> Rs, Ts, Ns;
  int solutions = decomposeHomographyMat(H, K, Rs, Ts, Ns);
  if (solutions == 0) return false;

  int bestIdx = -1;
  int bestFront = -1;

  // For each solution count number of points with positive depth
  for (int i = 0; i < solutions; i++) {
    Mat R = Rs[i];
    Mat t = Ts[i];

    int frontCount = 0;
    vector<Point2f> proj;
    // Project 3D plane points with candidate pose
    projectPoints(planePts3D, R, t, K, Mat(), proj);
    for (size_t j = 0; j < proj.size(); j++) {
      // Reconstruct depth sign by transforming a 3D point and looking at z in
      // camera frame
      Mat X = (Mat_<double>(3, 1) << planePts3D[j].x, planePts3D[j].y,
               planePts3D[j].z);
      Mat Xcam = R * X + t;
      if (Xcam.at<double>(2, 0) > 0) frontCount++;
    }
    if (frontCount > bestFront) {
      bestFront = frontCount;
      bestIdx = i;
    }
  }

  if (bestIdx < 0) return false;
  bestR = Rs[bestIdx].clone();
  bestT = Ts[bestIdx].clone();
  return true;
}

bool poseFromHomograph(const std::vector<cv::Point3f>& points3D,
                       const std::vector<cv::Point2f>& points2D,
                       const cv::Mat& K, gtsam::Pose3& Gw) {
  // Ensure we have enough correspondences
  if (points3D.size() < 4 || points3D.size() != points2D.size()) {
    return false;
  }

  // Convert 3D points to 2D by projecting onto XY plane (Z=0)
  std::vector<cv::Point2f> points3D_2D;
  for (const auto& pt : points3D) {
    points3D_2D.push_back(cv::Point2f(pt.x, pt.y));
  }

  // Find homography matrix
  cv::Mat H = cv::findHomography(points3D_2D, points2D, cv::RANSAC, 3.0);

  if (H.empty()) {
    throw std::runtime_error("Failed to compute homography");
  }

  // Decompose homography to get rotation and translation
  // H = K * [r1 r2 t] where r1, r2 are first two columns of rotation matrix
  cv::Mat K_inv = K.inv();

  // Normalize: H_normalized = K^(-1) * H
  cv::Mat H_normalized = K_inv * H;

  // Extract columns
  cv::Mat col1 = H_normalized.col(0);
  cv::Mat col2 = H_normalized.col(1);
  cv::Mat col3 = H_normalized.col(2);

  // Normalize rotation columns
  double lambda1 = cv::norm(col1);
  double lambda2 = cv::norm(col2);
  double lambda = (lambda1 + lambda2) / 2.0;

  cv::Mat r1 = col1 / lambda;
  cv::Mat r2 = col2 / lambda;
  cv::Mat t = col3 / lambda;

  // Compute r3 as cross product of r1 and r2
  cv::Mat r3 = r1.cross(r2);

  // Build rotation matrix
  cv::Mat R = cv::Mat(3, 3, CV_64F);
  r1.copyTo(R.col(0));
  r2.copyTo(R.col(1));
  r3.copyTo(R.col(2));

  // Ensure R is a valid rotation matrix using SVD
  cv::Mat W, U, Vt;
  cv::SVD::compute(R, W, U, Vt);
  R = U * Vt;

  // Convert OpenCV matrices to Eigen
  Eigen::Matrix3d R_eigen;
  Eigen::Vector3d t_eigen;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      R_eigen(i, j) = R.at<double>(i, j);
    }
    t_eigen(i) = t.at<double>(i, 0);
  }

  // Create 4x4 homogeneous transformation matrix
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform.block<3, 3>(0, 0) = R_eigen;
  transform.block<3, 1>(0, 3) = t_eigen;

  Gw = gtsam::Pose3(transform);
  return true;
}

bool poseFromPnP(const std::vector<cv::Point3f>& points3D,
                 const std::vector<cv::Point2f>& points2D, const cv::Mat& K,
                 gtsam::Pose3& Gw, cv::Mat& inliers) {
  // Ensure we have enough correspondences
  if (points3D.size() < 4 || points3D.size() != points2D.size()) {
    return false;
  }

  // Output vectors
  Mat rvec1, tvec1;
  cv::Mat distCoeffs = cv::Mat::zeros(1, 4, CV_64FC1);
  cv::Mat inliers1;

  // --- 1. Run solvePnPRansac ---
  bool success_ippe =
      solvePnPRansac(points3D, points2D, K, distCoeffs, rvec1, tvec1,
                     false,  // useExtrinsicGuess
                     500,    // iterationsCount (number of RANSAC iterations)
                     0.4,    // reprojectionError (max allowed error in pixels)
                     0.99,   // confidence
                     inliers1,      // Output for inlier indices
                     SOLVEPNP_IPPE  // PnP method (EPnP is a good default)
      );

  Mat rvec2, tvec2, inliers2;
  bool success_epnp =
      solvePnPRansac(points3D, points2D, K, distCoeffs, rvec2, tvec2,
                     false,  // useExtrinsicGuess
                     500,    // iterationsCount (number of RANSAC iterations)
                     0.4,    // reprojectionError (max allowed error in pixels)
                     0.99,   // confidence
                     inliers2,      // Output for inlier indices
                     SOLVEPNP_AP3P  // PnP method (EPnP is a good default)
      );

  Mat rvec, tvec;
  if (success_ippe && success_epnp) {
    LOG(INFO) << "Solved both object motions with ippe and epnp"
                 " IPPE inliers: "
              << inliers1.rows << " EPnP inliers: " << inliers2.rows;
    ;

    if (inliers1.rows > inliers2.rows) {
      // IPPE better
      rvec = rvec1;
      tvec = tvec1;
      inliers = inliers1;
    } else {
      rvec = rvec2;
      tvec = tvec2;
      inliers = inliers2;
    }
  } else if (success_ippe) {
    LOG(INFO) << "Only solved for IPPE";
    rvec = rvec1;
    tvec = tvec1;
    inliers = inliers1;
  } else if (success_epnp) {
    LOG(INFO) << "Only solved for EPnP";
    rvec = rvec2;
    tvec = tvec2;
    inliers = inliers2;
  } else {
    LOG(WARNING) << "Failed to solve object motion EPnP or IPPE";
    return false;
  }

  // cout << "Pose calculation successful using " << inliers.rows << " inliers."
  // << endl;

  // --- 2. Convert rvec to a 3x3 Rotation Matrix (R) ---
  Mat R_mat;               // OpenCV Mat for the 3x3 rotation matrix
  Rodrigues(rvec, R_mat);  // rvec (Rodrigues form) -> R_mat (3x3 matrix)

  // --- 3. Assemble the Homogeneous Matrix (T) using Eigen ---
  Eigen::Matrix4d T_homo =
      Eigen::Matrix4d::Identity();  // Start with an Identity matrix

  // Copy Rotation (R) from OpenCV Mat to Eigen 3x3 block
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      // R_mat is CV_64F (double) by default from Rodrigues
      T_homo(i, j) = R_mat.at<double>(i, j);
    }
  }

  // Copy Translation (t) from OpenCV Mat to Eigen 3x1 block
  T_homo(0, 3) = tvec.at<double>(0, 0);  // T_x
  T_homo(1, 3) = tvec.at<double>(1, 0);  // T_y
  T_homo(2, 3) = tvec.at<double>(2, 0);  // T_z

  Gw = gtsam::Pose3(T_homo);
  return true;
}

void testing::ExtendedKalmanFilterGTSAM::update(
    const vector<Point3>& P_w_list, const vector<Point2>& z_obs_list,
    const gtsam::Pose3& X_W_k) {
  if (P_w_list.empty() || P_w_list.size() != z_obs_list.size()) {
    cerr << "EKF Update skipped: Input lists are empty or mismatched." << endl;
    return;
  }

  const size_t num_measurements = P_w_list.size();
  const size_t total_rows =
      num_measurements * 2;  // Each measurement has 2 dimensions (u, v)

  // Stacked Jacobian H (M x 6) and Stacked Residual y (M x 1)
  gtsam::Matrix H_stacked = gtsam::Matrix::Zero(total_rows, 6);
  gtsam::Vector y_stacked = gtsam::Vector::Zero(total_rows);

  // Stacked Measurement Noise R_stacked (M x M)
  gtsam::Matrix R_stacked = gtsam::Matrix::Zero(total_rows, total_rows);
  // camera with fixed calibration so HPose is of dims 2x6
  //  gtsam::PinholePose<Cal3_S2> camera(G_w_, K_gtsam_);

  // auto project = [&](const gtsam::Point3& P_w, const gtsam::Pose3& H_w) ->
  // gtsam::Point2 {
  //       const gtsam::Pose3 Gw = X_W_k.inverse() * H_w;
  //       gtsam::PinholePose<Cal3_S2> camera(Gw.inverse(), K_gtsam_);
  //       // return camera.project2(P_w);
  //       auto [p, success] = camera.projectSafe(P_w);
  //       return p;
  // };

  // Calculate the combined pose T_cw = X^{-1} * H
  // Pose3 G_w = X_W_k.inverse() * H_w_;

  // gtsam::PinholePose<Cal3_S2> camera(G_w, K_gtsam_);
  // Calculate the Adjoint Matrix of X^{-1}, which links delta_h to delta_t_cw
  gtsam::Matrix Ad_X_inv = X_W_k.inverse().AdjointMap();  // 6x6 matrix

  Pose3 G_w = X_W_k.inverse() * H_w_;
  gtsam::PinholePose<Cal3_S2> camera(G_w.inverse(), K_gtsam_);
  // gtsam::Matrix Ad_X = X_W_k.AdjointMap(); // 6x6 matrix
  // 3. Calculate Correction Factor: J_corr = d(delta_t_wc)/d(delta_w) =
  // -Ad_{T_cw} This corrects the Jacobian returned by GTSAM (w.r.t. T_wc) to be
  // w.r.t. the state W.
  gtsam::Matrix Ad_G_w = G_w.AdjointMap();
  gtsam::Matrix CorrectionFactor = -Ad_G_w;

  for (size_t i = 0; i < num_measurements; ++i) {
    const gtsam::Point3& P_w = P_w_list[i];
    const gtsam::Point2& z_obs = z_obs_list[i];

    gtsam::Matrix26 H_pose;  // 2x6 Jacobian for this measurement (This was
                             // missing/corrupted)

    // // 1. Calculate Predicted Measurement (h(x)) and Jacobian (H)
    // gtsam::Point2 z_pred;
    // try {
    //     // // GTSAM's project() automatically computes the 2x6 Jacobian w.r.t
    //     the camera pose.
    //     // // The Jacobian is stored in H_pose.
    //     // z_pred = camera.project2(P_w, H_pose, boost::none);
    //     H_pose = gtsam::numericalDerivative22<gtsam::Vector2, gtsam::Point3,
    //     gtsam::Pose3>(
    //       project, P_w, H_w_
    //     );

    //   z_pred = project(P_w, H_w_);

    // } catch (const gtsam::CheiralityException& e) {
    //     // Handle points behind the camera
    //     cerr << "Warning: Point " << i << " behind camera. Skipping." <<
    //     endl; continue;
    // }
    // J_pi: Jacobian of projection w.r.t. T_cw perturbation (delta_t_cw)
    gtsam::Matrix J_pi;  // 2x6 matrix

    // 1. Calculate Predicted Measurement (h(H)) and Jacobian (J_pi)
    gtsam::Point2 z_pred;
    try {
      // GTSAM's project() computes: z_pred = pi(T_cw * P_w), J_pi =
      // d(pi)/d(delta_t_cw)
      z_pred = camera.project(P_w, J_pi);
    } catch (const gtsam::CheiralityException& e) {
      // Handle points behind the camera
      cerr << "Warning: Point " << i << " behind camera. Skipping." << endl;
      continue;
    }
    // 2. Calculate Residual (Error) y = z_obs - z_pred
    gtsam::Vector2 y_i = z_obs - z_pred;  // 2x1 residual vector

    // 3. Calculate EKF Jacobian: H_EKF = J_pi * Ad_X_inv
    // H_EKF = d(z)/d(delta_h) = (d(z)/d(delta_t_cw)) *
    // (d(delta_t_cw)/d(delta_h))
    gtsam::Matrix H_ekf_i = J_pi * CorrectionFactor;  // 2x6 matrix

    // 3. Stack H and y
    H_stacked.block<2, 6>(i * 2, 0) = H_ekf_i;
    y_stacked.segment<2>(i * 2) = y_i;

    // 4. Stack R (R_stacked is block diagonal with R)
    R_stacked.block<2, 2>(i * 2, i * 2) = R_;
  }

  // --- EKF Formulas using stacked matrices ---

  // 1. Innovation Covariance (S)
  // S = H_stacked * P * H_stacked^T + R_stacked
  gtsam::Matrix S = H_stacked * P_ * H_stacked.transpose() + R_stacked;

  // 2. Kalman Gain (K)
  // K = P * H_stacked^T * S^-1
  // We use FullLinearSolver for robustness with potentially large S
  Eigen::LLT<gtsam::Matrix> ldlt(S);
  gtsam::Matrix S_inv;
  {
    utils::TimingStatsCollector timer("ekfgtsam.inv");
    S_inv = ldlt.solve(Eigen::MatrixXd::Identity(S.rows(), S.cols()));
  }
  // gtsam::Matrix S_inv = ldlt.solve(Eigen::MatrixXd::Identity(S.rows(),
  // S.cols())); gtsam::Matrix S_inv = S.inverse();
  gtsam::Matrix K = P_ * H_stacked.transpose() * S_inv;  // 6 x M

  // 3. State Update (Perturbation Vector delta_x)
  // delta_x = K * y_stacked
  PerturbationVector delta_x = K * y_stacked;  // 6 x 1

  // 4. State Retraction (Update T_cw)
  // T_new = T_old.retract(delta_x)
  // GTSAM's retract is the generalized exponential map on the manifold.
  H_w_ = H_w_.retract(delta_x);

  // 5. Covariance Update
  // P = (I - K * H_stacked) * P
  gtsam::Matrix I = gtsam::Matrix66::Identity();
  // P_ = (I - K * H_stacked) * P_;
  gtsam::Matrix I_KH = I - K * H_stacked;
  P_ = I_KH * P_ * I_KH.transpose() + K * R_stacked * K.transpose();

  // cout << "\n--- EKF Update Complete ---" << endl;
  // cout << "Number of measurements processed: " << num_measurements << endl;
  // cout << "Total Residual Norm (||y||): " << y_stacked.norm() << endl;
  // cout << "Perturbation (delta_x) [t, phi]:\n" << delta_x.transpose() <<
  // endl;
}

}  // namespace testing

ObjectMotionSolverFilter::ObjectMotionSolverFilter(
    const ObjectMotionSolverFilter::Params& params,
    const CameraParams& camera_params)
    : ObjectMotionSovlerF2F(params, camera_params) {}

bool ObjectMotionSolverFilter::solveImpl(Frame::Ptr frame_k,
                                         Frame::Ptr frame_k_1,
                                         ObjectId object_id,
                                         MotionEstimateMap& motion_estimates) {
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

  const auto K = camera_params_.getCameraMatrix();

  auto rgbd_camera = frame_k->camera_->safeGetRGBDCamera();
  CHECK_NOTNULL(rgbd_camera);

  if (!filters_.exists(object_id)) {
    testing::MeasurementCovariance R =
        testing::MeasurementCovariance::Identity() * (1 * 1.0);

    // Initial State Covariance P (6x6)
    testing::StateCovariance P = testing::StateCovariance::Identity() * 0.3;

    // filters_.insert2(object_id,
    //                  std::make_shared<testing::ExtendedKalmanFilterGTSAM>(
    //                      gtsam::Pose3::Identity(), P,
    //                      camera_params_.getCameraMatrixEigen(), R));

    filters_.insert2(object_id,
                     std::make_shared<testing::SquareRootInfoFilterGTSAM>(
                         gtsam::Pose3::Identity(), P,
                         camera_params_.getCameraMatrixEigen(), R));
    filters_.at(object_id)->R_stereo_noise_ =
        testing::MeasurementCovarianceStereo::Identity() * 1.0;
  }
  auto filter = filters_.at(object_id).get();

  if (filter->stereo_calib_ == nullptr) {
    filter->stereo_calib_ = rgbd_camera->getFakeStereoCalib();
  }

  CHECK_NOTNULL(filter);

  // std::vector<cv::Point3f> object_points;
  // std::vector<cv::Point2f> image_points;

  // for (const auto& corres : dynamic_correspondences) {
  //   Landmark lmk = corres.ref_;
  //   Keypoint kp = corres.cur_;

  //   object_points.push_back(
  //       cv::Point3f((float)lmk(0), (float)lmk(1), (float)lmk(2)));
  //   image_points.push_back(utils::gtsamPointToCv(kp));
  // };

  std::vector<gtsam::Point3> object_points;
  std::vector<gtsam::Point2> image_points;
  std::vector<gtsam::StereoPoint2> stereo_measurements;

  for (TrackletId inlier_tracklet : geometric_result.inliers) {
    const Feature::Ptr feature_k_1 = frame_k_1->at(inlier_tracklet);
    Feature::Ptr feature_k = frame_k->at(inlier_tracklet);

    if (feature_k->hasDepth()) {
      bool right_projection_result = rgbd_camera->projectRight(feature_k);
      if (!right_projection_result) {
        // TODO: for now mark as outlier and ignore point
        feature_k->markOutlier();
        continue;
      }
    }
    const Keypoint& L = feature_k->keypoint();
    const Keypoint& R = feature_k->rightKeypoint();

    gtsam::StereoPoint2 stereo_measurement(L(0), R(0), L(1));

    // (rgbd_camera && f->hasDepth()) {
    //         bool right_projection_result = rgbd_camera->projectRight(f);
    //         if (!right_projection_result) {
    //           // TODO: for now mark as outlier and ignore point
    //           f->markOutlier();
    //           continue;
    //         }

    CHECK_NOTNULL(feature_k_1);
    CHECK_NOTNULL(feature_k);

    const Keypoint kp_k = feature_k->keypoint();
    const gtsam::Point3 lmk_k_1_world =
        frame_k_1->backProjectToWorld(inlier_tracklet);

    object_points.push_back(lmk_k_1_world);
    image_points.push_back(kp_k);
    stereo_measurements.push_back(stereo_measurement);
  }

  // update and predict should be one step so that if we dont have enough points
  // we dont predict?
  filter->predict();
  {
    utils::TimingStatsCollector timer("motion_solver.ekf_update");
    filter->update(object_points, image_points, frame_k->getPose());
    // filter->updateStereo(object_points, stereo_measurements,
    // frame_k->getPose());
  }

  // gtsam::Pose3 G_w;
  // cv::Mat inliers_ransac;
  // bool homography_result =
  //     testing::poseFromPnP(object_points, image_points, K, G_w,
  //     inliers_ransac);

  bool homography_result = false;

  if (geometric_result.status == TrackingStatus::VALID) {
    TrackletIds refined_inlier_tracklets = geometric_result.inliers;
    // if (homography_result) {
    // gtsam::Pose3 G_w = geometric_result.best_result.inverse();
    // gtsam::Pose3 H_w = filter->getPose();
    gtsam::Pose3 H_w_filter = filter->getStatePoseW();
    gtsam::Pose3 G_w_filter_inv =
        (frame_k->getPose().inverse() * H_w_filter).inverse();
    // gtsam::Pose3 H_w = frame_k->getPose() * G_w;

    // OpticalFlowAndPoseOptimizer flow_optimizer(
    //       object_motion_params.joint_of_params);

    // auto flow_opt_result = flow_optimizer.optimizeAndUpdate<CalibrationType>(
    //       frame_k_1, frame_k, refined_inlier_tracklets,
    //       G_w_filter_inv);

    // // still need to take the inverse as we get the inverse of G out
    // const auto G_w_flow = flow_opt_result.best_result.refined_pose.inverse();
    // // inliers should be a subset of the original refined inlier tracks
    // refined_inlier_tracklets = flow_opt_result.inliers;
    // gtsam::Pose3 H_w = frame_k->getPose() * G_w_flow;

    gtsam::Pose3 H_w = H_w_filter;

    // camera at frame_k->getPose()
    auto gtsam_camera = frame_k->getFrameCamera();

    // calcuate reprojection error
    // double inlier_error = 0, outlier_error = 0;
    // int inlier_count = 0, outlier_count = 0;
    // for(const AbsolutePoseCorrespondence& corr : dynamic_correspondences) {
    //   gtsam::Point3 lmk_W_k_1 = corr.ref_;
    //   gtsam::Point3 lmk_W_k = H_w * lmk_W_k_1;
    //   Keypoint kp_k_measured = corr.cur_;

    //   Keypoint kp_k_projected = gtsam_camera.project2(lmk_W_k);
    //   double repr = (kp_k_measured - kp_k_projected).norm();

    //   cv::Scalar colour;

    //   auto it = std::find(geometric_result.inliers.begin(),
    //   geometric_result.inliers.end(), corr.tracklet_id_); if(it !=
    //   geometric_result.inliers.end()) {
    //     //inlier
    //     inlier_error += repr;
    //     inlier_count++;
    //     colour = Color::green().bgra();
    //   }
    //   else {
    //     //outlier
    //     outlier_error += repr;
    //     outlier_count++;
    //     colour = Color::red().bgra();
    //   }
    //   cv::arrowedLine(viz, utils::gtsamPointToCv(kp_k_measured),
    //                   utils::gtsamPointToCv(kp_k_projected), colour, 1, 8, 0,
    //                   0.1);
    // cv::circle(viz, utils::gtsamPointToCv(kp_k_measured), 2, colour, -1);
    // }

    // LOG(INFO) << "Inlier repr " << inlier_error/(double)inlier_count <<
    // " outlier rpr " << outlier_error/(double)outlier_count;

    // cv::imshow("Inlier/Outlier", viz);
    // cv::waitKey(0);

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
    return true;
  } else {
    return false;
  }
}

}  // namespace dyno
