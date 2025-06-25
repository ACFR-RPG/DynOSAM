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
#include "dynosam/backend/FactorGraphTools.hpp"  //TODO: clean
#include "dynosam/common/Flags.hpp"
#include "dynosam/common/Types.hpp"
#include "dynosam/factors/LandmarkMotionTernaryFactor.hpp"
#include "dynosam/factors/Pose3FlowProjectionFactor.h"
#include "dynosam/frontend/vision/VisionTools.hpp"
#include "dynosam/utils/Accumulator.hpp"
#include "dynosam/utils/GtsamUtils.hpp"
#include "dynosam/utils/Numerical.hpp"
#include "dynosam/utils/TimingStats.hpp"

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
  
  field(config.initial_R_world_camera, "initial_R_world_camera");
  field(config.initial_t_world_camera, "initial_t_world_camera");
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

  const double reprojection_error = params_.ransac_threshold_pnp;
  const double avg_focal_length =
      0.5 * static_cast<double>(camera_params_.fx() + camera_params_.fy());
  const double threshold =
      1.0 - std::cos(std::atan(std::sqrt(2.0) * reprojection_error /
                               avg_focal_length));
  // const double threshold = params_.ransac_threshold_pnp;

  AbsolutePoseAdaptor adapter(bearing_vectors, points);

  gtsam::Pose3 best_result;
  std::vector<int> ransac_inliers;

  bool success = runRansac<AbsolutePoseProblem>(
      std::make_shared<AbsolutePoseProblem>(adapter,
                                            AbsolutePoseProblem::KNEIP),
      threshold, params_.ransac_iterations, params_.ransac_probability,
      params_.optimize_3d2d_pose_from_inliers, best_result, ransac_inliers);

  if (success) {
    constructTrackletInliers(result.inliers, result.outliers, correspondences,
                             ransac_inliers, tracklets);

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
  // //original flow image that goes from k to k+1 (gross, im sorry!)
  // TODO: use flow_is_future param
  const cv::Mat& flow_image =
      frame_k->image_container_.get<ImageType::OpticalFlow>();
  const cv::Mat& motion_mask =
      frame_k->image_container_.get<ImageType::MotionMask>();

  auto camera = frame_k->camera_;
  const auto& refined_inliers = result.inliers;
  const auto& refined_flows = result.best_result.refined_flows;

  // outliers from the result. We will update this vector with new outliers
  auto refined_outliers = result.outliers;

  for (size_t i = 0; i < refined_inliers.size(); i++) {
    TrackletId tracklet_id = refined_inliers.at(i);
    gtsam::Point2 refined_flow = refined_flows.at(i);

    const Feature::Ptr feature_k_1 = frame_k_1->at(tracklet_id);
    Feature::Ptr feature_k = frame_k->at(tracklet_id);

    CHECK_EQ(feature_k->objectId(), result.best_result.object_id);

    const Keypoint kp_k_1 = feature_k_1->keypoint();
    Keypoint refined_keypoint = kp_k_1 + refined_flow;

    // check boundaries?
    if (!camera->isKeypointContained(refined_keypoint)) {
      refined_outliers.push_back(tracklet_id);
      continue;
    }

    feature_k->keypoint(refined_keypoint);
    ObjectId predicted_label =
        functional_keypoint::at<ObjectId>(refined_keypoint, motion_mask);
    if (predicted_label != result.best_result.object_id) {
      refined_outliers.push_back(tracklet_id);
      // TODO: other fields of the feature does not get updated? Inconsistencies
      // as measured flow, predicted kp etc are no longer correct!!?
      continue;
    }

    // we now have to update the prediced keypoint using the original flow!!
    // TODO: code copied from feature tracker
    const int x = functional_keypoint::u(refined_keypoint);
    const int y = functional_keypoint::v(refined_keypoint);
    double flow_xe = static_cast<double>(flow_image.at<cv::Vec2f>(y, x)[0]);
    double flow_ye = static_cast<double>(flow_image.at<cv::Vec2f>(y, x)[1]);
    // the measured flow after the origin has been updated
    OpticalFlow new_measured_flow(flow_xe, flow_ye);
    feature_k->measuredFlow(new_measured_flow);
    // TODO: check predicted flow is within image
    Keypoint predicted_kp = Feature::CalculatePredictedKeypoint(
        refined_keypoint, new_measured_flow);
    feature_k->predictedKeypoint(predicted_kp);
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
        [&](const std::pair<ObjectId, DynamicObjectObservation>& pair) {
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
    frame_k->dynamic_features_.markOutliers(result.outliers);

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
      VLOG(10) << "Refining object motion pose with joint of";
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

  return motion_result;
}

// Pose3SolverResult ObjectMotionSovler::geometricOutlierRejection3d3d(
//     Frame::Ptr frame_k_1, Frame::Ptr frame_k, const gtsam::Pose3& T_world_k,
//     ObjectId object_id) {
//   PointCloudCorrespondences dynamic_correspondences;
//   // get the corresponding feature pairs
//   bool corr_result = frame_k->getDynamicCorrespondences(
//       dynamic_correspondences, *frame_k_1, object_id,
//       frame_k->landmarkWorldPointCloudCorrespondance());

//   const size_t& n_matches = dynamic_correspondences.size();

//   Pose3SolverResult result;
//   result =
//       EgoMotionSolver::geometricOutlierRejection3d3d(dynamic_correspondences);

//   Motion3SolverResult motion_result;
//   motion_result.status = result.status;

//   if (result.status == TrackingStatus::VALID) {
//     const gtsam::Pose3 G_w = result.best_result.inverse();
//     const gtsam::Pose3 H_w = T_world_k * G_w;
//     result.best_result = H_w;
//   }

//   // if not valid, return motion result as is
//   return motion_result;
// }

// // TODO: dont actually need all these variables
// Pose3SolverResult ObjectMotionSovler::motionModelOutlierRejection3d2d(
//     const AbsolutePoseCorrespondences& dynamic_correspondences,
//     Frame::Ptr frame_k_1, Frame::Ptr frame_k, const gtsam::Pose3& T_world_k,
//     ObjectId object_id) {
//   Pose3SolverResult result;

//   // get object motions in previous frame (so k-2 to k-1)
//   const MotionEstimateMap& motion_estiamtes_k_1 =
//   frame_k_1->motion_estimates_; if (!motion_estiamtes_k_1.exists(object_id))
//   {
//     result.status = TrackingStatus::INVALID;
//     return result;
//   }

//   // previous motion model: k-2 to k-1 in w
//   const gtsam::Pose3 motion_model = motion_estiamtes_k_1.at(object_id);

//   using Calibration = Camera::CalibrationType;
//   const auto calibration =
//       camera_params_.constructGtsamCalibration<Calibration>();

//   auto I = gtsam::traits<gtsam::Pose3>::Identity();
//   gtsam::PinholeCamera<Calibration> camera(I, calibration);

//   const double reprojection_error = params_.ransac_threshold_pnp;

//   TrackletIds tracklets;
//   TrackletIds inlier_tracklets;

//   utils::Accumulatord total_repr_error, inlier_repr_error;

//   const size_t& n_matches = dynamic_correspondences.size();
//   for (size_t i = 0u; i < n_matches; i++) {
//     const AbsolutePoseCorrespondence& corres = dynamic_correspondences.at(i);
//     tracklets.push_back(corres.tracklet_id_);

//     const Keypoint& kp_k = corres.cur_;
//     // the landmark int the world frame at k-1
//     const Landmark& w_lmk_k_1 = corres.ref_;

//     // using the motion, put the lmk in the world frame at k
//     const Landmark w_lmk_k = motion_model * w_lmk_k_1;
//     // using camera pose, put the lmk in the camera frame at k
//     const Landmark c_lmk_c = T_world_k.inverse() * w_lmk_k;

//     try {
//       double repr_error = camera.reprojectionError(c_lmk_c,
//       kp_k).squaredNorm();

//       total_repr_error.Add(repr_error);
//       if (repr_error < reprojection_error) {
//         inlier_tracklets.push_back(corres.tracklet_id_);
//         inlier_repr_error.Add(repr_error);
//       }
//     } catch (const gtsam::CheiralityException&) {
//     }
//   }

//   TrackletIds outliers;
//   determineOutlierIds(inlier_tracklets, tracklets, outliers);

//   VLOG(20) << "(Object) motion model inliers/total(error) "
//            << inlier_tracklets.size() << "(" << inlier_repr_error.Mean()
//            << ") / " << tracklets.size() << "(" << total_repr_error.Mean()
//            << ")";

//   result.best_result = motion_model;
//   result.inliers = inlier_tracklets;
//   result.outliers = outliers;
//   result.status = TrackingStatus::VALID;
//   return result;
// }

ObjectMotionSolverSAM::ObjectMotionSolverSAM(
    const ObjectMotionSovlerF2F::Params& geometric_motion_sovler_params,
    const CameraParams& camera_params, const gtsam::ISAM2Params& isam2_params)
    : isam2_params_(isam2_params),
      output_style_(MotionRepresentationStyle::F2F) {
  ObjectMotionSovlerF2F::Params motion_params = geometric_motion_sovler_params;
  // ensure we dont use the 3D motion solver,
  motion_params.refine_motion_with_3d = false;

  geometric_solver_ =
      std::make_shared<ObjectMotionSovlerF2F>(motion_params, camera_params);
}

ObjectMotionSolverSAM::Result ObjectMotionSolverSAM::solve(
    Frame::Ptr frame_k, Frame::Ptr frame_k_1) {
  // MotionEstimateMap motion_map;
  auto solve_impl =
      [&](const std::pair<ObjectId, DynamicObjectObservation>& pair) -> bool {
    const auto object_id = pair.first;
    VLOG(5) << "Solving Object Motion SAM for j=" << object_id;

    // do geometric solve
    const Motion3SolverResult geometric_sovler_result =
        initialPnPSolve(frame_k, frame_k_1, object_id);

    if (geometric_sovler_result.status == TrackingStatus::VALID) {
      frame_k->dynamic_features_.markOutliers(geometric_sovler_result.outliers);
      frame_k_1->dynamic_features_.markOutliers(
          geometric_sovler_result.outliers);

      const Motion3ReferenceFrame& motion_reference =
          geometric_sovler_result.best_result;
      CHECK_EQ(motion_reference.style(), MotionRepresentationStyle::F2F);

      CHECK_GE(geometric_sovler_result.inliers.size(), 5);

      DecoupledObjectSAM::Ptr estimator = getEstimator(object_id);
      CHECK_NOTNULL(estimator);
      auto map = estimator->map();
      // lambda function -> we want to avoid adding measurements twice
      // and ideally this function will run incrementally (without dropping
      // frames) so ideally we should add the k-1 measurements once (the first
      // time this function is called) and only at measurements at k after that
      // however, in reality, we DO drop measurements, so we should also add k-1
      // when we've never added them before
      auto should_add_previous_measurements =
          [&frame_k_1, &map](ObjectId object_id) -> bool {
        // object does not exist yet, therefore this is the first time we've
        // seen the object
        const auto& object_node = map->getObject(object_id);
        if (!object_node) return true;

        // likwise if the frame node does not exist yet, add measurements to add
        // it!!
        const auto& frame_k_1_node = map->getFrame(frame_k_1->getFrameId());
        if (!frame_k_1_node) return true;

        // if object has been observed, then measurements should already be
        // added if not, add new measurements at this frame!
        return !frame_k_1_node->objectObserved(object_id);
      };

      if (should_add_previous_measurements(object_id)) {
        GenericTrackedStatusVector<LandmarkKeypointStatus> measurements_k_1 =
            createMeasurementVector(frame_k_1, object_id);

        map->updateObservations(measurements_k_1);
        // No noise model!!
        map->updateSensorPoseMeasurement(
            frame_k_1->getFrameId(), Pose3Measurement(frame_k_1->getPose()));

        LOG(INFO) << "Adding previous measurerments for "
                  << info_string(frame_k_1->getFrameId(), object_id);

      } else {
        const auto& object_node = map->getObject(object_id);
        CHECK_NOTNULL(object_node);
        FrameId last_updated = object_node->getLastSeenFrame();

        // TODO: should also check that first seen frame is L?
        //  check we're updating consequative frames
        //  TODO: what about the case we drop frames?
        CHECK_GT(frame_k->getFrameId(), last_updated)
            << "Consistent frame failure for j=" << object_id;
      }

      GenericTrackedStatusVector<LandmarkKeypointStatus> measurements =
          createMeasurementVector(frame_k, object_id);
      estimator->update(frame_k->getFrameId(), measurements, frame_k->getPose(),
                        geometric_sovler_result.best_result);
      return true;

    } else {
      // Should be formulations job to handle and object tracking that is not
      // perfectly frame-to-frame just need to make sure this points still go to
      // the backend ;)
      return false;
    }
  };

  {
    dyno::utils::TimingStatsCollector timer("decoupled_object_sam.optimize");
    tbb::parallel_for_each(frame_k->object_observations_.begin(),
                           frame_k->object_observations_.end(), solve_impl);
  }

  return mergeObjectMaps();
}

ObjectMotionSolverSAM::Result ObjectMotionSolverSAM::mergeObjectMaps() const {
  ObjectPoseMap all_object_poses;
  ObjectMotionMap all_object_motions;

  std::function<ObjectMotionMap(DecoupledObjectSAM::Ptr)>
      get_object_motions_func = getObjectMotionMapFunc();

  for (const auto& [object_id, estimator] : sam_estimators_) {
    const ObjectPoseMap per_object_poses = estimator->getObjectPoses();
    const ObjectMotionMap per_object_motions =
        get_object_motions_func(estimator);
    // should only have one object in it
    CHECK_EQ(per_object_poses.size(), 1u);
    CHECK(per_object_poses.exists(object_id));

    CHECK_EQ(per_object_motions.size(), 1u);
    CHECK(per_object_motions.exists(object_id));

    all_object_poses += per_object_poses;
    all_object_motions += per_object_motions;
  }
  return {all_object_motions, all_object_poses};
}

// TODO: really horrible I have to remake this data-structure again and
// again.... this relies on only gettting the inliers in both frames
GenericTrackedStatusVector<LandmarkKeypointStatus>
ObjectMotionSolverSAM::createMeasurementVector(Frame::Ptr frame,
                                               ObjectId object_id) const {
  auto dynamic_features_iterator = FeatureFilterIterator(
      const_cast<FeatureContainer&>(frame->dynamic_features_),
      [object_id](const Feature::Ptr& f) -> bool {
        return Feature::IsUsable(f) && f->objectId() == object_id;
      });

  GenericTrackedStatusVector<LandmarkKeypointStatus> collection;
  for (const Feature::Ptr& f : dynamic_features_iterator) {
    const Keypoint& kp = f->keypoint();
    const TrackletId tracklet_id = f->trackletId();
    const Landmark lmk = frame->backProjectToCamera(tracklet_id);
    const FrameId frame_id = f->frameId();
    CHECK_EQ(frame_id, frame->getFrameId());

    LandmarkKeypointStatus status(LandmarkKeypoint(lmk, kp), frame_id,
                                  tracklet_id, object_id,
                                  ReferenceFrame::LOCAL);
    collection.push_back(status);
  }

  return collection;
}

Motion3SolverResult ObjectMotionSolverSAM::initialPnPSolve(Frame::Ptr frame_k,
                                                           Frame::Ptr frame_k_1,
                                                           ObjectId object_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  return geometric_solver_->geometricOutlierRejection3d2d(
      frame_k_1, frame_k, frame_k->getPose(), object_id);
}

DecoupledObjectSAM::Ptr ObjectMotionSolverSAM::getEstimator(
    ObjectId object_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  // make new estimator if needed
  if (!sam_estimators_.exists(object_id)) {
    LOG(INFO) << "Making new DecoupledObjectSAM for object " << object_id;

    FormulationHooks hooks;
    hooks.ground_truth_packets_request =
        geometric_solver_->objectMotionParams().ground_truth_packets_request;

    BackendParams backend_params;  // TODO: for now
    DecoupledObjectSAM::Params params;
    params.isam = isam2_params_;
    auto noise_models = NoiseModels::fromBackendParams(backend_params);
    // make this prior not SO small
    noise_models.initial_pose_prior =
        gtsam::noiseModel::Isotropic::Sigma(6u, 0.001);
    sam_estimators_.insert2(
        object_id, std::make_shared<DecoupledObjectSAM>(params, object_id,
                                                        noise_models, hooks));
  }

  return sam_estimators_.at(object_id);
}

std::function<ObjectMotionMap(DecoupledObjectSAM::Ptr)>
ObjectMotionSolverSAM::getObjectMotionMapFunc() const {
  std::function<ObjectMotionMap(DecoupledObjectSAM::Ptr)> func;
  if (output_style_ == MotionRepresentationStyle::F2F) {
    func = [](DecoupledObjectSAM::Ptr agent) -> ObjectMotionMap {
      return agent->getFrame2FrameMotions();
    };
  } else {
    func = [](DecoupledObjectSAM::Ptr agent) -> ObjectMotionMap {
      return agent->getKeyFramedMotions();
    };
  }
  return func;
}

}  // namespace dyno
