#include "dynosam2/frontend/Frontend.hpp"

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>

#include "dynosam2/frontend/GeometryUpdater.hpp"
#include "dynosam_common/utils/TimingStats.hpp"

namespace dyno2 {

Frontend::Frontend(const FrontendParameters& parameters,
                   dyno::Camera::Ptr camera)
    : parameters_(parameters),
      camera_(camera),
      feature_tracker_(parameters, camera) {}

void Frontend::detectAndTrack(FeatureBlockContainer*& features_out,
                              dyno::ObjectDetectionResult& object_detection_out,
                              ImageContainer& container_in_out) {
  auto tracking_result = feature_tracker_.track(container_in_out);
  features_out = tracking_result.featues;
  CHECK_NOTNULL(features_out);
  object_detection_out = tracking_result.object_detection;
}

void Frontend::solve(SlamBackend& estimator, FeatureBlockContainer& features,
                     const ImageContainer& container,
                     UpdatedStates& updated_states, ObjectIds& solved_objects) {
  // build intermediate geometry
  FeatureBlockGeometryMap feature_block_geometries;
  GeometryUpdater(camera_, container, features)
      .calcPoints(feature_block_geometries);
}

bool Frontend::solve3d2dRansac(gtsam::Pose3& pose, std::vector<bool>& inliers,
                               FrameAbsoluteAdaptor& adaptor,
                               double ransac_threshold,
                               int max_iterations) const {
  dyno::utils::ChronoTimingStats pnp_t("frontend.pnp");
  const size_t num_matches = adaptor.getNumberCorrespondences();
  if (num_matches < 5) {
    VLOG(10) << "3D2D RANSAC failed: not enough matches";
    return false;
  }

  using AbsolutePoseProblem =
      opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem;
  auto problem = std::make_shared<AbsolutePoseProblem>(
      adaptor, AbsolutePoseProblem::KNEIP);

  opengv::sac::Ransac<AbsolutePoseProblem> ransac;
  ransac.sac_model_ = problem;
  ransac.threshold_ = ransac_threshold;
  ransac.max_iterations_ = max_iterations;

  // run the ransac
  ransac.computeModel(0);

  int ransac_inliers = int(ransac.inliers_.size());
  float ransac_ratio = float(ransac_inliers) / float(num_matches);
  const bool ransac_success = ransac_inliers > 10 && ransac_ratio > 0.6;

  if (!ransac_success) {
    VLOG(10) << "3D2D RANSAC failed: not enough inliers (" << ransac_inliers
             << ") or ransac ratio (" << ransac_ratio << ") failed!";
    return false;
  }

  pose = utils::openGvTfToGtsamPose3(ransac.model_coefficients_);
  std::vector<bool> inliers_v(num_matches, false);
  for (size_t k = 0; k < (size_t)ransac_inliers; k++) {
    inliers_v.at(size_t(ransac.inliers_.at(k))) = true;
  }
  inliers = std::move(inliers_v);
  return true;
}

}  // namespace dyno2
