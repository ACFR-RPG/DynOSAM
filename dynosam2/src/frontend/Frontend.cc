#include "dynosam2/frontend/Frontend.hpp"

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>

#include "dynosam2/frontend/FlowRefinement.hpp"
#include "dynosam2/frontend/GeometryUpdater.hpp"
#include "dynosam2/frontend/MatchingAdaptor.hpp"
#include "dynosam_common/utils/TimingStats.hpp"

namespace dyno2 {

static double computeOpenGvRansacThreshold(const double repr_error,
                                           dyno::Camera::Ptr camera) {
  const auto& camera_params = camera->getParams();
  const double fx = camera_params.fx();
  const double fy = camera_params.fy();

  const double avg_focal_length = 0.5 * static_cast<double>(fx + fy);
  double ransac_threshold_3d2d =
      1.0 - std::cos(std::atan(std::sqrt(2.0) * repr_error / avg_focal_length));
  return ransac_threshold_3d2d;
}

Frontend::Frontend(const FrontendParameters& parameters,
                   dyno::Camera::Ptr camera)
    : parameters_(parameters),
      camera_(camera),
      feature_tracker_(parameters, camera),
      vio_ransac_threshold_3d_2d_(computeOpenGvRansacThreshold(2.0, camera)),
      do_ransac_threshold_3d_2d_(computeOpenGvRansacThreshold(2.0, camera)) {}

void Frontend::detectAndTrack(FeatureBlockContainer*& features_out,
                              dyno::ObjectDetectionResult& object_detection_out,
                              ImageContainer& container_in_out) {
  auto tracking_result = feature_tracker_.track(container_in_out);
  features_out = tracking_result.featues;
  CHECK_NOTNULL(features_out);
  object_detection_out = tracking_result.object_detection;
}

// void Frontend::solve(SlamBackend& estimator, FeatureBlockContainer& features,
//                      const ImageContainer& container,
//                      UpdatedStates& updated_states, ObjectIds&
//                      solved_objects) {
//   // build intermediate geometry
//   FeatureBlockGeometryMap feature_block_geometries;
//   GeometryUpdater(camera_, container, features)
//       .calcPoints(feature_block_geometries);
// }

void Frontend::solveVisualOdometry(VioGraphEstimator& estimator,
                                   FeatureBlockContainer& features,
                                   FeatureBlockGeometry& geometry,
                                   ViFrame& frame, AyncVO& async_context,
                                   std::vector<FrameId>& frames_updated,
                                   bool& success, const ViFrames& frames,
                                   const ImageContainer& image_container) {
  const FrameId frame_id = image_container.frameId();
  const Timestamp timestamp = image_container.timestamp();

  // previous frame as the refernece fmrae
  // the estimator does not yet have a solution for the current frame
  // and therefore we need to initalise form the previous frame
  const FrameId previous_frame_id = estimator.currentFrameId();
  const ViFrame& vi_frame_previous = frames.at(previous_frame_id);

  // reference frame id
  // for now we expect this to be the k-1
  CHECK_EQ(previous_frame_id, frame_id - 1u);
  CHECK(estimator.poseExists(previous_frame_id));

  LandmarkMap previous_landmarks_C;
  // build local geometry from previous frame and use with PnP initially
  // this just seems WAY more stabls
  // observations in previous frame and in current

  const gtsam::Pose3& X_W_km1 = estimator.getPose(previous_frame_id);
  const gtsam::Pose3 X_W_km1_inv = X_W_km1.inverse();
  for (size_t i = 0; i < vi_frame_previous.ids.size(); i++) {
    // all previous landmarks or just ones in map?
    previous_landmarks_C.setLandmark(vi_frame_previous.ids[i],
                                     vi_frame_previous.lmks_C[i]);
  }

  FrameAbsoluteAdaptor adapter(camera_, geometry, previous_landmarks_C,
                               features);
  size_t num_matches = adapter.getNumberCorrespondences();
  LOG(INFO) << "Solving VO RANSAC with n matches:" << num_matches;

  // solve relative motion
  CHECK(async_context.vo);
  // relative motion from previous frame (j) to current frame (i)
  gtsam::Pose3 T_ij;
  std::vector<bool> inliers;
  dyno::utils::ChronoTimingStats ransac_t("frontend.vo.ransac");
  async_context.vo_valid =
      solve3d2dRansac(T_ij, inliers, adapter, vio_ransac_threshold_3d_2d_, 100);
  ransac_t.stop();

  if (!async_context.vo_valid) {
    {
      std::lock_guard<std::mutex> lock(async_context.vo_mutex);
      // alert awaiting threads that the solve was complete even if it was
      // invalid
      async_context.vo_ready = true;
    }

    success = false;
    // alert awaiting threads
    async_context.vo_cv.notify_all();

    LOG(WARNING) << "3D2D RANSAC failed for VO!";
    // TODO: still fill VI frame?
    return;
  }

  if (!vi_tracking_condition_.is_initalized) {
    vi_tracking_condition_.is_initalized = true;
    LOG(INFO) << "Initialized VI Map!";
  }

  CHECK_EQ(inliers.size(), num_matches);
  for (size_t k = 0; k < num_matches; k++) {
    if (!inliers[k]) {
      dyno::TrackletId i = adapter.trackletId(k);
      Index fc_index = geometry.getFeatureContainerIndex(i);
      CHECK_EQ(features.ids[fc_index], i);
      features.inlier[fc_index] = 0;
    }
  }

  // refine with flow
  FlowRefinement(camera_, image_container, adapter).refine(T_ij, T_ij);
  // update geometry
  GeometryUpdater(camera_, image_container, features).updateGeometry(geometry);

  gtsam::Pose3 X_W = X_W_km1 * T_ij;
  // NOTE: this is not the best available value!
  LOG(INFO) << "Solved VO k= " << frame_id;
  {
    std::lock_guard<std::mutex> lock(async_context.vo_mutex);
    *async_context.vo = X_W;
    async_context.vo_ready = true;
    CHECK(async_context.vo_valid);
  }
  // alert awaiting threads
  async_context.vo_cv.notify_all();

  frame.frame_id = frame_id;
  frame.timestamp = timestamp;
  frame.fill(geometry, features);

  // initalise estimator with new state (eventually maybe from VIO?)
  // we now have an initial pose
  gtsam::Vector3 sigmas;
  sigmas << 2, 2, 2;
  StereoMeasurementStatusVector stereo_measurements;
  stereo_measurements.reserve(adapter.getNumberCorrespondences());
  // add measurements for keypoints that were tracked and already initalised
  // this will help with the BA to get intermediate frame measurements
  for (size_t idx = 0; idx < adapter.getNumberCorrespondences(); idx++) {
    TrackletId id = adapter.trackletId(idx);
    if (adapter.isInlier(idx) && estimator.landmarkExists(id)) {
      auto stereo_measurement =
          StereoMeasurement::FromSigmas(frame.stereoPointById(id), sigmas);

      stereo_measurements.push_back(
          StereoMeasurementStatus(stereo_measurement, frame_id, timestamp, id,
                                  0, ReferenceFrame::LOCAL));
    }
  }

  estimator.initaliseStates(frame_id, stereo_measurements, X_W);
  frames_updated.push_back(frame_id);

  success = true;
}

void Frontend::solveObjectOdometry(
    ObjectId object_id, const VioGraphEstimator& vo_estimator,
    const ImageContainer& image_container, const DoFrames& frames,
    DoGraphEstimator& estimator, FeatureBlockContainer& features,
    FeatureBlockGeometry& geometry, DoFrame& frame, AyncVO& async_context,
    std::vector<FrameId>& frames_updated, uchar& success) {
  const FrameId frame_id = image_container.frameId();
  const Timestamp timestamp = image_container.timestamp();
  const FrameId previous_frame_id = frame_id - 1u;

  LandmarkMap previous_landmarks_C;

  // for now reeject if not in previous frame
  if (!frames.exists(previous_frame_id)) {
    // still have to fill frame
    frame.fill(geometry, features);
    success = false;
    return;
  }

  CHECK(frames.exists(previous_frame_id));
  // observations in previous frame and in current
  const DoFrame& ref_object_frame = frames.at(previous_frame_id);
  for (size_t i = 0; i < ref_object_frame.ids.size(); i++) {
    // all previous landmarks or just ones in map?
    previous_landmarks_C.setLandmark(ref_object_frame.ids[i],
                                     ref_object_frame.lmks_C[i]);
  }

  FrameAbsoluteAdaptor adapter(camera_, geometry, previous_landmarks_C,
                               features);
  size_t num_matches = adapter.getNumberCorrespondences();
  LOG(INFO) << "Solving doo j= " << object_id << " # matches= " << num_matches;

  gtsam::Pose3 G_i_ij_inv;
  std::vector<bool> inliers;
  bool ransac_success = solve3d2dRansac(G_i_ij_inv, inliers, adapter,
                                        do_ransac_threshold_3d_2d_, 100);

  if (!ransac_success) {
    // TODO: still fill DoFrame
    LOG(WARNING) << "3D2D RANSAC failed for DOO j=" << object_id;
    frame.fill(geometry, features);
    success = 0;
    return;
  }

  gtsam::Point3Vector reference_lmks;
  reference_lmks.reserve(num_matches);
  CHECK_EQ(inliers.size(), num_matches);
  for (size_t k = 0; k < num_matches; k++) {
    if (!inliers[k]) {
      TrackletId i = adapter.trackletId(k);
      Index fc_index = geometry.getFeatureContainerIndex(i);
      CHECK_EQ(features.ids[fc_index], i);
      features.inlier[fc_index] = 0;
    } else {
      reference_lmks.push_back(adapter.getPoint(k));
    }
  }

  // refine with flow
  FlowRefinement(camera_, image_container, adapter)
      .refine(G_i_ij_inv, G_i_ij_inv);
  // update geometry
  GeometryUpdater(camera_, image_container, features).updateGeometry(geometry);

  // wait until camera pose is ready!
  {
    std::unique_lock<std::mutex> lock(async_context.vo_mutex);
    async_context.vo_cv.wait(lock, [&] { return async_context.vo_ready; });
  }

  if (!async_context.vo_valid) {
    LOG(WARNING) << "Unable to process DOO j=" << object_id
                 << " as visual odometry failed!";
    success = 0;
    return;
  }

  const dyno::FrameId frame_id_j = image_container.frameId();
  const dyno::Timestamp timestamp_j = image_container.timestamp();

  // previous frame and same frame as the reference geometry (hopefully!)
  const dyno::FrameId frame_id_i = previous_frame_id;
  const dyno::Timestamp timestamp_i = ref_object_frame.timestamp;

  // in this place we replace k with i, where k is the current frame
  const gtsam::Pose3 X_W_i = vo_estimator.getPose(previous_frame_id);
  const gtsam::Pose3 X_W_j = *async_context.vo;
  // reltive motion of object in the camera frame i, from i to j
  // without accounting for the additional camera motion between i and j
  dyno::Motion3ReferenceFrame G_i_ij(
      G_i_ij_inv.inverse(), dyno::MotionRepresentationStyle::F2F,
      dyno::ReferenceFrame::GLOBAL, frame_id_i, frame_id_j);

  // TODO: add measurements for the tracked ones!
  frame.fill(geometry, features);

  if (estimator.numStates() > 0) {
    LOG(INFO) << "j=" << object_id << " solved k=" << frame_id_j
              << " last state k=" << estimator.currentFrameId();
  }

  if (!estimator.stateExists(frame_id_i)) {
    gtsam::Point3 t_W_i_centroid =
        X_W_i * dyno::computeCentroid(reference_lmks);
    gtsam::Pose3 L_W_i_centroid(gtsam::Rot3::Identity(), t_W_i_centroid);
    estimator.initalizeNewAnchorFrame(
        frame_id_i, dyno::StereoMeasurementStatusVector{}, L_W_i_centroid);

    frames_updated.push_back(frame_id_i);
  }
  estimator.initalizeStatesPropogate(
      frame_id_j, dyno::StereoMeasurementStatusVector{}, G_i_ij, X_W_j, X_W_i);

  frames_updated.push_back(frame_id_j);
  success = true;
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
