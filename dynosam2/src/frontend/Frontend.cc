#include "dynosam2/frontend/Frontend.hpp"

#include <gtsam/geometry/CameraSet.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>

#include "dynosam2/frontend/FlowRefinement.hpp"
#include "dynosam2/frontend/GeometryUpdater.hpp"
#include "dynosam2/frontend/MatchingAdaptor.hpp"
#include "dynosam_common/utils/TimingStats.hpp"
#include "dynosam_core/PointCloudProcess.hpp"
#include "dynosam_cv/CvUtils.hpp"

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
    : camera_(camera),
      feature_tracker_(parameters, camera),
      vio_ransac_threshold_3d_2d_(computeOpenGvRansacThreshold(2.0, camera)),
      do_ransac_threshold_3d_2d_(computeOpenGvRansacThreshold(2.0, camera)) {}

void Frontend::detectAndTrack(FeatureBlockContainer*& features_out,
                              dyno::ObjectDetectionResult& object_detection_out,
                              dyno::ImageContainer::Ptr container_in_out) {
  auto tracking_result = feature_tracker_.track(container_in_out);
  features_out = tracking_result.featues;
  CHECK_NOTNULL(features_out);
  object_detection_out = tracking_result.object_detection;
}

void Frontend::solveVisualOdometry(
    VioGraphEstimator& estimator, FeatureBlockContainer& features,
    FeatureBlockGeometry& geometry, AyncVO& async_context,
    std::vector<FrameId>& frames_updated,
    std::vector<KeypointIdentifier>& new_measurements, bool& success,
    const ImageContainer& image_container) {
  dyno::utils::ChronoTimingStats vo_t("frontend.vo");

  const FrameId frame_id = image_container.frameId();
  const Timestamp timestamp = image_container.timestamp();

  // previous frame as the refernece fmrae
  // the estimator does not yet have a solution for the current frame
  // and therefore we need to initalise form the previous frame
  const FrameId previous_frame_id = estimator.currentFrameId();
  // const ViFrame& vi_frame_previous = frames.at(previous_frame_id);
  const ViFrame::Ptr vi_frame_previous_ptr =
      estimator.getFramePtr(previous_frame_id);
  CHECK_NOTNULL(vi_frame_previous_ptr);
  const ViFrame& vi_frame_previous = *vi_frame_previous_ptr;

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
  gtsam::Pose3 T_ij_pnp;
  std::vector<bool> inliers;
  dyno::utils::ChronoTimingStats ransac_t("frontend.vo.ransac");
  async_context.vo_valid = solve3d2dRansac(T_ij_pnp, inliers, adapter,
                                           vio_ransac_threshold_3d_2d_, 100);
  ransac_t.stop();

  if (!async_context.vo_valid) {
    {
      std::lock_guard<std::mutex> lock(async_context.vo_mutex);
      // alert awaiting threads that the solve was complete even if it was
      // invalid
      async_context.vo_ready = true;
    }

    ViFrame::Ptr vi_frame = std::make_shared<ViFrame>(frame_id, timestamp);
    vi_frame->fill(geometry, features);

    // propogate use constant velocity model
    // or IMU!
    const VioGraphState& vio_state_previous =
        estimator.getVioState(previous_frame_id);
    const dyno::Timestamp dt = timestamp - vio_state_previous.timestamp;
    const gtsam::Pose3& previous_pose_W = vio_state_previous.pose_W;
    gtsam::Pose3 predicted_pose = predictPoseConstantVelocity(
        previous_pose_W, vio_state_previous.velocity_W, dt);

    vi_frame->setRelativePose(Motion3ReferenceFrame(
        previous_pose_W.inverse() * predicted_pose, ReferenceFrame::LOCAL,
        vio_state_previous.frame_id, vi_frame->getFrameId()));

    estimator.initaliseStates(vi_frame, predicted_pose,
                              vio_state_previous.velocity_W);

    success = false;
    // alert awaiting threads
    async_context.vo_cv.notify_all();

    LOG(WARNING) << "3D2D RANSAC failed for VO! k=" << frame_id;
    return;
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

  dyno::Pose3Measurement T_ij;
  FlowRefinement(camera_, image_container, adapter).refine(T_ij_pnp, T_ij);
  // update geometry
  GeometryUpdater(camera_, image_container, features).updateGeometry(geometry);

  gtsam::Pose3 X_W = X_W_km1 * T_ij.measurement();
  ViFrame::Ptr vi_frame = std::make_shared<ViFrame>(frame_id, timestamp);
  vi_frame->fill(geometry, features);
  vi_frame->setRelativePose(
      Motion3ReferenceFrame(T_ij.measurement(), ReferenceFrame::LOCAL,
                            previous_frame_id, vi_frame->getFrameId()));

  estimator.initaliseStates(vi_frame, X_W);
  // initalise estimator with new state (eventually maybe from VIO?)
  // we now have an initial pose
  std::vector<dyno::TrackletId> observation_ids;
  observation_ids.reserve(adapter.getNumberCorrespondences());
  new_measurements.reserve(adapter.getNumberCorrespondences());
  // add measurements for keypoints that were tracked and already initalised
  // this will help with the BA to get intermediate frame measurements
  for (size_t idx = 0; idx < adapter.getNumberCorrespondences(); idx++) {
    TrackletId id = adapter.trackletId(idx);
    if (adapter.isInlier(idx) && estimator.landmarkExists(id)) {
      observation_ids.push_back(id);
      new_measurements.emplace_back(frame_id, id, dyno::background_label);
    }
  }
  // all measurements are for the same frame use this function call which runs a
  // bit faster!
  estimator.addMeasurements(vi_frame, observation_ids);

  std::set<dyno::FrameId> frames_updated_set;
  frames_updated_set.insert(frame_id);

  if (!vi_tracking_condition_.is_initalized) {
    // as this is the first initalised frame, now also add the previous
    // measurements
    estimator.addMeasurements(vi_frame_previous_ptr, observation_ids);
    vi_tracking_condition_.is_initalized = true;
    LOG(INFO) << "Initialized VI Map!";
    frames_updated_set.insert(vi_frame_previous_ptr->getFrameId());
  }

  // TODO: if sensor_rig has IMU

  // alert waiting threads AFTER we initalise states as they may need the pose!
  LOG(INFO) << "Solved VO k= " << frame_id;
  {
    std::lock_guard<std::mutex> lock(async_context.vo_mutex);
    *async_context.vo = X_W;
    async_context.vo_ready = true;
    CHECK(async_context.vo_valid);
  }
  // alert awaiting threads
  async_context.vo_cv.notify_all();

  const bool should_be_keyframe = shouldBeViKeyFrame(estimator, vi_frame);
  // TODO: must bound keyframes (only for estimator and covisibility checking)
  //  somehow this must be different from full SLAM and trajectory output!
  //  full SLAM will use Keyframe MAP which represents the full SLAM problem!
  //  this is just maybe the local problem
  if (should_be_keyframe) {
    // initlaise new points in the map by attempting to triangulate points
    // at all active keyframes
    const size_t init_triangulation =
        matchToVioMap(estimator, vi_frame, new_measurements);
    // LOG(INFO) << "Init triangulation: " << init_triangulation;
    LOG(INFO) << "Made VIO keyframe k=" << frame_id;
    estimator.setKeyframe(frame_id, true);
  }

  if (estimator.keyFrames().size() > 2) {
    utils::ChronoTimingStats timer_t("frontend.vo.refine_map");
    std::vector<FrameId> frames_affected;
    if (should_be_keyframe) {
      // optimize with structure
      estimator.optimize(frame_id, frames_affected, false);
    } else {
      // optimize only states
      estimator.optimize(frame_id, frames_affected, true);
    }

    for (FrameId frame_id : frames_affected) {
      frames_updated_set.insert(frame_id);
    }
  }

  frames_updated = std::vector<dyno::FrameId>(frames_updated_set.begin(),
                                              frames_updated_set.end());

  success = true;
}

void Frontend::solveObjectOdometry(
    ObjectId object_id, const VioGraphEstimator& vo_estimator,
    const EstimationParameters& params, const ImageContainer& image_container,
    DoGraphEstimator& estimator, FeatureBlockContainer& features,
    FeatureBlockGeometry& geometry, AyncVO& async_context,
    std::vector<FrameId>& frames_updated, uchar& success) {
  const FrameId frame_id = image_container.frameId();
  const Timestamp timestamp = image_container.timestamp();
  const FrameId previous_frame_id = frame_id - 1u;

  DoFrame::Ptr do_frame_i = estimator.getFramePtr(previous_frame_id);
  // for now reeject if not in previous frame
  if (!do_frame_i) {
    DoFrame::Ptr frame =
        std::make_shared<DoFrame>(frame_id, timestamp, object_id);
    frame->fill(geometry, features);
    estimator.initaliseFrame(frame);
    success = false;
    return;
  }

  LandmarkMap previous_landmarks_C;
  previous_landmarks_C.reserve(do_frame_i->numKeypoints());
  for (size_t i = 0; i < do_frame_i->numKeypoints(); i++) {
    // all previous landmarks or just ones in map?
    previous_landmarks_C.setLandmark(do_frame_i->ids[i], do_frame_i->lmks_C[i]);
  }

  FrameAbsoluteAdaptor adapter(camera_, geometry, previous_landmarks_C,
                               features);
  size_t num_matches = adapter.getNumberCorrespondences();
  LOG(INFO) << "Solving doo j= " << object_id << " # matches= " << num_matches;

  gtsam::Pose3 G_i_ij_inv_pnp;
  std::vector<bool> inliers;
  bool ransac_success = solve3d2dRansac(G_i_ij_inv_pnp, inliers, adapter,
                                        do_ransac_threshold_3d_2d_, 100);

  if (!ransac_success) {
    LOG(WARNING) << "3D2D RANSAC failed for DOO j=" << object_id;
    DoFrame::Ptr frame =
        std::make_shared<DoFrame>(frame_id, timestamp, object_id);
    // still have to fill frame
    frame->fill(geometry, features);
    estimator.initaliseFrame(frame);
    success = 0;
    return;
  }

  CHECK_EQ(inliers.size(), num_matches);
  for (size_t k = 0; k < num_matches; k++) {
    TrackletId i = adapter.trackletId(k);
    if (!inliers[k]) {
      Index fc_index = geometry.getFeatureContainerIndex(i);
      CHECK_EQ(features.ids[fc_index], i);
      features.inlier[fc_index] = 0;
    }
  }

  // refine with flow
  dyno::Pose3Measurement G_i_ij_inv;
  FlowRefinement(camera_, image_container, adapter)
      .refine(G_i_ij_inv_pnp, G_i_ij_inv);
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
  const dyno::FrameId frame_id_i = do_frame_i->getFrameId();

  // in this place we replace k with i, where k is the current frame
  const gtsam::Pose3 X_W_i = vo_estimator.getPose(previous_frame_id);
  const gtsam::Pose3 X_W_j = *async_context.vo;
  // reltive motion of object in the camera frame i, from i to j
  // without accounting for the additional camera motion between i and j
  const gtsam::Matrix66 Adj_G_i_ij_inv = G_i_ij_inv.measurement().AdjointMap();
  gtsam::Pose3 G_i_ij_pose = G_i_ij_inv.measurement().inverse();
  // run covariance through adjoint
  const gtsam::Matrix66 G_i_ij_cov =
      Adj_G_i_ij_inv * G_i_ij_inv.covariance() * Adj_G_i_ij_inv.transpose();

  dyno::Motion3ReferenceFrame G_i_ij(G_i_ij_pose, dyno::ReferenceFrame::GLOBAL,
                                     frame_id_i, frame_id_j);

  DoFrame::Ptr do_frame_j =
      std::make_shared<DoFrame>(frame_id_j, timestamp_j, object_id);
  do_frame_j->fill(geometry, features);

  std::unordered_set<dyno::FrameId> frames_updated_set;

  if (estimator.numStates() > 0) {
    LOG(INFO) << "j=" << object_id << " solved k=" << frame_id_j
              << " last state k=" << estimator.currentFrameId();
  }

  if (!estimator.stateExists(frame_id_i)) {
    TrackletIds measurements_i;
    gtsam::Point3Vector reference_lmks;
    TrackletIds reference_lmk_ids;

    const size_t num_matches = adapter.getNumberCorrespondences();
    measurements_i.reserve(num_matches);
    reference_lmks.reserve(num_matches);
    reference_lmk_ids.reserve(num_matches);
    for (size_t idx = 0; idx < num_matches; idx++) {
      TrackletId id = adapter.trackletId(idx);
      if (!adapter.isInlier(idx)) {
        continue;
      }
      if (do_frame_i->observedLandmark(id) && estimator.landmarkExists(id)) {
        measurements_i.push_back(id);
      }

      reference_lmks.push_back(adapter.getPoint(idx));
      reference_lmk_ids.push_back(id);
    }

    // We didnt get enough tracks to init this object
    if (reference_lmks.size() < 3) {
      // CANNOT INIT!
    }

    // construct the object reference frame from the centroid of the object
    // apply some basic statistical outlier rejection to avoid skewing the
    // centroid as much as possible!
    std::vector<uint8_t> point_status;
    dyno::statisticalOutlierRemoval(reference_lmks, point_status);

    gtsam::Point3Vector reference_lmks_inliers;
    reference_lmks_inliers.reserve(point_status.size());
    for (size_t i = 0; i < point_status.size(); i++) {
      if (point_status[i]) {
        reference_lmks_inliers.push_back(reference_lmks[i]);
      } else {
        // mark as outlier
        Index fc_index =
            geometry.getFeatureContainerIndex(reference_lmk_ids[i]);
        features.inlier[fc_index] = 0;
      }
    }

    gtsam::Point3 t_W_i_centroid =
        X_W_i * dyno::computeCentroid(reference_lmks_inliers);
    gtsam::Pose3 L_W_i_centroid(gtsam::Rot3::Identity(), t_W_i_centroid);

    LOG(INFO) << "obj j=" << object_id << " made Anchor KF " << frame_id_i;
    estimator.initalizeNewAnchorState(do_frame_i, L_W_i_centroid);
    estimator.addMeasurements(do_frame_i, measurements_i);
    frames_updated_set.insert(frame_id_i);
  }

  std::vector<dyno::TrackletId> measurements_j;
  measurements_j.reserve(adapter.getNumberCorrespondences());
  // add measurements for keypoints that were tracked and already initalised
  // this will help with the BA to get intermediate frame measurements
  for (size_t idx = 0; idx < adapter.getNumberCorrespondences(); idx++) {
    TrackletId id = adapter.trackletId(idx);
    if (adapter.isInlier(idx) && estimator.landmarkExists(id)) {
      measurements_j.push_back(id);
    }
  }

  estimator.initalizeStatesPropogate(do_frame_j, G_i_ij, X_W_j, X_W_i);
  estimator.addMeasurements(do_frame_j, measurements_j);
  frames_updated_set.insert(frame_id_j);

  const bool is_keyframe = shouldBeDoKeyFrame(estimator, do_frame_j);
  if (is_keyframe) {
    LOG(INFO) << "obj j=" << object_id << " made new Regular KF";
    // if keyframe for now just initalise keyframe in object frame

    // use points from reference frame as we will have had a chance to optimize
    // it
    estimator.setRegularKeyframe(frame_id_j);

    // TODO: before adding points do radial outlier rejection
    // TODO: then mark those ones as outliers
    //  and remove measurements
    // TODO: just do with PCL
    const gtsam::Pose3& L_W_i = estimator.getPose(frame_id_i);
    // transform a point in camera frame to a point in object frame
    const gtsam::Pose3 T_LC_i = L_W_i.inverse() * X_W_i;

    std::vector<dyno::TrackletId> measurements_j;
    std::vector<dyno::TrackletId> measurements_i;

    const size_t num_matches = adapter.getNumberCorrespondences();
    measurements_j.reserve(num_matches);
    measurements_i.reserve(num_matches);

    // add all measurements (that have not yet been added) to the BA
    // problem as this is a new keyframe!
    for (size_t idx = 0; idx < num_matches; idx++) {
      TrackletId id = adapter.trackletId(idx);
      if (!adapter.isInlier(idx)) {
        continue;
      }

      if (!estimator.landmarkExists(id)) {
        const gtsam::Point3& m_C = adapter.getPoint(idx);
        const gtsam::Point3& m_L = T_LC_i * m_C;
        estimator.setLandmark(id, m_L);
      }

      if (!estimator.hasMeasurement(frame_id_i, id)) {
        measurements_i.push_back(id);
      }

      if (!estimator.hasMeasurement(frame_id_j, id)) {
        measurements_j.push_back(id);
      }
    }

    estimator.addMeasurements(do_frame_i, measurements_i);
    estimator.addMeasurements(do_frame_j, measurements_j);
  }

  std::vector<dyno::FrameId> affected_in_opt;
  estimator.optimize(frame_id_j, vo_estimator, affected_in_opt,
                     params.num_dynamic_keyframes);

  frames_updated_set.insert(affected_in_opt.begin(), affected_in_opt.end());
  frames_updated = std::vector<dyno::FrameId>(frames_updated_set.begin(),
                                              frames_updated_set.end());
  success = true;
}

struct StereoMeasurementFiller {
  const gtsam::Pose3 left_Pose_right;
  const gtsam::Cal3_S2 mono_cal;

  using GtsamCamera = dyno::Camera::CameraImpl;

  StereoMeasurementFiller(const dyno::Camera::Ptr& camera)
      : left_Pose_right(dyno::RGBDCamera::StereoExtrinsics(camera)),
        mono_cal(dyno::Camera::Calibration(camera)) {}

  bool fill(dyno::TrackletId id, gtsam::CameraSet<GtsamCamera>& camera_set,
            gtsam::Point2Vector& measurements, const ViFrame::Ptr& frame_k,
            const gtsam::Pose3& X_W_k) const {
    if (!frame_k->observedLandmark(id)) {
      return false;
    }

    const gtsam::Pose3& left_pose = X_W_k;
    const GtsamCamera left_camera_i(left_pose, mono_cal);

    const gtsam::Pose3 rightPose = left_pose.compose(left_Pose_right);
    const GtsamCamera right_camera_i(rightPose, mono_cal);

    const gtsam::StereoPoint2& z = frame_k->stereoPointById(id);
    camera_set.push_back(left_camera_i);
    measurements.push_back(z.point2());

    if (!std::isnan(z.uR())) {  // if right point is valid
      camera_set.push_back(right_camera_i);
      measurements.push_back(z.right());
    }

    return true;
  }
};

size_t Frontend::matchToVioMap(
    VioGraphEstimator& estimator, ViFrame::Ptr frame,
    std::vector<KeypointIdentifier>& new_measurements) {
  const size_t num_threads = std::max(1u, std::thread::hardware_concurrency());
  const size_t num_tracklets = frame->numKeypoints();
  const size_t chunk_size = (num_tracklets + num_threads - 1) / num_threads;
  const dyno::FrameId frame_id = frame->getFrameId();

  using GtsamCamera = dyno::Camera::CameraImpl;
  gtsam::SharedNoiseModel model = gtsam::noiseModel::Isotropic::Sigma(2, 2.0);

  gtsam::TriangulationParameters triangulation_params;
  triangulation_params.noiseModel = model;
  // set maximum average reprojection threshold
  triangulation_params.dynamicOutlierRejectionThreshold = 2.0;

  // get active keyframes
  const std::set<dyno::FrameId> all_frames = estimator.keyFrames();
  std::vector<FrameId> query_frames_for_triangulation(all_frames.begin(),
                                                      all_frames.end());

  const StereoMeasurementFiller stereo_measurement_filler(camera_);

  dyno::utils::ChronoTimingStats triangulate_t("frontend.vo.triangulate");
  AlignedVector<MatchInfo> match_infos(num_tracklets);
  std::vector<dyno::FrameIds> matching_frames(num_tracklets);

  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, num_tracklets, chunk_size),
      [&](const tbb::blocked_range<size_t>& range) {
        const std::vector<TrackletId>& ids = frame->ids;
        for (size_t i = range.begin(); i != range.end(); ++i) {
          const dyno::TrackletId tracklet_id = ids[i];

          if (estimator.landmarkExists(tracklet_id)) {
            continue;
          }

          gtsam::CameraSet<GtsamCamera> camera_set;
          gtsam::Point2Vector measurements;
          // frames used for triangulation
          FrameIds used_frames;

          // Search all candidate frames for this tracklet.
          for (const auto other_frame_id : query_frames_for_triangulation) {
            CHECK(estimator.poseExists(other_frame_id));

            const ViFrame::Ptr& other_vi_frame =
                estimator.getFramePtr(other_frame_id);
            CHECK_NOTNULL(other_vi_frame);

            const gtsam::Pose3& X_W_other = estimator.getPose(other_frame_id);

            // add measurements for all keyframes
            if (stereo_measurement_filler.fill(tracklet_id, camera_set,
                                               measurements, other_vi_frame,
                                               X_W_other)) {
              used_frames.push_back(other_frame_id);
            }
          }

          // add measurements for current frame
          if (stereo_measurement_filler.fill(tracklet_id, camera_set,
                                             measurements, frame,
                                             estimator.getPose(frame_id))) {
            used_frames.push_back(frame->getFrameId());
          }

          auto m_W_result = dyno::triangulateSafe<GtsamCamera>(
              camera_set, measurements, triangulation_params, true);

          // LOG(INFO) << "i=" << tracked_id << " " << count;
          // LOG(INFO) << m_W_result;

          // TODO: add extra checks
          if (m_W_result) {
            gtsam::Point3 m_W = *m_W_result;
            match_infos[i] = MatchInfo{m_W, tracklet_id, true, 0.0};
            matching_frames[i] = used_frames;
          }
        }
      });

  std::vector<dyno::KeypointIdentifier> observations;
  observations.reserve(match_infos.size());

  for (size_t i = 0; i < match_infos.size(); i++) {
    const MatchInfo& match_info = match_infos[i];
    if (!match_info.is_valid) {
      continue;
    }

    CHECK(!estimator.landmarkExists(match_info.id));
    estimator.setLandmark(match_info.id, match_info.m_W);

    // keyframes used for triangulation
    const dyno::FrameIds& matched_frames = matching_frames[i];
    for (dyno::FrameId k0 : matched_frames) {
      // check we havent already added this measurement
      if (!estimator.hasMeasurement(k0, match_info.id)) {
        observations.push_back(dyno::KeypointIdentifier{k0, match_info.id, 0});
      }
    }
  }
  estimator.addMeasurements(observations);
  new_measurements.insert(new_measurements.begin(), observations.begin(),
                          observations.end());
  triangulate_t.stop();

  return observations.size();
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

bool Frontend::shouldBeViKeyFrame(const VioGraphEstimator& estimator,
                                  ViFrame::Ptr frame) const {
  dyno::utils::ChronoTimingStats timer_t("frontend.vi_kf_selection");
  if (frame->frame_id < 4) {
    // just starting, so yes, we need this as a new keyframe
    return true;
  }

  if (!vi_tracking_condition_.is_initalized) return false;

  // TODO: not initalised!
  const auto& camera_params = camera_->getParams();
  const int rows = camera_params.ImageHeight() / 10;
  const int cols = camera_params.ImageWidth() / 10;

  const double kptradius_ = 0.09;
  const double radius = double(std::min(rows, cols)) * kptradius_;

  // TODO: careful this does not grow forever as this will just take up more and
  // more time!
  std::set<FrameId> all_frames = estimator.keyFrames();

  const size_t num_keypoints = frame->numKeypoints();

  cv::Mat matches = cv::Mat::zeros(rows, cols, CV_8UC1);
  cv::Mat detections = cv::Mat::zeros(rows, cols, CV_8UC1);

  int intersection_count = 0;
  int union_count = 0;

  // lets test without overlap of current frame
  // because we have not yet initalised points in the map based on point in the
  // current frame when we call this function: just for tesgin!
  cv::Point2f keypoint;
  for (size_t i = 0; i < num_keypoints; i++) {
    frame->getCvKeypointByIndex(i, keypoint);
    cv::circle(detections, keypoint * 0.1, int(radius), cv::Scalar(255),
               cv::FILLED);

    TrackletId id = frame->ids[i];
    if (estimator.landmarkExists(id)) {
      cv::circle(matches, keypoint * 0.1, int(radius), cv::Scalar(255),
                 cv::FILLED);
    }

    cv::Mat intersectionMask, unionMask;
    cv::bitwise_and(matches, detections, intersectionMask);
    cv::bitwise_or(matches, detections, unionMask);
    intersection_count = cv::countNonZero(intersectionMask);
    union_count = cv::countNonZero(unionMask);
  }

  // // overlap of current frame with landmarks!
  double overlap = double(intersection_count) / double(union_count);
  // double overlap = 100;

  double overlap_others = 0.0;
  for (auto other_frame_id : all_frames) {
    cv::Mat matches = cv::Mat::zeros(rows, cols, CV_8UC1);
    cv::Mat detections = cv::Mat::zeros(rows, cols, CV_8UC1);

    ViFrame::Ptr other_frame = estimator.getFramePtr(other_frame_id);
    CHECK_NOTNULL(other_frame);

    const size_t num_other_keypoints = other_frame->numKeypoints();

    cv::Point2f keypoint;
    for (size_t i = 0; i < num_other_keypoints; i++) {
      other_frame->getCvKeypointByIndex(i, keypoint);
      cv::circle(detections, keypoint * 0.1, int(radius), cv::Scalar(255),
                 cv::FILLED);

      // check both if we have a match AND that the landmark is in the BA (ie
      // has been tracked and initalised)
      TrackletId id = other_frame->ids[i];
      if (estimator.landmarkExists(id) && frame->observedLandmark(id)) {
        cv::circle(matches, keypoint * 0.1, int(radius), cv::Scalar(255),
                   cv::FILLED);
      }
    }

    // IoU
    cv::Mat intersectionMask, unionMask;
    cv::bitwise_and(matches, detections, intersectionMask);
    cv::bitwise_or(matches, detections, unionMask);
    int intersection_count = cv::countNonZero(intersectionMask);
    int union_count = cv::countNonZero(unionMask);

    overlap_others = std::max(overlap_others,
                              double(intersection_count) / double(union_count));
  }

  overlap = std::min(overlap_others, overlap);
  // take a decision
  if (num_keypoints < 14) {
    // a respectable keyframe needs some detections...
    return false;
  }

  const double overlap_thresh = 0.55;  // spatial redundancy
  if (overlap > overlap_thresh
      /*&& double(numMatches)/double(numKeypoints) > 0.35*/) {
    return false;
  } else {
    return true;
  }
}

void Frontend::ObjectKeyframeMetrics::computeCovariance(
    const std::vector<Eigen::Vector2d>& pts, Eigen::Matrix2d& cov,
    Eigen::Vector2d& mean) {
  const int N = static_cast<int>(pts.size());

  if (N == 0) {
    cov.setIdentity();
    mean.setZero();
    return;
  }

  // // --- stack into matrix (2 x N)
  // Eigen::Matrix<double, 2, Eigen::Dynamic> X(2, N);

  // for (int i = 0; i < N; ++i) {
  //   X.col(i) = pts[i];
  // }

  // // --- mean (vectorized)
  // mean = X.rowwise().mean();

  // // --- center in one shot
  // Eigen::Matrix<double, 2, Eigen::Dynamic> Xc = X.colwise() - mean;

  // // --- covariance (fully batched)
  // cov = (Xc * Xc.transpose()) / double(N);
  // Compute mean.
  mean.setZero();
  for (const auto& pt : pts) {
    mean += pt;
  }
  mean /= static_cast<double>(N);

  // Compute covariance.
  cov.setZero();
  for (const auto& pt : pts) {
    const Eigen::Vector2d centered = pt - mean;
    cov.noalias() += centered * centered.transpose();
  }
  cov /= static_cast<double>(N);
}

double Frontend::ObjectKeyframeMetrics::computeCoverage(
    const std::vector<Eigen::Vector2d>& pts_ref,
    const std::vector<Eigen::Vector2d>& pts_target, double radius) {
  if (pts_ref.empty() || pts_target.empty()) return 0.0;

  const int N = pts_target.size();
  const double r2 = radius * radius;

  // --- pack current points into matrix (2 x N)
  Eigen::Matrix2Xd C(2, N);
  for (int i = 0; i < N; ++i) C.col(i) = pts_target[i];

  int covered = 0;

  for (const auto& k : pts_ref) {
    // broadcast subtraction (vectorized)
    Eigen::ArrayXd dx = C.row(0).array() - k.x();
    Eigen::ArrayXd dy = C.row(1).array() - k.y();

    // squared distances (SIMD)
    Eigen::ArrayXd d2 = dx.square() + dy.square();

    // check if any point is within radius
    if ((d2 < r2).any()) {
      covered++;
    }
  }

  return static_cast<double>(covered) / pts_ref.size();
}

bool Frontend::shouldBeDoKeyFrame(const DoGraphEstimator& estimator,
                                  DoFrame::Ptr frame) const {
  const std::set<dyno::FrameId>& keyframes = estimator.activeKeyframes();
  if (keyframes.size() < 3) {
    return true;
  }

  dyno::FrameId last_keyframe_id = *keyframes.rbegin();
  const DoFrame::Ptr& last_keyframe = estimator.getFramePtr(last_keyframe_id);
  CHECK_NOTNULL(last_keyframe);

  std::vector<Eigen::Vector2d> pts_kf;
  std::vector<Eigen::Vector2d> pts_cur;

  const size_t num_keypoints = frame->numKeypoints();
  const std::vector<TrackletId>& ids = frame->ids;
  for (size_t i = 0; i < num_keypoints; i++) {
    TrackletId id = ids[i];

    // TODO: and is in estimator!!
    if (last_keyframe->observedLandmark(id)) {
      dyno::Keypoint keypoint_curr;
      frame->getKeypointByIndex(i, keypoint_curr);

      const dyno::Keypoint keypoint_kp = last_keyframe->keypointById(id);

      pts_cur.push_back(keypoint_curr);
      pts_kf.push_back(keypoint_kp);
    }
  }

  Eigen::Matrix2d cov_kf, cov_cur;
  Eigen::Vector2d mean_kf, mean_cur;

  ObjectKeyframeMetrics::computeCovariance(pts_kf, cov_kf, mean_kf);
  ObjectKeyframeMetrics::computeCovariance(pts_cur, cov_cur, mean_cur);

  // --- eigenvalues
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver_kf(cov_kf);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver_cur(cov_cur);

  const Eigen::Vector2d eig_kf = solver_kf.eigenvalues();
  const Eigen::Vector2d eig_cur = solver_cur.eigenvalues();

  // --- scale ratio (0..1-ish)
  const double scale_ratio = std::sqrt((eig_cur.sum()) / (eig_kf.sum() + 1e-9));

  // --- shape score (0..1)
  // Eigen guarantees ascending order
  const double kf_min = eig_kf(0);
  const double kf_max = eig_kf(1);

  const double cur_min = eig_cur(0);
  const double cur_max = eig_cur(1);

  const double ratio_kf = kf_min / (kf_max + 1e-9);
  const double ratio_cur = cur_min / (cur_max + 1e-9);

  const double shape_score =
      std::min(ratio_cur / (ratio_kf + 1e-9), ratio_kf / (ratio_cur + 1e-9));

  const double coverage =
      ObjectKeyframeMetrics::computeCoverage(pts_kf, pts_cur);

  bool distribution_degraded = shape_score < 0.3;  // structure collapsed

  bool coverage_lost = coverage < 0.3;  // drift outside expected region

  // 1.0 (ish) means no scale change
  // > 1 is increase in size, < 1 is decrease in size.
  // scale change is the delta in change
  double scale_change = std::abs(scale_ratio - 1.0);

  bool large_scale_change = scale_change > 0.8;

  bool need_new_keyframe =
      distribution_degraded || coverage_lost || large_scale_change;

  // need at least some detections to be a good keyframe
  if (num_keypoints < 10) {
    need_new_keyframe = false;
  }

  // // but also need at least some initial points
  // if (all_m_L_points_.size() < 10) {
  //   need_new_keyframe = false;
  // }

  LOG(INFO) << "KF stats j=" << frame->getObjectId() << " cov " << coverage
            << " scale_change " << scale_change << " shape_score "
            << shape_score;

  return need_new_keyframe;
}

}  // namespace dyno2
