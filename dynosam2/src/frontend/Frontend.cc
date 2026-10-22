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
#include "dynosam_cv/CvUtils.hpp"
#include "dynosam_opt/factors/HybridFormulationFactors.hpp"

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

void Frontend::solveVisualOdometry(VioGraphEstimator& estimator,
                                   FeatureBlockContainer& features,
                                   FeatureBlockGeometry& geometry,
                                   AyncVO& async_context,
                                   std::vector<FrameId>& frames_updated,
                                   bool& success,
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

    ViFrame::Ptr vi_frame = std::make_shared<ViFrame>(frame_id, timestamp);
    vi_frame->fill(geometry, features);

    // propogate use constant velocity model
    // or IMU!
    const VioGraphState& vio_state_previous =
        estimator.getVioState(previous_frame_id);
    const dyno::Timestamp dt = timestamp - vio_state_previous.timestamp;
    gtsam::Pose3 predicted_pose = predictPoseConstantVelocity(
        vio_state_previous.pose_W, vio_state_previous.velocity_W, dt);
    // estimator.initaliseStates(frame_id, timestamp,
    //                           StereoMeasurementStatusVector{},
    //                           predicted_pose);
    estimator.initaliseStates(vi_frame, predicted_pose);

    success = false;
    // alert awaiting threads
    async_context.vo_cv.notify_all();

    LOG(WARNING) << "3D2D RANSAC failed for VO! k=" << frame_id;
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
  // frame.frame_id = frame_id;
  // frame.timestamp = timestamp;
  // frame.fill(geometry, features);
  ViFrame::Ptr vi_frame = std::make_shared<ViFrame>(frame_id, timestamp);
  vi_frame->fill(geometry, features);
  estimator.initaliseStates(vi_frame, X_W);

  // initalise estimator with new state (eventually maybe from VIO?)
  // we now have an initial pose
  std::vector<dyno::TrackletId> observation_ids;
  observation_ids.reserve(adapter.getNumberCorrespondences());
  // add measurements for keypoints that were tracked and already initalised
  // this will help with the BA to get intermediate frame measurements
  for (size_t idx = 0; idx < adapter.getNumberCorrespondences(); idx++) {
    TrackletId id = adapter.trackletId(idx);
    if (adapter.isInlier(idx) && estimator.landmarkExists(id)) {
      observation_ids.push_back(id);
    }
  }
  // all measurements are for the same frame use this function call which runs a
  // bit faster!
  estimator.addMeasurements(vi_frame, observation_ids);

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

  std::set<dyno::FrameId> frames_updated_set;
  frames_updated_set.insert(frame_id);

  const bool should_be_keyframe = shouldBeViKeyFrame(estimator, vi_frame);
  // TODO: must bound keyframes (only for estimator and covisibility checking)
  //  somehow this must be different from full SLAM and trajectory output!
  //  full SLAM will use Keyframe MAP which represents the full SLAM problem!
  //  this is just maybe the local problem
  if (should_be_keyframe) {
    const size_t num_threads =
        std::max(1u, std::thread::hardware_concurrency());
    const size_t num_tracklets = vi_frame->ids.size();
    const size_t chunk_size = (num_tracklets + num_threads - 1) / num_threads;

    using GtsamCamera = Camera::CameraImpl;
    gtsam::SharedNoiseModel model = gtsam::noiseModel::Isotropic::Sigma(2, 3.0);

    gtsam::TriangulationParameters triangulation_params;
    triangulation_params.noiseModel = model;
    // set maximum average reprojection threshold
    triangulation_params.dynamicOutlierRejectionThreshold = 4.0;

    std::set<dyno::FrameId> all_frames = estimator.keyFrames();
    std::vector<FrameId> query_frames_for_triangulation(all_frames.begin(),
                                                        all_frames.end());

    std::shared_ptr<dyno::RGBDCamera> rgbd_camera =
        camera_->safeGetRGBDCamera();
    StereoCalibPtr K_stereo = rgbd_camera->getFakeStereoCalib();
    CHECK_NOTNULL(K_stereo);
    const gtsam::Cal3_S2& mono_cal = K_stereo->calibration();
    const gtsam::Pose3 left_Pose_right = gtsam::Pose3(
        gtsam::Rot3(), gtsam::Point3(K_stereo->baseline(), 0.0, 0.0));

    dyno::utils::ChronoTimingStats triangulate_t("frontend.vo.triangulate");
    AlignedVector<MatchInfo> match_infos(num_tracklets);
    std::vector<dyno::FrameIds> matching_frames(num_tracklets);

    auto prepareStereoMeasurements =
        [&left_Pose_right, &mono_cal](
            dyno::TrackletId id, gtsam::CameraSet<GtsamCamera>& camera_set,
            gtsam::Point2Vector& measurements, const ViFrame& frame_k,
            const gtsam::Pose3& X_W_k) -> bool {
      if (!frame_k.observedLandmark(id)) {
        return false;
      }

      const gtsam::Pose3& left_pose = X_W_k;
      const GtsamCamera left_camera_i(left_pose, mono_cal);

      const gtsam::Pose3 rightPose = left_pose.compose(left_Pose_right);
      const GtsamCamera right_camera_i(rightPose, mono_cal);

      const gtsam::StereoPoint2& z = frame_k.stereoPointById(id);
      camera_set.push_back(left_camera_i);
      measurements.push_back(z.point2());
      if (!std::isnan(z.uR())) {  // if right point is valid
        camera_set.push_back(right_camera_i);
        measurements.push_back(z.right());
      }

      return true;
    };

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, num_tracklets, chunk_size),
        [&](const tbb::blocked_range<size_t>& range) {
          for (size_t i = range.begin(); i != range.end(); ++i) {
            const dyno::TrackletId tracklet_id = vi_frame->ids[i];

            if (estimator.landmarkExists(tracklet_id)) {
              continue;
            }

            gtsam::CameraSet<GtsamCamera> camera_set;
            gtsam::Point2Vector measurements;
            // frames used for triangulation
            FrameIds used_frames;

            size_t count{0};
            // Search all candidate frames for this tracklet.
            for (const auto other_frame_id : query_frames_for_triangulation) {
              // CHECK(frames.exists(other_frame_id));
              CHECK(estimator.poseExists(other_frame_id));

              // const ViFrame& other_vi_frame = frames.at(other_frame_id);
              const ViFrame::Ptr other_vi_frame_ptr =
                  estimator.getFramePtr(other_frame_id);
              CHECK_NOTNULL(other_vi_frame_ptr);
              const ViFrame& other_vi_frame = *other_vi_frame_ptr;

              const gtsam::Pose3& X_W_other = estimator.getPose(other_frame_id);

              // add measurements for all keyframes
              if (prepareStereoMeasurements(tracklet_id, camera_set,
                                            measurements, other_vi_frame,
                                            X_W_other)) {
                ++count;
                used_frames.push_back(other_frame_id);
              }
            }

            // add measurements for current frame
            if (prepareStereoMeasurements(tracklet_id, camera_set, measurements,
                                          *vi_frame, X_W)) {
              used_frames.push_back(vi_frame->frame_id);
              ++count;
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

    size_t init_triangulation{0};
    for (size_t i = 0; i < match_infos.size(); i++) {
      const MatchInfo& match_info = match_infos[i];
      if (!match_info.is_valid) {
        continue;
      }

      CHECK(!estimator.landmarkExists(match_info.id));
      estimator.setLandmark(match_info.id, match_info.m_W);
      init_triangulation++;

      // keyframes used for triangulation
      const dyno::FrameIds& matched_frames = matching_frames[i];
      for (dyno::FrameId k0 : matched_frames) {
        // CHECK(frames.exists(k0));
        // const ViFrame& vi_frame = frames.at(k0);
        // const ViFrame::Ptr
        // CHECK_EQ(vi_frame.frame_id, k0);

        // check we havent already added this measurement
        if (!estimator.hasObservation(k0, match_info.id)) {
          observations.push_back(
              dyno::KeypointIdentifier{k0, match_info.id, 0});
        }
      }
    }

    dyno::utils::ChronoTimingStats add_m_t2("frontend.vo.add2");
    estimator.addMeasurements(observations);
    add_m_t2.stop();
    triangulate_t.stop();

    LOG(INFO) << "Init triangulation: " << init_triangulation;
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
  dyno::TrackletIds reference_ids;
  reference_lmks.reserve(num_matches);
  reference_ids.reserve(num_matches);
  CHECK_EQ(inliers.size(), num_matches);
  for (size_t k = 0; k < num_matches; k++) {
    TrackletId i = adapter.trackletId(k);
    if (!inliers[k]) {
      Index fc_index = geometry.getFeatureContainerIndex(i);
      CHECK_EQ(features.ids[fc_index], i);
      features.inlier[fc_index] = 0;
    } else {
      reference_lmks.push_back(adapter.getPoint(k));
      reference_ids.push_back(i);
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

  std::set<dyno::FrameId> frames_updated_set;

  gtsam::Vector3 sigmas;
  sigmas << 2, 2, 2;

  if (estimator.numStates() > 0) {
    LOG(INFO) << "j=" << object_id << " solved k=" << frame_id_j
              << " last state k=" << estimator.currentFrameId();
  }

  std::unordered_set<dyno::TrackletId> added_measurements_i;
  if (!estimator.stateExists(frame_id_i)) {
    gtsam::Point3 t_W_i_centroid =
        X_W_i * dyno::computeCentroid(reference_lmks);
    gtsam::Pose3 L_W_i_centroid(gtsam::Rot3::Identity(), t_W_i_centroid);

    StereoMeasurementStatusVector stereo_measurements;
    stereo_measurements.reserve(adapter.getNumberCorrespondences());
    for (size_t idx = 0; idx < adapter.getNumberCorrespondences(); idx++) {
      TrackletId id = adapter.trackletId(idx);
      if (adapter.isInlier(idx) && ref_object_frame.observedLandmark(id) &&
          estimator.landmarkExists(id)) {
        auto stereo_measurement = StereoMeasurement::FromSigmas(
            ref_object_frame.stereoPointById(id), sigmas);

        stereo_measurements.push_back(
            StereoMeasurementStatus(stereo_measurement, frame_id_i, timestamp_i,
                                    id, object_id, ReferenceFrame::LOCAL));

        added_measurements_i.insert(id);
      }
    }
    LOG(INFO) << "obj j=" << object_id << " made Anchor KF " << frame_id_i;
    estimator.initalizeNewAnchorFrame(frame_id_i, stereo_measurements,
                                      L_W_i_centroid);
    frames_updated_set.insert(frame_id_i);
  }

  std::unordered_set<dyno::TrackletId> added_measurements_j;
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
          StereoMeasurementStatus(stereo_measurement, frame_id_j, timestamp_j,
                                  id, object_id, ReferenceFrame::LOCAL));

      added_measurements_j.insert(id);
    }
  }

  estimator.initalizeStatesPropogate(frame_id_j, stereo_measurements, G_i_ij,
                                     X_W_j, X_W_i);
  frames_updated_set.insert(frame_id_j);

  const bool is_keyframe = shouldBeDoKeyFrame(estimator, frame, frames);
  if (is_keyframe) {
    LOG(INFO) << "obj j=" << object_id << " made new Regular KF";
    // if keyframe for now just initalise keyframe in object frame

    // use points from reference frame as we will have had a chance to optimize
    // it
    estimator.setRegularKeyframe(frame_id_j);

    const StereoBaMap& observations = estimator.getObservations();

    // TODO: before adding points do radial outlier rejection
    // TODO: then mark those ones as outliers
    //  and remove measurements
    // TODO: just do with PCL
    const gtsam::Pose3& L_W_i = estimator.getPose(frame_id_i);

    StereoMeasurementStatusVector stereo_measurements;
    stereo_measurements.reserve(2.0 * adapter.getNumberCorrespondences());

    for (size_t i = 0; i < reference_ids.size(); i++) {
      const dyno::TrackletId id = reference_ids[i];
      if (!estimator.landmarkExists(id)) {
        const gtsam::Point3& m_C = reference_lmks[i];
        const gtsam::Point3& m_L = L_W_i.inverse() * X_W_i * m_C;
        estimator.setLandmark(id, m_L);
      }

      // previous and current measurement
      // avoid adding dupliated
      if (!added_measurements_i.count(id)) {
        auto stereo_measurement_i = StereoMeasurement::FromSigmas(
            ref_object_frame.stereoPointById(id), sigmas);
        stereo_measurements.push_back(StereoMeasurementStatus(
            stereo_measurement_i, frame_id_i, timestamp_i, id, object_id,
            ReferenceFrame::LOCAL));
      }

      if (!added_measurements_j.count(id)) {
        auto stereo_measurement_j =
            StereoMeasurement::FromSigmas(frame.stereoPointById(id), sigmas);
        stereo_measurements.push_back(StereoMeasurementStatus(
            stereo_measurement_j, frame_id_j, timestamp_j, id, object_id,
            ReferenceFrame::LOCAL));
      }
      // check we have measurements of this!
      // CHECK_GT(observations.numMeasurements(id), 0);
    }
  }

  std::vector<dyno::FrameId> affected_in_opt;
  // estimator.optimize(frame_id_j, vo_estimator, affected_in_opt, 2);

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
      : left_Pose_right(Extrinsics(camera)), mono_cal(Calibration(camera)) {}

  bool fill(dyno::TrackletId id, gtsam::CameraSet<GtsamCamera>& camera_set,
            gtsam::Point2Vector& measurements, const ViFrame& frame_k,
            const gtsam::Pose3& X_W_k) const {
    if (!frame_k.observedLandmark(id)) {
      return false;
    }

    const gtsam::Pose3& left_pose = X_W_k;
    const GtsamCamera left_camera_i(left_pose, mono_cal);

    const gtsam::Pose3 rightPose = left_pose.compose(left_Pose_right);
    const GtsamCamera right_camera_i(rightPose, mono_cal);

    const gtsam::StereoPoint2& z = frame_k.stereoPointById(id);
    camera_set.push_back(left_camera_i);
    measurements.push_back(z.point2());

    if (!std::isnan(z.uR())) {  // if right point is valid
      camera_set.push_back(right_camera_i);
      measurements.push_back(z.right());
    }

    return true;
  }

  static gtsam::Pose3 Extrinsics(const dyno::Camera::Ptr& camera) {
    std::shared_ptr<dyno::RGBDCamera> rgbd_camera = camera->safeGetRGBDCamera();
    StereoCalibPtr K_stereo = rgbd_camera->getFakeStereoCalib();
    CHECK_NOTNULL(K_stereo);
    return gtsam::Pose3(gtsam::Rot3(),
                        gtsam::Point3(K_stereo->baseline(), 0.0, 0.0));
  }

  static gtsam::Cal3_S2 Calibration(const dyno::Camera::Ptr& camera) {
    std::shared_ptr<dyno::RGBDCamera> rgbd_camera = camera->safeGetRGBDCamera();
    StereoCalibPtr K_stereo = rgbd_camera->getFakeStereoCalib();
    CHECK_NOTNULL(K_stereo);
    const gtsam::Cal3_S2& mono_cal = K_stereo->calibration();
    return mono_cal;
  }
};

// size_t Frontend::matchToVioMap(VioGraphEstimator& estimator,
//                                const ViFrames& frames, ViFrame& frame) {
//   const size_t num_threads = std::max(1u,
//   std::thread::hardware_concurrency()); const size_t num_tracklets =
//   frame.ids.size(); const size_t chunk_size = (num_tracklets + num_threads -
//   1) / num_threads; const dyno::FrameId frame_id = frame.frame_id;

//   using GtsamCamera = dyno::Camera::CameraImpl;
//   gtsam::SharedNoiseModel model =
//   gtsam::noiseModel::Isotropic::Sigma(2, 2.0);

//   gtsam::TriangulationParameters triangulation_params;
//   triangulation_params.noiseModel = model;
//   // set maximum average reprojection threshold
//   triangulation_params.dynamicOutlierRejectionThreshold = 2.0;

//   // get active keyframes
//   const std::set<dyno::FrameId> all_frames = estimator.keyFrames();
//   std::vector<FrameId> query_frames_for_triangulation(all_frames.begin(),
//                                                       all_frames.end());

//   const StereoMeasurementFiller stereo_measurement_filler(camera_);

//   dyno::utils::ChronoTimingStats triangulate_t("frontend.vo.triangulate");
//   AlignedVector<MatchInfo> match_infos(num_tracklets);
//   std::vector<dyno::FrameIds> matching_frames(num_tracklets);

//   tbb::parallel_for(
//       tbb::blocked_range<size_t>(0, num_tracklets, chunk_size),
//       [&](const tbb::blocked_range<size_t>& range) {
//         for (size_t i = range.begin(); i != range.end(); ++i) {
//           const dyno::TrackletId tracklet_id = frame.ids[i];

//           if (estimator.landmarkExists(tracklet_id)) {
//             continue;
//           }

//           gtsam::CameraSet<GtsamCamera> camera_set;
//           gtsam::Point2Vector measurements;
//           // frames used for triangulation
//           FrameIds used_frames;

//           size_t count{0};
//           // Search all candidate frames for this tracklet.
//           for (const auto other_frame_id : query_frames_for_triangulation) {
//             CHECK(frames.exists(other_frame_id));
//             CHECK(estimator.poseExists(other_frame_id));

//             const ViFrame& other_vi_frame = frames.at(other_frame_id);
//             const gtsam::Pose3& X_W_other =
//             estimator.getPose(other_frame_id);

//             // add measurements for all keyframes
//             if (stereo_measurement_filler.fill(tracklet_id, camera_set,
//                                                measurements, other_vi_frame,
//                                                X_W_other)) {
//               ++count;
//               used_frames.push_back(other_frame_id);
//             }
//           }

//           // add measurements for current frame
//           if (stereo_measurement_filler.fill(tracklet_id, camera_set,
//                                              measurements, frame,
//                                              estimator.getPose(frame_id))) {
//             used_frames.push_back(frame.frame_id);
//             ++count;
//           }

//           auto m_W_result = dyno::triangulateSafe<GtsamCamera>(
//               camera_set, measurements, triangulation_params, true);

//           // LOG(INFO) << "i=" << tracked_id << " " << count;
//           // LOG(INFO) << m_W_result;

//           // TODO: add extra checks
//           if (m_W_result) {
//             gtsam::Point3 m_W = *m_W_result;
//             match_infos[i] = MatchInfo{m_W, tracklet_id, true, 0.0};
//             matching_frames[i] = used_frames;
//           }
//         }
//       });

//   StereoMeasurementStatusVector stereo_measurements;
//   stereo_measurements.reserve(match_infos.size());

//   size_t init_triangulation{0};
//   for (size_t i = 0; i < match_infos.size(); i++) {
//     const MatchInfo& match_info = match_infos[i];
//     if (!match_info.is_valid) {
//       continue;
//     }

//     CHECK(!estimator.landmarkExists(match_info.id));
//     estimator.setLandmark(match_info.id, match_info.m_W);
//     init_triangulation++;

//     // keyframes used for triangulation
//     const dyno::FrameIds& matched_frames = matching_frames[i];
//     for (dyno::FrameId k0 : matched_frames) {
//       CHECK(frames.exists(k0));
//       const ViFrame& vi_frame = frames.at(k0);
//       CHECK_EQ(vi_frame.frame_id, k0);

//       // check we havent already added this measurement
//       if (!estimator.hasObservation(k0, match_info.id)) {
//         // auto stereo_measurement = StereoMeasurement::FromSigmas(
//         //     vi_frame.stereoPointById(match_info.id), sigmas);

//         // stereo_measurements.push_back(StereoMeasurementStatus(
//         //     stereo_measurement, k0, vi_frame.timestamp, match_info.id, 0,
//         //     ReferenceFrame::LOCAL));
//       }
//     }
//   }

//   estimator.addMeasurements(stereo_measurements);
//   triangulate_t.stop();

//   return stereo_measurements.size();
// }

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

    // CHECK(frames.exists(other_frame_id));
    // const ViFrame& other_frame = frames.at(other_frame_id);
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

  LOG(INFO) << overlap;
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
                                  const DoFrame& frame,
                                  const DoFrames& frames) const {
  const std::set<dyno::FrameId>& keyframes = estimator.activeKeyframes();
  if (keyframes.size() < 3) {
    return true;
  }

  dyno::FrameId last_keyframe_id = *keyframes.rbegin();
  CHECK(frames.exists(last_keyframe_id));

  const DoFrame& last_keyframe = frames.at(last_keyframe_id);

  std::vector<Eigen::Vector2d> pts_kf;
  std::vector<Eigen::Vector2d> pts_cur;

  const size_t num_keypoints = frame.numKeypoints();
  for (size_t i = 0; i < num_keypoints; i++) {
    TrackletId id = frame.ids[i];

    // TODO: and is in estimator!!
    if (last_keyframe.observedLandmark(id)) {
      dyno::Keypoint keypoint_curr;
      frame.getKeypointByIndex(i, keypoint_curr);

      const dyno::Keypoint keypoint_kp = last_keyframe.keypointById(id);

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

  LOG(INFO) << "KF stats j=" << frame.object_id << " cov " << coverage
            << " scale_change " << scale_change << " shape_score "
            << shape_score;

  return need_new_keyframe;
}

}  // namespace dyno2
