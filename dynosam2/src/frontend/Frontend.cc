#include "dynosam2/frontend/Frontend.hpp"

#include <gtsam/geometry/CameraSet.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>

#include "dynosam2/estimator/ImuError.hpp"
#include "dynosam2/frontend/FlowRefinement.hpp"
#include "dynosam2/frontend/GeometryUpdater.hpp"
#include "dynosam2/frontend/MatchingAdaptor.hpp"
#include "dynosam_common/anms/anms/nanoflann.hpp"
#include "dynosam_common/utils/Numerical.hpp"
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
      do_ransac_threshold_3d_2d_(computeOpenGvRansacThreshold(2.0, camera)) {
  // describer_ = cv::BRISK::create();
  describer_ = cv::xfeatures2d::BriefDescriptorExtractor::create();
}

void Frontend::detectAndTrack(FeatureBlockContainer*& features_out,
                              dyno::ObjectDetectionResult& object_detection_out,
                              dyno::ImageContainer::Ptr container_in_out,
                              const gtsam::Rot3& R_k_km1) {
  // TODO: I think track should take a real value R and not an optional!
  auto tracking_result = feature_tracker_.track(container_in_out, R_k_km1);
  features_out = tracking_result.featues;
  CHECK_NOTNULL(features_out);
  object_detection_out = tracking_result.object_detection;
}

void Frontend::solveVisualOdometry(
    VioGraphEstimator& estimator, FeatureBlockContainer& features,
    FeatureBlockGeometry& geometry, AyncVO& async_context,
    std::vector<FrameId>& frames_updated,
    std::vector<KeypointIdentifier>& new_measurements,
    std::vector<TrackletId>& correspondences, bool& success,
    const ImageContainer& image_container,
    const VioMotionPropagation& T_k_km1_propagated) {
  dyno::utils::ChronoTimingStats vo_t("frontend.vo");

  CHECK(T_k_km1_propagated.success);

  const FrameId frame_id = image_container.frameId();
  const Timestamp timestamp = image_container.timestamp();

  // previous frame as the refernece fmrae
  // the estimator does not yet have a solution for the current frame
  // and therefore we need to initalise form the previous frame
  const FrameId previous_frame_id = estimator.currentFrameId();
  VLOG(20) << "VIO tracking flow against frame k=" << previous_frame_id;
  // const ViFrame& vi_frame_previous = frames.at(previous_frame_id);
  const ViFrame::Ptr vi_frame_previous_ptr =
      estimator.getFramePtr(previous_frame_id);
  CHECK_NOTNULL(vi_frame_previous_ptr);
  const ViFrame& vi_frame_previous = *vi_frame_previous_ptr;

  CHECK_EQ(previous_frame_id, T_k_km1_propagated.relative_motion_error->from());

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
  const bool ransac_success = solve3d2dRansac(T_ij_pnp, inliers, adapter,
                                              vio_ransac_threshold_3d_2d_, 500);
  ransac_t.stop();

  if (!ransac_success) {
    ViFrame::Ptr vi_frame = std::make_shared<ViFrame>(frame_id, timestamp);
    vi_frame->fill(geometry, features);
    vi_frame->setRelativePose(T_k_km1_propagated.relative_motion_error);

    estimator.initaliseStates(vi_frame, T_k_km1_propagated.predicted_pv);

    success = true;
    async_context.markReadyAndValidAndNotify(
        T_k_km1_propagated.predicted_pv.pose);

    LOG(WARNING) << "3D2D RANSAC failed for VO! k=" << frame_id
                 << " but state propogated via motion!";
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

  // TODO: test!
  //  if(T_k_km1_propagated.source == VioMotionPropagation::Source::IMU) {
  //    CHECK_EQ(T_k_km1_propagated.relative_motion_error->to(), frame_id);
  //    CHECK_EQ(T_k_km1_propagated.relative_motion_error->from(),
  //    previous_frame_id);
  //    vi_frame->setRelativePose(T_k_km1_propagated.relative_motion_error);
  //  }
  //  else {
  //     vi_frame->setRelativePose(std::make_shared<RelativePoseError>(
  //      T_ij, previous_frame_id, vi_frame->getFrameId()));
  //  }
  vi_frame->setRelativePose(std::make_shared<RelativePoseError>(
      T_ij, previous_frame_id, vi_frame->getFrameId()));

  estimator.initaliseStates(vi_frame, X_W);

  LOG(INFO) << "Solved VO k= " << frame_id;
  async_context.markReadyAndValidAndNotify(X_W);

  // initalise estimator with new state (eventually maybe from VIO?)
  // we now have an initial pose
  std::vector<dyno::TrackletId> observation_ids;
  std::unordered_set<TrackletId> correspondences_set;
  observation_ids.reserve(adapter.getNumberCorrespondences());
  correspondences_set.reserve(adapter.getNumberCorrespondences());

  std::unordered_set<KeypointIdentifier> new_measurements_set;
  // Computed flow between tracked points. This is used to help determine if
  // there is a zero velocity update
  std::vector<double> flows;
  flows.reserve(adapter.getNumberCorrespondences());
  // add measurements for keypoints that were tracked and already initalised
  // this will help with the BA to get intermediate frame measurements
  for (size_t idx = 0; idx < adapter.getNumberCorrespondences(); idx++) {
    TrackletId id = adapter.trackletId(idx);
    if (adapter.isInlier(idx)) {
      // indicate point as well tracked
      // may or may not be in the map!
      correspondences_set.insert(id);
      flows.push_back(
          (adapter.keypoint(idx) - adapter.keypointPrev(idx)).norm());

      // add new measurements to BA as these points are well tracked and we
      // already
      //  have this as a map poing
      if (estimator.landmarkExists(id)) {
        observation_ids.push_back(id);
        new_measurements_set.emplace(frame_id, id, dyno::background_label);
      }
    }
  }

  const double median_flow = dyno::calculateMedian(flows);
  const bool zero_velocity_update = median_flow < 1.5;
  if (zero_velocity_update) {
    LOG(INFO) << "Zero velocity update found! Skipping adding new points!";
    frames_updated.push_back(frame_id);
    new_measurements.clear();
    correspondences = dyno::TrackletIds(correspondences_set.begin(),
                                        correspondences_set.end());
    success = true;
    return;
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

  vi_frame->setImages(image_container);
  const bool should_be_keyframe = shouldBeViKeyFrame(estimator, vi_frame);
  if (should_be_keyframe) {
    // add images to frame. These will get released when the frame is pruned
    // vi_frame->setImages(image_container);
    // vi_frame->describe(describer_);
    // CHECK(vi_frame->hasDescriptors());

    // //TODO:
    // std::unordered_map<dyno::TrackletId, dyno::TrackletId> merged_lmks;
    // matchToVioMap(estimator, image_container, vi_frame, merged_lmks);

    // // hacky and probably slow for the moment
    // // update tracklet labels in FeatureContainer AND in new measurements AND
    // in correspences for(size_t i = 0; i < features.ids.size(); i++) {
    //   auto tid = features.ids[i];

    //   auto it = merged_lmks.find(tid);
    //   if(it != merged_lmks.end()) {
    //     LOG(INFO) << "Updatinf tids: " << features.ids[i] << " -> " <<
    //     it->second; features.ids[i] = it->second;
    //   }
    // }

    // for(size_t i = 0; i < correspondences.size(); i++) {
    //   auto tid = correspondences[i];

    //   auto it = merged_lmks.find(tid);
    //   if(it != merged_lmks.end()) {
    //     correspondences[i] = it->second;
    //   }
    // }

    // for(size_t i = 0; i < new_measurements.size(); i++) {
    //   auto tid = new_measurements[i].id;

    //   auto it = merged_lmks.find(tid);
    //   if(it != merged_lmks.end()) {
    //     new_measurements[i].id = it->second;
    //   }
    // }

    // initlaise new points in the map by attempting to triangulate points
    // at all active keyframes
    const size_t init_triangulation =
        temporalVioTriangultion(estimator, vi_frame, new_measurements_set);
    // LOG(INFO) << "Init triangulation: " << init_triangulation;
    LOG(INFO) << "Made VIO keyframe k=" << frame_id;
    estimator.setKeyframe(frame_id, true);
  }

  if (estimator.keyFrames().size() > 2) {
    std::vector<TrackletId> outliers;
    // we need images at this point in the vi_frame
    estimator.removeOutliers(vi_frame, outliers);

    VLOG(20) << "VIO removed " << outliers.size() << " map outliers";
    for (dyno::TrackletId tid : outliers) {
      new_measurements_set.erase(
          dyno::KeypointIdentifier(frame_id, tid, dyno::background_label));

      // Index fc_index = geometry.getFeatureContainerIndex(tid);
      // CHECK_EQ(features.ids[fc_index], tid);
      // features.inlier[fc_index] = 0;
    }

    utils::ChronoTimingStats timer_t("frontend.vo.refine_map");
    std::vector<FrameId> frames_affected;
    if (should_be_keyframe) {
      // optimize with structure
      estimator.optimize(frame_id, frames_affected, false);
    } else {
      // optimize only states
      estimator.optimize(frame_id, frames_affected, true);
      // if this frame is not a keyframe we can release the image memory here
      // since we will not use it again they keyframe image memory will be
      // cleaned during pruning
      vi_frame->releaseImages();
    }

    for (FrameId frame_id : frames_affected) {
      frames_updated_set.insert(frame_id);
    }
  }

  frames_updated = std::vector<dyno::FrameId>(frames_updated_set.begin(),
                                              frames_updated_set.end());
  correspondences =
      dyno::TrackletIds(correspondences_set.begin(), correspondences_set.end());
  new_measurements = std::vector<KeypointIdentifier>(
      new_measurements_set.begin(), new_measurements_set.end());

  success = true;
}

void Frontend::solveObjectOdometry(
    ObjectId object_id, const VioGraphEstimator& vo_estimator,
    const EstimationParameters& params, const ImageContainer& image_container,
    DoGraphEstimator& estimator, FeatureBlockContainer& features,
    FeatureBlockGeometry& geometry, AyncVO& async_context,
    std::vector<FrameId>& frames_updated,
    std::vector<TrackletId>& correspondences, uchar& success) {
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
  previous_landmarks_C.reserveLandmarks(do_frame_i->numKeypoints());
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
  const gtsam::Pose3 G_i_ij_pose = G_i_ij_inv.measurement().inverse();
  const Motion3ReferenceFrame G_i_ij(G_i_ij_pose, dyno::ReferenceFrame::GLOBAL,
                                     frame_id_i, frame_id_j);
  // run covariance through adjoint
  const gtsam::Matrix66 G_i_ij_cov =
      Adj_G_i_ij_inv * G_i_ij_inv.covariance() * Adj_G_i_ij_inv.transpose();

  const HeavyPose3Measurement G_i_ij_measurement(
      dyno::Pose3Measurement(G_i_ij_pose, G_i_ij_cov),
      dyno::ReferenceFrame::GLOBAL, frame_id_i, frame_id_j);

  DoFrame::Ptr do_frame_j =
      std::make_shared<DoFrame>(frame_id_j, timestamp_j, object_id);
  do_frame_j->fill(geometry, features);
  // must be set when we have successful tracks
  do_frame_j->setRelativeMotion(G_i_ij_measurement);

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
      VLOG(50) << "Cannot init object pose from centroid as not enough "
                  "reference points!";
      success = false;
      return;
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

    if (reference_lmks_inliers.size() < 3) {
      VLOG(50) << "Cannot init object pose from centroid as not enough inlier "
                  "points!";
      // estimator.initaliseFrame(frame);
      success = false;
      return;
    }

    gtsam::Point3 t_W_i_centroid =
        X_W_i * dyno::computeCentroid(reference_lmks_inliers);
    gtsam::Pose3 L_W_i_centroid(gtsam::Rot3::Identity(), t_W_i_centroid);

    LOG(INFO) << "obj j=" << object_id << " made Anchor KF " << frame_id_i;

    // set images for this keyframe
    // memory will be cleared from inside the estimator as new keyframes are
    // added
    // TODO: unfortunately we dont have access to image_container_i (previous)
    // do_frame_i->setImages(image_container);
    estimator.initalizeNewAnchorState(do_frame_i, L_W_i_centroid);
    estimator.addMeasurements(do_frame_i, measurements_i);
    frames_updated_set.insert(frame_id_i);
  }

  std::vector<dyno::TrackletId> measurements_j;
  measurements_j.reserve(adapter.getNumberCorrespondences());
  correspondences.reserve(adapter.getNumberCorrespondences());
  // add measurements for keypoints that were tracked and already initalised
  // this will help with the BA to get intermediate frame measurements
  for (size_t idx = 0; idx < adapter.getNumberCorrespondences(); idx++) {
    TrackletId id = adapter.trackletId(idx);
    if (adapter.isInlier(idx)) {
      correspondences.push_back(id);

      if (estimator.landmarkExists(id)) {
        measurements_j.push_back(id);
      }
    }
  }

  estimator.initalizeStatesPropogate(do_frame_j, G_i_ij, X_W_j, X_W_i);
  estimator.addMeasurements(do_frame_j, measurements_j);
  frames_updated_set.insert(frame_id_j);

  const bool is_keyframe = shouldBeDoKeyFrame(estimator, do_frame_j);
  if (is_keyframe) {
    LOG(INFO) << "obj j=" << object_id << " made new Regular KF";
    // if keyframe for now just initalise keyframe in object frame

    // set images for this keyframe
    do_frame_j->setImages(image_container);
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

  estimator.removeOutliersByDetection(vo_estimator);

  frames_updated_set.insert(affected_in_opt.begin(), affected_in_opt.end());
  frames_updated = std::vector<dyno::FrameId>(frames_updated_set.begin(),
                                              frames_updated_set.end());
  success = true;
}

VioMotionPropagation Frontend::predictFromMotion(
    const dyno::FrameId frame_id, const dyno::Timestamp timestamp,
    const VioGraphEstimator& estimator, const ImuMeasurements& imu_measurements,
    const dyno::Timestamp constant_motion_max_dt) {
  const dyno::FrameId last_frame_id = estimator.currentFrameId();
  const VioGraphState& vio_state_previous =
      estimator.getVioState(last_frame_id);

  // TODO: but we add states even when we have no tracks so previous state
  //  will likely always be the previous state!
  const dyno::Timestamp dt = timestamp - vio_state_previous.timestamp;

  VioMotionPropagation propagation;
  propagation.success = false;
  if (auto imu_params = estimator.getImuConfig()) {
    ImuError::Ptr imu_error = std::make_shared<ImuError>(
        imu_measurements, imu_params, last_frame_id, frame_id);
    // TODO: IMU bias
    const gtsam::NavState predicted_nav_state =
        imu_error->predict(vio_state_previous.navState(), ImuError::Bias{});
    propagation.predicted_pv.pose = predicted_nav_state.pose();
    propagation.predicted_pv.velocity.linear = predicted_nav_state.velocity();
    // TODO: leaving angular blank!
    propagation.relative_motion_error = imu_error;
    propagation.success = true;
    propagation.source = VioMotionPropagation::Source::IMU;
  } else if (dt < constant_motion_max_dt) {
    // TODO: can only predict with constant velocity so bar back!
    const gtsam::Pose3& previous_pose_W = vio_state_previous.pose_W;
    gtsam::Pose3 X_W_k = predictPoseConstantVelocity(
        previous_pose_W, vio_state_previous.velocity_W, dt);
    propagation.predicted_pv.pose = X_W_k;
    propagation.predicted_pv.velocity = vio_state_previous.velocity_W;

    gtsam::Matrix66 sqrt_information;
    sqrt_information.setIdentity();

    const dyno::Timestamp sqrt_dt = sqrt(dt);
    // fill rotation
    sqrt_information.block<3, 3>(0, 0) *= 0.10 / sqrt_dt;
    // fill translation
    sqrt_information.block<3, 3>(3, 3) *= 1.0 / sqrt_dt;
    const gtsam::Matrix66 cov =
        (sqrt_information.transpose() * sqrt_information).inverse();

    dyno::Pose3Measurement T_ij(previous_pose_W.inverse() * X_W_k, cov);
    propagation.relative_motion_error =
        std::make_shared<RelativePoseError>(T_ij, last_frame_id, frame_id);
    propagation.success = true;
    propagation.source = VioMotionPropagation::Source::PRIOR_MOTION;
  }

  return propagation;
}

class KeypointCloud {
 public:
  explicit KeypointCloud(gtsam::Point2Vector points)
      : points_(std::move(points)) {}

  std::size_t kdtree_get_point_count() const { return points_.size(); }

  double kdtree_get_pt(const std::size_t index,
                       const std::size_t dimension) const {
    return dimension == 0 ? points_[index].x() : points_[index].y();
  }

  double kdtree_distance(const double* query_point, const std::size_t index,
                         std::size_t size) const {
    double distance = 0.0;

    for (std::size_t i = 0; i < size; ++i) {
      const double diff = query_point[i] - points_[index](i);
      distance += diff * diff;
    }

    return distance;
  }

  template <class BoundingBox>
  bool kdtree_get_bbox(BoundingBox&) const {
    return false;
  }

  const gtsam::Point2& operator[](std::size_t index) const {
    return points_[index];
  }

 private:
  gtsam::Point2Vector points_;
};

using KeypointKDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, KeypointCloud>, KeypointCloud, 2,
    std::size_t>;

class KeypointSpatialIndex {
 public:
  explicit KeypointSpatialIndex(const gtsam::Point2Vector& points)
      : cloud_(points),
        tree_(2, cloud_, nanoflann::KDTreeSingleIndexAdaptorParams(10)) {
    tree_.buildIndex();
  }

  // Returns keypoint indices within radius (pixels).
  std::vector<std::size_t> radiusSearch(const gtsam::Point2& query,
                                        double radius) const {
    const double query_point[2] = {query.x(), query.y()};

    // nanoflann uses squared L2 distance.
    const double radius_squared = radius * radius;

    std::vector<std::pair<std::size_t, double>> results;

    nanoflann::SearchParams search_params;
    tree_.radiusSearch(query_point, radius_squared, results, search_params);

    std::vector<std::size_t> indices;
    indices.reserve(results.size());

    for (const auto& result : results) {
      indices.push_back(result.first);
    }

    return indices;
  }

 private:
  KeypointCloud cloud_;
  KeypointKDTree tree_;
};

inline int hammingDistance(const uint8_t* a, const uint8_t* b,
                           int descriptor_size) {
  int distance = 0;

  for (int i = 0; i < descriptor_size; ++i) {
    distance += __builtin_popcount(static_cast<unsigned>(a[i] ^ b[i]));
  }

  return distance;
}

struct LandmarkDescriptorObservation {
  dyno::FrameId frame_id;
  std::vector<uint8_t> descriptor;
};

struct LandmarkDescriptorSet {
  dyno::TrackletId landmark_id;
  std::vector<LandmarkDescriptorObservation> observations;
};

struct VioMapMatch {
  Index keypoint_index;
  dyno::TrackletId landmark_id;
  int hamming_distance;
};

void Frontend::matchToVioMap(
    VioGraphEstimator& estimator, const ImageContainer& container,
    ViFrame::Ptr frame,
    std::unordered_map<dyno::TrackletId, dyno::TrackletId>& merged_lmks) {
  utils::ChronoTimingStats timer_t("frontend.vo.match_to_map");
  if (!frame->hasDescriptors()) {
    return;
  }

  std::vector<VioMapMatch> matches;

  // ah this should probably smaller than the feature detection min distance
  // between detected and tracked points so we dont try and search for lmks that
  // are already tracked alternatively we will just explicitly exclude from
  // matching observations in the current frame with an associated map point!
  constexpr float kSearchRadius = 15.0f;
  constexpr float kRatioThreshold = 0.8f;
  constexpr float kMaxHammingFraction = 0.25f;
  constexpr float kMinDepth = 0.2f;

  constexpr size_t kMaxObservationsPerLandmark = 5;

  const int descriptor_size = frame->descriptorSize();
  CHECK_GT(descriptor_size, 0);

  const int max_hamming_distance =
      static_cast<int>(descriptor_size * 8 * kMaxHammingFraction);

  // search last 'hopefully covisible' keyframe and most recent keyframe
  //  that is not this keyframe
  const std::set<dyno::FrameId> keyframes = estimator.keyFrames();

  if (keyframes.empty()) {
    return;
  }

  const dyno::FrameId oldest_fid = *keyframes.begin();
  const dyno::FrameId latest_fid = *keyframes.rbegin();

  CHECK_NE(latest_fid, frame->getFrameId());

  VLOG(10) << "Matching against local map KF's: " << oldest_fid << " "
           << latest_fid << " to fid=" << frame->getFrameId();

  const ViFrame::Ptr oldest_kf = estimator.getFramePtr(oldest_fid);
  const ViFrame::Ptr latest_kf = estimator.getFramePtr(latest_fid);

  // ---------------------------------------------------------------------------
  // 2. Collect multiple descriptors per landmark
  // ---------------------------------------------------------------------------

  std::vector<LandmarkDescriptorSet> landmark_descriptors;

  std::unordered_map<dyno::TrackletId, size_t> landmark_lookup;

  auto collectDescriptors = [&](const ViFrame::Ptr& keyframe) {
    if (!keyframe || !keyframe->hasDescriptors()) {
      return;
    }

    CHECK_EQ(keyframe->descriptorSize(), descriptor_size);

    for (size_t i = 0; i < keyframe->numKeypoints(); ++i) {
      const dyno::TrackletId tid = keyframe->ids[i];

      MapPoint map_point;

      if (!estimator.getMapPoint(tid, map_point)) {
        continue;
      }

      const auto [valid, descriptor] = keyframe->getDescriptorByIndex(i);

      if (!valid || descriptor == nullptr) {
        continue;
      }

      // Find or create the landmark descriptor set.
      auto it = landmark_lookup.find(tid);

      if (it == landmark_lookup.end()) {
        const size_t index = landmark_descriptors.size();

        landmark_lookup.emplace(tid, index);

        landmark_descriptors.push_back({tid, {}});

        it = landmark_lookup.find(tid);
      }

      auto& observations = landmark_descriptors[it->second].observations;

      // Don't store duplicate observations from the same frame.
      const bool already_observed =
          std::any_of(observations.begin(), observations.end(),
                      [&](const LandmarkDescriptorObservation& observation) {
                        return observation.frame_id == keyframe->getFrameId();
                      });

      if (already_observed) {
        continue;
      }

      LandmarkDescriptorObservation observation;
      observation.frame_id = keyframe->getFrameId();
      observation.descriptor.assign(descriptor, descriptor + descriptor_size);

      observations.push_back(std::move(observation));

      // Retain the most recent observations.
      if (observations.size() > kMaxObservationsPerLandmark) {
        observations.erase(observations.begin());
      }
    }
  };

  // Oldest first, latest second.
  // The latest descriptor is therefore included when a landmark appears
  // in both keyframes.
  collectDescriptors(oldest_kf);

  if (latest_fid != oldest_fid) {
    collectDescriptors(latest_kf);
  }

  if (landmark_descriptors.empty()) {
    return;
  }

  // ---------------------------------------------------------------------------
  // 3. Build KD-tree over the new frame's keypoints
  // ---------------------------------------------------------------------------

  gtsam::Point2Vector keypoint_locations;
  keypoint_locations.reserve(frame->numKeypoints());

  for (size_t i = 0; i < frame->numKeypoints(); ++i) {
    gtsam::Point2 pt;
    frame->getKeypointByIndex(i, pt);
    keypoint_locations.push_back(pt);
  }

  CHECK_EQ(keypoint_locations.size(), frame->numKeypoints());

  const KeypointSpatialIndex spatial_index(keypoint_locations);

  // ---------------------------------------------------------------------------
  // 4. Project landmarks and perform spatially constrained matching
  // ---------------------------------------------------------------------------

  struct BestTwoMatches {
    int best_distance = std::numeric_limits<int>::max();
    int second_best_distance = std::numeric_limits<int>::max();

    dyno::TrackletId best_landmark_id{};
    bool has_best = false;
  };

  // Record all keyframes in which each landmark/tracklet ID was observed.
  std::unordered_map<dyno::TrackletId, std::unordered_set<dyno::FrameId>>
      observation_frames;

  // recomputing covisability again...
  std::set<FrameId> keyframes_and_current = keyframes;
  // add which ever frames are used for BA!
  // TODO: this should just be a function insoide VioEstimator! (also allows us
  // to retrieve frames involved in the ba!)
  keyframes_and_current.insert(frame->getFrameId());
  keyframes_and_current.insert(frame->getFrameId() - 1u);
  for (const dyno::FrameId fid : keyframes_and_current) {
    const ViFrame::Ptr keyframe = estimator.getFramePtr(fid);
    LOG(INFO) << "Adding fid= " << fid << " to observration frame check";

    for (const dyno::TrackletId id : keyframe->ids) {
      observation_frames[id].insert(fid);
    }
  }
  // Reject associations if the two IDs have ever been observed together
  // in a keyframe. Such observations indicate distinct landmarks.
  const auto coObservedInKeyframe = [&observation_frames](
                                        dyno::TrackletId id_a,
                                        dyno::TrackletId id_b) {
    const auto it_a = observation_frames.find(id_a);
    const auto it_b = observation_frames.find(id_b);

    if (it_a == observation_frames.end() || it_b == observation_frames.end()) {
      return false;
    }

    // Iterate over the smaller set for efficiency.
    const auto* smaller = &it_a->second;
    const auto* larger = &it_b->second;

    if (smaller->size() > larger->size()) {
      std::swap(smaller, larger);
    }

    for (const dyno::FrameId fid : *smaller) {
      if (larger->count(fid)) {
        return true;
      }
    }

    return false;
  };

  // Indexed by original frame keypoint index.
  std::vector<BestTwoMatches> best_matches(frame->numKeypoints());
  const gtsam::Pose3 T_W_C = estimator.getPose(frame->getFrameId());
  const gtsam::Pose3 T_C_W = T_W_C.inverse();

  const BoundsChecker bounds_checker(container.objectMotionMask());

  std::unordered_map<dyno::TrackletId, Keypoint> projected_landmarks;
  projected_landmarks.reserve(landmark_descriptors.size());

  const auto& cam_params = camera_->getParams();
  const double vfov = cam_params.vFov();
  const double hfov = cam_params.hFov();

  const double max_rad_fov = (hfov > vfov) ? std::atan(hfov) : std::atan(vfov);
  const double view_th = std::cos(max_rad_fov);

  for (const auto& landmark_set : landmark_descriptors) {
    MapPoint map_point;

    if (!estimator.getMapPoint(landmark_set.landmark_id, map_point)) {
      continue;
    }

    // avoid trying to match against landmarks that have a tracked point in them
    // this helps avoid the case where a well tracked feature and an initalised
    // landmark gets matched aginast a different tracklet!
    if (frame->observedLandmark(landmark_set.landmark_id)) {
      continue;
    }

    // Transform world landmark into the new camera frame.
    const gtsam::Point3 p_C = T_C_W * map_point.p_W;

    // Reject points behind or too close to the camera.
    if (p_C.z() <= kMinDepth) {
      continue;
    }

    const double view_angle = p_C.z() / p_C.norm();
    if (std::abs(view_angle) < view_th) {
      continue;
    }

    // Project into the image.
    Keypoint projected_keypoint;
    camera_->project(p_C, &projected_keypoint);

    if (!bounds_checker(projected_keypoint)) {
      continue;
    }

    // Find nearby keypoints.
    const auto nearby_indices =
        spatial_index.radiusSearch(projected_keypoint, kSearchRadius);

    // maybe expand search radius and try again?
    if (nearby_indices.empty()) {
      continue;
    }

    projected_landmarks.emplace(landmark_set.landmark_id, projected_keypoint);

    // Compare against each nearby keypoint.
    for (const size_t keypoint_idx : nearby_indices) {
      // check none of these are already in the map?
      dyno::TrackletId tid = frame->ids[keypoint_idx];
      // for now skip already mapped keypoints since we only initalise
      // map-points on a keyframe!
      if (estimator.landmarkExists(tid)) {
        continue;
      }

      // reject candiates that have already been observed alongside the
      // projected map landmark in any historical keyframe.
      if (coObservedInKeyframe(tid, landmark_set.landmark_id)) {
        continue;
      }

      const auto [valid, query_descriptor] =
          frame->getDescriptorByIndex(keypoint_idx);

      if (!valid || query_descriptor == nullptr) {
        continue;
      }

      // Find the best descriptor distance across all observations
      // of this landmark.
      int landmark_distance = std::numeric_limits<int>::max();

      for (const auto& observation : landmark_set.observations) {
        const int distance = hammingDistance(
            query_descriptor, observation.descriptor.data(), descriptor_size);

        landmark_distance = std::min(landmark_distance, distance);
      }

      if (landmark_distance > max_hamming_distance) {
        continue;
      }

      // Update the best two DISTINCT landmark candidates for this
      // new-frame keypoint.
      auto& best = best_matches[keypoint_idx];

      if (landmark_distance < best.best_distance) {
        best.second_best_distance = best.best_distance;
        best.best_distance = landmark_distance;
        best.best_landmark_id = landmark_set.landmark_id;
        best.has_best = true;

      } else if (landmark_distance < best.second_best_distance) {
        best.second_best_distance = landmark_distance;
      }
    }
  }

  // ---------------------------------------------------------------------------
  // 5. Ratio test
  // ---------------------------------------------------------------------------

  struct AcceptedMatch {
    size_t keypoint_idx;
    dyno::TrackletId landmark_id;
    int distance;
  };

  std::vector<AcceptedMatch> accepted_matches;

  for (size_t i = 0; i < best_matches.size(); ++i) {
    const auto& best = best_matches[i];

    if (!best.has_best) {
      continue;
    }

    // Ratio test, when a second distinct landmark candidate exists.
    if (best.second_best_distance != std::numeric_limits<int>::max()) {
      if (best.best_distance >= kRatioThreshold * best.second_best_distance) {
        continue;
      }
    }

    accepted_matches.push_back({i, best.best_landmark_id, best.best_distance});
  }

  // ---------------------------------------------------------------------------
  // 6. Enforce one-to-one associations
  // ---------------------------------------------------------------------------

  std::sort(accepted_matches.begin(), accepted_matches.end(),
            [](const AcceptedMatch& a, const AcceptedMatch& b) {
              return a.distance < b.distance;
            });

  std::unordered_set<size_t> matched_keypoints;
  std::unordered_set<dyno::TrackletId> matched_landmarks;

  for (const auto& match : accepted_matches) {
    if (!matched_keypoints.insert(match.keypoint_idx).second) {
      continue;
    }

    if (!matched_landmarks.insert(match.landmark_id).second) {
      continue;
    }

    matches.push_back({static_cast<Index>(match.keypoint_idx),
                       match.landmark_id, match.distance});
  }

  cv::Mat debug_image = container.rgb().clone();

  CHECK_EQ(debug_image.type(), CV_8UC3);

  // make copy of orignial allowing merging of lmk ids
  const std::vector<dyno::TrackletId> original_tids = frame->ids;
  for (const auto& match : matches) {
    const size_t keypoint_idx = static_cast<size_t>(match.keypoint_index);

    CHECK_LT(keypoint_idx, frame->numKeypoints());

    MapPoint map_point;

    if (!estimator.getMapPoint(match.landmark_id, map_point)) {
      continue;
    }

    const auto projected_it = projected_landmarks.find(match.landmark_id);

    if (projected_it == projected_landmarks.end()) {
      continue;
    }

    const Keypoint& projected = projected_it->second;
    const gtsam::Point2& observed = keypoint_locations[keypoint_idx];

    const cv::Point projected_pt(cvRound(projected.x()),
                                 cvRound(projected.y()));

    const cv::Point observed_pt(cvRound(observed.x()), cvRound(observed.y()));

    // Colour corresponds to the map point's object ID.
    const dyno::Color color = dyno::Color::uniqueId(map_point.id);

    const cv::Scalar cv_color(color.b, color.g, color.r);

    // Draw the projected map point.
    cv::circle(debug_image, projected_pt, 5, cv_color, 2, cv::LINE_AA);

    // Draw the observed keypoint.
    cv::circle(debug_image, observed_pt, 4, cv_color, -1, cv::LINE_AA);

    // Connect projection and observation.
    cv::line(debug_image, projected_pt, observed_pt, cv_color, 1, cv::LINE_AA);

    // A tracklet ID different from the map point ID indicates that
    // the frontend track has not yet been associated with this map point.
    if (original_tids[keypoint_idx] != map_point.id) {
      // ++stats.tracklet_id_mismatches;

      // merge landmarks
      // we assume we only need to iterate since the last keyframe to the
      // current frame to reset keypoints
      dyno::TrackletId from_id = original_tids[keypoint_idx];
      dyno::TrackletId to_id = map_point.id;

      VLOG(30) << "Matched VO keypoint: " << from_id << " -> " << to_id;

      // need to do it for all these frames becuase the BA will include the most
      // recent frames
      //  too not just the keyframes
      //  for(dyno::FrameId k0 = latest_fid; k0 <= frame->getFrameId(); k0++) {

      // hack for now! dont update all frames just the ones that will be in the
      // next bundle adjustmnet this will result in inconsistent ids in older
      // frames
      for (dyno::FrameId k0 : keyframes_and_current) {
        auto vi_frame = estimator.getFramePtr(k0);
        auto& idx_map = vi_frame->local_indices;

        const auto from_it = idx_map.find(from_id);
        const auto to_it = idx_map.find(to_id);

        // TODO: not a permanent slution as if both IDS exist in the same frame,
        // we will need
        //  somehow to resovle diplicate observations and retain one!

        if (from_it != idx_map.end()) {
          if (to_it != idx_map.end()) {
            // we have a match but this frame has somehow seen both ids!
            auto from_idx = from_it->second;
            auto to_idx = to_it->second;

            Keypoint from_kp;
            vi_frame->getKeypointByIndex(from_idx, from_kp);
            Keypoint to_kp;
            vi_frame->getKeypointByIndex(to_idx, to_kp);

            std::cout << "from_id: " << from_id << ", index: " << from_idx
                      << ", measurement: " << from_kp.transpose() << std::endl;

            std::cout << "to_id: " << to_id << ", index: " << to_idx
                      << ", measurement: " << to_kp.transpose() << std::endl;
          }
          CHECK(to_it == idx_map.end())
              << "Cannot merge tracklets: both IDs already exist in frame "
              << k0 << ". from_id=" << from_id << ", to_id=" << to_id;

          const Index index = from_it->second;

          idx_map.erase(from_it);
          idx_map.emplace(to_id, index);

          CHECK_EQ(vi_frame->ids[index], from_id);
          vi_frame->ids[index] = to_id;
        }
      }

      merged_lmks[from_id] = to_id;
      // merge observations of from into to, and remove the from landmark from
      // the map!
      estimator.mergeLandmark(from_id, to_id);

      // Red cross around the observed keypoint.
      cv::drawMarker(debug_image, observed_pt, cv::Scalar(0, 0, 255),
                     cv::MARKER_CROSS, 14, 2, cv::LINE_AA);
    }
  }

  // stats.accepted_matches = matches.size();

  cv::imshow("VIO Map Matches", debug_image);
  cv::waitKey(1);
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

size_t Frontend::temporalVioTriangultion(
    VioGraphEstimator& estimator, ViFrame::Ptr frame,
    std::unordered_set<KeypointIdentifier>& new_measurements) {
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
  new_measurements.insert(observations.begin(), observations.end());
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

void AyncVO::markReadyButInvalidAndNotify() {
  {
    std::lock_guard<std::mutex> lock(vo_mutex);
    vo_ready = true;
    vo_valid = false;
  }
  vo_cv.notify_all();
}

void AyncVO::markReadyAndValidAndNotify(const gtsam::Pose3& pose) {
  {
    std::lock_guard<std::mutex> lock(vo_mutex);
    vo_ready = true;
    vo_valid = true;
    *vo = pose;
  }
  vo_cv.notify_all();
}

std::ostream& operator<<(std::ostream& os,
                         const VioMotionPropagation::Source& source) {
  static const std::unordered_map<VioMotionPropagation::Source, std::string>
      map = {{VioMotionPropagation::Source::PRIOR_MOTION, "Prior Motion"},
             {VioMotionPropagation::Source::IMU, "IMU"}};
  os << map.at(source);
  return os;
}

}  // namespace dyno2
