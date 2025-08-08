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

#include "dynosam/frontend/RGBDInstanceFrontendModule.hpp"

#include <glog/logging.h>

#include <opencv4/opencv2/opencv.hpp>

#include "dynosam/common/Flags.hpp"  //for common flags
#include "dynosam/frontend/RGBDInstance-Definitions.hpp"
#include "dynosam/frontend/vision/MotionSolver.hpp"
#include "dynosam/frontend/vision/Vision-Definitions.hpp"
#include "dynosam/logger/Logger.hpp"
#include "dynosam/utils/OpenCVUtils.hpp"
#include "dynosam/utils/SafeCast.hpp"
#include "dynosam/utils/TimingStats.hpp"

DEFINE_bool(use_frontend_logger, false,
            "If true, the frontend logger will be used");
DEFINE_bool(use_dynamic_track, true,
            "If true, the dynamic tracking will be used");

DEFINE_bool(log_projected_masks, false,
            "If true, projected masks will be saved at every frame");

namespace dyno {

RGBDInstanceFrontendModule::RGBDInstanceFrontendModule(
    const DynoParams& params, Camera::Ptr camera,
    ImageDisplayQueue* display_queue)
    : FrontendModule(params, display_queue),
      camera_(camera),
      motion_solver_(params.frontend_params_.ego_motion_solver_params,
                     camera->getParams()),
      imu_frontend_(params.frontend_params_.imu_params) {
  CHECK_NOTNULL(camera_);
  tracker_ = std::make_unique<FeatureTracker>(getFrontendParams(), camera_,
                                              display_queue);

  if (FLAGS_use_frontend_logger) {
    LOG(INFO) << "Using front-end logger!";
    logger_ = std::make_unique<RGBDFrontendLogger>();
  }

  ObjectMotionSovlerF2F::Params object_motion_solver_params =
      getFrontendParams().object_motion_solver_params;
  // add ground truth hook
  object_motion_solver_params.ground_truth_packets_request = [&]() {
    return this->shared_module_info.getGroundTruthPackets();
  };
  object_motion_solver_params.refine_motion_with_3d = false;

  object_motion_solver_ = std::make_unique<ObjectMotionSovlerF2F>(
      object_motion_solver_params, camera->getParams());
}

RGBDInstanceFrontendModule::~RGBDInstanceFrontendModule() {
  if (FLAGS_save_frontend_json) {
    LOG(INFO) << "Saving frontend output as json";
    const std::string file_path =
        getOutputFilePath(kRgbdFrontendOutputJsonFile);
    // JsonConverter::WriteOutJson(output_packet_record_, file_path,
    //                             JsonConverter::Format::BSON);
  }
}

FrontendModule::ImageValidationResult
RGBDInstanceFrontendModule::validateImageContainer(
    const ImageContainer::Ptr& image_container) const {
  return ImageValidationResult(image_container->hasDepth(),
                               "Depth is required");
}

FrontendModule::SpinReturn RGBDInstanceFrontendModule::boostrapSpin(
    FrontendInputPacketBase::ConstPtr input) {
  ImageContainer::Ptr image_container = input->image_container_;

  std::set<ObjectId> keyframed_objects;
  Frame::Ptr frame = tracker_->track(input->getFrameId(), input->getTimestamp(),
                                     *image_container, keyframed_objects);
  (void)keyframed_objects;
  CHECK(frame->updateDepths());

  return {State::Nominal, nullptr};
}

FrontendModule::SpinReturn RGBDInstanceFrontendModule::nominalSpin(
    FrontendInputPacketBase::ConstPtr input) {
  ImageContainer::Ptr image_container = input->image_container_;

  const bool has_imu = input->imu_measurements.has_value();
  const bool has_stereo = image_container->hasRightRgb();

  std::optional<gtsam::Rot3> R_curr_ref;
  ImuFrontend::PimPtr pim;
  if (has_imu) {
    pim = imu_frontend_.preintegrateImuMeasurements(
        input->imu_measurements.value());

    nav_state_ =
        pim->predict(previous_nav_state_, gtsam::imuBias::ConstantBias{});

    last_imu_nav_state_update_ = input->getFrameId();

    R_curr_ref =
        previous_nav_state_.attitude().inverse() * nav_state_.attitude();
  }

  std::set<ObjectId> keyframed_objects;
  Frame::Ptr frame =
      tracker_->track(input->getFrameId(), input->getTimestamp(),
                      *image_container, keyframed_objects, R_curr_ref);

  Frame::Ptr previous_frame = tracker_->getPreviousFrame();
  CHECK(previous_frame);

  VLOG(1) << to_string(tracker_->getTrackerInfo());

  {
    utils::TimingStatsCollector update_depths_timer("depth_updater");
    frame->updateDepths();
  }

  bool stereo_result = false;
  static const double base_line = 0.05;  // VIODE
  // static const double base_line = 0.12;  // ZED

  if (has_stereo) {
    const cv::Mat& left_rgb = image_container->rgb();
    const cv::Mat& right_rgb = image_container->rightRgb();

    FeaturePtrs stereo_features_1;
    stereo_result =
        tracker_->stereoTrack(stereo_features_1, frame->static_features_,
                              left_rgb, right_rgb, base_line);
  }

  // this includes the refine correspondances with joint optical flow
  // TODO: lots of internal logic around how the actual pose gets predicted.
  // should streamline this and tell backend how pose was selected!!
  if (!solveCameraMotion(frame, previous_frame, R_curr_ref)) {
    LOG(ERROR) << "Could not solve for camera";
  }

  if (has_stereo && stereo_result) {
    // need to match aagain after optical flow used to update the keypoints
    // wow this seems to make a pretty big difference!!
    const cv::Mat& left_rgb = image_container->rgb();
    const cv::Mat& right_rgb = image_container->rightRgb();
    FeaturePtrs stereo_features_2;
    stereo_result &=
        tracker_->stereoTrack(stereo_features_2, frame->static_features_,
                              left_rgb, right_rgb, base_line);
  }

  // VERY important calculation
  const gtsam::Pose3 T_k_1_k =
      previous_nav_state_.pose().inverse() * frame->T_world_camera_;
  vo_velocity_ = T_k_1_k;

  // we currently use the frame pose as the nav state - this value can come from
  // either the VO OR the IMU, depending on the result from the
  // solveCameraMotion this is only relevant since we dont solve incremental so
  // the backend is not immediately updating the frontend at which point we can
  // just use the best estimate in the case of the VO, the nav_state velocity
  // will be wrong (currently!!)
  previous_nav_state_ =
      gtsam::NavState(frame->T_world_camera_, nav_state_.velocity());

  if (R_curr_ref) {
    imu_frontend_.resetIntegration();
  }

  if (FLAGS_use_dynamic_track) {
    utils::TimingStatsCollector track_dynamic_timer("tracking_dynamic");
    vision_tools::trackDynamic(getFrontendParams(), *previous_frame, frame);
  }

  const auto [object_motions, object_poses] =
      object_motion_solver_->solve(frame, previous_frame);

  VisionImuPacket::Ptr vision_imu_packet = std::make_shared<VisionImuPacket>();
  vision_imu_packet->frameId(frame->getFrameId());
  vision_imu_packet->timestamp(frame->getTimestamp());
  vision_imu_packet->pim(pim);
  vision_imu_packet->groundTruthPacket(input->optional_gt_);
  fillOutputPacketWithTracks(vision_imu_packet, *frame, T_k_1_k, object_motions,
                             object_poses);

  DebugImagery debug_imagery;
  debug_imagery.tracking_image =
      createTrackingImage(frame, previous_frame, object_poses);
  const ImageContainer& processed_image_container = frame->image_container_;
  debug_imagery.rgb_viz =
      ImageType::RGBMono::toRGB(processed_image_container.rgb());
  debug_imagery.flow_viz =
      ImageType::OpticalFlow::toRGB(processed_image_container.opticalFlow());
  debug_imagery.mask_viz = ImageType::MotionMask::toRGB(
      processed_image_container.objectMotionMask());
  debug_imagery.depth_viz =
      ImageType::Depth::toRGB(processed_image_container.depth());

  if (display_queue_)
    display_queue_->push(
        ImageToDisplay("Tracks", debug_imagery.tracking_image));

  debug_imagery.detected_bounding_boxes = frame->drawDetectedObjectBoxes();
  vision_imu_packet->debugImagery(debug_imagery);

  // // const cv::Mat& board_detection_mask =
  // tracker_->getBoarderDetectionMask(); PointCloudLabelRGB::Ptr
  // dense_labelled_cloud =
  //     frame->projectToDenseCloud(&board_detection_mask);
  PointCloudLabelRGB::Ptr dense_labelled_cloud = nullptr;

  // if (FLAGS_save_frontend_json)
  //   output_packet_record_.insert({output->getFrameId(), output});

  sendToFrontendLogger(frame, vision_imu_packet);
  // if (FLAGS_log_projected_masks)
  //   vision_tools::writeOutProjectMaskAndDepthMap(
  //       frame->image_container_.depth(),
  //       frame->image_container_.objectMotionMask(), *frame->getCamera(),
  //       frame->getFrameId());

  return {State::Nominal, vision_imu_packet};
}

bool RGBDInstanceFrontendModule::solveCameraMotion(
    Frame::Ptr frame_k, const Frame::Ptr& frame_k_1,
    std::optional<gtsam::Rot3> R_curr_ref) {
  utils::TimingStatsCollector timer("frontend.solve_camera_motion");
  Pose3SolverResult result;

  const auto& frontend_params = getFrontendParams();
  if (frontend_params.use_ego_motion_pnp) {
    result = motion_solver_.geometricOutlierRejection3d2d(frame_k_1, frame_k,
                                                          R_curr_ref);
  } else {
    // TODO: untested
    LOG(FATAL) << "Not tested";
    // result = motion_solver_.geometricOutlierRejection3d3d(frame_k_1,
    // frame_k);
  }

  VLOG(15) << (frontend_params.use_ego_motion_pnp ? "3D2D" : "3D3D")
           << "camera pose estimate at frame " << frame_k->frame_id_
           << (result.status == TrackingStatus::VALID ? " success "
                                                      : " failure ")
           << ":\n"
           << "- Tracking Status: " << to_string(result.status) << '\n'
           << "- Total Correspondences: "
           << result.inliers.size() + result.outliers.size() << '\n'
           << "\t- # inliers: " << result.inliers.size() << '\n'
           << "\t- # outliers: " << result.outliers.size() << '\n';

  // collect all usable tracklets
  TrackletIds tracklets = frame_k->static_features_.collectTracklets();
  CHECK_GE(tracklets.size(),
           result.inliers.size() +
               result.outliers.size());  // tracklets shoudl be more (or same
                                         // as) correspondances as there will
                                         // be new points untracked
  frame_k->static_features_.markOutliers(result.outliers);

  // was 60!
  if (result.status != TrackingStatus::VALID || result.inliers.size() < 30) {
    // TODO: fix code structure - nav state should be passed in?
    // use nav state which we assume is updated by IMU
    std::stringstream ss;
    ss << "Number usable static feature < 30 or status is invalid. ";

    // check if we have a nav state update from the IMU (this is also a cheap
    // way of checking that we HAVE an imu). If we do we can use the nav state
    // directly to update the current pose as the nav state is the forward
    // prediction from the IMU
    if (last_imu_nav_state_update_ == frame_k_1->getFrameId()) {
      frame_k->T_world_camera_ = nav_state_.pose();
      ss << "Nav state was previous updated with IMU. Using predicted pose to "
            "set camera transform; k"
         << frame_k->getFrameId();
    } else {
      // no IMU for forward prediction, use constant velocity model to propogate
      // pose expect previous_nav_state_ to always be updated with the best
      // pose!
      frame_k->T_world_camera_ = previous_nav_state_.pose() * vo_velocity_;
      ss << "Nav state has no information from imu. Using constant velocity "
            "model to propofate pose; k"
         << frame_k->getFrameId();
    }

    VLOG(10) << ss.str();

    // if fails should we mark current inliers as outliers?

    // TODO: should almost definitely do this in future, but right now we use
    // measurements to construct a framenode in the backend so if there are no
    // measurements we get a frame_node null.... for now... make hack and set
    // all ages of inliers to 1!!! since we need n measurements in the backend
    // this will ensure that they dont get added to the
    //  optimisation problem but will get added to the map...
    for (const auto& inlier : result.inliers) {
      frame_k->static_features_.getByTrackletId(inlier)->age(1u);
    }
    // frame_k->static_features_.markOutliers(result.inliers);

    // for some reason using tracklets to mark all features gives error as a
    // tracklet id
    // seems to be not actually in static features. Dont know why
    // maybe remeber that tracklets (fromc collectTracklets) is actually just
    // the usable tracklets...?
    // frame_k->static_features_.markOutliers(tracklets);
    return false;
  } else {
    frame_k->T_world_camera_ = result.best_result;

    if (frontend_params.refine_camera_pose_with_joint_of) {
      VLOG(10) << "Refining camera pose with joint of";
      OpticalFlowAndPoseOptimizer flow_optimizer(
          frontend_params.object_motion_solver_params.joint_of_params);

      auto flow_opt_result = flow_optimizer.optimizeAndUpdate<CalibrationType>(
          frame_k_1, frame_k, result.inliers, result.best_result);
      frame_k->T_world_camera_ = flow_opt_result.best_result.refined_pose;
      VLOG(15) << "Refined camera pose with optical flow - error before: "
               << flow_opt_result.error_before.value_or(NaN)
               << " error_after: " << flow_opt_result.error_after.value_or(NaN);
    }
    return true;
  }
}

void RGBDInstanceFrontendModule::fillOutputPacketWithTracks(
    VisionImuPacket::Ptr vision_imu_packet, const Frame& frame,
    const gtsam::Pose3& T_k_1_k, const ObjectMotionMap& object_motions,
    const ObjectPoseMap& object_poses) const {
  CHECK(vision_imu_packet);
  const auto frame_id = frame.getFrameId();
  // construct image tracks
  const double& static_pixel_sigma =
      params_.backend_params_.static_pixel_noise_sigma;
  const double& static_point_sigma =
      params_.backend_params_.static_point_noise_sigma;

  const double& dynamic_pixel_sigma =
      params_.backend_params_.dynamic_pixel_noise_sigma;
  const double& dynamic_point_sigma =
      params_.backend_params_.dynamic_point_noise_sigma;

  gtsam::Vector2 static_pixel_sigmas;
  static_pixel_sigmas << static_pixel_sigma, static_pixel_sigma;

  gtsam::Vector2 dynamic_pixel_sigmas;
  dynamic_pixel_sigmas << dynamic_pixel_sigma, dynamic_pixel_sigma;

  auto& camera = *this->camera_;
  auto fill_camera_measurements =
      [&camera](FeatureFilterIterator it,
                CameraMeasurementStatusVector* measurements, FrameId frame_id,
                const gtsam::Vector2& pixel_sigmas, double depth_sigma) {
        for (const Feature::Ptr& f : it) {
          const TrackletId tracklet_id = f->trackletId();
          const Keypoint& kp = f->keypoint();
          const ObjectId object_id = f->objectId();
          CHECK_EQ(f->objectId(), object_id);
          CHECK(Feature::IsUsable(f));

          MeasurementWithCovariance<Keypoint> kp_measurement =
              MeasurementWithCovariance<Keypoint>::FromSigmas(kp, pixel_sigmas);
          CameraMeasurement camera_measurement(kp_measurement);

          // This can come from either stereo or rgbd
          if (f->hasDepth()) {
            MeasurementWithCovariance<Landmark> landmark_measurement(
                // assume sigma_u and sigma_v are identical
                vision_tools::backProjectAndCovariance(
                    *f, camera, pixel_sigmas(0), depth_sigma));
            camera_measurement.landmark(landmark_measurement);
          }

          if (f->hasRightKeypoint()) {
            CHECK(f->hasDepth())
                << "Right keypoint set for feature but no depth!";
            MeasurementWithCovariance<Keypoint> right_kp_measurement =
                MeasurementWithCovariance<Keypoint>::FromSigmas(
                    f->rightKeypoint(), pixel_sigmas);
            camera_measurement.rightKeypoint(right_kp_measurement);
          }

          if (f->keypointType() == KeyPointType::STATIC) {
            CHECK_EQ(object_id, background_label);
          } else {
            CHECK_NE(object_id, background_label);
          }

          measurements->push_back(
              CameraMeasurementStatus(camera_measurement, frame_id, tracklet_id,
                                      object_id, ReferenceFrame::LOCAL));
        }
      };

  // TODO: fill ttracking status?
  VisionImuPacket::CameraTracks camera_tracks;
  auto* static_measurements = &camera_tracks.measurements;
  fill_camera_measurements(frame.usableStaticFeaturesBegin(),
                           static_measurements, frame_id, static_pixel_sigmas,
                           static_point_sigma);
  camera_tracks.X_W_k = frame.getPose();
  camera_tracks.T_k_1_k = T_k_1_k;
  vision_imu_packet->cameraTracks(camera_tracks);

  // First collect all dynamic measurements then split them by object
  // This is a bit silly
  CameraMeasurementStatusVector dynamic_measurements;
  fill_camera_measurements(frame.usableDynamicFeaturesBegin(),
                           &dynamic_measurements, frame_id,
                           dynamic_pixel_sigmas, dynamic_point_sigma);

  VisionImuPacket::ObjectTrackMap object_tracks;
  // motions in this frame (ie. new motions!!)
  MotionEstimateMap motion_estimates = object_motions.toEstimateMap(frame_id);
  auto pose_estimates = object_poses.toEstimateMap(frame_id);

  // fill object tracks based on valid motions
  for (const auto& [object_id, motion_reference_estimate] : motion_estimates) {
    CHECK(pose_estimates.exists(object_id))
        << "Object pose missing " << info_string(frame_id, object_id)
        << " but frontend motion available";
    const auto& L_W_k = pose_estimates.at(object_id);

    VisionImuPacket::ObjectTracks object_track;
    object_track.H_W_k_1_k = motion_reference_estimate;
    object_track.L_W_k = L_W_k;
    object_tracks.insert2(object_id, object_track);
  }

  for (const auto& dm : dynamic_measurements) {
    const auto& object_id = dm.objectId();
    // throw out features detected on objects where the tracking failed
    if (object_tracks.exists(object_id)) {
      VisionImuPacket::ObjectTracks& object_track = object_tracks.at(object_id);
      object_track.measurements.push_back(dm);
    }
  }
  vision_imu_packet->objectTracks(object_tracks);
}

void RGBDInstanceFrontendModule::sendToFrontendLogger(
    const Frame::Ptr& frame, const VisionImuPacket::Ptr& vision_imu_packet) {
  if (logger_) {
    auto ground_truths = this->shared_module_info.getGroundTruthPackets();
    logger_->logCameraPose(frame->getFrameId(), vision_imu_packet->cameraPose(),
                           ground_truths);
    logger_->logObjectMotion(frame->getFrameId(),
                             vision_imu_packet->objectMotions(), ground_truths);
    logger_->logObjectPose(frame->getFrameId(),
                           vision_imu_packet->objectPoses(), ground_truths);
    logger_->logTrackingLengthHistogram(frame);
    logger_->logPoints(frame->getFrameId(), vision_imu_packet->cameraPose(),
                       vision_imu_packet->dynamicLandmarkMeasurements());
    logger_->logFrameIdToTimestamp(frame->getFrameId(), frame->getTimestamp());
  }
}

cv::Mat RGBDInstanceFrontendModule::createTrackingImage(
    const Frame::Ptr& frame_k, const Frame::Ptr& frame_k_1,
    const ObjectPoseMap& object_poses) const {
  cv::Mat tracking_image = tracker_->computeImageTracks(
      *frame_k_1, *frame_k, getFrontendParams().image_tracks_vis_params);

  const auto& camera_params = camera_->getParams();
  const auto& K = camera_params.getCameraMatrix();
  const auto& D = camera_params.getDistortionCoeffs();

  const gtsam::Pose3& X_k = frame_k->getPose();

  // poses are expected to be in the world frame
  gtsam::FastMap<ObjectId, gtsam::Pose3> poses_k_map =
      object_poses.collectByFrame(frame_k->getFrameId());
  std::vector<gtsam::Pose3> poses_k_vec;
  std::transform(poses_k_map.begin(), poses_k_map.end(),
                 std::back_inserter(poses_k_vec),
                 [&X_k](const std::pair<ObjectId, gtsam::Pose3>& pair) {
                   // put object pose into the camera frame so it can be
                   // projected into the image
                   return X_k.inverse() * pair.second;
                 });

  // TODO: bring back when visualisation is unified with incremental solver!!
  //  utils::drawObjectPoseAxes(tracking_image, K, D, poses_k_vec);
  return tracking_image;
}

}  // namespace dyno
