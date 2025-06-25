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
    const FrontendParams& frontend_params, Camera::Ptr camera,
    ImageDisplayQueue* display_queue)
    : FrontendModule(frontend_params, display_queue),
      camera_(camera),
      motion_solver_(frontend_params.ego_motion_solver_params,
                     camera->getParams()) {
  // object_motion_solver_(frontend_params.object_motion_solver_params,
  //                       camera->getParams()) {

  CHECK_NOTNULL(camera_);
  tracker_ =
      std::make_unique<FeatureTracker>(frontend_params, camera_, display_queue);

  if (FLAGS_use_frontend_logger) {
    LOG(INFO) << "Using front-end logger!";
    logger_ = std::make_unique<RGBDFrontendLogger>();
  }

  // gtsam::ISAM2Params isam2_params;
  // isam2_params.keyFormatter = DynoLikeKeyFormatter;
  // isam2_params.evaluateNonlinearError = true;

  object_motion_solver_params_ =
      frontend_params.object_motion_solver_params;

  // add ground truth hook
  object_motion_solver_params_.ground_truth_packets_request = [&]() {
    return this->shared_module_info.getGroundTruthPackets();
  };
  object_motion_solver_params_.refine_motion_with_3d = false;

  // object_motion_solver_ = std::make_unique<ObjectMotionSolverSAM>(
  //     object_motion_solver_params, camera->getParams(), isam2_params);
  object_motion_solver_ = std::make_unique<ObjectMotionSovlerF2F>(
      object_motion_solver_params_, camera->getParams());

  ego_motion_solver_params_ = frontend_params.ego_motion_solver_params;

}

RGBDInstanceFrontendModule::~RGBDInstanceFrontendModule() {
  if (FLAGS_save_frontend_json) {
    LOG(INFO) << "Saving frontend output as json";
    const std::string file_path =
        getOutputFilePath(kRgbdFrontendOutputJsonFile);
    JsonConverter::WriteOutJson(output_packet_record_, file_path,
                                JsonConverter::Format::BSON);
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
  Frame::Ptr frame = tracker_->track(input->getFrameId(), input->getTimestamp(),
                                     *image_container);


  // gtsam::Rot3(R[0],)
    
  std::vector<double> initial_R_world_camera = ego_motion_solver_params_.initial_R_world_camera;
  std::vector<double> initial_t_world_camera = ego_motion_solver_params_.initial_t_world_camera;

  // gtsam::Rot3 R = gtsam::Rot3::RzRyRx(0.0, 0, 0.0);

  gtsam::Rot3 R = gtsam::Rot3(initial_R_world_camera[0],initial_R_world_camera[1],initial_R_world_camera[2],
                              initial_R_world_camera[3],initial_R_world_camera[4],initial_R_world_camera[5],
                              initial_R_world_camera[6],initial_R_world_camera[7],initial_R_world_camera[8]);

  gtsam::Point3 t = {initial_t_world_camera[0],initial_t_world_camera[1],initial_t_world_camera[2]};
  // std::cout << "T_world_camera rotation: " << R << "T_world_camera translation: " <<  t << std::endl;

  frame->T_world_camera_ = gtsam::Pose3(R, t);
  CHECK(frame->updateDepths());

  return {State::Nominal, nullptr};
}

FrontendModule::SpinReturn RGBDInstanceFrontendModule::nominalSpin(
    FrontendInputPacketBase::ConstPtr input) {
  ImageContainer::Ptr image_container = input->image_container_;

  Frame::Ptr frame = nullptr;
  {
    utils::TimingStatsCollector tracking_timer("tracking_timer");
    frame = tracker_->track(input->getFrameId(), input->getTimestamp(),
                            *image_container);
  }
  CHECK(frame);

  Frame::Ptr previous_frame = tracker_->getPreviousFrame();
  CHECK(previous_frame);

  VLOG(1) << to_string(tracker_->getTrackerInfo());

  {
    utils::TimingStatsCollector update_depths_timer("depth_updater");
    frame->updateDepths();
  }
  // updates frame->T_world_camera_
  if (!solveCameraMotion(frame, previous_frame)) {
    LOG(ERROR) << "Could not solve for camera";
  }

  if (FLAGS_use_dynamic_track) {
    // TODO: bring back byte tracker??
    utils::TimingStatsCollector track_dynamic_timer("tracking_dynamic");
    vision_tools::trackDynamic(base_params_, *previous_frame, frame);
  }

  ObjectMotionMap object_motions;
  ObjectPoseMap object_poses;
  // MotionEstimateMap motion_estimates;
  std::tie(object_motions, object_poses) =
      object_motion_solver_->solve(frame, previous_frame);

  // motions in this frame (ie. new motions!!)
  MotionEstimateMap motion_estimates =
      object_motions.toEstimateMap(frame->getFrameId());

  const cv::Mat& board_detection_mask = tracker_->getBoarderDetectionMask();
  PointCloudLabelRGB::Ptr dense_labelled_cloud =
      frame->projectToDenseCloud(&board_detection_mask);
  // PointCloudLabelRGB::Ptr  dense_labelled_cloud = nullptr;

  if (logger_) {
    auto ground_truths = this->shared_module_info.getGroundTruthPackets();
    logger_->logCameraPose(frame->getFrameId(), frame->getPose(),
                           ground_truths);
    // TODO: this is now wrong as we dont log all the corrected motions!!!!
    logger_->logObjectMotion(frame->getFrameId(), motion_estimates,
                             ground_truths);
    logger_->logTrackingLengthHistogram(frame);
    logger_->logFrameIdToTimestamp(frame->getFrameId(), frame->getTimestamp());
  }

  DebugImagery debug_imagery;
  debug_imagery.tracking_image =
      createTrackingImage(frame, previous_frame, object_poses);
  if (display_queue_)
    display_queue_->push(
        ImageToDisplay("tracks", debug_imagery.tracking_image));

  debug_imagery.detected_bounding_boxes = frame->drawDetectedObjectBoxes();

  const ImageContainer& processed_image_container = frame->image_container_;
  debug_imagery.rgb_viz = ImageType::RGBMono::toRGB(
      processed_image_container.get<ImageType::RGBMono>());
  debug_imagery.flow_viz = ImageType::OpticalFlow::toRGB(
      processed_image_container.get<ImageType::OpticalFlow>());
  debug_imagery.mask_viz = ImageType::MotionMask::toRGB(
      processed_image_container.get<ImageType::MotionMask>());
  debug_imagery.depth_viz = ImageType::Depth::toRGB(
      processed_image_container.get<ImageType::Depth>());

  // if (display_queue_)
  //   display_queue_->push(ImageToDisplay("Motion Mask",
  //   debug_imagery.mask_viz));

  RGBDInstanceOutputPacket::Ptr output = constructOutput(
      *frame, object_motions, object_poses, frame->T_world_camera_,
      input->optional_gt_, debug_imagery, dense_labelled_cloud);

  if (FLAGS_save_frontend_json)
    output_packet_record_.insert({output->getFrameId(), output});

  if (FLAGS_log_projected_masks)
    vision_tools::writeOutProjectMaskAndDepthMap(
        frame->image_container_.get<ImageType::Depth>(),
        frame->image_container_.get<ImageType::MotionMask>(),
        *frame->getCamera(), frame->getFrameId());

  if (logger_) {
    auto ground_truths = this->shared_module_info.getGroundTruthPackets();
    logger_->logPoints(output->getFrameId(), output->T_world_camera_,
                       output->dynamic_landmarks_);
    // object_poses_ are in frontend module
    logger_->logObjectPose(output->getFrameId(), object_poses, ground_truths);
    logger_->logObjectBbxes(output->getFrameId(), output->getObjectBbxes());
  }
  return {State::Nominal, output};
}

bool RGBDInstanceFrontendModule::solveCameraMotion(
    Frame::Ptr frame_k, const Frame::Ptr& frame_k_1) {
  Pose3SolverResult result;
  if (base_params_.use_ego_motion_pnp) {
    result = motion_solver_.geometricOutlierRejection3d2d(frame_k_1, frame_k);
  } else {
    // TODO: untested
    LOG(FATAL) << "Not tested";
    // result = motion_solver_.geometricOutlierRejection3d3d(frame_k_1,
    // frame_k);
  }

  VLOG(15) << (base_params_.use_ego_motion_pnp ? "3D2D" : "3D3D")
           << "camera pose estimate at frame " << frame_k->frame_id_
           << (result.status == TrackingStatus::VALID ? " success "
                                                      : " failure ")
           << ":\n"
           << "- Tracking Status: " << to_string(result.status) << '\n'
           << "- Total Correspondences: "
           << result.inliers.size() + result.outliers.size() << '\n'
           << "\t- # inliers: " << result.inliers.size() << '\n'
           << "\t- # outliers: " << result.outliers.size() << '\n';

  if (result.status == TrackingStatus::VALID) {
    frame_k->T_world_camera_ = result.best_result;
    TrackletIds tracklets = frame_k->static_features_.collectTracklets();
    CHECK_GE(tracklets.size(),
             result.inliers.size() +
                 result.outliers.size());  // tracklets shoudl be more (or same
                                           // as) correspondances as there will
                                           // be new points untracked
    frame_k->static_features_.markOutliers(result.outliers);

    if (base_params_.refine_camera_pose_with_joint_of) {
      VLOG(10) << "Refining camera pose with joint of";
      OpticalFlowAndPoseOptimizer flow_optimizer(
          base_params_.object_motion_solver_params.joint_of_params);

      auto flow_opt_result = flow_optimizer.optimizeAndUpdate<CalibrationType>(
          frame_k_1, frame_k, result.inliers, result.best_result);
      frame_k->T_world_camera_ = flow_opt_result.best_result.refined_pose;
      VLOG(15) << "Refined camera pose with optical flow - error before: "
               << flow_opt_result.error_before.value_or(NaN)
               << " error_after: " << flow_opt_result.error_after.value_or(NaN);
    }
    std::cerr << frame_k->T_world_camera_ << "T world camera valid track" << std::endl;
    return true;
  } else {
    frame_k->T_world_camera_ = gtsam::Pose3::Identity();
    std::cerr << frame_k->T_world_camera_ << "T world camera invalid track" << std::endl;
    return false;
  }
}

RGBDInstanceOutputPacket::Ptr RGBDInstanceFrontendModule::constructOutput(
    const Frame& frame, const ObjectMotionMap& object_motions,
    const ObjectPoseMap& object_poses, const gtsam::Pose3& T_world_camera,
    const GroundTruthInputPacket::Optional& gt_packet,
    const DebugImagery::Optional& debug_imagery,
    const PointCloudLabelRGB::Ptr dense_labelled_cloud) {
  StatusKeypointVector static_keypoint_measurements;
  StatusLandmarkVector static_landmarks;
  for (const Feature::Ptr& f : frame.usableStaticFeaturesBegin()) {
    const TrackletId tracklet_id = f->trackletId();
    const Keypoint kp = f->keypoint();
    CHECK(f->isStatic());
    CHECK(Feature::IsUsable(f));

    // dont include features that have only been seen once as we havent had a
    // chance to validate it yet
    if (f->age() < 1) {
      continue;
    }

    MeasurementWithCovariance<Keypoint> kp_measurement(kp);
    MeasurementWithCovariance<Landmark> landmark_measurement(
        vision_tools::backProjectAndCovariance(*f, *camera_, 0.2, 0.1));

    static_keypoint_measurements.push_back(KeypointStatus::StaticInLocal(
        kp_measurement, frame.getFrameId(), tracklet_id));

    static_landmarks.push_back(LandmarkStatus::StaticInLocal(
        landmark_measurement, frame.getFrameId(), tracklet_id));
  }

  StatusKeypointVector dynamic_keypoint_measurements;
  StatusLandmarkVector dynamic_landmarks;
  for (const auto& [object_id, obs] : frame.object_observations_) {
    CHECK_EQ(object_id, obs.instance_label_);
    // TODO: add back in?
    //  CHECK(obs.marked_as_moving_);

    for (const TrackletId tracklet : obs.object_features_) {
      if (frame.isFeatureUsable(tracklet)) {
        const Feature::Ptr f = frame.at(tracklet);
        CHECK(!f->isStatic());
        CHECK_EQ(f->objectId(), object_id);

        // dont include features that have only been seen once as we havent had
        // a chance to validate it yet
        if (f->age() < 1) {
          continue;
        }

        const TrackletId tracklet_id = f->trackletId();
        const Keypoint kp = f->keypoint();

        MeasurementWithCovariance<Keypoint> kp_measurement(kp);
        MeasurementWithCovariance<Landmark> landmark_measurement(
            vision_tools::backProjectAndCovariance(*f, *camera_, 0.2, 0.1));

        dynamic_keypoint_measurements.push_back(KeypointStatus::DynamicInLocal(
            kp_measurement, frame.frame_id_, tracklet_id, object_id));

        dynamic_landmarks.push_back(LandmarkStatus::DynamicInLocal(
            landmark_measurement, frame.frame_id_, tracklet_id, object_id));
      }
    }
  }

  // update trajectory of camera poses to be visualised by the frontend viz
  // module
  camera_poses_.push_back(T_world_camera);

  // create timestamps for the provided frames
  // TODO: maybe a more efficient way later!!!
  const FrameIdTimestampMap& all_timestamp_map =
      shared_module_info.getTimestampMap();
  FrameIdTimestampMap involved_timestamps;
  for (const FrameId& frame_id : object_motions.getInvovledFrameIds()) {
    CHECK(all_timestamp_map.exists(frame_id));
    involved_timestamps.insert2(frame_id, all_timestamp_map.at(frame_id));
  }

  return std::make_shared<RGBDInstanceOutputPacket>(
      static_keypoint_measurements, dynamic_keypoint_measurements,
      static_landmarks, dynamic_landmarks, T_world_camera, frame.timestamp_,
      frame.frame_id_, object_motions, object_poses, camera_poses_, camera_,
      gt_packet, debug_imagery, dense_labelled_cloud, involved_timestamps);
}

cv::Mat RGBDInstanceFrontendModule::createTrackingImage(
    const Frame::Ptr& frame_k, const Frame::Ptr& frame_k_1,
    const ObjectPoseMap& object_poses) const {
  cv::Mat tracking_image = tracker_->computeImageTracks(
      *frame_k_1, *frame_k, base_params_.image_tracks_vis_params);

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

// TrackingInputImages RGBDInstanceFrontendModule::constructTrackingImages(
//     const ImageContainer::Ptr image_container) {
//   // if we only have instance semgentation (not motion) then we need to make
//   a
//   // motion mask out of the semantic mask we cannot do this for the first
//   frame
//   // so we will just treat the semantic mask and the motion mask and then
//   // subsequently elimate non-moving objects later on
//   TrackingInputImages tracking_images;
//   if (image_container->hasSemanticMask()) {
//     CHECK(!image_container->hasMotionMask());
//     // TODO: some bug when going from semantic mask to motion mask as motion
//     // mask is empty in the tracker after this process!!! its becuase we dont
//     // actually use the tracking_images!!
//     auto intermediate_tracking_images =
//         image_container->makeSubset<ImageType::RGBMono,
//         ImageType::OpticalFlow,
//                                     ImageType::SemanticMask>();
//     tracking_images = TrackingInputImages(
//         intermediate_tracking_images.getImageWrapper<ImageType::RGBMono>(),
//         intermediate_tracking_images.getImageWrapper<ImageType::OpticalFlow>(),
//         ImageWrapper<ImageType::MotionMask>(
//             intermediate_tracking_images.get<ImageType::SemanticMask>()));
//   } else {
//     tracking_images =
//         image_container->makeSubset<ImageType::RGBMono,
//         ImageType::OpticalFlow,
//                                     ImageType::MotionMask>();
//   }
//   return tracking_images;
// }

}  // namespace dyno
