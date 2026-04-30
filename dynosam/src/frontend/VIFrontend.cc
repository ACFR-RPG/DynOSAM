#include "dynosam/frontend/VIFrontend.hpp"

namespace dyno {

DEFINE_bool(use_frontend_logger, false,
            "If true, the frontend logger will be used");

VIFrontendLogger::VIFrontendLogger(const std::string& logger_name)
    : EstimationModuleLogger(logger_name),
      tracking_length_hist_file_name_(
          getOutputFilePath("tracklet_length_hist.json")) {}

void VIFrontendLogger::logTrackingLengthHistogram(const Frame::Ptr frame) {
  gtsam::FastMap<ObjectId, Histogram> histograms =
      vision_tools::makeTrackletLengthHistorgram(frame);
  // collect histograms per object and then nest them per frame
  // must cast keys (object id, frame id) to string to get the json library to
  // properly construct nested maps
  json per_object_hist;
  for (const auto& [object_id, hist] : histograms) {
    per_object_hist[std::to_string(object_id)] = hist;
  }
  tracklet_length_json_[std::to_string(frame->getFrameId())] = per_object_hist;
}

VIFrontendLogger::~VIFrontendLogger() {
  JsonConverter::WriteOutJson(tracklet_length_json_,
                              tracking_length_hist_file_name_);
}

Frontend::Frontend(const std::string& name, const DynoParams& params,
                   ImageDisplayQueue* display_queue,
                   const SharedGroundTruth& shared_ground_truth)
    : Base(name),
      dyno_params_(params),
      display_queue_(display_queue),
      shared_ground_truth_(shared_ground_truth) {
  if (FLAGS_use_frontend_logger) {
    LOG(INFO) << "Using front-end logger!";
    logger_ = std::make_unique<VIFrontendLogger>(name);
  }
}

bool Frontend::pushImageToDisplayQueue(const std::string& wname,
                                       const cv::Mat& image) {
  if (!display_queue_) {
    return false;
  }

  display_queue_->push(ImageToDisplay(wname, image));
  return true;
}

void Frontend::logRealTimeOutput(const RealtimeOutput::Ptr& output) {
  if (logger_) {
    auto ground_truths = shared_ground_truth_.access();
    const auto& dyno_state = output->state;
    const auto frame_id = dyno_state.frame_id;

    // TODO: right now we only log per frame!! Does not account for the updated
    // trajectory!
    logger_->logCameraPose(frame_id, dyno_state.camera_trajectory,
                           ground_truths);

    logger_->logObjectTrajectory(frame_id, dyno_state.object_trajectories,
                                 ground_truths);

    // TODO: not logging map points?

    // TODO: historgram?
  }
}

void Frontend::validateInput(const VIFrontendInput::ConstPtr& input) const {
  const auto image_container = input->image_container_;

  if (!image_container) {
    throw DynosamException("Image container is null!");
  }

  const bool has_rgb = image_container->hasRgb();
  const bool has_depth_image = image_container->hasDepth();
  const bool has_stereo = image_container->hasRightRgb();

  if (!has_rgb) {
    throw InvalidImageContainerException(*image_container, "Missing RGB");
  }

  if (!has_depth_image && !has_stereo) {
    throw InvalidImageContainerException(*image_container,
                                         "Missing Depth or Stereo");
  }
}

VIFrontend::VIFrontend(const std::string& name, const DynoParams& params,
                       Camera::Ptr camera, ImageDisplayQueue* display_queue,
                       const SharedGroundTruth& shared_ground_truth)
    : Frontend(name, params, display_queue, shared_ground_truth),
      camera_(CHECK_NOTNULL(camera)),
      pnp_ransac_(params.frontend_params_.ego_motion_pnp_ransac_params,
                  camera->getParams()),
      optical_flow_pose_solver_(OpticalFlowAndPoseSolverParams{}),
      imu_frontend_(params.frontend_params_.imu_params) {
  const auto& frontend_params = dyno_params_.frontend_params_;
  tracker_ =
      std::make_unique<FeatureTracker>(frontend_params, camera_, display_queue);

  rgbd_camera_ = camera_->safeGetRGBDCamera();
  CHECK_NOTNULL(rgbd_camera_);

  // measurement sigmas
  static_point_sigma_ = dyno_params_.backend_params_.static_point_noise_sigma;
  dynamic_point_sigma_ = dyno_params_.backend_params_.dynamic_point_noise_sigma;

  const double& static_pixel_sigma =
      dyno_params_.backend_params_.static_pixel_noise_sigma;
  static_pixel_sigmas_ << static_pixel_sigma, static_pixel_sigma;

  const double& dynamic_pixel_sigma =
      dyno_params_.backend_params_.dynamic_pixel_noise_sigma;
  dynamic_pixel_sigmas_ << dynamic_pixel_sigma, dynamic_pixel_sigma;
}

Frame::Ptr VIFrontend::featureTrack(const VIFrontendInput::ConstPtr input,
                                    std::optional<gtsam::Rot3> R_km1_k) {
  ImageContainer::Ptr image_container = input->image_container_;
  Frame::Ptr frame = tracker_->track(input->getFrameId(), input->getTimestamp(),
                                     *image_container, R_km1_k);

  if (image_container->hasDepth()) CHECK(frame->updateDepths());

  // TODO: for now (testing with retroactive frame)
  //  should onlly do this for the set of features we re-troactively tracked
  //  and update the depth - definitely for stereo too!
  Frame::Ptr frame_km1 = tracker_->getPreviousFrame();
  if (frame_km1 && frame_km1->imageContainer().hasDepth()) {
    frame_km1->updateDepths();
  }
  return frame;
}

std::optional<gtsam::NavState> VIFrontend::tryPropogateImu(
    const VIFrontendInput::ConstPtr input,
    const gtsam::NavState& nav_state_lIMU, ImuFrontend::PimPtr& pim_out) {
  if (!input->imu_measurements.has_value()) {
    return {};
  }

  auto imu_measurements = input->imu_measurements.value();

  pim_out = imu_frontend_.preintegrateImuMeasurements(imu_measurements);
  return pim_out->predict(nav_state_lIMU, gtsam::imuBias::ConstantBias{});
}

bool VIFrontend::stereoMatch(Frame::Ptr frame) {
  const ImageContainer& container = frame->imageContainer();

  if (!container.hasRightRgb()) {
    return false;
  }

  // collect all features
  FeatureContainer features;
  for (const auto& f : frame->static_features_.usableIterator()) {
    features.add(f);
  }

  for (const auto& f : frame->dynamic_features_.usableIterator()) {
    features.add(f);
  }

  return tracker_->stereoTrack(features, container);
}

void VIFrontend::fillDebugImagery(DebugImagery& debug_imagery,
                                  const Frame::Ptr& frame_k,
                                  const Frame::Ptr& frame_km1) const {
  debug_imagery.tracking_image = tracker_->computeFeatureTracks(
      *frame_km1, *frame_k,
      dyno_params_.frontend_params_.image_tracks_vis_params);

  const ImageContainer& processed_ic = frame_k->image_container_;

  if (processed_ic.hasRgb()) {
    debug_imagery.rgb_viz = ImageType::RGBMono::toRGB(processed_ic.rgb());
  }

  if (processed_ic.hasDepth()) {
    debug_imagery.depth_viz = ImageType::Depth::toRGB(processed_ic.depth());
  }

  if (processed_ic.hasOpticalFlow()) {
    debug_imagery.flow_viz =
        ImageType::OpticalFlow::toRGB(processed_ic.opticalFlow());
  }

  if (processed_ic.hasObjectMask()) {
    debug_imagery.mask_viz =
        ImageType::MotionMask::toRGB(processed_ic.objectMotionMask());
  }

  // const auto& camera_params = camera_->getParams();
  // const auto& K = camera_params.getCameraMatrix();
  // const auto& D = camera_params.getDistortionCoeffs();

  // const gtsam::Pose3& X_k = frame_k->getPose();

  // // poses are expected to be in the world frame
  // gtsam::FastMap<ObjectId, gtsam::Pose3> poses_k_map =
  //     object_poses.collectByFrame(frame_k->getFrameId());
  // std::vector<gtsam::Pose3> poses_k_vec;
  // std::transform(poses_k_map.begin(), poses_k_map.end(),
  //                 std::back_inserter(poses_k_vec),
  //                 [&X_k](const std::pair<ObjectId, gtsam::Pose3>& pair) {
  //                 // put object pose into the camera frame so it can be
  //                 // projected into the image
  //                 return X_k.inverse() * pair.second;
  //                 });

  // TODO: bring back when visualisation is unified with incremental solver!!
  //  utils::drawObjectPoseAxes(tracking_image, K, D, poses_k_vec);
  // return tracking_image;
}

}  // namespace dyno
