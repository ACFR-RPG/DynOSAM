#include "dynosam_ros/CameraSystem.hpp"

#include <config_utilities/config_utilities.h>
#include <config_utilities/parsing/yaml.h>

#include <deque>
#include <dynosam_common/Types.hpp>
#include <dynosam_cv/StereoCamera.hpp>
#include <unordered_set>

#include "dynosam_ros/RosUtils.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"

namespace dyno {

bool hasStreamType(const std::vector<StreamConfig>& configs,
                   StreamConfig::Types query_type) {
  for (size_t i = 0; i < configs.size(); i++) {
    if (configs.at(i).type == query_type) {
      return true;
    }
  }
  return false;
}

// Helper function to split a valid string by '+' using string_views (no
// allocations)
std::vector<std::string> splitByPlus(const std::string& sensor_mode) {
  std::vector<std::string> tokens;
  size_t start = 0;
  size_t end = sensor_mode.find('+');

  while (end != std::string_view::npos) {
    tokens.push_back(sensor_mode.substr(start, end - start));
    start = end + 1;
    end = sensor_mode.find('+', start);
  }
  tokens.push_back(sensor_mode.substr(start));  // Add the final token
  return tokens;
}

std::ostream& operator<<(std::ostream& os, const StreamConfig& config) {
  os << "StreamConfig {" << config.name
     << ", needed for depth: " << std::boolalpha << config.needed_for_depth
     << ", assume aligned: " << config.assume_aligned << "}";
  return os;
}

SensorMode::SensorMode(const std::string& sensor_mode)
    : raw_sensor_mode_(sensor_mode) {
  parse(sensor_mode);
}

const std::vector<StreamConfig>& SensorMode::configs() const {
  return image_configs_;
}

bool SensorMode::useImu() const { return use_imu_; }

DepthRigType SensorMode::depthRigType() const { return depth_rig_type_; }

bool SensorMode::reconfigure(DynoParams& dyno_params) const {
  bool any_changed = false;
  auto& tracker_params = dyno_params.frontend_params_.tracker_params;
  if (tracker_params.prefer_provided_optical_flow &&
      !hasStreamType(configs(), StreamConfig::Types::OpticalFlow)) {
    LOG(WARNING) << "Sensor mode:" << raw_sensor_mode_
                 << " does not include optical flow "
                 << " but prefer_provided_optical_flow=true! DynoParams will "
                    "be reconfigured";
    tracker_params.prefer_provided_optical_flow = false;
    any_changed = true;
  }

  if (tracker_params.prefer_provided_object_detection &&
      !hasStreamType(configs(), StreamConfig::Types::Mask)) {
    LOG(WARNING) << "Sensor mode:" << raw_sensor_mode_
                 << " does not include object mask "
                 << " but prefer_provided_object_detection=true! DynoParams "
                    "will be reconfigured!\n"
                 << " NOTE: ground truth object id's may now no longer match "
                    "the tracked id's from Dynosam!!!";
    tracker_params.prefer_provided_object_detection = false;
    any_changed = true;
  }
  return any_changed;
}

void SensorMode::parse(const std::string& sensor_mode) {
  // split string by +
  const auto string_configs = splitByPlus(sensor_mode);

  // would be far more elegant to match the configs to the types in the image
  // container... but we use stereo to mean multiple streams etc and mask in
  // general could be many thing...?
  static const std::unordered_set<std::string> known_configs = {
      "rgb", "depth", "stereo", "opticalflow", "mask", "imu"};

  const std::string aligned_prefix = "aligned_";

  std::deque<StreamConfig> image_configs;
  bool found_rgb = false;
  bool found_depth = false;
  bool assume_depth_aligned = false;
  bool found_stereo = false;
  bool any_depth_source = false;
  for (const auto& provided_config : string_configs) {
    std::string config = provided_config;
    bool assume_aligned = false;
    if (config.rfind(aligned_prefix, 0) == 0) {
      // strip the prefix
      config.erase(0, aligned_prefix.length());
      assume_aligned = true;
    }

    if (known_configs.find(config) == known_configs.end()) {
      LOG(FATAL) << "Unknown config: " << provided_config;
    }

    if (config == "rgb") {
      found_rgb = true;
    } else if (config == "depth") {
      found_depth = true;
      any_depth_source = true;
      // only really important in the rgbd case as this will
      // inform the loader if we need to load the depth image camera params
      assume_depth_aligned = assume_aligned;
    } else if (config == "stereo") {
      found_stereo = true;
      any_depth_source = true;
    } else if (config == "imu") {
      use_imu_ = true;
    } else {
      // a misc image config like mask or optical flow
      StreamConfig image_config;
      image_config.name = config;
      image_config.needed_for_depth = false;
      // if we assume the image is aligned then we do not need to get params
      image_config.assume_aligned = assume_aligned;

      if (config == "mask") {
        image_config.type = StreamConfig::Types::Mask;
      } else if (config == "opticalflow") {
        image_config.type = StreamConfig::Types::OpticalFlow;
      } else {
        // default treat as rgb
        image_config.type = StreamConfig::Types::RGBMono;
      }

      image_configs.push_back(image_config);
    }
  }

  if (!any_depth_source) {
    DYNO_THROW_MSG(InvalidSensorSystem)
        << "No valid depth rig setup found (RBGD or Stereo) in sensor mode:"
        << raw_sensor_mode_;
  }

  // if found stereo and found depth throw error?
  if (found_stereo && found_depth) {
    DYNO_THROW_MSG(InvalidSensorSystem)
        << "Multiple depth rig setup's found (RBGD and Stereo) in sensor mode:"
        << raw_sensor_mode_;
  }

  if (found_stereo) {
    // right image
    StreamConfig image_config;
    image_config.type = StreamConfig::Types::RGBMono;
    image_config.name = "image_1";
    image_config.needed_for_depth = true;
    image_config.assume_aligned = false;
    image_configs.push_front(image_config);

    // left image
    image_config.name = "image_0";
    // aligned with itself
    image_config.assume_aligned = true;
    image_configs.push_front(image_config);
    depth_rig_type_ = DepthRigType::Stereo;
  }

  if (found_depth) {
    CHECK(found_rgb);
    // depth image
    StreamConfig image_config;
    image_config.name = "depth";
    image_config.needed_for_depth = true;
    image_config.type = StreamConfig::Types::Depth;
    image_config.assume_aligned = assume_depth_aligned;
    image_configs.push_front(image_config);

    // rgb image
    image_config.name = "rgb";
    // aligned with itself
    image_config.assume_aligned = true;
    image_config.type = StreamConfig::Types::RGBMono;
    image_configs.push_front(image_config);
    depth_rig_type_ = DepthRigType::RGBD;
  }

  // very important the main camera (either left or rgb) is at the start of the
  // configs
  image_configs_.insert(image_configs_.begin(), image_configs.begin(),
                        image_configs.end());
}

SensorSystem::SensorSystem(std::shared_ptr<rclcpp::Node> node,
                           DepthRigType depth_rig_type,
                           const std::string& path_to_params,
                           const bool load_cameras_from_ros)
    : node_(node),
      depth_rig_type_(depth_rig_type),
      path_to_params_(path_to_params),
      load_cameras_from_ros_(load_cameras_from_ros),
      tf_buffer_(node->get_clock()),
      tf_listener_(tf_buffer_),
      enable_imu_(false),
      is_initalised_{false} {
  if (!load_cameras_from_ros_) {
    // assume loading source is path to dynosam paramter folder
    LOG(FATAL) << "Not implemented!";
  }
}

void SensorSystem::addCamera(const StreamConfig& config) {
  configs_.push_back(config);
}

void SensorSystem::enableImu(bool flag) { enable_imu_ = flag; }
bool SensorSystem::imuEnabled() const { return enable_imu_; }
size_t SensorSystem::numCameraStreams() const { return camera_params_.size(); }

CameraParams SensorSystem::getCanonicalParams() const {
  return cannonical_camera_params_;
}

ReferenceFrames SensorSystem::getReferenceFrames() const {
  return reference_frames_;
}

DepthRigType SensorSystem::depthRigType() const { return depth_rig_type_; }

std::string SensorSystem::streamName(unsigned int stream_index) const {
  return configs_.at(stream_index).name;
}

StreamConfig::Types SensorSystem::streamType(unsigned int stream_index) const {
  return configs_.at(stream_index).type;
}

bool SensorSystem::isInitalised() const { return is_initalised_; }

void SensorSystem::calibrateDetphRig(const cv::Mat& img0_src,
                                     const cv::Mat& img1_src, cv::Mat& img0_out,
                                     cv::Mat& img1_out) {
  CHECK(calibrate_depth_rig_);
  calibrate_depth_rig_(img0_src, img1_src, img0_out, img1_out);
}

void SensorSystem::finalise() {
  // load all requested params first
  // we must always load the main camera (ie config.at(0))
  CHECK_GT(configs_.size(), 0);
  std::vector<CameraParams> params_needed_for_depth;
  std::vector<std::string> camera_optical_frames;

  // now we have main camera params (including reference frame)
  // we can load all system reference frames from the ros params
  // reference frame values must be set before calling the loadSingleParams
  // function as the loadSingleParamsFromROS variant needs these values to set
  // the paramter extrinsics
  reference_frames_.base_frame =
      ParameterConstructor(node_.get(), "base_frame",
                           reference_frames_.base_frame)
          .description("ROS frame id for base link of the robot")
          .finish()
          .get<std::string>();

  reference_frames_.odom_frame =
      ParameterConstructor(node_.get(), "odom_frame",
                           reference_frames_.odom_frame)
          .description("ROS frame id for the static workd frame (ie. odometry)")
          .finish()
          .get<std::string>();

  // should either be rgb or image_0
  const CameraParams main_camera_params = loadSingleParams(configs_.at(0));
  reference_frames_.camera_frame = main_camera_params.referenceFrame();
  LOG(INFO) << "Using robot frame: " << reference_frames_.base_frame;
  LOG(INFO) << "Using odom frame: " << reference_frames_.odom_frame;
  LOG(INFO) << "Using camera (estimation) frame: "
            << reference_frames_.camera_frame;

  if (enable_imu_) {
    reference_frames_.imu_frame =
        ParameterConstructor(node_.get(), "imu_frame",
                             reference_frames_.imu_frame)
            .description("ROS frame id for the IMU frame")
            .finish()
            .get<std::string>();

    const auto& robot_frame = reference_frames_.base_frame;
    const auto& imu_frame = reference_frames_.imu_frame;
    getLatestTransform(robot_frame, imu_frame, T_RI_);

    // Estimation is all done in the camera (optical) frame
    // so the transform we use for the IMU params (while usually IMU to robot)
    // is actually going to be IMU to CAMERA
    const auto& camera_frame = reference_frames_.camera_frame;
    gtsam::Pose3 T_CI;
    getLatestTransform(camera_frame, imu_frame, T_CI);
  }

  camera_params_.push_back(main_camera_params);
  params_needed_for_depth.push_back(camera_params_.back());

  for (size_t i = 1; i < configs_.size(); i++) {
    const StreamConfig& image_config = configs_.at(i);
    // if aligned with the main image then assume the calibration is the same
    const bool needs_params = !image_config.assume_aligned;
    CameraParams camera_params;
    if (needs_params) {
      camera_params = loadSingleParams(image_config);
    } else {
      camera_params = main_camera_params;
    }
    camera_params_.push_back(camera_params);

    if (image_config.needed_for_depth) {
      params_needed_for_depth.push_back(camera_params_.back());
    }
  }

  // should be in rgb,depth or image0, image1 order
  // where rgb and image0 are the main channels
  CHECK_EQ(params_needed_for_depth.size(), 2u);
  CHECK_GE(configs_.size(), 2u);

  CHECK_EQ(configs_.size(), camera_params_.size());

  GeneralParams general_params;
  loadGeneralParams(main_camera_params.imageSize(), general_params);

  // FOR NOW: assume that all non-depth needed params are aligned!

  CameraParams cannonical_camera_params;
  CalibrateDepthRig calibrate_depth_rig;
  // get/set tf extrinsics (if possible?)
  // construct calibrated camera (better name?)
  if (depth_rig_type_ == DepthRigType::RGBD && configs_.at(1).assume_aligned) {
    RGBDParams rgbd_params;
    loadRGBDSpecificParams(rgbd_params);

    // construct cannonical camera only from the main camera
    calibrateFromAlignedRGBD(main_camera_params, general_params, rgbd_params,
                             cannonical_camera_params, calibrate_depth_rig);
  } else if (depth_rig_type_ == DepthRigType::Stereo) {
    // cannoical camera needs full stereo calibration
    calibrateFromStereo(main_camera_params, params_needed_for_depth.at(1),
                        general_params, cannonical_camera_params,
                        calibrate_depth_rig);
  } else {
    LOG(FATAL) << "Other calibration routuines not implemented yet...";
  }
  // set is_initalised
  LOG(INFO) << "Cannonical camera params: "
            << cannonical_camera_params.toString();
  cannonical_camera_params_ = cannonical_camera_params;
  calibrate_depth_rig_ = calibrate_depth_rig;
  is_initalised_ = true;
}

CameraParams SensorSystem::loadSingleParams(const StreamConfig& config) const {
  if (load_cameras_from_ros_) {
    return loadSingleParamsFromROS(config);
  } else {
    LOG(FATAL) << "Not implemented!";
  }
}

CameraParams SensorSystem::loadSingleParamsFromROS(
    const StreamConfig& config) const {
  LOG(INFO) << "Getting camera params for " << config.name << " from ROS";

  CameraParams params =
      waitAndSetCameraParams(node_, "/dynosam/" + config.name + "/camera_info",
                             std::chrono::milliseconds(-1));

  // Set the optical frame, either from parameters or from the loaded
  // CameraParams which has its referecenFrame set from the camera info msg
  const std::string optical_frame =
      getCameraOpticalFrame(config.name, params.referenceFrame());
  params.referenceFrame(optical_frame);

  VLOG(5) << "Image " << config.name
          << " using reference frame: " << optical_frame;
  // assume reference values are set correctly
  const auto& robot_frame = reference_frames_.base_frame;

  // transform from camera -> robot
  // ie Z_r = T_RC * z_c where z_c is a measurement taken in the camera frame
  gtsam::Pose3 T_RC;
  getLatestTransform(robot_frame, optical_frame, T_RC);
  params.setExtrinsics(T_RC);

  return params;
}

ImuParams SensorSystem::loadImuParams(const gtsam::Pose3& T_CI) const {
  return ImuParams{};
}

std::string SensorSystem::getCameraOpticalFrame(
    const std::string& name, const std::string& default_optical_frame) const {
  // todo: descriptive naming
  const std::string key = name + "_optical_frame";
  auto detail = ParameterConstructor(node_.get(), key, default_optical_frame)
                    .description("Camera optical frame id")
                    .finish();
  auto result = detail.get<std::string>();

  if (VLOG_IS_ON(5)) {
    LOG(INFO) << "Requesting optical frame for camera " << name
              << (detail.isSet() ? " Using ROS param: "
                                 : " Using header frame id: ")
              << result;
  }

  return result;
}

void SensorSystem::getLatestTransform(const std::string& target,
                                      const std::string& source,
                                      gtsam::Pose3& pose) const {
  geometry_msgs::msg::TransformStamped transform_stamped;

  // Time out duration for TF tree lookup before throwing an exception.
  constexpr int32_t kTimeOutSeconds = 10;

  try {
    if (!tf_buffer_.canTransform(target, source, tf2::TimePointZero,
                                 tf2::durationFromSec(kTimeOutSeconds))) {
      RCLCPP_ERROR(
          node_->get_logger(),
          "Transform is impossible. canTransform(%s -> %s) returns false",
          target.c_str(), source.c_str());
    }
    transform_stamped =
        tf_buffer_.lookupTransform(target, source, tf2::TimePointZero,
                                   tf2::durationFromSec(kTimeOutSeconds));
    dyno::convert(transform_stamped, pose);
  } catch (tf2::TransformException& ex) {
    RCLCPP_INFO(node_->get_logger(), "Could not transform %s to %s: %s",
                source.c_str(), target.c_str(), ex.what());
    throw std::runtime_error("Could not find the requested transform!");
  }
}

// helper struct to hold calibration data for RGBD system
struct CalibrateData {
  cv::Mat mapx;
  cv::Mat mapy;
  double depth_scale;
};

void undistort(const CalibrateData calib_data, const cv::Mat& src,
               cv::Mat& dst) {
  cv::remap(src, dst, calib_data.mapx, calib_data.mapy, cv::INTER_LINEAR,
            cv::BORDER_REPLICATE);
  // output will have the same type as mapx/y so covnert back to required type
  dst.convertTo(dst, src.type());
}

void SensorSystem::calibrateFromAlignedRGBD(
    const CameraParams& main_camera_params, const GeneralParams& general_params,
    const RGBDParams& rgbd_params, CameraParams& cannonical_camera,
    CalibrateDepthRig& calibrate_depth_rig) {
  const auto& original_size = main_camera_params.imageSize();
  const auto& rescale_size = general_params.new_image_size;
  const auto original_K = main_camera_params.getCameraMatrix();
  const auto distortion = main_camera_params.getDistortionCoeffs();

  CalibrateData calib_data;
  calib_data.depth_scale = rgbd_params.depth_scale;

  static constexpr double kAlpha = 0.0;  // crop to valid region
  cv::Mat new_K = cv::getOptimalNewCameraMatrix(
      original_K, distortion, original_size, kAlpha, rescale_size);

  cv::initUndistortRectifyMap(original_K, distortion, cv::Mat(), new_K,
                              rescale_size, CV_32FC1, calib_data.mapx,
                              calib_data.mapy);

  dyno::CameraParams::IntrinsicsCoeffs intrinsics;
  cv::Mat K_double;
  new_K.convertTo(K_double, CV_64F);
  dyno::CameraParams::convertKMatrixToIntrinsicsCoeffs(K_double, intrinsics);
  dyno::CameraParams::DistortionCoeffs zero_distortion(4, 0);

  cannonical_camera = CameraParams(intrinsics, zero_distortion, rescale_size,
                                   main_camera_params.getDistortionModel(),
                                   main_camera_params.getExtrinsics(),
                                   main_camera_params.referenceFrame());

  cannonical_camera.setDepthParams(rgbd_params.virtual_baseline);

  // capture calib data by copy so it remains in-scope
  calibrate_depth_rig = [calib_data](const cv::Mat& rgb_src,
                                     const cv::Mat& depth_src, cv::Mat& rgb_out,
                                     cv::Mat& depth_out) -> void {
    undistort(calib_data, rgb_src, rgb_out);
    undistort(calib_data, depth_src, depth_out);

    // convert the depth map to metirc scale
    // data-type shoule match
    depth_out *= calib_data.depth_scale;
  };
}

void SensorSystem::calibrateFromStereo(const CameraParams& left_params,
                                       const CameraParams& right_params,
                                       const GeneralParams& general_params,
                                       CameraParams& cannonical_camera,
                                       CalibrateDepthRig& calibrate_depth_rig) {
  auto stereo_camera = std::make_shared<StereoCamera>(
      left_params, right_params, general_params.new_image_size);

  // should be the result of undistorting/rectifying the stereo pair
  // extrinnsics should be modified to account for stereo rectification
  // and baseline set appropiately
  cannonical_camera =
      stereo_camera->getUndistortedRectifiedCanonicalCameraParams();

  calibrate_depth_rig = [stereo_camera](
                            const cv::Mat& left_src, const cv::Mat& right_src,
                            cv::Mat& left_out, cv::Mat& right_out) -> void {
    stereo_camera->undistortRectifyImages(left_out, right_out, left_src,
                                          right_src);
  };
}

void SensorSystem::loadRGBDSpecificParams(
    SensorSystem::RGBDParams& params) const {
  params.depth_scale =
      ParameterConstructor(node_.get(), "depth_scale", 0.001)
          .description(
              "Value to scale the depth image from a disparity map "
              "to metric depth")
          .finish()
          .get<double>();

  params.virtual_baseline =
      ParameterConstructor(node_.get(), "baseline", 0.1)
          .description(
              "Stereo camera baseline needed for virtual-stereo system")
          .finish()
          .get<double>();
}
void SensorSystem::loadGeneralParams(
    const cv::Size& main_image_size,
    SensorSystem::GeneralParams& params) const {
  double rescale_width =
      ParameterConstructor(node_.get(), "rescale_width", main_image_size.width)
          .description(
              "Image width to rescale to. If not provided or -1 "
              "image will be inchanged")
          .finish()
          .get<int>();
  if (rescale_width == -1) {
    rescale_width = main_image_size.width;
  }

  double rescale_height =
      ParameterConstructor(node_.get(), "rescale_height",
                           main_image_size.height)
          .description(
              "Image height to rescale to. If not provided or -1 "
              "image will be inchanged")
          .finish()
          .get<int>();
  if (rescale_height == -1) {
    rescale_height = main_image_size.height;
  }

  params.new_image_size.height = rescale_height;
  params.new_image_size.width = rescale_width;
}

}  // namespace dyno
