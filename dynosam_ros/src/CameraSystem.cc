#include "dynosam_ros/CameraSystem.hpp"

#include <deque>
#include <unordered_set>

#include "dynosam_common/Types.hpp"
#include "dynosam_ros/RosUtils.hpp"

namespace dyno {

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

std::ostream& operator<<(std::ostream& os, const ImageConfig& config) {
  os << "ImageConfig {" << config.name
     << ", needed for depth: " << std::boolalpha << config.needed_for_depth
     << ", assume aligned: " << config.assume_aligned << "}";
  return os;
}

SensorMode::SensorMode(const std::string& sensor_mode)
    : raw_sensor_mode_(sensor_mode) {
  parse(sensor_mode);
}

const std::vector<ImageConfig>& SensorMode::configs() const {
  return image_configs_;
}

bool SensorMode::useImu() const { return use_imu_; }

DepthCameraMode SensorMode::depthCameraMode() const {
  return depth_camera_mode_;
}

void SensorMode::parse(const std::string& sensor_mode) {
  // split string by +
  const auto string_configs = splitByPlus(sensor_mode);

  static const std::unordered_set<std::string> known_configs = {
      "rgb", "depth", "stereo", "opticalflow", "mask", "imu"};

  const std::string aligned_prefix = "aligned_";

  std::deque<ImageConfig> image_configs;
  bool found_rgb = false;
  bool found_depth = false;
  bool assume_depth_aligned = false;
  bool found_stereo = false;
  bool any_depth_source = false;
  for (const auto& provided_config : string_configs) {
    if (known_configs.find(provided_config) == known_configs.end()) {
      LOG(FATAL) << "Unknown config: " << provided_config;
    }

    std::string config = provided_config;
    bool assume_aligned = false;
    if (config.starts_with(aligned_prefix)) {
      // strip the prefix
      config.erase(0, aligned_prefix.length());
      assume_aligned = true;
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
      ImageConfig image_config;
      image_config.name = config;
      image_config.needed_for_depth = false;
      // if we assume the image is aligned then we do not need to get params
      image_config.needs_params = !assume_aligned;
      image_configs.push_back(image_config);
    }
  }

  if (!any_depth_source) {
    LOG(FATAL) << "no depth sources specified";
  }

  // if found stereo and found depth throw error?
  if (found_stereo && found_depth) {
    LOG(FATAL) << "multiple depth sources found";
  }

  if (found_stereo) {
    // right image
    ImageConfig image_config;
    image_config.name = "image_1";
    image_config.needed_for_depth = true;
    image_config.assume_aligned = false;
    image_configs.push_front(image_config);

    // left image
    image_config.name = "image_0";
    // aligned with itself
    image_config.assume_aligned = true;
    image_configs.push_front(image_config);
    depth_camera_mode_ = DepthCameraMode::Stereo;
  }

  if (found_depth) {
    CHECK(found_rgb);
    // depth image
    ImageConfig image_config;
    image_config.name = "depth";
    image_config.needed_for_depth = true;
    image_config.assume_aligned = assume_depth_aligned;
    image_configs.push_front(image_config);

    // rgb image
    image_config.name = "rgb";
    // aligned with itself
    image_config.assume_aligned = true;
    image_configs.push_front(image_config);
    depth_camera_mode_ = DepthCameraMode::RGBD;
  }

  // very important the main camera (either left or rgb) is at the start of the
  // configs
  image_configs_.insert(image_configs_.begin(), image_configs.begin(),
                        image_configs.end());
}

SensorSystem::SensorSystem(rclcpp::Node* node,
                           DepthCameraMode depth_camera_mode,
                           const std::string& loading_source)
    : node_(node),
      depth_camera_mode_(depth_camera_mode),
      loading_source_(loading_source) {}

void SensorSystem::addCamera(const ImageConfig& config) {
  configs_.push_back(config);
}
void SensorSystem::finalise() {
  // load all requested params first
  // we must always load the main camera (ie config.at(0))
  CHECK_GT(configs_.size(), 0);
  std::vector<CameraParams> params_needed_for_depth;
  std::vector<std::string> camera_optical_frames;

  // should either be rgb or image_0
  CameraParams main_camera_params = loadSingleParams(configs_.at(0));
  // Set the main frames, either from parameters or from camera info message.
  const std::string main_optical_frame = getCameraOpticalFrame(
      main_camera_params.name, main_camera_params.referenceFrame());
  main_camera_params.referenceFrame(main_optical_frame);

  camera_params_.push_back(main_camera_params);
  params_needed_for_depth.push_back(camera_params_.back());

  for (size_t i = 1; i < configs_.size(); i++) {
    const ImageConfig& image_config = configs_.at(i);
    // if aligned with the main image then assume the calibration is the same
    const bool needs_params = !config.assume_aligned;
    CameraParams camera_params if (needs_params) {
      camera_params = loadSingleParams(image_config);
    }
    else {
      camera_params = main_camera_params;
    }
    // load the image config if possible or use the camera info message
    std::string optical_frame = getCameraOpticalFrame(
        image_config.name, camera_params.referenceFrame());
    // update the camera params
    camera_params.referenceFrame(optical_frame);
    camera_params_.push_back(camera_params);

    if (config.needed_for_depth) {
      params_needed_for_depth.push_back(camera_params_.back());
    }
  }

  // should be in rgb,depth or image0, image1 order
  // where rgb and image0 are the main channels
  CHECK_EQ(params_needed_for_depth.size(), 2u);
  CHECK_GE(configs_.size(), 2u);

  // load reference frame definitions
  // set extrinsics for all cameras
  // get/set tf extrinsics (if possible?)
  // construct calibrated camera (better name?)
  // register pre-processing calibration functions
  // set is_initalised
}

std::string SensorSystem::getCameraOpticalFrame(
    const std::string& name, const std::string& default_optical_frame) {
  // todo: descriptive naming
  return ParameterConstructor(node_, name + "_optical_frame",
                              default_optical_frame)
      .description("Camera optical frame id")
      .finish()
      .get<std::string>();
}

}  // namespace dyno
