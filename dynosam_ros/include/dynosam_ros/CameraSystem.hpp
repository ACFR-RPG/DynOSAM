#pragma once

#include <ostream>
#include <string>
#include <variant>
#include <vector>

#include "dynosam_cv/Camera.hpp"
#include "rclcpp/node.hpp"

namespace dyno {

/**
 * @brief Configuration needed to create a camera and if the camera is needed in
 * constructing the DepthCamera
 *
 */
struct ImageConfig {
  std::string name;
  //! Proxy for indicating it is a physical camera
  bool needed_for_depth{false};
  bool assume_aligned{false};
};

std::ostream& operator<<(std::ostream& os, const ImageConfig& config);

enum class DepthCameraMode { RGBD, Stereo };

class SensorMode {
 public:
  SensorMode(const std::string& sensor_mode);

  const std::vector<ImageConfig>& configs() const;

  bool useImu() const;
  DepthCameraMode depthCameraMode() const;

 private:
  void parse(const std::string& sensor_mode);

 private:
  std::string raw_sensor_mode_;
  //! How the depth will be computed
  DepthCameraMode depth_camera_mode_;
  bool use_imu_{false};
  std::vector<ImageConfig> image_configs_;
};

struct ReferenceFrames {
  std::string odom_frame = "odom";
  std::string base_frame = "camera_link";
  //! This is the one we actually publish in!
  std::string camera_frame = "camera_optical_frame";
  std::string imu_frame = "imu_frame";
};

// look up camera optical frame (only need 1 in RGBD, need 2 for stereo)
// after processing we assume all images will have a CameraParams that match the
// "target" params
class SensorSystem {
 public:
  SensorSystem(rclcpp::Node* node, DepthCameraMode depth_camera_mode,
               const std::string& loading_source = "");

  void addCamera(const ImageConfig& config);
  void finalise();

  bool isInitalised() const;

  // TODO: all extrinsics?

 private:
  CameraParams loadSingleParams(const ImageConfig config) const;
  std::string getCameraOpticalFrame(const std::string& name,
                                    const std::string& default_optical_frame);

 private:
  rclcpp::Node* node_;
  DepthCameraMode depth_camera_mode_;
  std::string loading_source_;

  std::vector<ImageConfig> configs_;
  std::vector<CameraParams> camera_params_;

  ReferenceFrames reference_frames_;

  bool is_initalised_{false};
};

}  // namespace dyno
