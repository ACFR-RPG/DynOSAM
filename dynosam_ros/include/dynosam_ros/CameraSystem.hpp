#pragma once

#include <functional>
#include <ostream>
#include <string>
#include <variant>
#include <vector>

#include "dynosam_cv/Camera.hpp"
#include "rclcpp/node.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

namespace dyno {

/**
 * @brief Configuration needed to create an image stream
 *
 */
struct StreamConfig {
  //! Echos the possible ImageType struct
  enum class Types { RGBMono, Depth, OpticalFlow, Mask };

  std::string name;
  //! Proxy for indicating it is a physical camera
  bool needed_for_depth{false};
  bool assume_aligned{false};
  Types type{Types::RGBMono};
};

std::ostream& operator<<(std::ostream& os, const StreamConfig& config);

enum class DepthRigType { RGBD, Stereo };

class SensorMode {
 public:
  SensorMode(const std::string& sensor_mode);

  const std::vector<StreamConfig>& configs() const;

  bool useImu() const;
  DepthRigType depthRigMode() const;

 private:
  void parse(const std::string& sensor_mode);

 private:
  std::string raw_sensor_mode_;
  //! How the depth will be computed
  DepthRigType depth_rig_type_;
  bool use_imu_{false};
  std::vector<StreamConfig> image_configs_;
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
  DYNO_POINTER_TYPEDEFS(SensorSystem)

  typedef std::function<void(const cv::Mat&, const cv::Mat&, cv::Mat&,
                             cv::Mat&)>
      CalibrateDepthRig;
  typedef std::function<void(const cv::Mat&, cv::Mat&)> CalibrateOther;

  // loading source expected to be empty (load from ROS) or path to parameter
  // folder
  SensorSystem(std::shared_ptr<rclcpp::Node> node,
               DepthRigType depth_camera_mode,
               const std::string& loading_source = "");

  void addCamera(const StreamConfig& config);
  void finalise();

  size_t numCameraStreams() const;
  bool isInitalised() const;

  // TODO: all extrinsics?

  /**
   * @brief Run the calibration rountine (undistory/rectify etc) as necessary
   * to convert the depth rig images (ie rgb+depth or img0+img1) to match
   * the canonical camera params
   *
   * @param img0_src
   * @param img1_src
   * @param img0_out
   * @param img1_out
   */
  void calibrateStereoRig(const cv::Mat& img0_src, const cv::Mat& img1_src,
                          cv::Mat& img0_out, cv::Mat& img1_out);

  const ReferenceFrames& referenceFrames() const;

  /* Returns canonical params representing a single virtual camera after
   * undistortion/rectification */
  CameraParams getCanonicalParams() const;

  std::string streamName(unsigned int stream_index) const;
  StreamConfig::Types streamType(unsigned int stream_index) const;

  DepthRigType depthRigMode() const;

 private:
  bool loadingSourceROS() const;

  /**
   * @brief Loads a single param from either ros params or YAML config.
   * Will additionally set the reference frame value either from ros params (if
   * provided) or from the camera info message.
   *
   * @param config const StreamConfig&
   * @return CameraParams
   */
  CameraParams loadSingleParams(const StreamConfig& config) const;

  // NOTE: reference_frames_ must be set correctly before using
  CameraParams loadSingleParamsFromROS(const StreamConfig& config) const;

  std::string getCameraOpticalFrame(
      const std::string& name, const std::string& default_optical_frame) const;

  /* Helper function to get child frame pose wrt parent frame from the tf tree*/
  void getLatestTransform(const std::string& target, const std::string& source,
                          gtsam::Pose3& pose) const;

  /* Specific params loaded when in RGBD mode. */
  struct RGBDParams {
    double depth_scale;
    double virtual_baseline;
  };

  struct GeneralParams {
    cv::Size new_image_size;
  };

  void loadRGBDSpecificParams(RGBDParams& params) const;
  void loadGeneralParams(const cv::Size& main_image_size,
                         GeneralParams& params) const;

  static void calibrateFromAlignedRGBD(const CameraParams& main_camera_params,
                                       const GeneralParams& general_params,
                                       const RGBDParams& rgbd_params,
                                       CameraParams& cannonical_camera,
                                       CalibrateDepthRig& calibrate_depth_rig);

  static void calibrateFromStereo(const CameraParams& left_params,
                                  const CameraParams& right_params,
                                  const GeneralParams& general_params,
                                  CameraParams& cannonical_camera,
                                  CalibrateDepthRig& calibrate_depth_rig);

 private:
  std::shared_ptr<rclcpp::Node> node_;
  DepthRigType depth_rig_type_;
  //! If empty, try loading camera params from camera info message
  std::string loading_source_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::vector<StreamConfig> configs_;
  std::vector<CameraParams> camera_params_;

  CameraParams cannonical_camera_params_;
  //! Preprocess the two images needed for the depth rig
  //! Functionality depends on mode
  CalibrateDepthRig calibrate_depth_rig_;

  ReferenceFrames reference_frames_;

  bool is_initalised_{false};
};

}  // namespace dyno
