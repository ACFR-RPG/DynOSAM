#pragma once

#include "dynosam_common/Types.hpp"
#include "dynosam_cv/CameraParams.hpp"

namespace dyno {

struct ReferenceFrames {
  std::string odom_frame = "odom";
  std::string base_frame = "camera_link";
  std::string camera_frame = "camera_optical_frame";
  std::string imu_frame = "imu_frame";
};

enum class DepthRigType { RGBD, Stereo };
std::ostream& operator<<(std::ostream& os, const DepthRigType& depth_rig_type);

/**
 * Represents an interface for the canonical sensor setup including canonical
 * camera and imu.
 */
class CanonicalSensorRig {
 public:
  DYNO_POINTER_TYPEDEFS(CanonicalSensorRig)

  virtual CameraParams getCanonicalParams() const = 0;
  virtual ReferenceFrames getReferenceFrames() const = 0;
  virtual DepthRigType depthRigType() const = 0;

  /**
   * @brief Get the extrinsics transform between between the base (robot) frame
   * and the canonical camera frame. This frame should be
   * ReferenceFrames::camera_frame.
   *
   * By default returns the extrinsics provided by
   * CanonicalSensorRig##getCanonicalParams
   *
   * @return gtsam::Pose3
   */
  virtual gtsam::Pose3 getCanonicalExtrinsics() const {
    return getCanonicalParams().getExtrinsics();
  }
};

}  // namespace dyno
