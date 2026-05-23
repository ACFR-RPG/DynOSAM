#pragma once

#include "dynosam_common/Types.hpp"
#include "dynosam_cv/CameraParams.hpp"

namespace dyno {

struct ReferenceFrames {
  std::string odom_frame = "odom";
  std::string base_frame = "camera_link";
  //! This is the one we actually publish in!
  std::string camera_frame = "camera_optical_frame";
  std::string imu_frame = "imu_frame";
};

enum class DepthRigType { RGBD, Stereo };
std::ostream& operator<<(std::ostream& os, const DepthRigType& depth_rig_type);

class SensorRigBase {
 public:
  DYNO_POINTER_TYPEDEFS(SensorRigBase)

  virtual CameraParams getCanonicalParams() const = 0;
  virtual ReferenceFrames getReferenceFrames() const = 0;
  virtual DepthRigType depthRigType() const = 0;
};

}  // namespace dyno
