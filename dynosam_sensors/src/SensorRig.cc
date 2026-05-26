#include "dynosam_sensors/SensorRig.hpp"

#include <ostream>

namespace dyno {

std::ostream& operator<<(std::ostream& os, const DepthRigType& depth_rig_type) {
  switch (depth_rig_type) {
    case DepthRigType::RGBD:
      os << "RGBD";
      break;
    case DepthRigType::Stereo:
      os << "Stereo";
      break;
    default:
      os << "Unknown DepthRigType";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const ReferenceFrames& frames) {
  os << "odom: " << frames.odom_frame << ", base (robot): " << frames.base_frame
     << ", camera: " << frames.camera_frame << ", IMU: " << frames.imu_frame;
  return os;
}

std::string CanonicalSensorRig::toString() const {
  std::stringstream ss;
  ss << "Canonical camera: " << this->getCanonicalParams().toString() << "\n";
  ss << "Reference frames: " << this->getReferenceFrames() << "\n";
  ss << "Depth Rig: " << this->depthRigType() << "\n";

  if (imuEnabled()) {
    ss << "IMU is enabled with params:\n" << this->getImuParams() << "\n";
  } else {
    ss << "IMU not enabled";
  }

  return ss.str();
}

ImuCalibration CanonicalSensorRig::getImuParams() const {
  return ImuCalibration{};
}

gtsam::Pose3 CanonicalSensorRig::getCanonicalExtrinsics() const {
  return getCanonicalParams().getExtrinsics();
}

std::ostream& operator<<(std::ostream& os,
                         const CanonicalSensorRig& sensor_rig) {
  os << sensor_rig.toString();
  return os;
}

}  // namespace dyno
