#include "dynosam_cv/SensorRig.hpp"

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

}  // namespace dyno
