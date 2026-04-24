#include "dynosam_common/MotionKeyFrame.hpp"

namespace dyno {

template <>
inline std::string to_string(const ObjectKeyFrameStatus& status) {
  std::string status_str = "";
  switch (status) {
    case ObjectKeyFrameStatus::NonKeyFrame: {
      status_str = "NonKeyFrame";
      break;
    }
    case ObjectKeyFrameStatus::RegularKeyFrame: {
      status_str = "RegularKeyFrame";
      break;
    }
    case ObjectKeyFrameStatus::AnchorKeyFrame: {
      status_str = "AnchorKeyFrame";
      break;
    }
  }
  return status_str;
}

std::ostream& operator<<(std::ostream& os, const ObjectKeyFrameStatus& status) {
  os << to_string(status);
  return os;
}

}  // namespace dyno
