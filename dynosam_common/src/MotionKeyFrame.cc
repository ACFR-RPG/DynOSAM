#include "dynosam_common/MotionKeyFrame.hpp"

namespace dyno {

const HybridKeyFrameUpdate::Object* HybridKeyFrameUpdate::getObject(
    ObjectId object_id) const {
  auto it = std::find_if(
      object_infos.begin(), object_infos.end(),
      [&object_id](const Object& obj) { return obj.object_id == object_id; });

  if (it != object_infos.end()) {
    return &(*it);
  } else {
    return nullptr;
  }
}

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
