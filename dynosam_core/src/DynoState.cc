#include "dynosam_core/DynoState.hpp"

#include "dynosam_common/Types.hpp"

namespace dyno {

template <>
std::string to_string(const DoKeyframeType& status) {
  static const std::unordered_map<DoKeyframeType, std::string> map = {
      {DoKeyframeType::NonKeyFrame, "NonKeyFrame"},
      {DoKeyframeType::RegularKeyFrame, "RegularKeyFrame"},
      {DoKeyframeType::AnchorKeyFrame, "AnchorKeyFrame"}};
  return map.at(status);
}

std::ostream& operator<<(std::ostream& os, const DoKeyframeType& status) {
  os << to_string(status);
  return os;
}

template <>
std::string to_string(const TrackingQuality& quality) {
  static const std::unordered_map<TrackingQuality, std::string> map = {
      {TrackingQuality::Good, "Good"},
      {TrackingQuality::Marginal, "Marginal"},
      {TrackingQuality::Lost, "Lost"},
  };
  return map.at(quality);
}

gtsam::Pose3 DynoState::cameraPose() const {
  return camera_trajectory.at(this->frame_id);
}
MotionEstimateMap DynoState::objectMotions() const {
  LOG(FATAL) << "Not implemented!";
}

}  // namespace dyno
