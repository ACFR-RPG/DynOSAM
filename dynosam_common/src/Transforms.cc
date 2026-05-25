#include "dynosam_common/Transforms.hpp"

namespace dyno {

gtsam::Pose3 robotToOpenCVTransform() {
  static gtsam::Matrix33 rot;
  rot << 0, -1, 0, 0, 0, -1, 1, 0, 0;
  static gtsam::Pose3 pose(gtsam::Rot3(rot), gtsam::Point3(0, 0, 0));
  return pose;
}

gtsam::Pose3 openCVToRobotTransform() {
  static gtsam::Pose3 transform(robotToOpenCVTransform().inverse());
  return transform;
}

}  // namespace dyno
