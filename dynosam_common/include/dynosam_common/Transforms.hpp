#pragma once

#include <gtsam/geometry/Pose3.h>

namespace dyno {

/**
 * @brief Transformation converting from
 * ROS Frame    (x-forward, y-left, z-up) to
 * OpenCV Frame (x-right, y-down, z-forward)
 * ROS    ->  OpenCV
 * x     ->     z
 * y     ->    -x
 * z     ->    -y
 *
 * @return gtsam::Pose3
 */
gtsam::Pose3 robotToOpenCVTransform();

/**
 * @brief Transform converting from OpenCV frame (z-forward) to ROS frame
 * convention (z-up)
 *
 * @return gtsam::Pose3
 */
gtsam::Pose3 openCVToRobotTransform();

/**
 * @brief  Helper funtion to change basis from frame source to frame target.
 *
 * @param target_frame const gtsam::Pose3. Transformation(Rotation only;
 * translation is zero) matrix between target and source.
 * @param source_frame const gtsam::Pose3 Transformation(Rotation and
 * translation) inside source frame. It is any arbritary transformation in
 * source frame.
 * @return gtsam::Pose3
 */
inline gtsam::Pose3 changeBasis(const gtsam::Pose3& target_frame,
                                const gtsam::Pose3& source_frame) {
  return target_frame * source_frame * target_frame.inverse();
}

}  // namespace dyno
