#pragma once

#include <gtsam/geometry/Pose3.h>

#include <eigen3/Eigen/Core>

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

// should be of type double and of size 3xN
template <typename Derived>
inline void transformTo(const gtsam::Pose3& T_ij,
                        Eigen::MatrixBase<Derived>& P) {
  static_assert(std::is_same_v<typename Derived::Scalar, double>);
  static_assert(Derived::RowsAtCompileTime == 3);

  const gtsam::Matrix33& R = T_ij.rotation().matrix();
  const gtsam::Vector3& t = T_ij.translation();

  P = R * P;
  P.colwise() += t;
}

/**
 * @brief Super fast batch implementation of pose * points using Eigen map
 * operations rather than linear operations.
 *
 * From simple benchmarks we get 60% improvements in speed vs a loop for vectors
 * as small as 200!
 *
 * @param T_ij transform from j to i (that is p_i = T_ij * p_j)
 * @param points
 * @return gtsam::Point3Vector
 */
inline void transformTo(const gtsam::Pose3& T_ij,
                        const gtsam::Point3Vector& points_j,
                        gtsam::Point3Vector& points_i) {
  if (points_j.empty()) {
    return;
  }
  points_i = points_j;
  Eigen::Map<Eigen::Matrix3Xd> P(points_i[0].data(), 3,
                                 static_cast<Eigen::Index>(points_i.size()));

  transformTo(T_ij, P);
}

}  // namespace dyno
