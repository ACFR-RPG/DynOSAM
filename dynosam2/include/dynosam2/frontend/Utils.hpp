#pragma once

#include <pcl/point_types.h>

#include <opencv4/opencv2/opencv.hpp>

#include "dynosam_common/Types.hpp"
#include "dynosam_sensors/CameraParams.hpp"

namespace dyno2 {

struct PoseVelocity {
  Eigen::Vector3d linear;   // World-frame linear velocity [m/s]
  Eigen::Vector3d angular;  // Angular velocity [rad/s]
};

inline PoseVelocity computeBodyVelocity(const gtsam::Pose3& T_WB_prev,
                                        const gtsam::Pose3& T_WB_curr,
                                        double dt) {
  if (dt <= 0.0) {
    throw std::invalid_argument("dt must be positive");
  }

  // Linear velocity in the world frame.
  const Eigen::Vector3d linear =
      (T_WB_curr.translation() - T_WB_prev.translation()) / dt;

  // Relative rotation.
  const gtsam::Rot3 R_delta =
      T_WB_prev.rotation().between(T_WB_curr.rotation());

  // Rotation vector: axis multiplied by angle.
  const Eigen::Vector3d angular = gtsam::Rot3::Logmap(R_delta) / dt;

  return {linear, angular};
}

inline bool checkBounds(const cv::Point2f& point, int rows, int cols) {
  const int x = cvRound(point.x);
  const int y = cvRound(point.y);
  return x >= 0 && x < cols && y >= 0 && y < rows;
}

inline bool checkBoundsAndLabel(const cv::Point2f& point,
                                const cv::Mat& object_masks,
                                ObjectId expected_id, int rows, int cols) {
  const int x = cvRound(point.x);
  const int y = cvRound(point.y);
  return x >= 0 && x < cols && y >= 0 && y < rows &&
         object_masks.at<ObjectId>(y, x) == expected_id;
}

/* @brief Super fast implementation of K^{-1} * kp, returning normalized bearing
 * vector. Basic benchmarking shows at least a 7x speed up over the raw eigen
 * version: (K_inv * gtsam::Vector3(kp(0), kp(1), 1.0)).normzlied().
 *
 * Implementation takes advantage of the structure of the K matrix.
 *
 * @param fx
 * @param fy
 * @param cx
 * @param cy
 * @param kp
 * @return gtsam::Vector3
 */
inline gtsam::Vector3 bearingOptimized(double fx, double fy, double cx,
                                       double cy, const gtsam::Point2& kp) {
  const double nx = (kp.x() - cx) / fx;
  const double ny = (kp.y() - cy) / fy;

  const double inv_norm = 1.0 / std::sqrt(nx * nx + ny * ny + 1.0);

  return gtsam::Vector3(nx * inv_norm, ny * inv_norm, inv_norm);
}

struct ComputeBearing {
  const double fx_, fy_, cx_, cy_;

  ComputeBearing(const CameraParams& params)
      : fx_(params.fx()),
        fy_(params.fy()),
        cx_(params.cu()),
        cy_(params.cv()) {}

  inline gtsam::Vector3 operator()(const gtsam::Point2& kp) const {
    return bearingOptimized(fx_, fy_, cx_, cy_, kp);
  }
};

}  // namespace dyno2
