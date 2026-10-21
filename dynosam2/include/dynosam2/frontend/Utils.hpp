#pragma once

#include <pcl/point_types.h>

#include <opencv4/opencv2/opencv.hpp>

#include "dynosam_common/Types.hpp"
#include "dynosam_sensors/CameraParams.hpp"

namespace dyno2 {

inline Velocity3 computeWorldVelocity(const gtsam::Pose3& T_WB_prev,
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

  // return linear and angular velocities in the world frame
  return {linear, T_WB_curr.rotation().rotate(angular)};
}

inline gtsam::Pose3 predictPoseConstantVelocity(const gtsam::Pose3& pose,
                                                const Velocity3& velocity_W,
                                                double dt) {
  if (dt < 0.0) {
    throw std::invalid_argument("Prediction horizon must be non-negative");
  }

  // Constant world-frame linear velocity.
  const Eigen::Vector3d predicted_translation =
      pose.translation() + velocity_W.linear * dt;

  // Constant world-frame angular velocity.
  const Eigen::Vector3d rotation_vector = velocity_W.angular * dt;
  const gtsam::Rot3 delta_rotation = gtsam::Rot3::Expmap(rotation_vector);

  // Left-compose because angular velocity is expressed in the world frame.
  const gtsam::Rot3 predicted_rotation =
      delta_rotation.compose(pose.rotation());

  return gtsam::Pose3(predicted_rotation, predicted_translation);
}

inline bool checkBounds(const cv::Point2f& point, int rows, int cols) {
  const int x = cvRound(point.x);
  const int y = cvRound(point.y);
  return x >= 0 && x < cols && y >= 0 && y < rows;
}

inline bool checkBoundsAndLabel(const cv::Point2f& point,
                                const cv::Mat& object_masks,
                                dyno::ObjectId expected_id, int rows,
                                int cols) {
  const int x = cvRound(point.x);
  const int y = cvRound(point.y);
  return x >= 0 && x < cols && y >= 0 && y < rows &&
         object_masks.at<dyno::ObjectId>(y, x) == expected_id;
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

  ComputeBearing(const dyno::CameraParams& params)
      : fx_(params.fx()),
        fy_(params.fy()),
        cx_(params.cu()),
        cy_(params.cv()) {}

  inline gtsam::Vector3 operator()(const gtsam::Point2& kp) const {
    return bearingOptimized(fx_, fy_, cx_, cy_, kp);
  }
};

}  // namespace dyno2
