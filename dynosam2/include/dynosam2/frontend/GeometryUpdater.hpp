#pragma once

#include "dynosam2/frontend/FeatureBlockContainer.hpp"
#include "dynosam2/frontend/Frame.hpp"
#include "dynosam2/frontend/Utils.hpp"
#include "dynosam_sensors/Camera.hpp"
#include "dynosam_sensors/ImageContainer.hpp"
#include "dynosam_sensors/RGBDCamera.hpp"

namespace dyno2 {

class GeometryUpdater {
 public:
  //! need more than 8 points for fundamental matrix calc with ransac
  constexpr static size_t kMinStereoMatches{8};

  GeometryUpdater(Camera::Ptr camera, const ImageContainer& images,
                  FeatureBlockContainer& features);

  void calcPoints(FeatureBlockGeometryMap& feature_block_geometries);
  void updateGeometry(FeatureBlockGeometry& feature_block_geometry);

 private:
  void calcPointsRGBD(FeatureBlockGeometryMap& feature_block_geometries);
  void calcPointsStereo(FeatureBlockGeometryMap& feature_block_geometries);

  void updateGeometryRGBD(FeatureBlockGeometry& feature_block_geometry);
  void updateGeometryStereo(FeatureBlockGeometry& feature_block_geometry);

  inline bool checkRGBDGeometry(const double depth) const {
    return std::isfinite(depth) && depth > 0.0;
  }

  // NOTE: keypoint sigma approximates the angular uncertainty
  //  of the pixel measurement (ie the ray can be anyway along the cone created
  //  by the sigma size)
  //  in OKVIS this is done with size0 / f0 * 0.125; where size0 is the keypoint
  //  size however we dont have this
  inline bool checkTwoViewGeometry(const gtsam::Point3& point,
                                   const gtsam::Point2& left_kp,
                                   const gtsam::Point2& right_kp,
                                   double max_reprojection_error = 2.0,
                                   double keypoint_sigma = 1.0) const {
    // Reject points that are too close to either camera.
    constexpr double min_depth = 0.2;
    if (!point.allFinite() || point.z() <= min_depth) {
      return false;
    }

    // // Left-camera reprojection.
    const double u_left =
        compute_bearing_.fx_ * point.x() / point.z() + compute_bearing_.cx_;

    const double v_left =
        compute_bearing_.fy_ * point.y() / point.z() + compute_bearing_.cy_;

    const double reprojection_error_left =
        std::hypot(u_left - left_kp.x(), v_left - left_kp.y());

    // Reject points with excessive reprojection error.
    if (reprojection_error_left > max_reprojection_error) {
      return false;
    }

    const Eigen::Vector3d ray_left = compute_bearing_(left_kp);
    const Eigen::Vector3d point_left = (point.z() / ray_left(2)) * ray_left;
    const Eigen::Vector3d direction_right =
        (point_left - T_LR_.translation()).normalized();
    const Eigen::Vector3d direction_left = ray_left;

    const double focal_length =
        0.5 * (compute_bearing_.fx_ + compute_bearing_.fy_);
    const double sigma = keypoint_sigma / focal_length;
    const double parallax = direction_left.dot(direction_right);

    // Reject insufficient parallax (rays are too nearly parallel).
    // this basically implies the minimum disparity is keypoint_sigma
    if (!std::isfinite(parallax) || parallax > std::cos(sigma)) {
      return false;
    }

    return true;
  }

  // assume all are part of the same object and t
  // matched* vectors will all have the same size (may be < left_kps)
  // as well matched_indexs and will say which index in the original left_kps
  // the matching is for!
  bool computeStereoMatching(const std::vector<cv::Point2f>& left_kps,
                             std::vector<cv::Point2f>& matched_right_kps,
                             std::vector<size_t>& matched_indexs);

  Camera::Ptr camera_;
  ImageContainer images_;
  FeatureBlockContainer& features_;

  // cached camera paramters
  ComputeBearing compute_bearing_;
  double baseline_;
  //! Transform from right to left stereo camera
  gtsam::Pose3 T_LR_;
};

}  // namespace dyno2
