#pragma once

#include "dynosam2/frontend/FeatureBlockContainer.hpp"
#include "dynosam2/frontend/Frame.hpp"

namespace dyno2 {

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

class GeometryUpdater {
 public:
  //! need more than 8 points for fundamental matrix calc with ransac
  constexpr static size_t kMinStereoMatches{8};

  GeometryUpdater(Camera::Ptr camera, const ImageContainer& images,
                  FeatureBlockContainer& features);

  // TODO: return outliers and mark features separately!
  void calcPoints(FeatureBlockGeometryMap& point_map);
  void updateGeometry(FeatureBlockGeometry& local_geometry);

 private:
  void calcPointsRGBD(FeatureBlockGeometryMap& point_map);
  void calcPointsStereo(FeatureBlockGeometryMap& point_map);

  void updateGeometryRGBD(FeatureBlockGeometry& local_geometry);
  void updateGeometryStereo(FeatureBlockGeometry& local_geometry);

  // NOTE: keypoint sigma approximates the angular uncertainty
  //  of the pixel measurement (ie the ray can be anyway along the cone created
  //  by the sigma size)
  //  in OKVIS this is done with size0 / f0 * 0.125; where size0 is the keypoint
  //  size however we dont have this
  inline bool checkTwoViewGeometry(const gtsam::Point3& point_left,
                                   const gtsam::Point2& left_kp,
                                   const gtsam::Point2& right_kp,
                                   double max_reprojection_error = 2.0,
                                   double keypoint_sigma = 1.0) {
    // Reject points that are too close to either camera.
    constexpr double min_depth = 0.2;
    if (!point_left.allFinite() || point_left.z() <= min_depth) {
      return false;
    }

    // // Left-camera reprojection.
    const double u_left =
        compute_bearing_.fx_ * point_left.x() / point_left.z() +
        compute_bearing_.cx_;

    const double v_left =
        compute_bearing_.fy_ * point_left.y() / point_left.z() +
        compute_bearing_.cv_;

    const double reprojection_error_left =
        std::hypot(u_left - left_kp.x(), v_left - left_kp.y());

    // Reject points with excessive reprojection error.
    if (reprojection_error_left > max_reprojection_error) {
      return false;
    }

    // Calculate the parallax angle between the two viewing rays.
    const Eigen::Vector3d ray_left = compute_bearing_(left_kp);
    const Eigen::Vector3d ray_right = compute_bearing_(right_kp);

    const double cos_parallax = std::clamp(
        ray_left.dot(ray_right) / (ray_left.norm() * ray_right.norm()), -1.0,
        1.0);

    const double sigma = keypoint_sigma_px / compute_bearing_.fx_;

    // Reject insufficient parallax (rays are too nearly parallel).
    if (!std::isfinite(cos_parallax) || cos_parallax > std::cos(6.0 * sigma)) {
      return false;
    }
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
};

}  // namespace dyno2
