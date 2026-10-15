#include "dynosam2/frontend/GeometryUpdater.hpp"

namespace dyno2 {

GeometryUpdater::GeometryUpdater(Camera::Ptr camera,
                                 const ImageContainer& images,
                                 FeatureBlockContainer& features)
    : camera_(camera),
      images_(images),
      features_(features),
      compute_bearing_(camera_->getParams()) {
  std::shared_ptr<RGBDCamera> rgbd_camera = camera_->safeGetRGBDCamera();
  CHECK(rgbd_camera);
  baseline_ = rgbd_camera->baseline();

  if (!std::isfinite(compute_bearing_.fx_) ||
      !std::isfinite(compute_bearing_.fy_) || compute_bearing_.fx_ <= 0.0 ||
      compute_bearing_.fy_ <= 0.0 || !std::isfinite(baseline_) ||
      baseline_ <= 0.0) {
    throw DynosamException("GeometryUpdater failed: invalid camera params!");
  }
}

void GeometryUpdater::calcPoints(FeatureBlockGeometryMap& point_map) {
  if (images_.hasRightRgb()) {
    return calcPointsStereo(point_map);
  } else if (images_.hasDepth()) {
    return calcPointsRGBD(point_map);
  }
}

void GeometryUpdater::updateGeometry(FeatureBlockGeometry& local_geometry) {}

void GeometryUpdater::calcPointsRGBD(FeatureBlockGeometryMap& point_map) {}
void GeometryUpdater::calcPointsStereo(FeatureBlockGeometryMap& point_map) {}

void GeometryUpdater::updateGeometryRGBD(FeatureBlockGeometry& local_geometry) {

}
void GeometryUpdater::updateGeometryStereo(
    FeatureBlockGeometry& local_geometry) {}

}  // namespace dyno2
