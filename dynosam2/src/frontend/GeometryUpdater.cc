#include "dynosam2/frontend/GeometryUpdater.hpp"

#include "dynosam2/frontend/SparseFlow.hpp"
#include "dynosam2/frontend/Utils.hpp"
#include "dynosam_common/utils/TimingStats.hpp"

namespace dyno2 {

GeometryUpdater::GeometryUpdater(Camera::Ptr camera,
                                 const ImageContainer& images,
                                 FeatureBlockContainer& features)
    : camera_(camera),
      images_(images),
      features_(features),
      compute_bearing_(camera_->getParams()),
      baseline_(dyno::RGBDCamera::Baseline(camera_)),
      T_LR_(dyno::RGBDCamera::StereoExtrinsics(camera_)) {
  if (!std::isfinite(compute_bearing_.fx_) ||
      !std::isfinite(compute_bearing_.fy_) || compute_bearing_.fx_ <= 0.0 ||
      compute_bearing_.fy_ <= 0.0 || !std::isfinite(baseline_) ||
      baseline_ <= 0.0) {
    throw DynosamException("GeometryUpdater failed: invalid camera params!");
  }
}

void GeometryUpdater::calcPoints(
    FeatureBlockGeometryMap& feature_block_geometries) {
  dyno::utils::ChronoTimingStats timer("geometry_updater.calc_points");
  if (images_.hasRightRgb()) {
    return calcPointsStereo(feature_block_geometries);
  } else if (images_.hasDepth()) {
    return calcPointsRGBD(feature_block_geometries);
  }
}

void GeometryUpdater::updateGeometry(
    FeatureBlockGeometry& feature_block_geometry) {
  if (images_.hasRightRgb()) {
    return updateGeometryStereo(feature_block_geometry);
  } else if (images_.hasDepth()) {
    return updateGeometryRGBD(feature_block_geometry);
  }
}

void GeometryUpdater::calcPointsRGBD(
    FeatureBlockGeometryMap& feature_block_geometries) {
  std::shared_ptr<RGBDCamera> rgbd_camera = camera_->safeGetRGBDCamera();

  const cv::Mat& depth_img = images_.depth();

  // iterate over by view so we can avoid lookup and memory allocation
  // for each FeatureBlockGeometry object
  for (auto& object_view : features_.objectViews()) {
    auto object_id = object_view.objectId();

    // create new FeatureBlockGeometry and allocate memory
    size_t num_points = object_view.size();

    FeatureBlockGeometry& feature_block_geometry =
        feature_block_geometries[object_id];
    feature_block_geometry.lmks_C.reserve(num_points);
    feature_block_geometry.left_kps.reserve(num_points);
    feature_block_geometry.left_kps_previous.reserve(num_points);
    feature_block_geometry.right_pixel.reserve(num_points);
    feature_block_geometry.ids.reserve(num_points);
    feature_block_geometry.fc_indices.reserve(num_points);

    auto points = object_view.points();
    // expect previous points to be updated in the container
    auto previous_points = object_view.previousPoints();
    auto ids = object_view.ids();
    auto inliers = object_view.inlier();

    // the starting offset of the block as stored in the feautre block container
    // allows us to retrieve the global index!
    size_t offset = object_view.layout().begin;
    for (size_t i = 0; i < num_points; i++) {
      if (!inliers[i]) {
        continue;
      }

      auto tracklet_id = ids[i];
      const auto& kp_cv = points[i];

      // global index in the feature container (ie. features.points[i])
      size_t global_index = i + offset;
      // sanity check out local index matches our global one
      CHECK_EQ(features_.ids[global_index], tracklet_id);

      const Depth depth = depth_img.at<Depth>(kp_cv);

      if (!checkRGBDGeometry(depth)) {
        inliers[i] = 0;
        continue;
      }

      const gtsam::Point2 left_kp = utils::cvPointToGtsam(kp_cv);
      const gtsam::Point2 right_kp = rgbd_camera->rightKeypoint(depth, left_kp);

      const gtsam::Point2 left_kp_previous =
          utils::cvPointToGtsam(previous_points[i]);

      if (!rgbd_camera->isKeypointContained(right_kp, depth)) {
        inliers[i] = 0;
        continue;
      }

      double right_pixel = right_kp(0);

      // fast version of the back projection function
      Landmark lmk(0, 0, 0);
      rgbd_camera->backProject2(left_kp, depth, lmk);

      if (!checkTwoViewGeometry(lmk, left_kp, right_kp)) {
        inliers[i] = 0;
        continue;
      }

      size_t local_index = feature_block_geometry.lmks_C.size();
      feature_block_geometry.lmks_C.push_back(lmk);
      feature_block_geometry.left_kps.push_back(left_kp);
      feature_block_geometry.left_kps_previous.push_back(left_kp_previous);
      feature_block_geometry.right_pixel.push_back(right_pixel);
      feature_block_geometry.ids.push_back(tracklet_id);
      feature_block_geometry.fc_indices.push_back(global_index);

      // mapping of tracklet id to location in the local points container
      feature_block_geometry.local_indices[tracklet_id] = local_index;
    }
  }
}

void GeometryUpdater::calcPointsStereo(FeatureBlockGeometryMap& point_map) {
  std::shared_ptr<RGBDCamera> rgbd_camera = camera_->safeGetRGBDCamera();

  for (auto& object_view : features_.objectViews()) {
    auto object_id = object_view.objectId();

    // create new FeatureBlockGeometry and allocate memory
    size_t num_points = object_view.size();

    auto points = object_view.points();
    // expect previous points to be updated in the container
    auto previous_points = object_view.previousPoints();
    auto ids = object_view.ids();
    auto inliers = object_view.inlier();

    std::vector<cv::Point2f> left_kps;
    left_kps.reserve(num_points);

    // object view index
    std::vector<size_t> left_kp_indices;
    left_kp_indices.reserve(num_points);

    TrackletIds tracklet_ids;
    tracklet_ids.reserve(num_points);
    // the starting offset of the block as stored in the feautre block container
    // allows us to retrieve the global index!
    size_t offset = object_view.layout().begin;
    for (size_t i = 0; i < num_points; i++) {
      if (!inliers[i]) {
        continue;
      }
      left_kps.push_back(points[i]);
      left_kp_indices.push_back(i);
      tracklet_ids.push_back(ids[i]);

      // maybe a hack?
      // mark all as outliers
      // then remark the matched ones as inliers after matching!
      inliers[i] = 0;
    }

    // perform stereo matching
    std::vector<cv::Point2f> matched_right_kps;
    // index in the left_kps/trackletids in which we have a match
    std::vector<size_t> matched_indexs;
    if (!computeStereoMatching(left_kps, matched_right_kps, matched_indexs)) {
      LOG(INFO) << "Stereo matchinf failed!";
      continue;
    }

    const size_t num_stereo_matches = matched_indexs.size();
    LOG(INFO) << "Stereo matches =" << num_stereo_matches;

    FeatureBlockGeometry& feature_block_geometry = point_map[object_id];
    feature_block_geometry.lmks_C.reserve(num_stereo_matches);
    feature_block_geometry.left_kps.reserve(num_stereo_matches);
    feature_block_geometry.left_kps_previous.reserve(num_stereo_matches);
    feature_block_geometry.right_pixel.reserve(num_stereo_matches);
    feature_block_geometry.ids.reserve(num_stereo_matches);
    feature_block_geometry.fc_indices.reserve(num_stereo_matches);

    for (size_t i = 0; i < matched_indexs.size(); i++) {
      const size_t matched_index = matched_indexs[i];
      // which index this point is found at in the local obect view
      const size_t object_view_index = left_kp_indices[matched_index];

      // global index in the feature container (ie. features.points[i])
      const size_t fc_index = object_view_index + offset;

      const TrackletId id = tracklet_ids[matched_index];
      // sanity check out local index matches our global one
      CHECK_EQ(features_.ids[fc_index], id);
      CHECK_EQ(ids[object_view_index], id);

      const gtsam::Point2 left_kp =
          utils::cvPointToGtsam(left_kps[matched_index]);
      const gtsam::Point2 right_kp =
          utils::cvPointToGtsam(matched_right_kps[i]);
      const gtsam::Point2 left_kp_previous =
          utils::cvPointToGtsam(previous_points[object_view_index]);

      const double uL = left_kp.x();
      const double uR = right_kp.x();
      const double disparity = uL - uR;
      const double depth = rgbd_camera->depthFromDisparity(disparity);

      // fast version of the back projection function
      Landmark lmk(0, 0, 0);
      rgbd_camera->backProject2(left_kp, depth, lmk);

      if (!checkTwoViewGeometry(lmk, left_kp, right_kp)) {
        // inliers[object_view_index] = 0;
        continue;
      }

      const size_t local_index = feature_block_geometry.lmks_C.size();
      feature_block_geometry.lmks_C.push_back(lmk);
      feature_block_geometry.left_kps.push_back(left_kp);
      feature_block_geometry.left_kps_previous.push_back(left_kp_previous);
      feature_block_geometry.right_pixel.push_back(uR);
      feature_block_geometry.ids.push_back(id);
      feature_block_geometry.fc_indices.push_back(fc_index);

      // mapping of tracklet id to location in the local points container
      feature_block_geometry.local_indices[id] = local_index;

      // remark as inlier
      inliers[object_view_index] = 1;
    }
  }
}

void GeometryUpdater::updateGeometryRGBD(
    FeatureBlockGeometry& feature_block_geometry) {
  std::shared_ptr<RGBDCamera> rgbd_camera = camera_->safeGetRGBDCamera();
  const cv::Mat& depth_img = images_.depth();

  size_t num_points = feature_block_geometry.lmks_C.size();
  for (size_t i = 0; i < num_points; i++) {
    const Index fc_index = feature_block_geometry.fc_indices[i];
    const TrackletId id = feature_block_geometry.ids[i];

    // check internal consistency
    CHECK_EQ(id, features_.ids[fc_index]);
    CHECK_EQ(feature_block_geometry.local_indices.at(id), i);
    if (!features_.inlier[fc_index]) {
      continue;
    }

    // should check consistency with what the geometry is MEANT to be
    // and if all featues are of the same id!
    const ObjectId object_id = features_.object_ids[fc_index];

    const gtsam::Point2 left_kp = feature_block_geometry.left_kps[i];
    const Depth depth =
        depth_img.at<Depth>(utils::gtsamPointToCv<float>(left_kp));

    if (!checkRGBDGeometry(depth)) {
      features_.inlier[i] = 0;
      continue;
    }

    const gtsam::Point2 right_kp = rgbd_camera->rightKeypoint(depth, left_kp);

    if (!rgbd_camera->isKeypointContained(right_kp, depth)) {
      features_.inlier[fc_index] = 0;
      continue;
    }

    // update geometry
    Landmark lmk(0, 0, 0);
    rgbd_camera->backProject2(left_kp, depth, lmk);

    if (!checkTwoViewGeometry(lmk, left_kp, right_kp)) {
      features_.inlier[fc_index] = 0;
      continue;
    }

    feature_block_geometry.lmks_C[i] = lmk;
    feature_block_geometry.right_pixel[i] = right_kp(0);
  }
}
void GeometryUpdater::updateGeometryStereo(
    FeatureBlockGeometry& feature_block_geometry) {
  std::shared_ptr<RGBDCamera> rgbd_camera = camera_->safeGetRGBDCamera();

  const size_t num_points = feature_block_geometry.lmks_C.size();
  std::vector<cv::Point2f> left_kps;
  left_kps.reserve(num_points);

  std::vector<size_t> left_kp_indices;
  left_kp_indices.reserve(num_points);

  TrackletIds tracklet_ids;
  tracklet_ids.reserve(num_points);

  for (size_t i = 0; i < num_points; i++) {
    Index fc_index = feature_block_geometry.fc_indices[i];
    TrackletId id = feature_block_geometry.ids[i];

    // check internal consistency
    CHECK_EQ(id, features_.ids[fc_index]);
    CHECK_EQ(feature_block_geometry.local_indices.at(id), i);
    if (!features_.inlier[fc_index]) {
      continue;
    }

    const gtsam::Point2 left_kp = feature_block_geometry.left_kps[i];
    left_kps.push_back(utils::gtsamPointToCv<float>(left_kp));
    left_kp_indices.push_back(i);
    tracklet_ids.push_back(id);

    // mark all featues as outliers here
    // then remark as inliers for those with matches!
    // this ensures that only features with matches are marked inliers
    // and is faster (probably) then computing the negative subset
    // to mark those without matches as inliers!
    features_.inlier[fc_index] = 0;
  }

  // perform stereo matching
  std::vector<cv::Point2f> matched_right_kps;
  // contains the index for which a right kp is matched too
  std::vector<size_t> matched_indexs;
  if (!computeStereoMatching(left_kps, matched_right_kps, matched_indexs)) {
    // failed!
    return;
  }

  for (size_t i = 0; i < matched_indexs.size(); i++) {
    const size_t matched_index = matched_indexs[i];
    const TrackletId id = tracklet_ids[matched_index];
    // index where the point can be found in the original feature geometry
    const size_t local_index = left_kp_indices[matched_index];

    const Index fc_index = feature_block_geometry.fc_indices[local_index];
    // sanity check out local index matches our global one
    CHECK_EQ(features_.ids[fc_index], id);

    const gtsam::Point2 left_kp = feature_block_geometry.left_kps[local_index];
    const gtsam::Point2 right_kp = utils::cvPointToGtsam(matched_right_kps[i]);

    // should check consistency with what the geometry is MEANT to be
    // and if all featues are of the same id!
    const ObjectId object_id = features_.object_ids[fc_index];

    const double uL = left_kp.x();
    const double uR = right_kp.x();
    const double disparity = uL - uR;
    const double depth = rgbd_camera->depthFromDisparity(disparity);

    // fast version of the back projection function
    Landmark lmk(0, 0, 0);
    rgbd_camera->backProject2(left_kp, depth, lmk);

    if (!checkTwoViewGeometry(lmk, left_kp, right_kp)) {
      // features_.inlier[fc_index] = 0;
      continue;
    }

    feature_block_geometry.lmks_C[local_index] = lmk;
    feature_block_geometry.right_pixel[local_index] = uR;

    // re mark as inlier!
    features_.inlier[fc_index] = 1;
  }
}

bool GeometryUpdater::computeStereoMatching(
    const std::vector<cv::Point2f>& left_kps,
    std::vector<cv::Point2f>& matched_right_kps,
    std::vector<size_t>& matched_indexs) {
  CHECK(images_.hasRightRgb());
  if (left_kps.size() < kMinStereoMatches) {
    return false;
  }

  OpticalFlowLK_CPU optical_flow(30);

  const WrappedRGBMono& wrapped_left = images_.rgb();
  const WrappedRGBMono& wrapped_right = images_.rightRgb();

  const int rows = wrapped_left.image().rows;
  const int cols = wrapped_left.image().cols;

  // // the image wrapper will cache the image pyramid!
  // const ImagePyramid left_image_pyr = wrapped_left.computeImagePyramid(
  //     optical_flow.win_size, optical_flow.max_level);

  // const ImagePyramid right_image_pyr = wrapped_right.computeImagePyramid(
  //     optical_flow.win_size, optical_flow.max_level);

  // track left to right
  // OpticalFlowLK_CPU::Result flow_result =
  //     optical_flow(left_image_pyr.levels, right_image_pyr.levels, left_kps);
  OpticalFlowLK_CPU::Result flow_result =
      optical_flow(wrapped_left, wrapped_right, left_kps);

  const size_t num_flows = flow_result.size();
  CHECK_EQ(num_flows, left_kps.size());

  std::vector<cv::Point2f> good_left, good_right;
  good_left.reserve(num_flows);
  good_right.reserve(num_flows);

  std::vector<size_t> good_indices;
  good_indices.reserve(num_flows);

  const std::vector<cv::Point2f>& predicted_points =
      flow_result.predictedPoints();
  const std::vector<cv::Point2f>& from_points = flow_result.fromPoints();
  for (size_t i = 0; i < flow_result.size(); i++) {
    if (!flow_result.isGood(i)) {
      continue;
    }

    const cv::Point2f& predicted_pt = predicted_points[i];
    if (!checkBounds(predicted_pt, rows, cols)) {
      continue;
    }

    good_left.push_back(from_points[i]);
    good_right.push_back(predicted_pt);
    good_indices.push_back(i);
  }

  size_t num_good = good_indices.size();
  if (good_indices.size() < kMinStereoMatches) {
    return false;
  }

  matched_right_kps.reserve(num_good);
  matched_indexs.reserve(num_good);

  std::vector<uchar> epipolar_inliers;
  cv::findFundamentalMat(good_left, good_right, cv::FM_RANSAC, 1.0, 0.99,
                         epipolar_inliers);

  CHECK_EQ(epipolar_inliers.size(), num_good);

  for (size_t i = 0; i < epipolar_inliers.size(); ++i) {
    if (epipolar_inliers[i]) {
      matched_right_kps.push_back(good_right[i]);

      // indices in the original entries (ie left_kps)
      matched_indexs.push_back(good_indices[i]);
    }
  }

  return matched_indexs.size() > kMinStereoMatches;
}

}  // namespace dyno2
