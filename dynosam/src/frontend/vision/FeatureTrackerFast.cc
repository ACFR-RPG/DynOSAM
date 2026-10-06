#include "dynosam/frontend/vision/FeatureTrackerFast.hpp"

#include "dynosam_common/Types.hpp"
#include "dynosam_nn/YoloV8ObjectDetector.hpp"

namespace dyno {

void FeatureBlockContainer::FeatureData::checkSizes() const {
  const size_t n = points.size();

  if (previous_points.size() != n || ids.size() != n || inlier.size() != n ||
      age.size() != n) {
    throw std::invalid_argument(
        "FeatureData: all feature arrays must have the same size");
  }
}

void FeatureBlockContainer::FeatureData::reserve(size_t n) {
  points.reserve(n);
  previous_points.reserve(n);
  ids.reserve(n);
  age.reserve(n);
  inlier.reserve(n);
  errors.reserve(n);
}

void FeatureBlockContainer::FeatureData::resize(size_t n) {
  points.resize(n);
  previous_points.resize(n);
  ids.resize(n);
  age.resize(n);
  inlier.resize(n);
  errors.resize(n);
}

// ---------------------------------------------------------------------
// Convenient bulk assignment
// ---------------------------------------------------------------------

void FeatureBlockContainer::BlockView::copyFrom(const FeatureData& data) {
  data.checkSizes();

  if (data.size() != size()) {
    throw std::invalid_argument(
        "FeatureSet::BlockView::copyFrom: "
        "FeatureData size does not match object size");
  }

  copyBlock(points(), data.points.data(), size());
  copyBlock(previousPoints(), data.previous_points.data(), size());
  copyBlock(ids(), data.ids.data(), size());
  copyBlock(age(), data.age.data(), size());
  copyBlock(inlier(), data.inlier.data(), size());
  copyBlock(errors(), data.errors.data(), size());
}

FeatureBlockContainer::FeatureBlockContainer(
    std::initializer_list<BlockDim> specs) {
  initialize(specs.begin(), specs.end());
}

FeatureBlockContainer::FeatureBlockContainer(
    const std::vector<BlockDim>& specs) {
  initialize(specs.begin(), specs.end());
}

size_t FeatureBlockContainer::size() const { return points.size(); }
size_t FeatureBlockContainer::objectCount() const {
  return block_layout_.size();
}

std::vector<FeatureBlockContainer::BlockView>
FeatureBlockContainer::objectViews() const {
  std::vector<BlockView> views;
  views.reserve(objectCount());

  for (const BlockLayout& block : block_layout_) {
    views.push_back(objectView(block.object_id));
  }

  return views;
}

bool FeatureBlockContainer::containsObject(ObjectId object_id) const {
  return object_lookup_.find(object_id) != object_lookup_.end();
}

FeatureBlockContainer FeatureBlockContainer::merge(
    const FeatureBlockContainer& other) const {
  // check for tracklet ids dupliactes
  // TODO: comment out for now - this makes creating empty or initalised but
  // inassigned FeatureSets invalid becuase all objectids/trackletids will
  // have the same id!
  //  std::unordered_set<int> feature_ids;
  //  feature_ids.reserve(ids.size() + other.ids.size());

  // for (const int id : ids)
  // {
  //     feature_ids.insert(id);
  // }

  // for (const int id : other.ids)
  // {
  //     if (!feature_ids.insert(id).second)
  //     {
  //         throw std::invalid_argument(
  //             "FeatureSet::merge: duplicate feature ID " +
  //             std::to_string(id));
  //     }
  // }

  // -------------------------------------------------------------------------
  // Build the resulting object layout.
  //
  // Existing objects retain their order.
  // New objects from `other` are appended in `other`'s order.
  // -------------------------------------------------------------------------

  std::vector<BlockDim> specs;
  specs.reserve(block_layout_.size() + other.block_layout_.size());

  // Existing objects.
  for (const BlockLayout& object : block_layout_) {
    const auto other_it = other.object_lookup_.find(object.object_id);

    const size_t other_size = other_it != other.object_lookup_.end()
                                  ? other.block_layout_[other_it->second].size()
                                  : 0;

    specs.push_back({object.object_id, object.size() + other_size});
  }

  // Objects which only exist in `other`.
  for (const BlockLayout& object : other.block_layout_) {
    if (object_lookup_.find(object.object_id) == object_lookup_.end()) {
      specs.push_back({object.object_id, object.size()});
    }
  }

  // -------------------------------------------------------------------------
  // Allocate the final FeatureSet exactly once.
  // -------------------------------------------------------------------------

  FeatureBlockContainer result(specs);

  // -------------------------------------------------------------------------
  // Copy the existing features into their final locations.
  // -------------------------------------------------------------------------

  for (const BlockLayout& object : block_layout_) {
    const BlockView source = objectView(object.object_id);

    BlockView destination = result.objectView(object.object_id);

    copyFeatures(destination, source);
  }

  // -------------------------------------------------------------------------
  // Append features from `other`.
  //
  // Existing objects are appended after their existing features.
  // New-only objects are copied starting at offset zero.
  // -------------------------------------------------------------------------

  for (const BlockLayout& other_object : other.block_layout_) {
    const BlockView source = other.objectView(other_object.object_id);

    BlockView destination = result.objectView(other_object.object_id);

    const auto existing_it = object_lookup_.find(other_object.object_id);

    const size_t destination_offset =
        existing_it != object_lookup_.end()
            ? block_layout_[existing_it->second].size()
            : 0;

    if (source.size() == 0) continue;

    copyFeatures(destination, source, destination_offset);
  }

  // object_ids are established by the FeatureSet constructor and therefore
  // don't need to be copied during the merge.

  result.checkInvariants();

  return result;
}

FeatureBlockContainer::BlockView FeatureBlockContainer::objectView(
    ObjectId object_id) {
  const BlockLayout& metadata = objectMetadata(object_id);
  return BlockView(this, metadata);
}

const FeatureBlockContainer::BlockView FeatureBlockContainer::objectView(
    ObjectId object_id) const {
  const BlockLayout& metadata = objectMetadata(object_id);
  return BlockView(const_cast<FeatureBlockContainer*>(this), metadata);
}

std::string FeatureBlockContainer::debugInfoString() const {
  std::stringstream ss;
  ss << "FeatureSet"
     << " | total_features=" << size() << " | objects=" << objectCount()
     << '\n';

  for (const BlockLayout& object : block_layout_) {
    ss << "  object_id=" << object.object_id << " | size=" << object.size()
       << " | range=[" << object.begin << ", " << object.end << ")" << '\n';
  }
  return ss.str();
}

void FeatureBlockContainer::checkInvariants() const {
#ifndef NDEBUG
  const size_t n = points.size();

  assert(previous_points.size() == n);
  assert(ids.size() == n);
  assert(object_ids.size() == n);
  assert(inlier.size() == n);
  assert(errors.size() == n);
  assert(age.size() == n);

  size_t expected_begin = 0;

  for (const BlockLayout& object : block_layout_) {
    assert(object.begin == expected_begin);
    assert(object.begin <= object.end);
    assert(object.end <= n);

    for (size_t i = object.begin; i < object.end; ++i) {
      assert(object_ids[i] == object.object_id);
    }

    expected_begin = object.end;
  }

  assert(expected_begin == n);
#endif
}

FeatureBlockContainer FeatureBlockContainer::reduceToInliers(
    IndexMapping* index_mapping) const {
  std::vector<BlockDim> specs;
  specs.reserve(block_layout_.size());

  std::vector<size_t> inlier_counts(block_layout_.size(), 0);

  size_t total_inliers = 0;

  for (size_t block_index = 0; block_index < block_layout_.size();
       ++block_index) {
    const BlockLayout& block = block_layout_[block_index];

    size_t count = 0;

    for (size_t i = block.begin; i < block.end; ++i) {
      count += static_cast<size_t>(inlier[i] != 0);
    }

    inlier_counts[block_index] = count;

    if (count > 0) {
      specs.push_back({block.object_id, count});
      total_inliers += count;
    }
  }

  // if no inliers no need to copy the data across
  // IndexMapping is not updated as no change to index's
  // TODO: (is this the desired behaviour?)
  if (total_inliers == this->size()) {
    return *this;
  }

  FeatureBlockContainer result(specs);

  if (index_mapping != nullptr) {
    index_mapping->clear();
  }

  size_t destination_block_index = 0;

  for (size_t source_block_index = 0; source_block_index < block_layout_.size();
       ++source_block_index) {
    const size_t count = inlier_counts[source_block_index];

    if (count == 0) {
      continue;
    }

    const BlockLayout& source_block = block_layout_[source_block_index];

    const BlockLayout& destination_block =
        result.block_layout_[destination_block_index];

    size_t destination_index = destination_block.begin;

    for (size_t source_index = source_block.begin;
         source_index < source_block.end; ++source_index) {
      if (inlier[source_index] == 0) {
        continue;
      }

      result.points[destination_index] = points[source_index];
      result.previous_points[destination_index] = previous_points[source_index];
      result.ids[destination_index] = ids[source_index];
      result.age[destination_index] = age[source_index];

      // initialize() already populated this correctly, but copying it
      // explicitly keeps this function independent of that behaviour.
      result.object_ids[destination_index] = object_ids[source_index];

      // Everything in the result is an inlier.
      result.inlier[destination_index] = 1;

      result.errors[destination_index] = errors[source_index];

      if (index_mapping != nullptr) {
        index_mapping->operator[](source_index) = destination_index;
      }

      ++destination_index;
    }

    ++destination_block_index;
  }

  result.checkInvariants();

  return result;
}

FeatureBlockContainer& FeatureBlockContainer::reduceToInliersInplace(
    IndexMapping* index_mapping) {
  FeatureBlockContainer filtered = reduceToInliers(index_mapping);
  *this = std::move(filtered);
  return *this;
}

void FeatureBlockContainer::copyFeatures(BlockView destination,
                                         const BlockView& source,
                                         size_t destination_offset) {
  copyBlock(destination.points() + destination_offset, source.points(),
            source.size());

  copyBlock(destination.previousPoints() + destination_offset,
            source.previousPoints(), source.size());

  copyBlock(destination.ids() + destination_offset, source.ids(),
            source.size());

  copyBlock(destination.age() + destination_offset, source.age(),
            source.size());

  copyBlock(destination.inlier() + destination_offset, source.inlier(),
            source.size());

  copyBlock(destination.errors() + destination_offset, source.errors(),
            source.size());
}

bool findMatches(const std::vector<FeatureBlockView>& first_views,
                 const std::vector<FeatureBlockView>& second_views,
                 IndexMatches& matches) {
  matches.clear();

  gtsam::FastMap<TrackletId, Index> first_index_map;

  for (const FeatureBlockView& view : first_views) {
    const auto* ids = view.ids();
    const auto* inliers = view.inlier();
    for (size_t i = 0; i < view.size(); i++) {
      if (inliers[i]) {
        first_index_map[ids[i]] = i;
      }
    }
  }

  matches.reserve(first_index_map.size());
  for (const FeatureBlockView& view : second_views) {
    const auto* ids = view.ids();
    const auto* inliers = view.inlier();

    for (size_t i = 0; i < view.size(); i++) {
      auto tracklet_id = ids[i];
      auto it = first_index_map.find(tracklet_id);
      if (inliers[i] && it != first_index_map.end()) {
        matches.push_back(std::make_pair(it->second, i));
      }
    }
  }

  return !matches.empty();
}

// FrameFast::FrameFast(Camera::Ptr camera,
//     const ImageContainer& image_container,
//     const ObjectDetectionResult& object_detection,
//     const FeatureBlockContainer& features)
//   : camera_(camera),
//     images_(image_container),
//     object_detection_(object_detection),
//     features_(features)
// {
//   computeDepths();
// }

// bool FrameFast::findMatches(const FrameFast& first, const FrameFast& second,
// IndexMatches& matches) {
//   return dyno::findMatches(first.features().objectViews(),
//   second.features().objectViews(), matches);
// }
// bool FrameFast::findMatches(const FrameFast& first, const FrameFast& second,
// ObjectId object_id, IndexMatches& matches) {
//   if(first.containsObject(object_id) && second.containsObject(object_id)) {
//     return dyno::findMatches({first.features().objectView(object_id)},
//     {second.features().objectView(object_id)}, matches);
//   }
//   return false;
// }

// void FrameFast::computeDepths() {
//   // for now just RGBD!
//   std::shared_ptr<RGBDCamera> rgbd_camera = camera_->safeGetRGBDCamera();
//   CHECK_NOTNULL(rgbd_camera);

//   const cv::Mat& depth = images_.depth();
//   for(size_t i = 0; i < features_.size(); i++) {
//     const Depth depth = depth.at<Depth>(features_.points[i]);
//   }
// }

DepthUpdaterFast::DepthUpdaterFast(const DepthThresholds& params,
                                   Camera::Ptr camera,
                                   const ImageContainer& images,
                                   FeatureBlockContainer& features)
    : params_(params), camera_(camera), images_(images), features_(features) {
  std::shared_ptr<RGBDCamera> rgbd_camera = camera_->safeGetRGBDCamera();
  CHECK(rgbd_camera);
  baseline_ = rgbd_camera->baseline();

  const auto& camera_params = camera_->getParams();
  fx_ = camera_params.fx();
  fy_ = camera_params.fy();
  cu_ = camera_params.cu();
  cv_ = camera_params.cv();
}

void DepthUpdaterFast::calcPoints(FrameGeometryMap& point_map) {
  if (images_.hasRightRgb()) {
    return calcPointsStereo(point_map);
  } else if (images_.hasDepth()) {
    return calcPointsRGBD(point_map);
  }
}

void DepthUpdaterFast::calcPointsRGBD(FrameGeometryMap& point_map) {
  utils::ChronoTimingStats feature_track_t("depth_updater_fast.calcPointsRGBD");
  std::shared_ptr<RGBDCamera> rgbd_camera = camera_->safeGetRGBDCamera();

  const cv::Mat& depth_img = images_.depth();

  // iterate over by view so we can avoid lookup and memory allocation
  // for each FrameGeometry object
  for (auto& object_view : features_.objectViews()) {
    auto object_id = object_view.objectId();

    // create new FrameGeometry and allocate memory
    size_t num_points = object_view.size();

    FrameGeometry& local_points = point_map[object_id];
    local_points.lmks_C.reserve(num_points);
    local_points.left_kps.reserve(num_points);
    local_points.left_kps_previous.reserve(num_points);
    local_points.right_pixel.reserve(num_points);
    local_points.ids.reserve(num_points);
    local_points.fc_indices.reserve(num_points);

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

      if (!checkRGBDDepth(depth, object_id)) {
        inliers[i] = 0;
      } else {
        gtsam::Point2 left_kp = utils::cvPointToGtsam(kp_cv);
        gtsam::Point2 right_kp = rgbd_camera->rightKeypoint(depth, left_kp);

        gtsam::Point2 left_kp_previous =
            utils::cvPointToGtsam(previous_points[i]);

        if (!rgbd_camera->isKeypointContained(right_kp, depth)) {
          inliers[i] = 0;
          continue;
        }

        double right_pixel = right_kp(0);

        // fast version of the back projection function
        Landmark lmk;
        rgbd_camera->backProject2(left_kp, depth, lmk);

        if (!checkTwoViewGeometry(lmk, left_kp, right_kp)) {
          inliers[i] = 0;
          continue;
        }

        size_t local_index = local_points.lmks_C.size();
        local_points.lmks_C.push_back(lmk);
        local_points.left_kps.push_back(left_kp);
        local_points.left_kps_previous.push_back(left_kp_previous);
        local_points.right_pixel.push_back(right_pixel);
        local_points.ids.push_back(tracklet_id);
        local_points.fc_indices.push_back(global_index);

        // mapping of tracklet id to location in the local points container
        local_points.local_indices[tracklet_id] = local_index;
      }
    }
  }
}

void DepthUpdaterFast::calcPointsStereo(FrameGeometryMap& point_map) {
  std::shared_ptr<RGBDCamera> rgbd_camera = camera_->safeGetRGBDCamera();

  for (auto& object_view : features_.objectViews()) {
    auto object_id = object_view.objectId();

    // create new FrameGeometry and allocate memory
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

    size_t num_stereo_matches = matched_indexs.size();
    LOG(INFO) << "Stereo matches =" << num_stereo_matches;

    FrameGeometry& local_points = point_map[object_id];
    local_points.lmks_C.reserve(num_stereo_matches);
    local_points.left_kps.reserve(num_stereo_matches);
    local_points.left_kps_previous.reserve(num_stereo_matches);
    local_points.right_pixel.reserve(num_stereo_matches);
    local_points.ids.reserve(num_stereo_matches);
    local_points.fc_indices.reserve(num_stereo_matches);

    for (size_t i = 0; i < matched_indexs.size(); i++) {
      size_t matched_index = matched_indexs[i];
      // which index this point is found at in the local obect view
      size_t object_view_index = left_kp_indices[matched_index];

      // global index in the feature container (ie. features.points[i])
      size_t fc_index = object_view_index + offset;

      TrackletId id = tracklet_ids[matched_index];
      // sanity check out local index matches our global one
      CHECK_EQ(features_.ids[fc_index], id);
      CHECK_EQ(ids[object_view_index], id);

      gtsam::Point2 left_kp = utils::cvPointToGtsam(left_kps[matched_index]);
      gtsam::Point2 right_kp = utils::cvPointToGtsam(matched_right_kps[i]);
      gtsam::Point2 left_kp_previous =
          utils::cvPointToGtsam(previous_points[object_view_index]);

      const double uL = left_kp.x();
      const double v = left_kp.y();
      const double uR = right_kp.x();

      const double disparity = uL - uR;
      double depth = rgbd_camera->depthFromDisparity(disparity);

      if (!checkStereoDepth(disparity, depth, object_id)) {
        // inliers[object_view_index] = 0;
        continue;
      }

      // fast version of the back projection function
      Landmark lmk;
      rgbd_camera->backProject2(left_kp, depth, lmk);

      if (!checkTwoViewGeometry(lmk, left_kp, right_kp)) {
        // inliers[object_view_index] = 0;
        continue;
      }

      size_t local_index = local_points.lmks_C.size();
      local_points.lmks_C.push_back(lmk);
      local_points.left_kps.push_back(left_kp);
      local_points.left_kps_previous.push_back(left_kp_previous);
      local_points.right_pixel.push_back(uR);
      local_points.ids.push_back(id);
      local_points.fc_indices.push_back(fc_index);

      // mapping of tracklet id to location in the local points container
      local_points.local_indices[id] = local_index;

      // remark as inlier
      inliers[object_view_index] = 1;
    }
  }
}

void DepthUpdaterFast::updateGeometry(FrameGeometry& local_geometry) {
  if (images_.hasRightRgb()) {
    return updateGeometryStereo(local_geometry);
  } else if (images_.hasDepth()) {
    return updateGeometryRGBD(local_geometry);
  }
}

void DepthUpdaterFast::updateGeometryRGBD(FrameGeometry& local_geometry) {
  std::shared_ptr<RGBDCamera> rgbd_camera = camera_->safeGetRGBDCamera();
  const cv::Mat& depth_img = images_.depth();

  size_t num_points = local_geometry.lmks_C.size();
  for (size_t i = 0; i < num_points; i++) {
    Index fc_index = local_geometry.fc_indices[i];
    TrackletId id = local_geometry.ids[i];

    // check internal consistency
    CHECK_EQ(id, features_.ids[fc_index]);
    CHECK_EQ(local_geometry.local_indices.at(id), i);
    if (!features_.inlier[fc_index]) {
      continue;
    }

    // should check consistency with what the geometry is MEANT to be
    // and if all featues are of the same id!
    const ObjectId object_id = features_.object_ids[fc_index];

    const gtsam::Point2 left_kp = local_geometry.left_kps[i];
    const Depth depth =
        depth_img.at<Depth>(utils::gtsamPointToCv<float>(left_kp));

    if (!checkRGBDDepth(depth, object_id)) {
      features_.inlier[fc_index] = 0;
    } else {
      gtsam::Point2 right_kp = rgbd_camera->rightKeypoint(depth, left_kp);

      if (!rgbd_camera->isKeypointContained(right_kp, depth)) {
        features_.inlier[fc_index] = 0;
        continue;
      }

      // update geometry
      Landmark lmk;
      rgbd_camera->backProject2(left_kp, depth, lmk);

      if (!checkTwoViewGeometry(lmk, left_kp, right_kp)) {
        features_.inlier[fc_index] = 0;
        continue;
      }

      local_geometry.lmks_C[i] = lmk;
      local_geometry.right_pixel[i] = right_kp(0);
    }
  }
}

void DepthUpdaterFast::updateGeometryStereo(FrameGeometry& local_geometry) {
  std::shared_ptr<RGBDCamera> rgbd_camera = camera_->safeGetRGBDCamera();

  size_t num_points = local_geometry.lmks_C.size();
  std::vector<cv::Point2f> left_kps;
  left_kps.reserve(num_points);

  std::vector<size_t> left_kp_indices;
  left_kp_indices.reserve(num_points);

  TrackletIds tracklet_ids;
  tracklet_ids.reserve(num_points);

  for (size_t i = 0; i < num_points; i++) {
    Index fc_index = local_geometry.fc_indices[i];
    TrackletId id = local_geometry.ids[i];

    // check internal consistency
    CHECK_EQ(id, features_.ids[fc_index]);
    CHECK_EQ(local_geometry.local_indices.at(id), i);
    if (!features_.inlier[fc_index]) {
      continue;
    }

    const gtsam::Point2 left_kp = local_geometry.left_kps[i];
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

  size_t num_stereo_matches = matched_indexs.size();

  for (size_t i = 0; i < matched_indexs.size(); i++) {
    size_t matched_index = matched_indexs[i];
    TrackletId id = tracklet_ids[matched_index];
    // index where the point can be found in the original feature geometry
    size_t local_index = left_kp_indices[matched_index];

    Index fc_index = local_geometry.fc_indices[local_index];
    // sanity check out local index matches our global one
    CHECK_EQ(features_.ids[fc_index], id);

    const gtsam::Point2 left_kp = local_geometry.left_kps[local_index];
    const gtsam::Point2 right_kp = utils::cvPointToGtsam(matched_right_kps[i]);

    // should check consistency with what the geometry is MEANT to be
    // and if all featues are of the same id!
    const ObjectId object_id = features_.object_ids[fc_index];

    const double uL = left_kp.x();
    const double v = left_kp.y();
    const double uR = right_kp.x();

    const double disparity = uL - uR;
    double depth = rgbd_camera->depthFromDisparity(disparity);

    if (!checkStereoDepth(disparity, depth, object_id)) {
      // features_.inlier[fc_index] = 0;
      continue;
    }

    // fast version of the back projection function
    Landmark lmk;
    rgbd_camera->backProject2(left_kp, depth, lmk);

    if (!checkTwoViewGeometry(lmk, left_kp, right_kp)) {
      // features_.inlier[fc_index] = 0;
      continue;
    }

    local_geometry.lmks_C[local_index] = lmk;
    local_geometry.right_pixel[local_index] = uR;

    // re mark as inlier!
    features_.inlier[fc_index] = 1;
  }
}

bool DepthUpdaterFast::computeStereoMatching(
    const std::vector<cv::Point2f>& left_kps,
    std::vector<cv::Point2f>& matched_right_kps,
    std::vector<size_t>& matched_indexs) {
  CHECK(images_.hasRightRgb());

  LOG(INFO) << "Attempting stereo match with " << left_kps.size();

  if (left_kps.size() < kMinStereoMatches) {
    return false;
  }

  OpticalFlowLK optical_flow(30);

  const WrappedRGBMono wrapped_left = images_.rgb();
  const WrappedRGBMono wrapped_right = images_.rightRgb();

  const int rows = wrapped_left.image().rows;
  const int cols = wrapped_left.image().cols;

  // the image wrapper will cache the image pyramid!
  const ImagePyramid left_image_pyr = wrapped_left.computeImagePyramid(
      optical_flow.win_size, optical_flow.max_level);

  const ImagePyramid right_image_pyr = wrapped_right.computeImagePyramid(
      optical_flow.win_size, optical_flow.max_level);

  // track left to right
  OpticalFlowLK::Result flow_result =
      optical_flow(left_image_pyr.levels, right_image_pyr.levels, left_kps);

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

struct FlowRefinement::ImplOptimizer {
  using Params = OpticalFlowAndPoseSolverParams;
  const Params params_;

  using Calibration = Camera::CalibrationType;
  const Calibration& calibration_;

  gtsam::noiseModel::mEstimator::Base::shared_ptr loss_;

  double flow_information_;
  double prior_information_;
  double flow_sqrt_information_;
  double prior_sqrt_information_;

  struct Inputs {
    gtsam::Point2Vector ref_kps;
    gtsam::Point3Vector ref_lmks;
    //! doubles as the measured flow from optical flow AND the initial value for
    //! the estimated flow
    gtsam::Point2Vector measured_flows;
  };

  // TODO: and timing!
  struct TerminationCriteria {
    size_t max_iterations = 10;
    double relative_error_tol{1e-5};
    double absolute_error_tol{1e-5};
    double error_tol{0.0};

    bool shouldTerminate(size_t iterations, double current_error,
                         double new_error) const {
      const bool exceeded_max_iterations = iterations >= max_iterations;
      const bool has_converged = gtsam::checkConvergence(
          relative_error_tol, absolute_error_tol, error_tol, current_error,
          new_error, gtsam::NonlinearOptimizerParams::Verbosity::SILENT);
      const bool has_infinite_error = std::isinf(current_error);

      const bool should_terminate =
          exceeded_max_iterations || has_infinite_error || has_converged;
      return should_terminate;
    }
  };

  struct Result {
    gtsam::Pose3 pose;
    gtsam::Point2Vector flows;
    // TODO: inliers!

    double error_before{0.0};
    double error_after{0.0};

    double total_time_ms{0.0};
    size_t iterations{0};
  };

  struct IterationStats {
    double avg_projection_error = 0.0;
    // GTSAM-style unnormalized nonlinear error:
    // sum 0.5 * ||whitened residual||^2
    double regular_error = 0.0;

    // Error after applying the robust loss.
    double robust_error = 0.0;

    double error_change = 0.0;

    // // Useful diagnostics.
    // double max_whitened_error = 0.0;
    // double pose_update_norm = 0.0;
    // double max_flow_update_norm = 0.0;
  };

  static void printIterationStats(const IterationStats& stats,
                                  std::size_t iteration) {
    LOG(INFO) << std::fixed << std::setprecision(6) << "Iteration " << iteration
              << '\n'
              << "  Projection error       : " << stats.avg_projection_error
              << '\n'
              << "  Regular error          : " << stats.regular_error << '\n'
              << "  Robust error           : " << stats.robust_error << '\n'
              << "  Error delta            : " << stats.error_change << '\n';
  }

  ImplOptimizer(const Params& params, const Calibration& calibration)
      : params_(params), calibration_(calibration) {
    // validateParams
    // setup loss
    flow_information_ = 1.0 / (params_.flow_sigma * params_.flow_sigma);
    prior_information_ =
        1.0 / (params_.flow_prior_sigma * params_.flow_prior_sigma);

    flow_sqrt_information_ = std::sqrt(flow_information_);
    prior_sqrt_information_ = std::sqrt(prior_information_);

    if (params_.use_robust) {
      loss_ = gtsam::noiseModel::mEstimator::Huber::Create(params_.k_huber);
    } else {
      loss_ = gtsam::noiseModel::mEstimator::Null::Create();
    }
  }

  ~ImplOptimizer() = default;

  Result optimize(const gtsam::Pose3& initial_pose, const Inputs& inputs,
                  const TerminationCriteria& criteria) const {
    CHECK_EQ(inputs.ref_kps.size(), inputs.ref_lmks.size());
    CHECK_EQ(inputs.ref_kps.size(), inputs.measured_flows.size());

    const size_t num_measurements = inputs.ref_kps.size();
    auto tic = utils::Timer::tic();

    Result result;
    result.pose = initial_pose;
    result.flows = inputs.measured_flows;

    gtsam::Point2Vector& refined_flows = result.flows;
    gtsam::Pose3& refined_pose = result.pose;

    std::vector<Pose3FlowProjectionResidual2> factors;
    factors.reserve(num_measurements);

    double current_error = 0.0;
    // build factors and compute initial error
    // there is some tiny overhead as we recompute the whitened
    // error here and then again during the iterations
    for (size_t i = 0; i < num_measurements; i++) {
      Pose3FlowProjectionResidual2 residual(inputs.ref_kps[i],
                                            inputs.ref_lmks[i], calibration_);
      gtsam::Vector2 unwhitened_error =
          residual(refined_flows[i], refined_pose);
      gtsam::Vector2 whitened_error = unwhitened_error * flow_sqrt_information_;
      current_error += 0.5 * whitened_error.squaredNorm();

      factors.push_back(std::move(residual));
    }

    struct LinearizedMeasurement {
      Matrix62 Hxf;
      Matrix22 Hff;
      Vector2 bf;
    };
    std::vector<LinearizedMeasurement> linearized(num_measurements);

    double new_error = current_error;
    size_t iterations = 0;
    std::vector<IterationStats> stats_per_iterations;
    do {
      stats_per_iterations.emplace_back();
      IterationStats& stats = stats_per_iterations.back();
      current_error = new_error;

      gtsam::Matrix66 H = gtsam::Matrix66::Zero();
      gtsam::Vector6 b = gtsam::Vector6::Zero();

      // temporary variables
      gtsam::Matrix H1;
      gtsam::Matrix H2;

      for (size_t i = 0; i < num_measurements; ++i) {
        Eigen::Matrix<double, 2, 2> Jf;
        Eigen::Matrix<double, 2, 6> Jx;

        const gtsam::Vector2& measured_flow = inputs.measured_flows.at(i);
        const gtsam::Vector2& esimated_flow = refined_flows.at(i);

        const auto& residual = factors.at(i);
        gtsam::Vector2 projection_error =
            residual(esimated_flow, refined_pose, H1, H2);
        Jf = H1;
        Jx = H2;

        stats.avg_projection_error += projection_error.norm();

        // whitened error
        gtsam::Vector2 whitened_error =
            projection_error * flow_sqrt_information_;
        // NOTE: this should actually sum to the current error since it is
        // recalculated from the same linerization point at the end of the last
        // iteration
        // TODO: we could actually cache the residuals then as we compute them
        // once at the end of each iteration and then again at the start!
        stats.regular_error += 0.5 * whitened_error.squaredNorm();

        // whitened Jacobians
        Jf *= flow_sqrt_information_;
        Jx *= flow_sqrt_information_;

        // --------------------------------------------------------------
        // Robust reweighting.
        //
        // This matches GTSAM's Block robust weighting:
        //
        //   w = HuberWeight(||r||)
        //
        // followed by:
        //
        //   J <- sqrt(w) J
        //   r <- sqrt(w) r
        //
        // where r is already whitened by sigma.
        // --------------------------------------------------------------
        const double whitened_distance = whitened_error.norm();
        const double robust_weight = loss_->weight(whitened_distance);
        const double sqrt_weight = std::sqrt(robust_weight);

        stats.robust_error += loss_->loss(whitened_distance);

        if (robust_weight != 1.0) {
          Jf *= sqrt_weight;
          Jx *= sqrt_weight;
          whitened_error *= sqrt_weight;
        }

        // --------------------------------------------------------------
        // Projection Hessian and gradient contributions.
        // --------------------------------------------------------------
        const gtsam::Matrix62 Jx_T = Jx.transpose();
        const gtsam::Matrix22 Jf_T = Jf.transpose();

        const gtsam::Matrix66 Hxx = Jx_T * Jx;
        const gtsam::Matrix62 Hxf = Jx_T * Jf;
        gtsam::Matrix22 Hff = Jf_T * Jf;

        const gtsam::Vector6 bx = Jx_T * whitened_error;
        gtsam::Vector2 bf = Jf_T * whitened_error;

        // Flow prior.  The prior is intentionally NOT robustified.
        const gtsam::Vector2 prior_r =
            (esimated_flow - measured_flow).template cast<double>();

        // Hessian: Jᵀ Λ J, with J = I.
        Hff.noalias() += prior_information_ * gtsam::Matrix22::Identity();

        // Gradient: Jᵀ Λ r, with J = I.
        const gtsam::Vector2 prior_gradient = prior_information_ * prior_r;
        bf.noalias() += prior_gradient;

        // Error: 0.5 rᵀ Λ r = 0.5 ||Λ½ r||².
        const gtsam::Vector2 whitened_prior_r =
            prior_sqrt_information_ * prior_r;

        const double prior_error = 0.5 * whitened_prior_r.squaredNorm();

        stats.regular_error += prior_error;
        stats.robust_error += prior_error;

        linearized[i].Hxf = Hxf;
        linearized[i].Hff = Hff;
        linearized[i].bf = bf;

        // Factor Hff only once.
        const Eigen::LDLT<Matrix22> Hff_ldlt(Hff);
        const gtsam::Matrix26 Hff_inv_Hfx = Hff_ldlt.solve(Hxf.transpose());
        const gtsam::Vector2 Hff_inv_bf = Hff_ldlt.solve(bf);

        H.noalias() += Hxx - Hxf * Hff_inv_Hfx;
        b.noalias() += bx - Hxf * Hff_inv_bf;
      }  // finish measurement loop

      if (num_measurements > 0) {
        stats.avg_projection_error /= (double)num_measurements;
      } else {
        stats.avg_projection_error = 0.0;
      }

      // printIterationStats(stats, iterations);

      // solve reduced system
      const gtsam::Vector6 dx = H.ldlt().solve(-b);
      // update pose
      refined_pose = refined_pose.retract(dx);

      new_error = 0.0;
      // update flows and recompute error term
      for (size_t i = 0; i < num_measurements; ++i) {
        const LinearizedMeasurement& lin = linearized[i];

        const Eigen::LDLT<Matrix22> Hff_ldlt(lin.Hff);

        const gtsam::Vector2 df =
            -Hff_ldlt.solve(lin.bf + lin.Hxf.transpose() * dx);

        refined_flows.at(i) += df;

        const auto& residual = factors.at(i);
        // TODO: probably could cache then and then use it for the start of the
        // next iteration!
        gtsam::Vector2 unwhitened_error =
            residual(refined_flows.at(i), refined_pose);
        gtsam::Vector2 whitened_error =
            unwhitened_error * flow_sqrt_information_;
        new_error += 0.5 * whitened_error.squaredNorm();
      }

      ++iterations;
      stats.error_change = new_error - current_error;

    } while (!criteria.shouldTerminate(iterations, current_error, new_error));

    auto toc = utils::Timer::toc(tic);
    double optimize_time_ms = utils::Timer::toUnits<std::milli>(toc);

    result.error_before = stats_per_iterations.front().regular_error;
    result.error_after = stats_per_iterations.back().regular_error;
    result.total_time_ms = optimize_time_ms;
    result.iterations = iterations;

    return result;
  }

  Result optimize(const gtsam::Pose3& initial_pose,
                  const Inputs& inputs) const {
    return this->optimize(initial_pose, inputs, TerminationCriteria{});
  }
};

FlowRefinement::FlowRefinement(const Camera::Ptr camera,
                               const ImageContainer& images,
                               MatchingAdaptorBase& adaptor)
    : camera_(camera), images_(images), adaptor_(adaptor) {}

FlowRefinement::~FlowRefinement() = default;

void FlowRefinement::refine(const OpticalFlowAndPoseSolverParams& params,
                            const gtsam::Pose3& pose_in,
                            gtsam::Pose3& pose_out) {
  utils::ChronoTimingStats timer("flow_refine.refine");
  // TODO: cahe!
  auto gtsam_calibration = *camera_->getGtsamCalibration();
  impl_ = std::make_unique<ImplOptimizer>(params, gtsam_calibration);

  ImplOptimizer::Inputs impl_inputs;

  // takes a few milliseconds to build...
  utils::ChronoTimingStats build_t("flow_refine.build");
  const size_t num_matches = adaptor_.numMatches();

  // store reference keypoints to avoid lookup
  gtsam::Point2Vector ref_kps;
  ref_kps.reserve(num_matches);

  std::vector<Index> matching_index;
  matching_index.reserve(num_matches);

  std::vector<uchar*> inlier_ptrs;
  inlier_ptrs.reserve(num_matches);

  for (size_t i = 0; i < num_matches; i++) {
    // if we recompute this should not be needed!
    uchar* inlier_ptr = adaptor_.inlierPtr(i);
    // dereference and check if valid
    if (!(*inlier_ptr)) {
      continue;
    }

    const gtsam::Point3& ref_lmk = adaptor_.landmark(i);
    // assuming the previous frame is the reference frame!
    const gtsam::Point2& ref_kp = adaptor_.keypointPrev(i);

    const gtsam::Point2& curr_kp = adaptor_.keypoint(i);
    const gtsam::Point2 initial_flow = curr_kp - ref_kp;

    matching_index.push_back(i);
    ref_kps.push_back(ref_kp);
    inlier_ptrs.push_back(inlier_ptr);

    impl_inputs.ref_kps.push_back(ref_kp);
    impl_inputs.ref_lmks.push_back(ref_lmk);
    impl_inputs.measured_flows.push_back(initial_flow);
  }

  build_t.stop();

  ImplOptimizer::TerminationCriteria criteria;
  criteria.max_iterations = 5;

  utils::ChronoTimingStats solve_t("flow_refine.shur_solve", 7);
  const auto result = impl_->optimize(pose_in, impl_inputs, criteria);
  solve_t.stop();

  CHECK_EQ(matching_index.size(), result.flows.size());
  const cv::Mat& object_masks = images_.objectMotionMask();

  for (size_t i = 0; i < matching_index.size(); i++) {
    gtsam::Point2 refined_keypoint = result.flows[i] + ref_kps[i];

    if (!camera_->isKeypointContained(refined_keypoint)) {
      *(inlier_ptrs[i]) = 0;
      continue;
    }

    // TODO: still within object mask!

    Index index = matching_index[i];
    // update both the gtsam representation and the opencv representation in
    // features!
    adaptor_.keypoint(index, refined_keypoint);
  }

  pose_out = result.pose;

  // VLOG(10) << "Initial error: " << result.error_before << " final error "
  //          << result.error_after << " time[ms] "
  //          << result.total_time_ms
  //          << " #iterations= " << result.iterations;
}

MatchingAdaptorBase::MatchingAdaptorBase(FrameGeometry& local_geometry,
                                         const LandmarkMap& reference_geometry,
                                         FeatureBlockContainer& features)
    : local_geometry_(local_geometry),
      reference_geometry_(reference_geometry),
      features_(features) {
  const size_t num_features = local_geometry.ids.size();
  matched_landmarks_ref_.reserve(num_features);

  size_t adaptor_index{0};
  for (size_t curr_idx = 0; curr_idx < num_features; curr_idx++) {
    TrackletId tracklet_id = local_geometry.ids[curr_idx];
    Index fc_curr_idx = local_geometry.fc_indices[curr_idx];

    size_t age = features_.age[fc_curr_idx];
    // seen at least twice (ie tracked from previous to current and not a new
    // feature!)
    if (age < 2) {
      continue;
    }

    if (!features_.inlier[fc_curr_idx]) {
      continue;
    }

    auto it = reference_geometry.find(tracklet_id);
    // we have a match in the local map!
    if (it != reference_geometry.end()) {
      // const gtsam::Point3& lmk_ref = reference_geometry_lmks.at(it->second);
      // check that the landmarks have the same id!
      Index ref_index = it->second;
      // CHECK_EQ(tracklet_id, reference_geometry.ids.at(ref_index));
      matches_[adaptor_index] = std::make_pair(curr_idx, ref_index);
      matched_landmarks_ref_.push_back(
          reference_geometry_.getLandmarkByIndex(ref_index));
      ++adaptor_index;
    }
  }
}

void MatchingAdaptorBase::recompute() {
  size_t adaptor_index{0};

  std::unordered_map<Index, IndexMatch> inlier_matches;
  gtsam::Point3Vector inlier_matched_landmarks_ref;
  for (size_t i = 0; i < numMatches(); i++) {
    IndexMatch match = matches_[i];
    Index local_index = match.first;

    Index fc_index = local_geometry_.fc_indices[local_index];
    if (features_.inlier[fc_index]) {
      inlier_matched_landmarks_ref.push_back(matched_landmarks_ref_[i]);
      inlier_matches[adaptor_index] = match;
      ++adaptor_index;
    }
  }

  matched_landmarks_ref_ = std::move(inlier_matched_landmarks_ref);
  matches_ = std::move(inlier_matches);

  // call virtual function now matches have been updated
  recomputeCache();
}

LocalVIOGraph::LocalVIOGraph(Camera::Ptr camera) {
  std::shared_ptr<RGBDCamera> rgbd_camera =
      CHECK_NOTNULL(camera->safeGetRGBDCamera());
  K_stereo_ = rgbd_camera->getFakeStereoCalib();
  CHECK_NOTNULL(K_stereo_);
  K_ = camera->getGtsamCalibration();
}

void LocalVIOGraph::optimize(FrameId frame_id,
                             std::vector<FrameId>* frames_affected) {
  // covisible tracklets
  //  auto frame_node_ptr = observations_->getFrame(frame_id);
  utils::ChronoTimingStats timer_t("vio_graph.opt");
  gtsam::FastMap<FrameId, gtsam::FastMap<FrameId, std::set<TrackletId>>>
      covisibilities;
  // collect all co-visible landmarks at selected frames!
  // for(const auto& [frame_i0, frame_node] : observations_->getFrames()) {
  //   auto landmarks = frame_node->staticLandmarks();
  //   for(const auto& lmk_node : landmarks) {
  //     TrackletId tracklet_id = lmk_node->trackletId();
  //     FrameId frame_i1 = frame_node->frameId();

  //     covisibilities[frame_i0][frame_i1].insert(tracklet_id);
  //   }
  // }

  utils::ChronoTimingStats covis_t("vio_graph.opt.covis");
  for (const auto& [tracklet_id, lmk_node] : observations_->getLandmarks()) {
    if (!landmarkExists(tracklet_id)) {
      continue;
    }

    FrameIds seen_frame_ids = lmk_node->getSeenFrameIds();
    for (FrameId frame_i0 : seen_frame_ids) {
      if (frame_i0 < frame_id - 10) {
        continue;
      }

      for (FrameId frame_i1 : seen_frame_ids) {
        // to ensure unique pairings!
        // that is frame a -> b is the same as b -> a!
        if (frame_i1 > frame_i0) {
          continue;
        }

        // and id is initalsied!?
        covisibilities[frame_i0][frame_i1].insert(tracklet_id);
      }
    }
  }
  covis_t.stop();

  gtsam::Values values;
  gtsam::NonlinearFactorGraph graph;

  std::vector<FrameId> frames_to_try(
      {frame_id - 7, frame_id - 3, frame_id - 1, frame_id});

  std::set<TrackletId> tracklets_with_update;
  std::set<FrameId> poses_with_update;

  utils::ChronoTimingStats build_t("vio_graph.opt.build");
  for (FrameId frame_i0 : frames_to_try) {
    auto it0 = covisibilities.find(frame_i0);
    if (it0 == covisibilities.end()) {
      continue;
    }

    // TODO: prior on first pose!
    CHECK(poseExists(frame_i0)) << frame_i0;
    gtsam::Symbol pose_sym_i0('x', frame_i0);

    const gtsam::Pose3& X_i0 = getPose(frame_i0);
    values.insert(pose_sym_i0, X_i0);

    if (frame_i0 == frames_to_try.front()) {
      graph.addPrior<gtsam::Pose3>(
          pose_sym_i0, X_i0, gtsam::noiseModel::Isotropic::Sigma(6u, 0.00001));
    }

    for (FrameId frame_i1 : frames_to_try) {
      gtsam::Symbol pose_sym_i1('x', frame_i1);

      auto it1 = it0->second.find(frame_i1);
      if (it1 != it0->second.end()) {
        std::set<TrackletId> shared_tracklets = it1->second;
        // LOG(INFO) << frame_i0 << " -> " << frame_i1 << " with n=" <<
        // shared_tracklets.size();

        for (TrackletId id : shared_tracklets) {
          auto lmk_node = observations_->getLandmark(id);

          gtsam::Symbol lmk_sym('l', id);

          if (!values.exists(lmk_sym)) {
            values.insert(lmk_sym, getLandmark(id));
          }

          tracklets_with_update.insert(id);

          // TODo: robustify
          //  only one measurement needed
          auto [measurement_i0, model_i0] = lmk_node->getMeasurement(frame_i0);
          model_i0 = factor_graph_tools::robustifyHuber(0.01, model_i0);

          auto factor_i0 = boost::make_shared<GenericStereoFactor>(
              measurement_i0, model_i0, pose_sym_i0, lmk_sym, K_stereo_);
          graph += factor_i0;

          poses_with_update.insert(frame_i0);

          // two measurements needed!
          if (frame_i0 != frame_i1) {
            auto [measurement_i1, model_i1] =
                lmk_node->getMeasurement(frame_i1);
            model_i1 = factor_graph_tools::robustifyHuber(0.01, model_i1);

            auto factor_i1 = boost::make_shared<GenericStereoFactor>(
                measurement_i1, model_i1, pose_sym_i1, lmk_sym, K_stereo_);
            graph += factor_i1;

            poses_with_update.insert(frame_i1);
          }
        }
      }
    }
  }
  build_t.stop();

  gtsam::GaussNewtonParams opt_params;
  // for speed
  opt_params.setMaxIterations(2);

  dyno::NonlinearOptimizer<gtsam::GaussNewtonOptimizer> solver(graph, values,
                                                               opt_params);

  NonlinearOptimizerSummary summary;
  NonlinearOptimizerOptions options;

  utils::ChronoTimingStats solve_recover_t("vio_graph.opt.solve_recover");
  gtsam::Values optimised_values = values;
  { CHECK(solver.solve(optimised_values, options, &summary)); }

  VLOG(10) << "Initial error: " << summary.initial_error << " final error "
           << summary.final_error << " time[s] "
           << summary.cumulative_time_in_seconds
           << " #iterations= " << summary.numIterations();

  if (frames_affected) frames_affected->reserve(poses_with_update.size());
  for (FrameId frame_id : poses_with_update) {
    gtsam::Symbol pose_sym('x', frame_id);
    gtsam::Pose3 pose = optimised_values.at<gtsam::Pose3>(pose_sym);
    setPose(frame_id, pose);

    if (frames_affected) frames_affected->push_back(frame_id);
  }

  for (TrackletId id : tracklets_with_update) {
    gtsam::Symbol lmk_sym('l', id);
    gtsam::Point3 lmk = optimised_values.at<gtsam::Point3>(lmk_sym);
    setLandmark(id, lmk);
  }

  // std::stringstream ss;
  // for(const auto& [frame_i0, seen_map] : covisibilities) {
  //   ss << "Frame " << frame_i0 << " seen frames: ";
  //   for(const auto& [frame_i1, tracks] : seen_map) {
  //     ss << frame_i1 <<  " [ " << tracks.size() << " ]\n";
  //   }
  // }
  //
  // LOG(INFO) << ss.str();
}

OpticalFlowLK::Result OpticalFlowLK::operator()(
    const std::vector<cv::Mat>& prev_pyr, const std::vector<cv::Mat>& next_pyr,
    const std::vector<cv::Point2f>& prev_points) const {
  CHECK_EQ(prev_pyr.size(), next_pyr.size());

  ImplResult forward;
  forward.next = prev_points;
  forward.prev = prev_points;

  cv::calcOpticalFlowPyrLK(prev_pyr, next_pyr, forward.prev, forward.next,
                           forward.status, forward.error, win_size, max_level,
                           criteria, 0);

  CHECK_EQ(forward.size(), prev_points.size());

  ImplResult reverse;
  // now do reverse flow
  reverse.status.resize(forward.size());
  reverse.error.resize(forward.size());
  reverse.next = forward.next;
  reverse.prev = forward.next;

  cv::calcOpticalFlowPyrLK(next_pyr, prev_pyr, forward.next, reverse.next,
                           reverse.status, reverse.error, win_size, max_level,
                           criteria, cv::OPTFLOW_USE_INITIAL_FLOW);

  return Result{forward, reverse};
}

FeatureTrackerFast::FeatureTrackerFast(const FrontendParams& params,
                                       Camera::Ptr camera,
                                       ImageDisplayQueue* display_queue)
    : FeatureTrackerBase(params.tracker_params, camera, display_queue),
      frontend_params_(params),
      tracklet_id_manager(TrackletIdManager::instance()),
      object_detection_impl_(this),
      feature_detector_(camera->getParams().imageSize()),
      optical_flow_impl_(50) {
  if (!trackerParams().prefer_provided_object_detection) {
    LOG(INFO) << "Creating object detection engine";
    dyno::YoloConfig yolo_config;
    dyno::ModelConfig model_config;
    model_config.model_file = "yolov8n-seg.pt";
    object_detection_engine_ =
        std::make_shared<dyno::YoloV8ObjectDetector>(model_config, yolo_config);
  }
}

TrackingResult FeatureTrackerFast::track(
    FrameId frame_id, Timestamp timestamp,
    const ImageContainer& image_container,
    const std::optional<gtsam::Rot3>& R_km1_k) {
  utils::ChronoTimingStats feature_track_t("fast_tracker.track");
  // object detect/psuedo object detection measurement

  utils::ChronoTimingStats object_detect_t("fast_tracker.object_detect");
  ObjectDetectionResult object_detection_result =
      object_detection_impl_.detectAndTrack(image_container);
  object_detect_t.stop();
  // update image container with motion mask
  const cv::Mat object_masks = object_detection_result.labelled_mask;
  ObjectIds object_ids = object_detection_result.objectIds();

  const WrappedRGBMono wrapped_rgb = image_container.rgb();
  const cv::Mat rgb = wrapped_rgb.image();

  // maintain an instance of the previous feature tracks
  // as previous_features_ will be overwritten once track()
  // is complete to become the 'current feature set'
  previous_tracked_features_ = previous_features_;

  // buildOpticalFlowPyramid(current_mono, curr_mono_pyr_);

  // build the image pyramid from the wrapped image
  // this allows us to cache the computation of the pyramid within the
  // image_container to save comptutation between frames as long as we keep the
  // same image pyramid around!
  const ImagePyramid image_pyramid = wrapped_rgb.computeImagePyramid(
      optical_flow_impl_.win_size, optical_flow_impl_.max_level);
  const cv::Mat current_mono = image_pyramid.mono;
  curr_mono_pyr_ = image_pyramid.levels;

  auto imageBoundingBox = [&current_mono]() -> cv::Rect {
    constexpr static int kBorder = 5;
    cv::Rect rect;
    rect.x = kBorder;
    rect.y = kBorder;
    rect.width = current_mono.cols - kBorder;
    rect.height = current_mono.rows - kBorder;
    return rect;
  };

  if (prev_mono_.empty()) {
    // fill out initial gfft params
    GfttDetector::Params feature_detection_params(object_ids.size() + 1);
    feature_detection_params.quality_level = 0.01;

    for (size_t i = 0; i < object_ids.size(); i++) {
      const auto object_id = object_ids[i];
      const auto& object_detection = object_detection_result.detections[i];
      // sanity check that object_ids and object_detection_result are stored in
      // the same order
      CHECK_EQ(object_id, object_detection.object_id);

      fillDetectionParam(object_id, object_detection.mask,
                         object_detection.bounding_box, 0,
                         feature_detection_params[i]);
    }

    // form detection params for static background
    cv::Mat object_masks_binary = object_masks > 0;
    cv::Mat static_mask;
    cv::bitwise_not(object_masks_binary, static_mask);

    // detection bounding box is the whole image
    fillDetectionParam(background_label, static_mask, imageBoundingBox(), 0,
                       feature_detection_params.back());
    object_ids.push_back(background_label);

    FeatureBlockContainer detected_features = feature_detector_.calc(
        current_mono, object_masks, feature_detection_params);

    prev_mono_ = current_mono;
    previous_features_ = detected_features;

    // buildOpticalFlowPyramid(prev_mono_, prev_mono_pyr_);
  } else {
    auto [tracked_features, tracking_stats] =
        trackGfftBatched(current_mono, object_masks);

    // fill out binary segmentation mask map to be used as detection mask
    gtsam::FastMap<ObjectId, cv::Mat> binary_detection_masks;
    // also hold bounding boxes as we will (probably) need them later
    gtsam::FastMap<ObjectId, cv::Rect> detected_bounding_boxes;

    for (const auto& object_detection : object_detection_result.detections) {
      binary_detection_masks[object_detection.object_id] =
          object_detection.mask;
      detected_bounding_boxes[object_detection.object_id] =
          object_detection.bounding_box;
    }

    const auto outer_thickness = 10;
    cv::Mat object_masks_binary = object_masks > 0;
    // to ensure the validatiy of the mask we will do some dilation
    // to both fill holes and to expant the object masks so we dont
    // attempt to track anywhere near the objects
    // this is slightly conservative but is helpful in practice
    cv::dilate(object_masks_binary, object_masks_binary,
               cv::getStructuringElement(
                   cv::MORPH_RECT,
                   cv::Size(2 * outer_thickness + 1, 2 * outer_thickness + 1)));
    cv::Mat static_mask;
    cv::bitwise_not(object_masks_binary, static_mask);
    binary_detection_masks[background_label] = static_mask;

    std::set<ObjectId> object_needs_detection;
    for (const auto& object_view : tracked_features.objectViews()) {
      const auto j = object_view.objectId();
      const auto& tracking_stats_j = tracking_stats[j];

      // should only be inliers!
      const auto num_tracked = object_view.size();
      const auto min_allowed_tracks = getMinAllowableTracks(j);
      const auto allowed_feature_distance = getMinFeatureDistance(j);
      const float max_feature_age = static_cast<float>(getMaxFeatureAge(j));

      LOG(INFO) << "Tracked features for j=" << j << " n=" << num_tracked;

      cv::Mat& binary_detection_mask_j = binary_detection_masks[j];
      auto object_points = object_view.points();

      // add mask over current features
      // improve masking for longer tracks by reducing supression for old tracks
      for (size_t i = 0; i < num_tracked; i++) {
        cv::circle(binary_detection_mask_j, object_points[i],
                   allowed_feature_distance, cv::Scalar(0), cv::FILLED);
      }

      const bool too_few_tracks =
          static_cast<int>(num_tracked) < min_allowed_tracks;

      const float survival_ratio = tracking_stats_j.survivalRatio();
      const bool poor_tracking = survival_ratio < 0.4;
      bool needs_detection = poor_tracking || too_few_tracks;

      const bool is_object = j > 0;
      if (is_object) {
        const cv::Rect& detected_bounding_box = detected_bounding_boxes[j];
        const cv::Rect tracked_bounding_box =
            cv::boundingRect(object_view.pointsMat());

        const double iou =
            utils::calculateIoU(detected_bounding_box, tracked_bounding_box);
        const auto min_iou = 0.5;

        const bool small_iou = iou < min_iou;

        needs_detection = needs_detection || small_iou;

        // if detection rectangle is tiny (less than 100 pixels in area) just
        // ignore
        // if (detected_bounding_box.area() < 80) {
        //   needs_detection = false;
        // }
      }

      if (needs_detection) {
        // hack for now
        LOG(INFO) << "Replacing tracks for poorly tracked object " << j;
        // per_object_tracks = detected_feature_map[j];
        object_needs_detection.insert(j);

      } else {
        LOG(INFO) << "Good tracks for object " << j;
      }
    }

    GfttDetector::Params feature_detection_params;
    ObjectIds objects_ids_for_detection;

    for (const auto& [j, masks_j] : binary_detection_masks) {
      LOG(INFO) << "Looking at mask j=" << j;
      // if does not exist in current tracking and we have detections
      // add as new object
      bool is_new = !tracked_features.containsObject(j);
      if (object_needs_detection.count(j) > 0 || is_new) {
        LOG(INFO) << "Object j " << j << " needs detection";
        objects_ids_for_detection.push_back(j);

        cv::Rect bbox;
        if (j > 0) {
          bbox = detected_bounding_boxes[j];
        } else {
          bbox = imageBoundingBox();
        }

        // Assuming that tracked features ONLY contain inliers!
        int current_tracks = is_new ? 0 : tracked_features.objectView(j).size();

        feature_detection_params.emplace_back();
        fillDetectionParam(j, masks_j, bbox, current_tracks,
                           feature_detection_params.back());
      }
    }

    if (!objects_ids_for_detection.empty()) {
      utils::ChronoTimingStats detect_t("fast_tracker.detect");
      // TODO: still need to handle tracklet ids!
      FeatureBlockContainer detected_features = feature_detector_.calc(
          current_mono, object_masks, feature_detection_params);
      detect_t.stop();
      previous_features_ = tracked_features.merge(detected_features);

    } else {
      previous_features_ = tracked_features;
    }
    // cv::imshow("batch Tracks", viz);
  }

  prev_mono_ = current_mono;
  // move data from current to previous
  // this prevents shallow copying such that updating the current pyramid
  // will update the previous pyramid
  prev_mono_pyr_ = std::move(curr_mono_pyr_);

  // always set previous object mask
  prev_object_mask_ = object_masks;

  return {object_detection_result, previous_features_};
}

FeatureTrackerFast::ObjectDetectionImpl::ObjectDetectionImpl(
    FeatureTrackerFast* parent_)
    : parent(parent_) {}

ObjectDetectionResult FeatureTrackerFast::ObjectDetectionImpl::detectAndTrack(
    const ImageContainer& image_container) const {
  if (parent->trackerParams().prefer_provided_object_detection) {
    if (!image_container.hasObjectMask()) {
      LOG(FATAL)
          << "Params specify prefer provided object mask but input is missing!";
    }
    return detectionViaObjectMask(image_container);

  } else {
    return detectionViaInference(image_container);
  }
}

ObjectDetectionResult
FeatureTrackerFast::ObjectDetectionImpl::detectionViaObjectMask(
    const ImageContainer& image_container) const {
  const cv::Mat object_masks = image_container.objectMotionMask();

  ObjectIds object_ids = vision_tools::getObjectLabels(object_masks);

  std::vector<cv::Rect> bounding_boxes;
  vision_tools::getObjectBoundingBoxes(object_masks, object_ids,
                                       bounding_boxes);

  CHECK_EQ(object_ids.size(), bounding_boxes.size());

  ObjectDetectionResult result;
  result.labelled_mask = object_masks;
  result.input_image = image_container.rgb();
  result.detections.reserve(object_ids.size());

  for (size_t i = 0; i < object_ids.size(); i++) {
    const auto object_id = object_ids[i];

    SingleDetectionResult detection_result;
    detection_result.mask = object_masks == object_id;
    detection_result.bounding_box = bounding_boxes[i];
    detection_result.confidence = 1.0;
    detection_result.object_id = object_id;
    detection_result.well_tracked = true;
    detection_result.source = SingleDetectionResult::Source::MASK;

    result.detections.push_back(detection_result);
  }
  return result;
}

ObjectDetectionResult
FeatureTrackerFast::ObjectDetectionImpl::detectionViaInference(
    const ImageContainer& image_container) const {
  auto& detection_engine = parent->object_detection_engine_;
  CHECK_NOTNULL(detection_engine);

  VLOG(50) << "Running object detection and tracking inference k="
           << image_container.frameId();
  return detection_engine->process(image_container.rgb());
}

FeatureTrackerFast::GfttDetector::GfttDetector(const cv::Size& size)
    : tracklet_id_manager_(TrackletIdManager::instance()) {
  // / ------------------------------------------------------------------
  // Page-locked host memory
  //
  // HostMem owns the allocations. The cv::Mat objects below are only
  // headers pointing into these allocations.
  // ------------------------------------------------------------------

  h_mono_ = cv::cuda::HostMem(size, CV_8UC1, cv::cuda::HostMem::PAGE_LOCKED);

  h_eig_ = cv::cuda::HostMem(size, CV_32FC1, cv::cuda::HostMem::PAGE_LOCKED);

  // Create cv::Mat headers over the pinned memory.
  mono_ = h_mono_.createMatHeader();
  eig_ = h_eig_.createMatHeader();

  // ------------------------------------------------------------------
  // GPU memory
  // ------------------------------------------------------------------

  d_mono_.create(size, CV_8UC1);
  d_eig_.create(size, CV_32FC1);

  // ------------------------------------------------------------------
  // CUDA corner detector
  //
  // Equivalent to:
  //
  // cv::cornerMinEigenVal(mono, eig, blockSize, 3);
  // ------------------------------------------------------------------

  detector_ = cv::cuda::createMinEigenValCorner(CV_8UC1, 3, 3);
}

// Structure to temporarily hold corner candidates for sorting
struct CornerCandidate {
  cv::Point2f pt;
  float score;

  // Sort descending by score
  bool operator>(const CornerCandidate& other) const {
    return score > other.score;
  }
};

// Structure to temporaily hold responses and keypoint in contiguous memory
struct CornerResponses {
  std::vector<cv::Point2f> keypoints;
  std::vector<float> responses;

  size_t size() const {
    CHECK_EQ(keypoints.size(), responses.size());
    return keypoints.size();
  }

  inline void reserve(size_t N) {
    keypoints.reserve(N);
    responses.reserve(N);
  }

  inline void resize(size_t N) {
    keypoints.resize(N);
    responses.resize(N);
  }

  inline void push_back(const CornerCandidate& candidate) {
    keypoints.push_back(candidate.pt);
    responses.push_back(candidate.score);
  }
};

FeatureBlockContainer FeatureTrackerFast::GfttDetector::calc(
    const cv::Mat& mono, const cv::Mat& object_mask, const Params& params) {
  using namespace dyno;
  utils::ChronoTimingStats r("fast_tracker.detector");
  CV_Assert(mono.type() == CV_8UC1 || mono.type() == CV_32FC1);
  // CV_Assert(masks.size() == maxCorners.size());

  if (params.empty()) {
    return {};
  }

  // --- STEP 1: Compute the Eigenvalue Map ONCE for the whole frame ---
  // cv::Mat eig;
  // cornerMinEigenVal handles the Sobel derivatives internally in a highly
  // optimized pass
  // utils::ChronoTimingStats t1("corner_min_eigen");
  // cv::cornerMinEigenVal(mono, eig, blockSize, 3);
  // t1.stop();
  // Copy normal CPU memory -> pinned memory.
  //
  utils::ChronoTimingStats t1("fast_tracker.corner_min_eigen");
  // If your camera/image pipeline can write directly into mono_,
  // this copy can be eliminated entirely.
  mono.copyTo(mono_);

  // Pinned host memory -> GPU.
  d_mono_.upload(mono_, stream_);

  // GPU min-eigenvalue corner response.
  detector_->compute(d_mono_, d_eig_, stream_);

  // GPU -> pinned host memory.
  d_eig_.download(eig_, stream_);

  // Because compute() returns a CPU cv::Mat, we need to wait before
  // returning it.
  stream_.waitForCompletion();
  t1.stop();

  // Find the global maximum corner score across the entire image
  utils::ChronoTimingStats tmin("fast_tracker.minMaxLoc");
  double maxVal = 0;
  cv::minMaxLoc(eig_, nullptr, &maxVal);
  tmin.stop();

  // Establish the baseline absolute threshold based on global max quality
  const float threshold = static_cast<float>(maxVal * params.quality_level);

  // --- STEP 2: Local Non-Maximum Suppression (NMS) via Dilation ---
  // OpenCV's internal GFTT uses a dilation trick to find local maxima
  // efficiently
  utils::ChronoTimingStats t2("dilate");
  cv::Mat localMax;
  cv::dilate(eig_, localMax, cv::Mat());
  t2.stop();

  // // calculate distance transform for each mask
  // utils::ChronoTimingStats distance_t("fast_tracker.distance masks");
  // // this can take up to 3-4ms
  // std::vector<cv::Mat> distanceTransformMasks(params.size());
  // for (size_t i = 0; i < params.size(); i++) {
  //   cv::distanceTransform(params[i].mask, distanceTransformMasks[i],
  //                         cv::DIST_L2, 3);
  // }
  // distance_t.stop();

  std::vector<CornerResponses> batchedResults(params.size());

  utils::ChronoTimingStats t3("fast_tracker.masks_loop");
  tbb::task_group task_group;
  for (size_t m = 0; m < params.size(); m++) {
    task_group.run([&, m] {
      const auto& param = params[m];
      const cv::Mat& mask = param.mask;

      cv::Mat dist;
      // no need to compute the mask for the static background?
      // can we save lots of computation here?
      // ideally should not be close to the dynamic points?
      // i think this is the biggest computation bottleneck!
      cv::distanceTransform(mask, dist, cv::DIST_L2, 3);

      CV_Assert(mask.type() == CV_8UC1);

      const int maxFeatureCount = param.max_corners;
      const float minDistance = param.min_distance;

      if (maxFeatureCount <= 0) return;

      // --------------------------------------------------------------
      // Restrict the entire detection process to the object bounding
      // box rather than scanning the entire image.
      // --------------------------------------------------------------

      const cv::Rect& bbox = param.bbox;

      if (bbox.empty()) return;

      const int x0 = bbox.x;
      const int y0 = bbox.y;
      const int x1 = bbox.x + bbox.width;
      const int y1 = bbox.y + bbox.height;

      const float minDistanceSq = minDistance * minDistance;

      // --------------------------------------------------------------
      // Grid
      // --------------------------------------------------------------

      const int cellSize = std::max(1, cvRound(minDistance));
      const float invCellSize = 1.0f / static_cast<float>(cellSize);

      const int gridWidth = (bbox.width + cellSize - 1) / cellSize;
      const int gridHeight = (bbox.height + cellSize - 1) / cellSize;

      // --------------------------------------------------------------
      // Candidate storage
      // --------------------------------------------------------------

      std::vector<CornerCandidate> candidates;
      candidates.reserve(256);

      // --------------------------------------------------------------
      // Find local maxima
      //
      // IMPORTANT:
      // Scan only the bounding box.
      // --------------------------------------------------------------
      for (int y = y0; y < y1; ++y) {
        const float* eigPtr = eig_.ptr<float>(y);
        const float* maxPtr = localMax.ptr<float>(y);

        const uchar* maskPtr = mask.empty() ? nullptr : mask.ptr<uchar>(y);
        const float* distPtr = dist.empty() ? nullptr : dist.ptr<float>(y);

        for (int x = x0; x < x1; ++x) {
          // Check mask FIRST.
          //
          // For small dynamic object masks this avoids reading
          // the distance transform for almost every background
          // pixel in the bounding box.
          if (maskPtr && !maskPtr[x]) {
            continue;
          }

          const float val = eigPtr[x];

          if (val <= threshold) continue;

          if (val != maxPtr[x]) continue;

          // Mask-edge constraint
          if (distPtr && distPtr[x] <= 5.0f) continue;

          candidates.push_back(
              {cv::Point2f(static_cast<float>(x), static_cast<float>(y)), val});
        }
      }

      if (candidates.empty()) return;

      // --------------------------------------------------------------
      // Sort strongest corners first.
      // --------------------------------------------------------------

      std::sort(candidates.begin(), candidates.end(),
                std::greater<CornerCandidate>());

      // --------------------------------------------------------------
      // Output
      // --------------------------------------------------------------

      CornerResponses& acceptedCorners = batchedResults[m];
      acceptedCorners.reserve(maxFeatureCount);

      // One linked-list head per spatial cell.
      std::vector<int> gridHeads(gridWidth * gridHeight, -1);
      std::vector<int> nextPointIdx;
      nextPointIdx.reserve(maxFeatureCount);

      // --------------------------------------------------------------
      // Greedy GFTT distance suppression
      // --------------------------------------------------------------

      for (const CornerCandidate& candidate : candidates) {
        if (static_cast<int>(acceptedCorners.size()) >= maxFeatureCount) {
          break;
        }

        // Coordinates relative to bounding box.
        const int localX = static_cast<int>(candidate.pt.x) - x0;

        const int localY = static_cast<int>(candidate.pt.y) - y0;

        const int xCell = static_cast<int>(localX * invCellSize);

        const int yCell = static_cast<int>(localY * invCellSize);

        const int x1Cell = std::max(0, xCell - 1);
        const int y1Cell = std::max(0, yCell - 1);
        const int x2Cell = std::min(gridWidth - 1, xCell + 1);
        const int y2Cell = std::min(gridHeight - 1, yCell + 1);

        bool good = true;

        for (int yy = y1Cell; yy <= y2Cell && good; ++yy) {
          const int rowOffset = yy * gridWidth;

          for (int xx = x1Cell; xx <= x2Cell; ++xx) {
            int pIdx = gridHeads[rowOffset + xx];

            while (pIdx != -1) {
              const cv::Point2f& accepted = acceptedCorners.keypoints[pIdx];

              const float dx = static_cast<int>(candidate.pt.x) -
                               static_cast<int>(accepted.x);
              const float dy = static_cast<int>(candidate.pt.y) -
                               static_cast<int>(accepted.y);

              if (dx * dx + dy * dy < minDistanceSq) {
                good = false;
                break;
              }

              pIdx = nextPointIdx[pIdx];
            }

            if (!good) break;
          }
        }

        if (!good) continue;

        const int cellIdx = yCell * gridWidth + xCell;

        const int pointIdx = static_cast<int>(acceptedCorners.size());

        nextPointIdx.push_back(gridHeads[cellIdx]);

        gridHeads[cellIdx] = pointIdx;

        acceptedCorners.push_back(candidate);
      }

      // do ANMS
      constexpr float anmsTolerance = 0.10f;
      static Eigen::MatrixXd binning_mask;

      AdaptiveNonMaximumSuppression non_maximum_supression(
          AnmsAlgorithmType::RangeTree);

      std::vector<cv::KeyPoint> keypoints;
      keypoints.reserve(acceptedCorners.size());

      LOG(INFO) << "kps before ANMS " << acceptedCorners.size();
      // LOG(INFO) << "keypoints needed " << param.num_corners_needed;

      for (size_t i = 0; i < acceptedCorners.size(); i++) {
        const auto& pt = acceptedCorners.keypoints[i];
        const auto& score = acceptedCorners.responses[i];
        keypoints.emplace_back(cv::Point2f(pt.x - static_cast<float>(x0),
                                           pt.y - static_cast<float>(y0)),
                               1.0f,   // size
                               -1.0f,  // angle
                               score   // response
        );
      }

      auto& selected_keypoints = keypoints;
      selected_keypoints = non_maximum_supression.suppressNonMax(
          keypoints, param.num_corners_needed, anmsTolerance, bbox.width,
          bbox.height, 5, 5, binning_mask);

      LOG(INFO) << "points after ANMS " << selected_keypoints.size();

      // re-allocate correct memory size
      acceptedCorners.resize(selected_keypoints.size());

      for (size_t i = 0; i < selected_keypoints.size(); i++) {
        auto kp = selected_keypoints[i];
        kp.pt.x += static_cast<float>(x0);
        kp.pt.y += static_cast<float>(y0);
        acceptedCorners.keypoints[i] = kp.pt;
        acceptedCorners.responses[i] = kp.response;
      }
    });
  }

  task_group.wait();
  t3.stop();

  utils::ChronoTimingStats t_terms("fast_tracker.make_blocks");
  std::vector<std::pair<ObjectId, FeatureBlockContainer::FeatureData>> terms(
      params.size());
  for (size_t i = 0; i < params.size(); i++) {
    terms[i].first = params[i].object_id;

    const CornerResponses& corner_responses = batchedResults[i];
    auto num_points = corner_responses.size();

    FeatureBlockContainer::FeatureData& feature_data = terms[i].second;
    feature_data.resize(num_points);
    feature_data.points = corner_responses.keypoints;
    feature_data.ids =
        tracklet_id_manager_.getAndIncrementTrackletIds(num_points);

    // start all with inliers
    std::fill(feature_data.inlier.begin(), feature_data.inlier.end(), 1);
    // for now dont use errors
    std::fill(feature_data.errors.begin(), feature_data.errors.end(), 0.0);

    // start new featur ages at 1
    std::fill(feature_data.age.begin(), feature_data.age.end(), 1);

    // dummy previous points value
    feature_data.previous_points.resize(num_points);

    // TODO: fill other values
  }

  FeatureBlockContainer feature_blocks(terms);
  t_terms.stop();

  // LOG(INFO) << "Detection: " << feature_blocks.debugInfoString();

  utils::ChronoTimingStats t4("fast_tracker.sub_pixe_refine");

  if (feature_blocks.size() == 0) {
    return feature_blocks;
  }

  const cv::Size window_size = cv::Size(5, 5);
  const cv::Size zero_zone = cv::Size(-1, -1);
  const cv::TermCriteria criteria = cv::TermCriteria(
      cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001);

  cv::cornerSubPix(mono, feature_blocks.points, window_size, zero_zone,
                   criteria);

  const int rows = mono.rows;
  const int cols = mono.cols;

  // after subpix refinement points may shift such that they no longer lie
  // within the image bounds or on the same object
  for (size_t i = 0; i < feature_blocks.size(); i++) {
    const auto& kp = feature_blocks.points[i];

    auto object_id = feature_blocks.object_ids[i];
    if (!checkBoundsAndLabel(kp, object_mask, object_id, rows, cols)) {
      // mark as outlier
      feature_blocks.inlier[i] = 0;
    }
  }

  // return the reduced version so that the tracker has access to
  // a contiguous set of points that are all valid
  return feature_blocks.reduceToInliers();
}

std::pair<FeatureBlockContainer, FeatureTrackerFast::FlowTrackingStatsMap>
FeatureTrackerFast::trackGfftBatched(const cv::Mat& mono,
                                     const cv::Mat& object_mask) {
  utils::ChronoTimingStats timer("fast_tracker.track_gfft");

  CHECK(!prev_mono_.empty());
  CV_Assert(prev_mono_.type() == CV_8UC1 && mono.type() == CV_8UC1);

  const int rows = mono.rows;
  const int cols = mono.cols;

  gtsam::FastMap<ObjectId, FeatureBlockContainer::FeatureData>
      tracks_per_object;
  FlowTrackingStatsMap tracking_stats;
  for (const auto& object_view : previous_features_.objectViews()) {
    size_t num_points = object_view.size();
    auto object_id = object_view.objectId();

    FeatureBlockContainer::FeatureData data;
    data.reserve(num_points);

    tracks_per_object[object_id] = data;
    tracking_stats[object_id].num_previous_tracks = num_points;

    LOG(INFO) << "Preparing featue tracking structures j=" << object_id
              << " n=" << num_points;
  }

  OpticalFlowLK::Result flow_result = optical_flow_impl_(
      prev_mono_pyr_, curr_mono_pyr_, previous_features_.points);

  const std::vector<cv::Point2f>& predicted_points =
      flow_result.predictedPoints();
  const std::vector<cv::Point2f>& from_points = flow_result.fromPoints();
  const std::vector<float>& error = flow_result.error();
  for (size_t i = 0; i < flow_result.size(); i++) {
    const cv::Point2f& predicted_pt = predicted_points[i];

    if (!flow_result.isGood(i)) {
      continue;
    }

    auto object_id = previous_features_.object_ids[i];
    if (!checkBoundsAndLabel(predicted_pt, object_mask, object_id, rows,
                             cols)) {
      continue;
    }

    auto tracklet_id = previous_features_.ids[i];
    auto new_age = previous_features_.age[i] + 1;
    const cv::Point2f from_pt = from_points[i];

    tracks_per_object[object_id].points.push_back(predicted_pt);
    tracks_per_object[object_id].previous_points.push_back(from_pt);
    tracks_per_object[object_id].ids.push_back(tracklet_id);
    tracks_per_object[object_id].age.push_back(new_age);

    tracks_per_object[object_id].inlier.push_back(1);
    tracks_per_object[object_id].errors.push_back(error[i]);
  }

  // // 2. --- SINGLE-PASS KLT EXECUTION ---
  // std::vector<cv::Point2f> flatNext = previous_features_.points;
  // const auto& flatPrev = previous_features_.points;
  // std::vector<uchar> forward_status;
  // std::vector<float> forward_err;

  // // One single call allows OpenCV to run hot loops across contiguous memory
  // // blocks
  // cv::calcOpticalFlowPyrLK(prev_mono_pyr_, curr_mono_pyr_, flatPrev,
  // flatNext,
  //                          forward_status, forward_err, win_size_,
  //                          max_level_, criteria_, 0);

  // // now do reverse flow
  // std::vector<uchar> reverse_status(flatNext.size());
  // std::vector<float> reverse_err(flatNext.size());

  // std::vector<cv::Point2f> flatReverse = flatNext;
  // cv::calcOpticalFlowPyrLK(curr_mono_pyr_, prev_mono_pyr_, flatNext,
  //                          flatReverse, reverse_status, reverse_err,
  //                          win_size_, max_level_, criteria_,
  //                          cv::OPTFLOW_USE_INITIAL_FLOW);

  // static constexpr float kMaxErr = 20.0f;
  // for (size_t i = 0; i < flatPrev.size(); ++i) {
  //   const bool both_status_good = forward_status.at(i) &&
  //   reverse_status.at(i); const bool within_distance =
  //       utils::distance(flatPrev.at(i), flatReverse.at(i)) <= 0.5;
  //   const bool within_error =
  //       reverse_err[i] < kMaxErr && forward_err[i] < kMaxErr;

  //   // Check if KLT tracking succeeded and point remains inside image
  //   // boundaries use 2i to check image boundaries
  //   if (both_status_good && within_distance && within_error) {
  //     forward_status.at(i) = 1;
  //   } else {
  //     forward_status.at(i) = 0;
  //   }
  //   // do proper rounding to integer to ensure bounds checks such that
  //   // we can access the images with a point2f value and not get OOB's errors
  //   const cv::Point2i kp_int(cvRound(flatNext[i].x), cvRound(flatNext[i].y));

  //   if (forward_status[i] && kp_int.x >= 0 && kp_int.x < (mono.cols - 1) &&
  //       kp_int.y >= 0 && kp_int.y < (mono.rows - 1)) {
  //     auto object_id = previous_features_.object_ids[i];
  //     auto tracklet_id = previous_features_.ids[i];
  //     auto new_age = previous_features_.age[i] + 1;

  //     if (object_id != object_mask.at<dyno::ObjectId>(kp_int)) {
  //       continue;
  //     }

  //     tracks_per_object[object_id].points.push_back(flatNext[i]);
  //     tracks_per_object[object_id].previous_points.push_back(flatPrev[i]);
  //     tracks_per_object[object_id].ids.push_back(tracklet_id);
  //     tracks_per_object[object_id].age.push_back(new_age);

  //     tracks_per_object[object_id].inlier.push_back(1);
  //     tracks_per_object[object_id].errors.push_back(forward_err[i]);
  //   }
  // }

  gtsam::FastMap<ObjectId, FeatureBlockContainer::FeatureData>
      verified_tracks_per_object;

  for (const auto& [object_id, good_tracks] : tracks_per_object) {
    auto num_good_points = good_tracks.size();

    tracking_stats[object_id].tracked_after_flow = num_good_points;
    LOG(INFO) << "j= " << object_id << " tracked points=" << num_good_points;

    static constexpr double kHomographyReprThreshold = 2.0;
    // limit the number of iterations for speed
    static constexpr double kHomographyMaxIters = 500;
    cv::Mat inlier_mask = vision_tools::findHomography(
        good_tracks.previous_points, good_tracks.points,
        kHomographyReprThreshold, kHomographyMaxIters);
    // cv::Mat inlier_mask = vision_tools::findHomography(
    //     good_tracks.previous_points, good_tracks.points);

    FeatureBlockContainer::FeatureData verified_tracks;
    verified_tracks.reserve(num_good_points);

    for (int i = 0; i < inlier_mask.rows; ++i) {
      if (inlier_mask.at<uchar>(i)) {
        verified_tracks.points.push_back(good_tracks.points[i]);
        verified_tracks.previous_points.push_back(
            good_tracks.previous_points[i]);
        verified_tracks.ids.push_back(good_tracks.ids[i]);
        verified_tracks.age.push_back(good_tracks.age[i]);
        verified_tracks.errors.push_back(good_tracks.errors[i]);
        verified_tracks.inlier.push_back(good_tracks.inlier[i]);
      }
    }

    LOG(INFO) << "j= " << object_id << "inlier/outlier "
              << verified_tracks.size() << "/" << num_good_points;
    // if we actually have any tracks
    // this will remove any objects with no tracks!
    if (verified_tracks.size() > 0) {
      // LOG(INFO) << "j= " << object_id << "inlier/outlier "
      //           << verified_tracks.size() << "/" << num_good_points;
      verified_tracks_per_object[object_id] = verified_tracks;
      tracking_stats[object_id].tracked_after_or = verified_tracks.size();
    }
  }

  FeatureBlockContainer tracked_features(verified_tracks_per_object);
  LOG(INFO) << "Tracked: " << tracked_features.debugInfoString();
  return {tracked_features, tracking_stats};
}

// FeatureBlockContainer FeatureTrackerFast::trackRetroactively(const
// FeatureBlockContainer& detected_features, const cv::Mat& object_mask)
// {
//   CHECK(!prev_mono_pyr_.empty());

//   //track from current detected featues (on current image)
//   // to previous image
//   std::vector<cv::Point2f> flatNext = previous_features_.points;
//   const auto& flatPrev = previous_features_.points;
//   std::vector<uchar> forward_status;
//   std::vector<float> forward_err;

//   // std::vector<cv::Mat> current_mono_pyr;
//   // buildOpticalFlowPyramid(mono, current_mono_pyr);

//   // One single call allows OpenCV to run hot loops across contiguous memory
//   // blocks
//   cv::calcOpticalFlowPyrLK(prev_mono_pyr_, curr_mono_pyr_, flatPrev,
//   flatNext,
//                            forward_status, forward_err, win_size_,
//                            max_level_, criteria_, 0);

//   // now do reverse flow
//   std::vector<uchar> reverse_status(flatNext.size());
//   std::vector<float> reverse_err(flatNext.size());

//   std::vector<cv::Point2f> flatReverse = flatNext;
//   cv::calcOpticalFlowPyrLK(curr_mono_pyr_, prev_mono_pyr_, flatNext,
//                            flatReverse, reverse_status, reverse_err,
//                            win_size_, max_level_, criteria_,
//                            cv::OPTFLOW_USE_INITIAL_FLOW);
// }

void FeatureTrackerFast::fillDetectionParam(
    ObjectId object_id, const cv::Mat& mask, const cv::Rect& bounding_box,
    int current_tracks,
    FeatureTrackerFast::GfttDetector::Param& detection_param) const {
  auto numCornersNeeded = [&](ObjectId object_id) -> int {
    auto desired_max_tracks = getMaxTrackingCorners(object_id);
    auto corners_needed = std::max(desired_max_tracks - current_tracks, 0);
    return corners_needed;
  };

  detection_param.object_id = object_id;
  detection_param.mask = mask;
  detection_param.bbox = bounding_box;
  // min distance between detections!
  // as suggested by a number of different implementations we could do 0.05 *
  // image cols so instead we will try 0.05 * bounding box! with 3 pixels as the
  // minimum possible distance (ie. we dont want such a small min distance that
  // the detection takes forever!) detection_param.min_distance =
  //     static_cast<float>(getMinFeatureDistance(object_id));
  // detection_param.min_distance =
  //     std::max(3.0f, 0.05f * static_cast<float>(bounding_box.width));
  detection_param.min_distance =
      std::max(3.0f, 0.02f * static_cast<float>(bounding_box.width));
  detection_param.max_corners = getMaxDetectionCorners(object_id);
  detection_param.num_corners_needed = numCornersNeeded(object_id);
}

cv::Mat drawBatchedFeatures(const cv::Mat& image,
                            const FeatureBlockContainer& batchedFeatures) {
  cv::Mat canvas;

  // Ensure we are drawing on a 3-channel color image
  if (image.channels() == 1) {
    cv::cvtColor(image, canvas, cv::COLOR_GRAY2BGR);
  } else {
    canvas = image.clone();
  }

  for (size_t i = 0; i < batchedFeatures.size(); ++i) {
    const auto object_id = batchedFeatures.object_ids[i];
    const auto current_point = batchedFeatures.points[i];

    const cv::Scalar color = dyno::Color::uniqueObjectId(object_id).bgra();

    // Draw optical-flow track
    if (batchedFeatures.age[i] > 1) {
      const auto previous_point = batchedFeatures.previous_points[i];
      cv::arrowedLine(canvas, previous_point, current_point, color, 1);

      // / Draw current feature location
      cv::circle(canvas, current_point, 4, color);
    }

    // // Draw current feature location
    // cv::circle(canvas, current_point, 4, color);

    // Optional white outer ring
    // cv::circle(canvas, current_point, 6, cv::Scalar(255, 255, 255), 1,
    //            cv::LINE_AA);
  }

  return canvas;
}

}  // namespace dyno
