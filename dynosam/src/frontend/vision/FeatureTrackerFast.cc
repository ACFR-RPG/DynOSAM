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

FeatureTrackerFast::FeatureTrackerFast(const FrontendParams& params,
                                       Camera::Ptr camera,
                                       ImageDisplayQueue* display_queue)
    : FeatureTrackerBase(params.tracker_params, camera, display_queue),
      frontend_params_(params),
      tracklet_id_manager(TrackletIdManager::instance()),
      win_size_(21, 21),
      max_level_(3),
      criteria_(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, 0.001),
      object_detection_impl_(this),
      feature_detector_(camera->getParams().imageSize()) {
  if (!trackerParams().prefer_provided_object_detection) {
    LOG(INFO) << "Creating object detection engine";
    dyno::YoloConfig yolo_config;
    dyno::ModelConfig model_config;
    model_config.model_file = "yolov8n-seg.pt";
    object_detection_engine_ =
        std::make_shared<dyno::YoloV8ObjectDetector>(model_config, yolo_config);
  }
}

void FeatureTrackerFast::track(FrameId frame_id, Timestamp timestamp,
                               const ImageContainer& image_container,
                               const std::optional<gtsam::Rot3>& R_km1_k) {
  utils::ChronoTimingStats feature_track_t("fast_tracker.track");

  const cv::Mat rgb = image_container.rgb();
  cv::Mat current_mono = ImageType::RGBMono::toMono(rgb);

  // object detect/psuedo object detection measurement
  ObjectDetectionResult object_detection_result =
      object_detection_impl_.detectAndTrack(image_container);
  // update image container with motion mask
  const cv::Mat object_masks = object_detection_result.labelled_mask;

  ObjectIds object_ids = object_detection_result.objectIds();

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
    cv::Rect static_bounding_box;
    static_bounding_box.x = 0;
    static_bounding_box.y = 0;
    static_bounding_box.width = current_mono.cols;
    static_bounding_box.height = current_mono.rows;

    fillDetectionParam(background_label, static_mask, static_bounding_box, 0,
                       feature_detection_params.back());
    object_ids.push_back(background_label);

    FeatureBlockContainer detected_features =
        feature_detector_.calc(current_mono, feature_detection_params);

    prev_mono_ = current_mono;
    previous_features_ = detected_features;

    buildOpticalFlowPyramid(prev_mono_, prev_mono_pyr_);
  } else {
    auto [tracked_features, tracking_stats] =
        trackGfftBatched(current_mono, object_masks);
    feature_track_t.stop();

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

      LOG(INFO) << "Tracked features for j=" << j << " n=" << num_tracked;

      cv::Mat binary_detection_mask_j = binary_detection_masks[j];
      auto object_points = object_view.points();
      for (size_t i = 0; i < num_tracked; i++) {
        // mark as location to ignore when doing feature detection
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
        if (detected_bounding_box.area() < 80) {
          needs_detection = false;
        }
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
          bbox.x = 0;
          bbox.y = 0;
          bbox.width = current_mono.cols;
          bbox.height = current_mono.rows;
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
      FeatureBlockContainer detected_features =
          feature_detector_.calc(current_mono, feature_detection_params);
      detect_t.stop();

      // TODO: ANMS to cut down to only the set of features we want!

      AdaptiveNonMaximumSuppression non_maximum_supression(
          AnmsAlgorithmType::RangeTree);

      static constexpr float kTolerance = 0.01;
      static Eigen::MatrixXd binning_mask;

      // TODO: We extract N max featues (not always as this makes the runtime
      // slower) and then cut down to the number of featues we want to track
      // using ANMS!

      // gtsam::FastMap<ObjectId, CornerTracks> detected_feature_map;
      // for (size_t i = 0; i < detected_features.size(); i++) {
      //   if (detected_features[i].size() > 0) {
      //     ObjectId object_id = objects_ids_for_detection[i];

      //     const int desired_tracks = object_id > 0 ? 300 : 800;
      //     int current_tracks =
      //         detailed_tracks.exists(object_id)
      //             ? detailed_tracks.at(object_id).corner_tracks.size()
      //             : 0;
      //     const int tracked_needed =
      //         std::max(desired_tracks - current_tracks, 0);

      //     std::vector<cv::Point2f>& new_keypoints =
      //         detected_features[i].keypoints;
      //     if (tracked_needed > 0) {
      //       std::vector<KeypointCV> keypoints =
      //           detected_features[i].toKeypoints();
      //       std::vector<KeypointCV>& max_keypoints = keypoints;

      //       max_keypoints = non_maximum_supression.suppressNonMax(
      //           keypoints, tracked_needed, kTolerance, current_mono.cols,
      //           current_mono.rows, 5, 5, binning_mask);

      //       LOG(INFO) << "J=" << object_id << " tracks = " <<
      //       current_tracks
      //                 << " needed=" << tracked_needed << " after anms "
      //                 << max_keypoints.size();

      //       // std::vector<cv::Point2f> points_after_anms;
      //       cv::KeyPoint::convert(max_keypoints, new_keypoints);
      //     }

      //     CornerTracks new_tracks;
      //     new_tracks.keypoints = new_keypoints;
      //     // fill new trackletids
      //     new_tracks.tracklet_ids.reserve(new_tracks.keypoints.size());
      //     for (auto i = 0u; i < new_tracks.keypoints.size(); i++) {
      //       TrackletId tracklet_to_use =
      //           tracklet_id_manager.getAndIncrementTrackletId();
      //       new_tracks.tracklet_ids.push_back(tracklet_to_use);
      //     }

      //     detected_feature_map[object_id] = new_tracks;

      //     TrackingInfo details;
      //     details.object_id = object_id;
      //     details.num_last_detected_features = new_tracks.keypoints.size();

      //     // TODO: missing last number of tracking
      //     tracking_infos_[object_id] = details;
      //   }
      // }

      // // for now just replace featues
      // for (const auto& [j, per_object_tracks] : detected_feature_map) {
      //   // tracked_features[j] = per_object_tracks;
      //   // add tracks to existing tracks!
      //   tracks[j] += per_object_tracks;
      // }
      previous_features_ = tracked_features.merge(detected_features);

    } else {
      previous_features_ = tracked_features;
    }
    prev_mono_ = current_mono;

    cv::Mat viz = drawBatchedFeatures(rgb, previous_features_);
    cv::imshow("batch Tracks", viz);
  }
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

  inline void reserve(size_t n) {
    keypoints.reserve(n);
    responses.reserve(n);
  }

  inline void push_back(const CornerCandidate& candidate) {
    keypoints.push_back(candidate.pt);
    responses.push_back(candidate.score);
  }
};

FeatureBlockContainer FeatureTrackerFast::GfttDetector::calc(
    const cv::Mat& mono, const Params& params) {
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

  // calculate distance transform for each mask
  utils::ChronoTimingStats distance_t("fast_tracker.distance masks");
  // this can take up to 3-4ms
  std::vector<cv::Mat> distanceTransformMasks(params.size());
  for (size_t i = 0; i < params.size(); i++) {
    cv::distanceTransform(params[i].mask, distanceTransformMasks[i],
                          cv::DIST_L2, 3);
  }
  distance_t.stop();

  std::vector<CornerResponses> batchedResults(params.size());

  utils::ChronoTimingStats t3("fast_tracker.masks_loop");
  cv::parallel_for_(
      cv::Range(0, static_cast<int>(params.size())),
      [&](const cv::Range& range) {
        for (int m = range.start; m < range.end; ++m) {
          const auto& param = params[m];

          const cv::Mat& mask = param.mask;
          const cv::Mat& dist = distanceTransformMasks[m];

          const int maxFeatureCount = param.max_corners;
          const float minDistance = param.min_distance;

          // terms[m].first = feature_detection_params[m].object_id;

          if (maxFeatureCount <= 0) continue;

          // --------------------------------------------------------------
          // Restrict the entire detection process to the object bounding
          // box rather than scanning the entire image.
          // --------------------------------------------------------------

          const cv::Rect& bbox = param.bbox;

          if (bbox.empty()) continue;

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
              if (maskPtr && !maskPtr[x]) continue;

              const float val = eigPtr[x];

              if (val <= threshold) continue;

              if (val != maxPtr[x]) continue;

              // Mask-edge constraint
              if (distPtr && distPtr[x] <= 5.0f) continue;

              candidates.push_back(
                  {cv::Point2f(static_cast<float>(x), static_cast<float>(y)),
                   val});
            }
          }

          if (candidates.empty()) continue;

          // --------------------------------------------------------------
          // Sort strongest corners first.
          // --------------------------------------------------------------

          std::sort(candidates.begin(), candidates.end(),
                    std::greater<CornerCandidate>());

          // --------------------------------------------------------------
          // Output
          // --------------------------------------------------------------

          CornerResponses& acceptedCorners = batchedResults[m];
          // std::vector<cv::Point2f>& accepted_corners =
          // terms[m].second.points;

          // acceptedCorners.clear();
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

                  const float dx = candidate.pt.x - accepted.x;

                  const float dy = candidate.pt.y - accepted.y;

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
        }
      });

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

    // start new featur ages at 0
    std::fill(feature_data.age.begin(), feature_data.age.end(), 0);

    // dummy previous points value
    feature_data.previous_points.resize(num_points);

    // TODO: fill other values
  }

  FeatureBlockContainer feature_blocks(terms);
  t_terms.stop();

  LOG(INFO) << "Detection: " << feature_blocks.debugInfoString();

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

  return feature_blocks;
}

std::pair<FeatureBlockContainer, FeatureTrackerFast::FlowTrackingStatsMap>
FeatureTrackerFast::trackGfftBatched(const cv::Mat& mono,
                                     const cv::Mat& object_mask) {
  utils::ChronoTimingStats timer("fast_tracker.track_gfft");

  CHECK(!prev_mono_.empty());
  CV_Assert(prev_mono_.type() == CV_8UC1 && mono.type() == CV_8UC1);

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

  // 2. --- SINGLE-PASS KLT EXECUTION ---
  std::vector<cv::Point2f> flatNext = previous_features_.points;
  const auto& flatPrev = previous_features_.points;
  std::vector<uchar> forward_status;
  std::vector<float> forward_err;

  std::vector<cv::Mat> current_mono_pyr;
  buildOpticalFlowPyramid(mono, current_mono_pyr);

  // One single call allows OpenCV to run hot loops across contiguous memory
  // blocks
  cv::calcOpticalFlowPyrLK(prev_mono_pyr_, current_mono_pyr, flatPrev, flatNext,
                           forward_status, forward_err, win_size_, max_level_,
                           criteria_, 0);

  // now do reverse flow
  std::vector<uchar> reverse_status;
  std::vector<float> reverse_err;

  std::vector<cv::Point2f> flatReverse = flatNext;
  cv::calcOpticalFlowPyrLK(current_mono_pyr, prev_mono_pyr_, flatNext,
                           flatReverse, reverse_status, reverse_err, win_size_,
                           max_level_, criteria_, cv::OPTFLOW_USE_INITIAL_FLOW);

  static constexpr float kMaxErr = 20.0f;
  for (size_t i = 0; i < flatPrev.size(); ++i) {
    const bool both_status_good = forward_status.at(i) && reverse_status.at(i);
    const bool within_distance =
        utils::distance(flatPrev.at(i), flatReverse.at(i)) <= 0.5;
    const bool within_error =
        reverse_err[i] < kMaxErr && forward_err[i] < kMaxErr;

    // Check if KLT tracking succeeded and point remains inside image
    // boundaries use 2i to check image boundaries
    if (both_status_good && within_distance && within_error) {
      forward_status.at(i) = 1;
    } else {
      forward_status.at(i) = 0;
    }

    cv::Point2i flextNextInt = static_cast<cv::Point2i>(flatNext[i]);
    if (forward_status[i] && flextNextInt.x >= 0 &&
        flextNextInt.x < mono.cols && flextNextInt.y >= 0 &&
        flextNextInt.y < mono.rows) {
      auto object_id = previous_features_.object_ids[i];
      auto tracklet_id = previous_features_.ids[i];
      auto new_age = previous_features_.age[i] + 1;

      // LOG(INFO) << "obj j " << j << " with curr object mask " <<
      // currObjectMask.at<dyno::ObjectId>(flatNext[i]);

      if (object_id != object_mask.at<dyno::ObjectId>(flextNextInt)) {
        continue;
      }

      tracks_per_object[object_id].points.push_back(flatNext[i]);
      tracks_per_object[object_id].previous_points.push_back(flatPrev[i]);
      tracks_per_object[object_id].ids.push_back(tracklet_id);
      tracks_per_object[object_id].age.push_back(new_age);

      tracks_per_object[object_id].inlier.push_back(1);
      tracks_per_object[object_id].errors.push_back(forward_err[i]);
    }
  }

  gtsam::FastMap<ObjectId, FeatureBlockContainer::FeatureData>
      verified_tracks_per_object;

  for (const auto& [object_id, good_tracks] : tracks_per_object) {
    auto num_good_points = good_tracks.size();

    tracking_stats[object_id].tracked_after_flow = num_good_points;

    static constexpr double kHomographyReprThreshold = 2.0;
    // limit the number of iterations for speed
    static constexpr double kHomographyMaxIters = 500;
    cv::Mat inlier_mask = vision_tools::findHomography(
        good_tracks.previous_points, good_tracks.points,
        kHomographyReprThreshold, kHomographyMaxIters);

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

    // if we actually have any tracks
    // this will remove any objects with no tracks!
    if (verified_tracks.size() > 0) {
      LOG(INFO) << "j= " << object_id << "inlier/outlier "
                << verified_tracks.size() << "/" << num_good_points;
      verified_tracks_per_object[object_id] = verified_tracks;
      tracking_stats[object_id].tracked_after_or = verified_tracks.size();
    }
  }

  // update previous image pyramid for reuse
  prev_mono_pyr_ = current_mono_pyr;

  FeatureBlockContainer tracked_features(verified_tracks_per_object);
  LOG(INFO) << tracked_features.debugInfoString();
  return {tracked_features, tracking_stats};
}

void FeatureTrackerFast::fillDetectionParam(
    ObjectId object_id, const cv::Mat& mask, const cv::Rect& bounding_box,
    int current_tracks,
    FeatureTrackerFast::GfttDetector::Param& detection_param) const {
  detection_param.object_id = object_id;
  detection_param.mask = object_id;
  detection_param.bbox = bounding_box;
  detection_param.min_distance = getMinFeatureDistance(object_id);

  auto max_corners = getMaxCorners(object_id);
  // sort of dont want to do this: should extract N and then use ANMS maybe to
  // cut down!
  detection_param.max_corners = std::max(max_corners - current_tracks, 0);
}

void FeatureTrackerFast::buildOpticalFlowPyramid(
    const cv::Mat& mono, std::vector<cv::Mat>& pyramid) const {
  pyramid.resize(max_level_ + 1);
  cv::buildOpticalFlowPyramid(mono, pyramid, win_size_, max_level_, false,
                              cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT,
                              true  // critical for reuse
  );
}

cv::Mat FeatureTrackerFast::drawBatchedFeatures(
    const cv::Mat& image, const FeatureBlockContainer& batchedFeatures) const {
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
    if (batchedFeatures.age[i] > 0) {
      const auto previous_point = batchedFeatures.previous_points[i];
      cv::arrowedLine(canvas, previous_point, current_point, color, 2,
                      cv::LINE_AA, 0, 0.25);
    }

    // Draw current feature location
    cv::circle(canvas, current_point, 4, color, -1, cv::LINE_AA);

    // Optional white outer ring
    cv::circle(canvas, current_point, 6, cv::Scalar(255, 255, 255), 1,
               cv::LINE_AA);
  }

  return canvas;
}

}  // namespace dyno
