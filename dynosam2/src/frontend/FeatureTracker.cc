#include "dynosam2/frontend/FeatureTracker.hpp"

#include <tbb/task_group.h>

#include "dynosam_common/anms/NonMaximumSuppression.h"
#include "dynosam_common/utils/Timing.hpp"
#include "dynosam_cv/CvUtils.hpp"
#include "dynosam_nn/YoloV8ObjectDetector.hpp"

namespace dyno2 {

static constexpr size_t kOpticalFlowIterations{50};

static cv::Matx33f GetCvKMatrix(const Camera::Ptr camera) {
  return camera->getParams().getCameraMatrix();
}

static cv::Matx33f GetCvKInvMatrix(const Camera::Ptr camera) {
  return cv::Matx33f(camera->getParams().getCameraMatrix()).inv();
}

FeatureTracker::FeatureTracker(const FrontendParameters& parameters,
                               Camera::Ptr camera)
    : parameters_(parameters),
      camera_(camera),
      feature_detector_(camera->getParams().imageSize()),
      object_detection_impl_(this),
      optical_flow_impl_(kOpticalFlowIterations),
      K_cv_(GetCvKMatrix(camera)),
      K_inv_cv_(GetCvKInvMatrix(camera)) {
  if (!parameters_.prefer_provided_object_detection) {
    LOG(INFO) << "Creating object detection engine";
    dyno::YoloConfig yolo_config;
    dyno::ModelConfig model_config;
    model_config.model_file = "yolov8n-seg.pt";
    object_detection_engine_ =
        std::make_shared<dyno::YoloV8ObjectDetector>(model_config, yolo_config);
  }
}

FeatureTracker::Result FeatureTracker::track(
    ImageContainer& image_container,
    const std::optional<gtsam::Rot3>& R_km1_k) {
  dyno::utils::ChronoTimingStats feature_track_t("feature_tracker.track");
  dyno::utils::ChronoTimingStats object_detect_t(
      "feature_tracker.track.object_detect");
  const auto [object_detection_result, update_object_mask] =
      object_detection_impl_.detectAndTrack(image_container);
  object_detect_t.stop();

  // update image container with motion mask
  const cv::Mat object_masks = object_detection_result.labelled_mask;
  if (update_object_mask) {
    image_container.objectMotionMask(object_masks);
  }

  const WrappedRGBMono wrapped_rgb = image_container.rgb();
  const cv::Mat rgb = wrapped_rgb.image();
  // build the image pyramid from the wrapped image
  // this allows us to cache the computation of the pyramid within the
  // image_container to save comptutation between frames as long as we keep the
  // same image pyramid around!
  const ImagePyramid image_pyramid = wrapped_rgb.computeImagePyramid(
      optical_flow_impl_.win_size, optical_flow_impl_.max_level);
  const cv::Mat current_mono = image_pyramid.mono;
  curr_mono_pyr_ = image_pyramid.levels;

  ObjectIds object_ids = object_detection_result.objectIds();

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
    dyno::utils::ChronoTimingStats previous_t("feature_tracker.track.previous");
    dyno::utils::ChronoTimingStats sparse_t("feature_tracker.track.sparse");
    auto [tracked_features, tracking_stats] =
        sparseFeatureTrack(current_mono, object_masks);
    sparse_t.stop();

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

    // const auto outer_thickness = 12;
    static constexpr double kReferenceBorderThickness = 15.0;
    static constexpr double kReferenceImageArea = 640.0 * 480.0;

    const double image_scale = static_cast<double>(current_mono.cols) *
                               static_cast<double>(current_mono.rows) /
                               kReferenceImageArea;

    const int scaled_border_thickness =
        std::round(kReferenceBorderThickness * std::sqrt(image_scale));

    dyno::utils::ChronoTimingStats make_masks_t(
        "feature_tracker.track.make_masks");
    cv::Mat object_masks_binary = object_masks > 0;
    // to ensure the validatiy of the mask we will do some dilation
    // to both fill holes and to expant the object masks so we dont
    // attempt to track anywhere near the objects
    // this is slightly conservative but is helpful in practice
    cv::dilate(object_masks_binary, object_masks_binary,
               cv::getStructuringElement(
                   cv::MORPH_RECT, cv::Size(2 * scaled_border_thickness + 1,
                                            2 * scaled_border_thickness + 1)));
    cv::Mat static_mask;
    cv::bitwise_not(object_masks_binary, static_mask);
    binary_detection_masks[background_label] = static_mask;
    make_masks_t.stop();

    dyno::utils::ChronoTimingStats needs_detection_t(
        "feature_tracker.track.needs_detection");
    std::unordered_set<ObjectId> object_needs_detection;
    for (const auto& object_view : tracked_features.objectViews()) {
      const auto j = object_view.objectId();
      const auto& tracking_stats_j = tracking_stats[j];

      // should only be inliers!
      const auto num_tracked = object_view.size();
      const auto min_allowed_tracks = getMinAllowableTracks(j);
      const auto allowed_feature_distance = getMinFeatureDistance(j);

      VLOG(20) << "Tracked features for j=" << j << " n=" << num_tracked;

      cv::Mat& binary_detection_mask_j = binary_detection_masks[j];
      auto object_points = object_view.points();

      // add mask over current features
      // improve masking for longer tracks by reducing supression for old tracks
      for (size_t i = 0; i < num_tracked; i++) {
        cv::circle(binary_detection_mask_j, object_points[i],
                   allowed_feature_distance, cv::Scalar(0), cv::FILLED);
      }

      const bool too_few_tracks = num_tracked < min_allowed_tracks;

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
        VLOG(20) << "Replacing tracks for poorly tracked object " << j;
        object_needs_detection.insert(j);

      } else {
        VLOG(20) << "Good tracks for object " << j;
      }
    }
    needs_detection_t.stop();

    GfttDetector::Params feature_detection_params;
    feature_detection_params.quality_level = 0.01;
    ObjectIds objects_ids_for_detection;

    for (const auto& [j, masks_j] : binary_detection_masks) {
      // if does not exist in current tracking and we have detections
      // add as new object
      bool is_new = !tracked_features.containsObject(j);
      if (object_needs_detection.count(j) > 0 || is_new) {
        VLOG(20) << "Object j " << j << " needs detection";
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
      utils::ChronoTimingStats detect_t("feature_tracker.track.re_detect");
      // TODO: still need to handle tracklet ids!
      FeatureBlockContainer detected_features = feature_detector_.calc(
          current_mono, object_masks, feature_detection_params);
      previous_features_ = tracked_features.merge(detected_features);
      detect_t.stop();

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

  return {object_detection_result, &previous_features_};
}

FeatureTracker::ObjectDetectionImpl::ObjectDetectionImpl(
    FeatureTracker* parent_)
    : parent(parent_) {}

std::pair<ObjectDetectionResult, bool>
FeatureTracker::ObjectDetectionImpl::detectAndTrack(
    const ImageContainer& image_container) const {
  if (parent->parameters_.prefer_provided_object_detection) {
    if (!image_container.hasObjectMask()) {
      LOG(FATAL)
          << "Params specify prefer provided object mask but input is missing!";
    }
    return detectionViaObjectMask(image_container);

  } else {
    return detectionViaInference(image_container);
  }
}

std::pair<ObjectDetectionResult, bool>
FeatureTracker::ObjectDetectionImpl::detectionViaObjectMask(
    const ImageContainer& image_container) const {
  const cv::Mat object_masks = image_container.objectMotionMask();

  ObjectIds object_ids = dyno::getObjectLabels(object_masks);

  std::vector<cv::Rect> bounding_boxes;
  dyno::getObjectBoundingBoxes(object_masks, object_ids, bounding_boxes);

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
  return {result, false};
}

std::pair<ObjectDetectionResult, bool>
FeatureTracker::ObjectDetectionImpl::detectionViaInference(
    const ImageContainer& image_container) const {
  auto& detection_engine = parent->object_detection_engine_;
  CHECK_NOTNULL(detection_engine);

  VLOG(50) << "Running object detection and tracking inference k="
           << image_container.frameId();
  return {detection_engine->process(image_container.rgb()), true};
}

FeatureTracker::GfttDetector::GfttDetector(const cv::Size& size)
    : id_count_(0) {
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

  // allocate GPU memory
  d_mono_.create(size, CV_8UC1);
  d_eig_.create(size, CV_32FC1);

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

FeatureBlockContainer FeatureTracker::GfttDetector::calc(
    const cv::Mat& mono, const cv::Mat& object_mask, const Params& params) {
  dyno::utils::ChronoTimingStats r("feature_tracker.detector");
  CV_Assert(mono.type() == CV_8UC1 || mono.type() == CV_32FC1);

  if (params.empty()) {
    return {};
  }

  utils::ChronoTimingStats t1("feature_tracker.corner_min_eigen");
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
  dyno::utils::ChronoTimingStats tmin("feature_tracker.minMaxLoc");
  double maxVal = 0;
  cv::minMaxLoc(eig_, nullptr, &maxVal);
  tmin.stop();

  // Establish the baseline absolute threshold based on global max quality
  const float threshold = static_cast<float>(maxVal * params.quality_level);

  // --- STEP 2: Local Non-Maximum Suppression (NMS) via Dilation ---
  // OpenCV's internal GFTT uses a dilation trick to find local maxima
  // efficiently
  cv::Mat localMax;
  cv::dilate(eig_, localMax, cv::Mat());

  std::vector<CornerResponses> batchedResults(params.size());

  const cv::Rect image_bounds(1, 1, mono.cols - 2, mono.rows - 2);
  // 11 approximatees a radius of 5 pixels from the edge
  // but we could do more?!
  // in pixels
  // should maybe be adaptive with object size? Whatever
  constexpr int boundary_distance = 5;
  const cv::Mat kernel = cv::getStructuringElement(
      cv::MORPH_RECT,
      cv::Size(2 * boundary_distance + 1, 2 * boundary_distance + 1));

  dyno::utils::ChronoTimingStats t3("feature_tracker.masks_loop");
  tbb::task_group task_group;
  for (size_t m = 0; m < params.size(); m++) {
    task_group.run([&, m] {
      const auto& param = params[m];
      const cv::Mat& mask = param.mask;

      cv::Mat safe_mask;
      cv::erode(mask, safe_mask, kernel);

      // cv::Mat dist;
      // // no need to compute the mask for the static background?
      // // can we save lots of computation here?
      // // ideally should not be close to the dynamic points?
      // // i think this is the biggest computation bottleneck!
      // cv::distanceTransform(mask, dist, cv::DIST_L2, 3);

      CV_Assert(mask.type() == CV_8UC1);

      const int maxFeatureCount = param.max_corners;
      const float minDistance = param.min_distance;

      if (maxFeatureCount <= 0) return;

      // --------------------------------------------------------------
      // Restrict the entire detection process to the object bounding
      // box rather than scanning the entire image.
      // --------------------------------------------------------------

      // clip bounding box so that it is not at the immediate edge of the image
      // and therefore out of bounds
      const cv::Rect bbox = param.bbox & image_bounds;

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
        const uchar* safePtr =
            safe_mask.empty() ? nullptr : safe_mask.ptr<uchar>(y);
        // const float* distPtr = dist.empty() ? nullptr : dist.ptr<float>(y);

        for (int x = x0; x < x1; ++x) {
          // Check mask FIRST.
          //
          // For small dynamic object masks this avoids reading
          // the distance transform for almost every background
          // pixel in the bounding box.
          if (maskPtr && !maskPtr[x]) {
            continue;
          }

          // Mask-edge constraint
          // if (distPtr && distPtr[x] <= 5.0f) continue;
          if (safePtr && safePtr[x] == 0) continue;

          const float val = eigPtr[x];

          if (val <= threshold) continue;

          if (val != maxPtr[x]) continue;

          // // Mask-edge constraint
          // if (distPtr && distPtr[x] <= 5.0f) continue;

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

  dyno::utils::ChronoTimingStats t_terms("feature_tracker.make_blocks");
  std::vector<std::pair<ObjectId, FeatureBlockContainer::FeatureData>> terms(
      params.size());
  for (size_t i = 0; i < params.size(); i++) {
    terms[i].first = params[i].object_id;

    const CornerResponses& corner_responses = batchedResults[i];
    auto num_points = corner_responses.size();

    FeatureBlockContainer::FeatureData& feature_data = terms[i].second;
    feature_data.resize(num_points);
    feature_data.points = corner_responses.keypoints;
    feature_data.ids = getNewIdsByChunk(num_points);

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

  dyno::utils::ChronoTimingStats t4("feature_tracker.sub_pixe_refine");

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

TrackletIds FeatureTracker::GfttDetector::getNewIdsByChunk(size_t N) {
  const TrackletId first_id = id_count_;
  id_count_ += N;

  std::vector<TrackletId> tracklet_ids(N);
  std::iota(tracklet_ids.begin(), tracklet_ids.end(), first_id);

  return tracklet_ids;
}

std::pair<FeatureBlockContainer, FeatureTracker::FlowTrackingStatsMap>
FeatureTracker::sparseFeatureTrack(const cv::Mat& mono,
                                   const cv::Mat& object_mask) {
  dyno::utils::ChronoTimingStats timer("feature_tracker.track_gfft");

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

    VLOG(30) << "Preparing featue tracking structures j=" << object_id
             << " n=" << num_points;
  }

  OpticalFlowLK::Result flow_result = optical_flow_impl_(
      prev_mono_pyr_, curr_mono_pyr_, previous_features_.points);

  const std::vector<cv::Point2f>& predicted_points =
      flow_result.predictedPoints();

  size_t flow_bad_count{0};
  size_t bounds_bad_count{0};
  const std::vector<cv::Point2f>& from_points = flow_result.fromPoints();
  const std::vector<float>& error = flow_result.error();
  for (size_t i = 0; i < flow_result.size(); i++) {
    const cv::Point2f& predicted_pt = predicted_points[i];

    if (!flow_result.isGood(i)) {
      flow_bad_count++;
      continue;
    }

    auto object_id = previous_features_.object_ids[i];
    if (!checkBoundsAndLabel(predicted_pt, object_mask, object_id, rows,
                             cols)) {
      bounds_bad_count++;
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

  gtsam::FastMap<ObjectId, FeatureBlockContainer::FeatureData>
      verified_tracks_per_object;

  dyno::utils::ChronoTimingStats lkt_outliers_t("feature_tracker.lkt_outliers");

  const cv::Mat K = camera_->getParams().getCameraMatrix();

  static constexpr double kThresholdE = 1.0;
  static constexpr double kEMaxIters = 100;

  // Build a stable list of objects to process. This also updates the tracking
  // statistics serially, avoiding concurrent writes to tracking_stats.
  struct ObjectTracks {
    ObjectId object_id;
    const FeatureBlockContainer::FeatureData* tracks;
  };

  std::vector<ObjectTracks> objects_to_process;
  objects_to_process.reserve(tracks_per_object.size());

  for (const auto& [object_id, good_tracks] : tracks_per_object) {
    tracking_stats[object_id].tracked_after_flow = good_tracks.size();

    objects_to_process.push_back({object_id, &good_tracks});
  }

  // Each task writes to its own result slot. We construct the final map
  // serially after all tasks complete, avoiding concurrent map insertions.
  std::vector<FeatureBlockContainer::FeatureData> verified_tracks(
      objects_to_process.size());

  tbb::task_group tasks;

  for (size_t idx = 0; idx < objects_to_process.size(); ++idx) {
    tasks.run([&, idx]() {
      const auto& [object_id, good_tracks] = objects_to_process[idx];

      const size_t num_good_points = good_tracks->size();

      const cv::Mat inlier_mask =
          dyno::findEssential(good_tracks->previous_points, good_tracks->points,
                              K, kThresholdE, kEMaxIters);

      FeatureBlockContainer::FeatureData verified;
      verified.reserve(num_good_points);

      for (int i = 0; i < inlier_mask.rows; ++i) {
        if (inlier_mask.at<uchar>(i)) {
          verified.points.push_back(good_tracks->points[i]);
          verified.previous_points.push_back(good_tracks->previous_points[i]);
          verified.ids.push_back(good_tracks->ids[i]);
          verified.age.push_back(good_tracks->age[i]);
          verified.errors.push_back(good_tracks->errors[i]);
          verified.inlier.push_back(good_tracks->inlier[i]);
        }
      }

      VLOG(30) << "j=" << object_id << " inlier/outlier " << verified.size()
               << "/" << num_good_points;

      // Each task exclusively owns this result slot.
      verified_tracks[idx] = std::move(verified);
    });
  }

  tasks.wait();
  lkt_outliers_t.stop();

  // Assemble the final map and update tracking statistics serially.
  for (size_t idx = 0; idx < objects_to_process.size(); ++idx) {
    const ObjectId object_id = objects_to_process[idx].object_id;
    auto& verified = verified_tracks[idx];

    // Exclude objects with no remaining inlier tracks.
    if (!verified.empty()) {
      tracking_stats[object_id].tracked_after_or = verified.size();

      verified_tracks_per_object.emplace(object_id, std::move(verified));
    }
  }

  FeatureBlockContainer tracked_features(verified_tracks_per_object);

  LOG(INFO) << "Tracked: " << tracked_features.debugInfoString();

  return {tracked_features, tracking_stats};
}

bool FeatureTracker::predictKeypointsGivenRotation(
    const std::vector<cv::Point2f>& previous_pts,
    const gtsam::Rot3& R_previous_curr,
    std::vector<cv::Point2f>& predicted_pts_curr) const {
  // Handle case when rotation is small: just copy prev_kps
  // Removed bcs even small rotations lead to huge optical flow at the borders
  // of the image.
  // Keep because then you save a lot of computation.
  static constexpr double kSmallRotationTol = 1e-4;
  if (std::abs(1.0 - std::abs(R_previous_curr.toQuaternion().w())) <
      kSmallRotationTol) {
    predicted_pts_curr = previous_pts;
    return true;
  }

  // // R is a relative rotation which takes a vector from the last frame to
  // // the current frame.
  // const cv::Matx33f R =
  // dyno::utils::gtsamMatrix3ToCvMat(R_previous_curr.matrix());
  // // Get bearing vector for kpt, rotate knowing frame to frame rotation,
  // // get keypoints again
  // const cv::Matx33f H = K_cv_ * R * K_inv_cv_;

  // const size_t& n_kps = previous_pts.size();
  // predicted_pts_curr.reserve(n_kps);
  // for (size_t i = 0u; i < n_kps; ++i) {
  //   // Create homogeneous keypoints.
  //   const auto& prev_kpt = previous_pts[i];
  //   cv::Vec3f p1(prev_kpt.x, prev_kpt.y, 1.0f);

  //   cv::Vec3f p2 = H * p1;

  //   // Project predicted bearing vectors to 2D again and re-homogenize.
  //   cv::Point2f new_kpt;
  //   if (p2[2] > 0.0f) {
  //     new_kpt = cv::Point2f(p2[0] / p2[2], p2[1] / p2[2]);
  //   } else {
  //     LOG(WARNING) << "Landmark behind the camera:\n"
  //                  << "- p1: " << p1 << '\n'
  //                  << "- p2: " << p2;
  //     new_kpt = prev_kpt;
  //   }
  //   // Check that keypoints remain inside the image boundaries!
  //   if (isWithinShrunkenImage(new_kpt)) {
  //     predicted_pts_curr.push_back(new_kpt);
  //   } else {
  //     // Otw copy-paste previous keypoint.
  //     predicted_pts_curr.push_back(prev_kpt);
  //   }
}

void FeatureTracker::fillDetectionParam(
    ObjectId object_id, const cv::Mat& mask, const cv::Rect& bounding_box,
    int current_tracks,
    FeatureTracker::GfttDetector::Param& detection_param) const {
  auto numCornersNeeded = [&](ObjectId object_id) -> int {
    auto desired_max_tracks =
        static_cast<int>(getMaxTrackingCorners(object_id));
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
  detection_param.max_corners =
      static_cast<int>(getMaxDetectionCorners(object_id));
  detection_param.num_corners_needed =
      static_cast<int>(numCornersNeeded(object_id));
}

size_t FeatureTracker::getMinFeatureDistance(ObjectId object_id) const {
  return object_id > background_label
             ? parameters_.dynamic_tracks.min_distance_btw_tracked_and_detected
             : parameters_.static_tracks.min_distance_btw_tracked_and_detected;
}
size_t FeatureTracker::getMaxDetectionCorners(ObjectId object_id) const {
  return object_id > background_label
             ? parameters_.dynamic_tracks.max_nr_kepoints_extaction
             : parameters_.static_tracks.max_nr_kepoints_extaction;
}

size_t FeatureTracker::getMaxTrackingCorners(ObjectId object_id) const {
  return object_id > background_label
             ? parameters_.dynamic_tracks.max_nr_desired_tracks
             : parameters_.static_tracks.max_nr_desired_tracks;
}

size_t FeatureTracker::getMinAllowableTracks(ObjectId object_id) const {
  return object_id > background_label ? parameters_.dynamic_tracks.min_nr_tracks
                                      : parameters_.static_tracks.min_nr_tracks;
}

}  // namespace dyno2
