#include "dynosam/frontend/vision/FeatureTrackerFast.hpp"

namespace dyno {

FeatureTrackerFast::FeatureTrackerFast(const FrontendParams& params,
                                       Camera::Ptr camera,
                                       ImageDisplayQueue* display_queue)
    : frontend_params_(params),
      tracklet_id_manager(TrackletIdManager::instance()),
      win_size_(21, 21),
      max_level_(3),
      criteria_(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, 0.001),
      object_detection_impl_(this) {
  // initDeviceMemory(camera.)
}

Frame::Ptr FeatureTrackerFast::track(
    FrameId frame_id, Timestamp timestamp,
    const ImageContainer& image_container,
    const std::optional<gtsam::Rot3>& R_km1_k) {
  cv::Mat current_mono = ImageType::RGBMono::toMono(image_container.rgb());

  // object detect/psuedo object detection measurement
  ObjectDetectionResult object_detection_result =
      object_detection_impl_.detectAndTrack(image_container);
  // update image container with motion mask
  const cv::Mat object_masks = object_detection_result.labelled_mask;

  ObjectIds object_ids = object_detection_result.objectIds();
  // detection mask with depth -> how to translate to binary masks for each
  // object + background
  if (prev_mono_.empty()) {
    std::vector<CornersPerDetectionParams> corner_detection_params(
        object_ids.size() + 1);
    for (const SingleDetectionResult& single_detection :
         object_detection_result.detections) {
      // corner_detection_params[i].object_id = single_detection.object_id;
      // corner_detection_params[i].mask = single_detection.mask;
      // corner_detection_params[i].bbox = single_detection.bounding_box;
      // corner_detection_params[i].max_corners = 600;
      // corner_detection_params[i].min_distance = 8;
    }

    cv::Mat object_masks_binary = object_masks > 0;
    cv::Mat static_mask;
    cv::bitwise_not(object_masks_binary, static_mask);

    cv::Rect image_rect;
    image_rect.x = 0;
    image_rect.y = 0;
    image_rect.width = current_mono.cols;
    image_rect.height = current_mono.rows;

    corner_detection_params[object_ids.size()].object_id = 0;
    corner_detection_params[object_ids.size()].mask = static_mask;
    corner_detection_params[object_ids.size()].bbox = image_rect;
    corner_detection_params[object_ids.size()].max_corners = 1000;
    corner_detection_params[object_ids.size()].min_distance = 15;
    object_ids.push_back(0);
  }
}

FeatureTrackerFast::ObjectDetectionImpl::ObjectDetectionImpl(
    FeatureTrackerFast* parent_)
    : parent(parent_) {}

ObjectDetectionResult FeatureTrackerFast::ObjectDetectionImpl::detectAndTrack(
    const ImageContainer& image_container) const {
  // if (parent->frontend_params_.prefer_provided_object_detection) {
  //     if (!image_container.hasObjectMask()) {
  //         LOG(FATAL) << "Params specify prefer provided object mask but input
  //         "
  //                     "is missing!";

  //     }
  //     return detectionViaObjectMask(image_container);

  // } else {
  //     return detectionViaInference(image_container);
  // }
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

}  // namespace dyno
