#include "dynosam/frontend/PoseChangeVIFrontendViz.hpp"

#include "dynosam_common/viz/Colour.hpp"

namespace dyno {

// declare helper function for Viz
void roundedRectangle(cv::Mat& img, const cv::Point& topLeft,
                      const cv::Point& bottomRight, const cv::Scalar& color,
                      int thickness = 2, int cornerRadius = 20);

ViTrackingViz::ViTrackingViz(const ImageTracksParams& viz_params)
    : viz_params_(viz_params) {}

cv::Mat ViTrackingViz::vizTracking(const Frame& frame_km1, const Frame& frame_k,
                                   const Data& data) {
  utils::ChronoTimingStats timer("pc-frontend.viz-tracking");
  const ImageWrapper<ImageType::RGBMono>& img_wrapper =
      frame_k.imageContainer().rgb();
  cv::Mat img_rgb = img_wrapper.toRGB().clone();

  std::string static_tracks_info_string;
  drawStaticTracks(img_rgb, static_tracks_info_string, frame_km1, frame_k,
                   data);

  std::string dynamic_tracks_info_string;
  drawDynamicTracks(img_rgb, dynamic_tracks_info_string, frame_km1, frame_k,
                    data);

  if (viz_params_.showFrameInfo()) {
    std::string info_string =
        static_tracks_info_string + dynamic_tracks_info_string;
    writeFrameInfo(img_rgb, info_string);
  }
  return img_rgb;
}

void ViTrackingViz::drawStaticTracks(cv::Mat& img, std::string& info,
                                     const Frame& frame_km1,
                                     const Frame& frame_k, const Data& data) {
  const bool debug = viz_params_.isDebug();
  const bool show_intermediate_tracking =
      viz_params_.showIntermediateTracking();
  const int static_point_thickness = viz_params_.featureThickness();

  static const cv::Scalar red(Color::red().bgra());
  static const cv::Scalar green(Color::green().bgra());
  static const cv::Scalar blue(Color::blue().bgra());

  size_t num_points_tracked = 0;

  // Add all keypoints in cur_frame with the tracks.
  for (const Feature::Ptr& feature : frame_k.static_features_) {
    const Keypoint& px_cur = feature->keypoint();
    const auto pc_cur = utils::gtsamPointToCv(px_cur);
    if (!feature->usable() &&
        show_intermediate_tracking) {  // Untracked landmarks are red.
      cv::circle(img, pc_cur, static_point_thickness, red, 2, cv::LINE_AA);
    } else {
      const Feature::Ptr& prev_feature =
          frame_km1.static_features_.getByTrackletId(feature->trackletId());
      if (prev_feature) {
        // If feature was in previous frame, display tracked feature with
        // green circle/line:
        cv::circle(img, pc_cur, static_point_thickness, green, 1);

        // draw the optical flow arrow
        const auto pc_prev = utils::gtsamPointToCv(prev_feature->keypoint());
        cv::arrowedLine(img, pc_prev, pc_cur, green, 1);

        num_points_tracked++;

      } else if (debug &&
                 show_intermediate_tracking) {  // New feature tracks are blue.
        cv::circle(img, pc_cur, 6, blue, 1);
      }
    }
  }

  if (data.keyframe_info.camera_keyframe) {
    CKF_count++;
  }

  std::stringstream ss;
  ss << "Frame: " << frame_k.getFrameId() << " ";
  ss << "[VO tracks: " << num_points_tracked << " ";
  ss << "Cam KFs: " << CKF_count << " ";
  ss << "quailty: " << to_string(data.camera_tracking_quality) << "]";

  info = ss.str();
}

void ViTrackingViz::drawDynamicTracks(cv::Mat& img, std::string& info,
                                      const Frame& frame_km1,
                                      const Frame& frame_k, const Data& data) {
  for (const Feature::Ptr& feature :
       frame_k.dynamic_features_.usableIterator()) {
    const auto px_cur = utils::gtsamPointToCv(feature->keypoint());
    const Feature::Ptr& prev_feature =
        frame_km1.dynamic_features_.getByTrackletId(feature->trackletId());
    if (prev_feature) {
      const auto px_prev = utils::gtsamPointToCv(prev_feature->keypoint());
      const cv::Scalar colour = Color::uniqueId(feature->objectId()).bgra();

      // cv::arrowedLine(img, px_prev,px_cur,colour, 1, 8, 0, 0.1);
      // cv::circle(img, utils::gtsamPointToCv(px_cur), 2, colour, -1);

      // draw feature as rectangle
      // 8 pixels size
      constexpr static int size = 6;
      int half = size / 2;
      cv::Point tl(px_cur.x - half, px_cur.y - half);
      cv::Point br(px_cur.x + half, px_cur.y + half);
      cv::rectangle(img, tl, br, colour, 2, -1);
    }
  }

  // mark new object keyframes
  for (const auto& kf_info : data.keyframe_info.object_keyframes) {
    ObjectId object_id = kf_info.object_id;
    if (!OKF_count_.exists(object_id)) {
      OKF_count_[object_id] = 0;
    }
    OKF_count_[object_id]++;
  }

  std::vector<ObjectId> objects_to_print;
  double now = frame_k.getTimestamp();

  for (const auto& [object_id, motion_track_status] :
       data.object_tracking_statuses) {
    // only draw well tracked objects
    if (motion_track_status != ObjectTrackingStatus::WellTracked) {
      continue;
    }

    std::optional<SingleDetectionResult> maybe_detection_result =
        frame_k.objectDetection(object_id);
    if (!maybe_detection_result) {
      continue;
    }
    const cv::Rect& bb = maybe_detection_result->bounding_box;

    if (bb.empty()) continue;

    auto& state = states_[object_id];
    objects_to_print.push_back(object_id);

    if (state.first_seen_time < 0.0) {
      state.first_seen_time = now;
      state.object_id = object_id;
    }

    state.last_seen_time = now;

    // time-based progress
    double elapsed = now - state.first_seen_time;
    state.appear_progress =
        std::min(static_cast<float>(elapsed / appear_duration_sec_), 1.0f);

    if (viz_params_.drawObjectBoundingBox()) {
      // const cv::Scalar colour = Color::uniqueId(object_id).bgra();
      // const std::string label = "object " + std::to_string(object_id);
      // utils::drawLabeledBoundingBox(img_rgb, label, colour, bb,
      // bbox_thickness);
      drawAnimatedBox(img, bb, state);
    }
  }

  if (viz_params_.drawObjectMask()) {
    const cv::Mat& object_mask = frame_k.imageContainer().objectMotionMask();

    constexpr static float kAlpha = 0.7;
    utils::labelMaskToRGB(object_mask, img, img, kAlpha);
  }

  std::stringstream ss;
  ss << " [Objects (KF): ";

  if (objects_to_print.empty()) {
    ss << "None";
  } else {
    for (size_t i = 0; i < objects_to_print.size(); ++i) {
      ss << objects_to_print[i] << " (" << OKF_count_[objects_to_print[i]]
         << ")";
      if (i != objects_to_print.size() - 1) {
        ss << ", ";  // Add comma between elements
      }
    }
  }
  ss << "]";

  info = ss.str();
}

void ViTrackingViz::writeFrameInfo(cv::Mat& img,
                                   const std::string& info_string) const {
  constexpr static double kFontScale = 0.4;
  constexpr static int kFontFace = cv::FONT_HERSHEY_SIMPLEX;
  constexpr static int kThickness = 1;

  int base_line;
  cv::Size text_size = cv::getTextSize(info_string, kFontFace, kFontScale,
                                       kThickness, &base_line);
  cv::Mat image_text =
      cv::Mat(img.rows + text_size.height + 10, img.cols, img.type());
  img.copyTo(image_text.rowRange(0, img.rows).colRange(0, img.cols));
  image_text.rowRange(img.rows, image_text.rows) =
      cv::Mat::zeros(text_size.height + 10, img.cols, img.type());
  cv::putText(image_text, info_string, cv::Point(5, image_text.rows - 5),
              kFontFace, kFontScale, cv::Scalar(255, 255, 255), kThickness);

  img = image_text;
}

void ViTrackingViz::drawAnimatedBox(cv::Mat& img, const cv::Rect& bbox,
                                    const TemporalObjectState& state) const {
  const float t = state.appear_progress;

  // --- slower, readable lock ---
  const float lock = 1.0f - std::exp(-3.5f * t);

  // optional: slow near end (magnetic feel)
  const float smooth_lock = lock * lock;

  std::array<cv::Point2f, 4> raw_corners = {
      cv::Point2f(bbox.x, bbox.y), cv::Point2f(bbox.x + bbox.width, bbox.y),
      cv::Point2f(bbox.x, bbox.y + bbox.height),
      cv::Point2f(bbox.x + bbox.width, bbox.y + bbox.height)};

  if (!state.corners_initialized) {
    state.filtered_corners = raw_corners;
    state.corners_initialized = true;
  }

  // ----------------------------
  // EMA smoothing (visual jitter reduction)
  // ----------------------------
  for (int i = 0; i < 4; ++i) {
    state.filtered_corners[i] =
        corner_smoothing_alpha_ * raw_corners[i] +
        (1.0f - corner_smoothing_alpha_) * state.filtered_corners[i];
  }

  const auto& filtered_tl = state.filtered_corners[0];
  const auto& filtered_br = state.filtered_corners[3];

  const cv::Rect filtered_bbox(filtered_tl, filtered_br);
  cv::Point2f filtered_center(filtered_bbox.x + filtered_bbox.width * 0.5f,
                              filtered_bbox.y + filtered_bbox.height * 0.5f);

  // --- corner growth ---
  const float corner_t = std::min(t * 1.8f, 1.0f);
  const int base_len = static_cast<int>(
      std::min(filtered_bbox.width, filtered_bbox.height) * 0.25f);
  const int len = static_cast<int>(base_len * corner_t);
  const float radius = len * 0.4f;

  // ----------------------------
  // far-to-near scale (lock-on feel)
  // ----------------------------
  const float start_scale = 1.6f;
  const float scale = 1.0f + (start_scale - 1.0f) * (1.0f - smooth_lock);

  // ----------------------------
  // flash effect (subtle acquisition cue)
  // ----------------------------
  const float flash = std::exp(-4.0f * t);

  const cv::Scalar base_color = Color::uniqueId(state.object_id).bgra();

  cv::Scalar color(std::min(255.0, base_color[0] + 255.0 * flash),
                   std::min(255.0, base_color[1] + 255.0 * flash),
                   std::min(255.0, base_color[2] + 255.0 * flash),
                   base_color[3]);

  static constexpr int thickness = 5;

  const cv::Point2f dir_tl = filtered_tl - filtered_center;
  const cv::Point2f p_tl = filtered_center + dir_tl * scale;

  const cv::Point2f dir_br = filtered_br - filtered_center;
  const cv::Point2f p_br = filtered_center + dir_br * scale;

  roundedRectangle(img, p_tl, p_br, color, thickness, radius);
}

static inline cv::Point pt(int x, int y) { return cv::Point(x, y); }

// mostly vibe-coded function to draw a target lock with rounded corners
void roundedRectangle(cv::Mat& img, const cv::Point& topLeft,
                      const cv::Point& bottomRight, const cv::Scalar& color,
                      int thickness, int cornerRadius) {
  static constexpr int lineType = cv::LINE_AA;

  int x1 = topLeft.x;
  int y1 = topLeft.y;
  int x2 = bottomRight.x;
  int y2 = bottomRight.y;

  // ------------------------------------------------------------
  // corners (same layout as Python version)
  // p1 - p2
  // |     |
  // p4 - p3
  // ------------------------------------------------------------
  const cv::Point p1(x1, y1);
  const cv::Point p2(x2, y1);
  const cv::Point p3(x2, y2);
  const cv::Point p4(x1, y2);

  const int r = cornerRadius;

  // ============================================================
  // TOP LEFT
  // ============================================================
  cv::line(img, pt(p1.x + r, p1.y), pt(p1.x + 2 * r, p2.y), color, thickness,
           lineType);

  cv::line(img, pt(p1.x, p1.y + r), pt(p1.x, p2.y + 2 * r), color, thickness,
           lineType);

  // ============================================================
  // TOP RIGHT
  // ============================================================
  cv::line(img, pt(p2.x - r, p2.y), pt(p2.x - 2 * r, p1.y), color, thickness,
           lineType);

  cv::line(img, pt(p2.x, p2.y + r), pt(p2.x, p1.y + 2 * r), color, thickness,
           lineType);

  // ============================================================
  // BOTTOM LEFT
  // ============================================================
  cv::line(img, pt(p4.x + r, p4.y), pt(p4.x + 2 * r, p3.y), color, thickness,
           lineType);

  cv::line(img, pt(p4.x, p4.y - r), pt(p4.x, p3.y - 2 * r), color, thickness,
           lineType);

  // ============================================================
  // BOTTOM RIGHT
  // ============================================================
  cv::line(img, pt(p3.x - r, p3.y), pt(p3.x - 2 * r, p4.y), color, thickness,
           lineType);

  cv::line(img, pt(p3.x, p3.y - r), pt(p3.x, p4.y - 2 * r), color, thickness,
           lineType);

  // ============================================================
  // ARCS (same quadrant logic as Python/OpenCV version)
  // ============================================================

  cv::ellipse(img, p1 + cv::Point(r, r), cv::Size(r, r), 0, 180, 270, color,
              thickness, lineType);

  cv::ellipse(img, p2 + cv::Point(-r, r), cv::Size(r, r), 0, 270, 360, color,
              thickness, lineType);

  cv::ellipse(img, p3 + cv::Point(-r, -r), cv::Size(r, r), 0, 0, 90, color,
              thickness, lineType);

  cv::ellipse(img, p4 + cv::Point(r, -r), cv::Size(r, r), 0, 90, 180, color,
              thickness, lineType);
}

}  // namespace dyno
