#include "dynosam2/frontend/FrontendViz.hpp"

#include "dynosam_common/utils/GtsamUtils.hpp"
#include "dynosam_common/utils/TimingStats.hpp"

namespace dyno2 {

FrontendViz::FrontendViz(const SlamBackend& estimator)
    : estimator_(estimator) {}

cv::Mat FrontendViz::draw(FrontendViz::Data::Ptr data) {
  dyno::utils::ChronoTimingStats timer("frontend.viz-tracking");
  const dyno::FrameId fid = data->images->frameId();
  const dyno::FrameId timestamp = data->images->timestamp();

  const ImageWrapper<ImageType::RGBMono>& img_wrapper = data->images->rgb();
  cv::Mat img_rgb = img_wrapper.toRGB().clone();

  // draw static correspondences
  using DoFrameTracks = FrontendViz::Data::DoFrameTracks;
  using ViFrameTracks = FrontendViz::Data::ViFrameTracks;

  const VioGraphEstimator& vi_estimator = estimator_.vioEstimator();
  const ViFrameTracks& vi_frame_tracks = data->vi_tracks;
  const ViFrame::Ptr vi_frame = vi_frame_tracks.frame;
  auto relative_motion = vi_frame->getRelativePose();
  CHECK(relative_motion);
  // sanity check
  CHECK_EQ(relative_motion->to(), fid);

  // frame against which we tracked
  const dyno::FrameId ref_frame = relative_motion->from();
  const ViFrame::Ptr ref_vi_frame = vi_estimator.getFramePtr(ref_frame);
  CHECK_NOTNULL(ref_vi_frame);

  for (dyno::TrackletId tid : vi_frame_tracks.correspondences) {
    const dyno::Keypoint pt = vi_frame_tracks.frame->keypointById(tid);
    const cv::Point2f pt_cv = dyno::utils::gtsamPointToCv(pt);

    // this point is in the map
    // TODO: draw projection
    if (vi_estimator.landmarkExists(tid)) {
      cv::circle(img_rgb, pt_cv, 2, blue, cv::FILLED, cv::LINE_AA);
    } else {
      // this should always be in the successful tracking case
      // TODO: ie tracking quality is good! (becuase if trackign quality is good
      // we should have tracks EACTLY from this frame!)
      // TODO: otherwise we should have no/few tracks (ie motion propogation)
      auto it = ref_vi_frame->local_indices.find(tid);
      if (it != ref_vi_frame->local_indices.end()) {
        cv::Point2f ref_pt_cv;
        ref_vi_frame->getCvKeypointByIndex(it->second, ref_pt_cv);

        cv::circle(img_rgb, pt_cv, 2, red, 1);

        cv::arrowedLine(img_rgb, ref_pt_cv, pt_cv, red, 1);
      }
    }
  }

  dyno::ObjectDetectionResult& object_detection = data->object_detection;
  const std::unordered_map<dyno::ObjectId, DoFrameTracks>& do_tracks =
      data->do_tracks;
  for (const dyno::TrackedObjectDetection& detection :
       object_detection.detections) {
    const dyno::ObjectId oid = detection.object_id;
    auto it = do_tracks.find(oid);
    if (it == do_tracks.end()) {
      // we have detection but no track
      continue;
    }

    // we should only have tracks for valid objects (ie those with a motion
    // estimate) therefore each should have a relative motion as well
    const DoGraphEstimator& do_estimator = estimator_.doEstimator(oid);
    const DoFrameTracks& do_track = it->second;
    const DoFrame::Ptr do_frame = do_track.frame;

    const auto& G = do_frame->getRelativeMotion();
    // sanity check
    CHECK_EQ(G.to(), fid);
    // frame against which we tracked
    const dyno::FrameId ref_frame = G.from();
    const DoFrame::Ptr ref_do_frame = do_estimator.getFramePtr(ref_frame);
    CHECK_NOTNULL(ref_do_frame);

    const dyno::Color object_colour = dyno::Color::uniqueId(oid);
    const cv::Scalar lmk_colour = object_colour.brightness(0.9f).bgra();
    const cv::Scalar tracked_colour = object_colour.brightness(0.5f).bgra();

    for (dyno::TrackletId tid : do_track.correspondences) {
      const dyno::Keypoint pt = do_frame->keypointById(tid);
      const cv::Point2f pt_cv = dyno::utils::gtsamPointToCv(pt);

      // this point is in the map
      // TODO: draw projection
      if (do_estimator.landmarkExists(tid)) {
        cv::circle(img_rgb, pt_cv, 3, lmk_colour, cv::FILLED, cv::LINE_AA);
      } else {
        // this should always be in the successful tracking case
        // TODO: ie tracking quality is good! (becuase if trackign quality is
        // good we should have tracks EACTLY from this frame!)
        // TODO: otherwise we should have no/few tracks (ie motion propogation)
        auto it = ref_do_frame->local_indices.find(tid);
        if (it != ref_do_frame->local_indices.end()) {
          cv::Point2f ref_pt_cv;
          ref_do_frame->getCvKeypointByIndex(it->second, ref_pt_cv);

          cv::circle(img_rgb, pt_cv, 2, tracked_colour, 1);
          cv::arrowedLine(img_rgb, ref_pt_cv, pt_cv, tracked_colour, 1);
        }
      }
    }

    updateObjectTrackingBox(img_rgb, detection, timestamp);
  }

  return img_rgb;
}

struct TargetLockStyle {
  int thickness = 2;
  int corner_radius = 10;

  // Glow configuration
  float blur_sigma = 4.0f;
  float glow_strength = 0.5f;
  int glow_thickness = 8;

  // Glow colour (can differ from the primary colour)
  cv::Scalar glow_color = cv::Scalar(255, 255, 255);

  // Optional colour gradient
  bool use_gradient = false;
  cv::Scalar gradient_color = cv::Scalar(255, 0, 255);
};

void drawTargetLock(cv::Mat& img, const cv::Point& topLeft,
                    const cv::Point& bottomRight, const cv::Scalar& color,
                    const TargetLockStyle& style);

void FrontendViz::updateObjectTrackingBox(
    cv::Mat& canvas, const dyno::TrackedObjectDetection& detection,
    dyno::Timestamp now) {
  const cv::Rect& bbox = detection.bounding_box;
  if (bbox.empty()) return;

  const dyno::ObjectId oid = detection.object_id;
  TemporalObjectState& state = states_[oid];
  if (state.first_seen_time < 0.0) {
    state.first_seen_time = now;
    state.object_id = oid;
  }

  state.last_seen_time = now;
  // time-based progress
  const dyno::Timestamp elapsed = now - state.first_seen_time;
  state.appear_progress =
      std::min(static_cast<float>(elapsed / appear_duration_sec_), 1.0f);

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

  const cv::Scalar base_color = Color::uniqueId(oid).bgra();

  cv::Scalar color(std::min(255.0, base_color[0] + 255.0 * flash),
                   std::min(255.0, base_color[1] + 255.0 * flash),
                   std::min(255.0, base_color[2] + 255.0 * flash),
                   base_color[3]);

  const cv::Point2f dir_tl = filtered_tl - filtered_center;
  const cv::Point2f p_tl = filtered_center + dir_tl * scale;

  const cv::Point2f dir_br = filtered_br - filtered_center;
  const cv::Point2f p_br = filtered_center + dir_br * scale;

  TargetLockStyle style;
  style.thickness = 2;
  style.corner_radius = radius;
  style.glow_thickness = 10;
  style.glow_strength = 0.35f;

  drawTargetLock(canvas, p_tl, p_br, color, style);
}

constexpr int kLineType = cv::LINE_AA;
static inline cv::Point pt(int x, int y) { return cv::Point(x, y); }

void drawTargetLock(cv::Mat& img, const cv::Point& topLeft,
                    const cv::Point& bottomRight, const cv::Scalar& color,
                    int thickness, int cornerRadius) {
  const int x1 = topLeft.x;
  const int y1 = topLeft.y;
  const int x2 = bottomRight.x;
  const int y2 = bottomRight.y;

  const cv::Point p1(x1, y1);
  const cv::Point p2(x2, y1);
  const cv::Point p3(x2, y2);
  const cv::Point p4(x1, y2);

  const int r = cornerRadius;

  // Top left
  cv::line(img, pt(p1.x + r, p1.y), pt(p1.x + 2 * r, p2.y), color, thickness,
           kLineType);
  cv::line(img, pt(p1.x, p1.y + r), pt(p1.x, p2.y + 2 * r), color, thickness,
           kLineType);

  // Top right
  cv::line(img, pt(p2.x - r, p2.y), pt(p2.x - 2 * r, p1.y), color, thickness,
           kLineType);
  cv::line(img, pt(p2.x, p2.y + r), pt(p2.x, p1.y + 2 * r), color, thickness,
           kLineType);

  // Bottom left
  cv::line(img, pt(p4.x + r, p4.y), pt(p4.x + 2 * r, p3.y), color, thickness,
           kLineType);
  cv::line(img, pt(p4.x, p4.y - r), pt(p4.x, p3.y - 2 * r), color, thickness,
           kLineType);

  // Bottom right
  cv::line(img, pt(p3.x - r, p3.y), pt(p3.x - 2 * r, p4.y), color, thickness,
           kLineType);
  cv::line(img, pt(p3.x, p3.y - r), pt(p3.x, p4.y - 2 * r), color, thickness,
           kLineType);

  // Arcs
  cv::ellipse(img, p1 + cv::Point(r, r), cv::Size(r, r), 0, 180, 270, color,
              thickness, kLineType);

  cv::ellipse(img, p2 + cv::Point(-r, r), cv::Size(r, r), 0, 270, 360, color,
              thickness, kLineType);

  cv::ellipse(img, p3 + cv::Point(-r, -r), cv::Size(r, r), 0, 0, 90, color,
              thickness, kLineType);

  cv::ellipse(img, p4 + cv::Point(r, -r), cv::Size(r, r), 0, 90, 180, color,
              thickness, kLineType);
}

void drawTargetLock(cv::Mat& img, const cv::Point& topLeft,
                    const cv::Point& bottomRight, const cv::Scalar& color,
                    const TargetLockStyle& style) {
  CV_Assert(img.type() == CV_8UC3);

  // ------------------------------------------------------------
  // 1. Determine the ROI affected by the glow.
  // ------------------------------------------------------------

  const int margin = static_cast<int>(std::ceil(3.0f * style.blur_sigma)) +
                     style.glow_thickness / 2 + 1;

  const cv::Rect image_bounds(0, 0, img.cols, img.rows);

  const cv::Rect rectangle_bounds(std::min(topLeft.x, bottomRight.x),
                                  std::min(topLeft.y, bottomRight.y),
                                  std::abs(bottomRight.x - topLeft.x) + 1,
                                  std::abs(bottomRight.y - topLeft.y) + 1);

  // Construct an expanded ROI, clipped to the image.
  const cv::Rect roi(rectangle_bounds.x - margin, rectangle_bounds.y - margin,
                     rectangle_bounds.width + 2 * margin,
                     rectangle_bounds.height + 2 * margin);

  const cv::Rect clipped_roi = roi & image_bounds;

  if (clipped_roi.empty()) {
    return;
  }

  cv::Mat image_roi = img(clipped_roi);

  // Convert global coordinates into ROI-local coordinates.
  const cv::Point offset(-clipped_roi.x, -clipped_roi.y);

  const cv::Point local_top_left = topLeft + offset;
  const cv::Point local_bottom_right = bottomRight + offset;

  // ------------------------------------------------------------
  // 2. Generate single-channel geometry mask.
  // ------------------------------------------------------------

  cv::Mat mask = cv::Mat::zeros(clipped_roi.size(), CV_8UC1);

  drawTargetLock(mask, local_top_left, local_bottom_right, cv::Scalar(255),
                 style.glow_thickness, style.corner_radius);

  // ------------------------------------------------------------
  // 3. Apply Gaussian blur.
  // ------------------------------------------------------------

  cv::Mat blurred_mask;

  if (style.blur_sigma > 0.0f) {
    cv::GaussianBlur(mask, blurred_mask, cv::Size(0, 0), style.blur_sigma);
  } else {
    blurred_mask = mask;
  }

  // ------------------------------------------------------------
  // 4. Colourise and composite the glow.
  // ------------------------------------------------------------

  const float inv_range = 1.0f / std::max(1, bottomRight.y - topLeft.y);

  for (int y = 0; y < clipped_roi.height; ++y) {
    // // Vertical gradient interpolation.
    // const float t = std::clamp(
    //     (clipped_roi.y + y - topLeft.y) * inv_range,
    //     0.0f, 1.0f);

    cv::Vec3b glow_color;
    for (int c = 0; c < 3; ++c) {
      glow_color[c] = cv::saturate_cast<uchar>(style.glow_color[c]);
    }

    // if (style.use_gradient) {
    //   for (int c = 0; c < 3; ++c) {
    //     glow_color[c] = cv::saturate_cast<uchar>(
    //         style.glow_color_start[c] * (1.0f - t) +
    //         style.glow_color_end[c] * t);
    //   }
    // } else {
    //   for (int c = 0; c < 3; ++c) {
    //     glow_color[c] = cv::saturate_cast<uchar>(
    //         style.glow_color_start[c]);
    //   }
    // }

    const uchar* alpha_ptr = blurred_mask.ptr<uchar>(y);

    cv::Vec3b* image_ptr = image_roi.ptr<cv::Vec3b>(y);

    for (int x = 0; x < clipped_roi.width; ++x) {
      const float alpha = (alpha_ptr[x] / 255.0f) * style.glow_strength;

      if (alpha <= 0.0f) {
        continue;
      }

      for (int c = 0; c < 3; ++c) {
        image_ptr[x][c] =
            cv::saturate_cast<uchar>(image_ptr[x][c] + glow_color[c] * alpha);
      }
    }
  }

  // ------------------------------------------------------------
  // 5. Draw crisp rectangle on top.
  // ------------------------------------------------------------

  drawTargetLock(img, topLeft, bottomRight, color, style.thickness,
                 style.corner_radius);
}

}  // namespace dyno2
