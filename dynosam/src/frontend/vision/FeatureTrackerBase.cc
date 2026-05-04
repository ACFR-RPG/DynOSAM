/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a
 copy
 *   of this software and associated documentation files (the "Software"), to
 deal
 *   in the Software without restriction, including without limitation the
 rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in
 all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE
 *   SOFTWARE.
 */

#include "dynosam/frontend/vision/FeatureTrackerBase.hpp"

#include "dynosam/frontend/anms/NonMaximumSuppression.h"
#include "dynosam_common/utils/GtsamUtils.hpp"
#include "dynosam_common/utils/OpenCVUtils.hpp"
#include "dynosam_common/viz/Colour.hpp"

namespace dyno {

decltype(TrackletIdManager::instance_) TrackletIdManager::instance_;

FeatureTrackerBase::FeatureTrackerBase(const TrackerParams& params,
                                       Camera::Ptr camera,
                                       ImageDisplayQueue* display_queue)
    : params_(params),
      img_size_(camera->getParams().imageSize()),
      camera_(camera),
      display_queue_(display_queue) {
  setImageBounds();
}

void FeatureTrackerBase::setImageBounds() {
  const int shrink_row = std::max(0, params_.shrink_row);
  const int shrink_col = std::max(0, params_.shrink_col);
  const int image_rows = img_size_.height;
  const int image_cols = img_size_.width;

  shrunken_row_min_ = shrink_row;
  shrunken_row_max_ = image_rows - shrink_row;

  shrunken_col_min_ = shrink_col;
  shrunken_col_max_ = image_cols - shrink_col;
}

PyramidBuilder::PyramidBuilder(const cv::Size& win_size, int max_level)
    : win_size_(win_size), max_level_(max_level) {}

bool PyramidBuilder::build(const ImageContainer& container,
                           const std::string& img_key,
                           OpticalFlowPyramid& pyr) const {
  const FrameId frame_id = container.frameId();

  if (!container.exists(img_key)) {
    DYNO_THROW_MSG(DynosamException)
        << "Cannot build image pyramid for k=" << frame_id
        << ": requested image " << img_key
        << " does not exist in the container!";
  }

  bool needs_building = false;
  // Ensure correct size (no reallocation if already correct)
  if (static_cast<int>(pyr.levels.size()) != max_level_ + 1) {
    pyr.levels.resize(max_level_ + 1);
    needs_building = true;
  }

  const std::string name = std::to_string(frame_id) + "+" + img_key;

  if (name != pyr.name) {
    needs_building = true;
  }

  if (needs_building) {
    // parse the image as an RGB image regardless of what it is!
    cv::Mat image = ImageType::RGBMono::toMono(container.at(img_key));

    cv::buildOpticalFlowPyramid(image, pyr.levels, win_size_, max_level_, false,
                                cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT,
                                true  // critical for reuse
    );
    pyr.name = name;
  }

  return needs_building;
}

SparseLKTracker::SparseLKTracker(const cv::Size& win_size, int max_level,
                                 int expected_max_features,
                                 const std::string& img1_key,
                                 const std::string& img2_key)
    : img1_key_(img1_key),
      img2_key_(img2_key),
      pyr_builder_(win_size, max_level),
      criteria_(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01) {
  img1_pyr_.reserve(max_level);
  img2_pyr_.reserve(max_level);
  workspace_.reserve(expected_max_features);
  reverse_workspace_.reserve(expected_max_features);
}

const LKWorkspace& SparseLKTracker::track(
    const ImageContainer& image_container_1,
    const ImageContainer& image_container_k,
    const std::vector<cv::Point2f>& img1_pts,
    const std::vector<cv::Point2f>* img2_pts) {
  pyr_builder_.build(image_container_1, img1_key_, img1_pyr_);
  pyr_builder_.build(image_container_k, img2_key_, img2_pyr_);

  // prepare workspace
  // should not allocate memory if workspace is already at correct size
  workspace_.reserve(img1_pts.size());

  // used as flags argument for calcOpticalFlowPyrLK - initially starts as
  // default (0) flag
  int klt_flags = 0;
  if (img2_pts) {
    CHECK_EQ(img2_pts->size(), img1_pts.size());
    // prepare workspace for initial flow
    // allocate all data and then set the
    // initial points using the input
    workspace_.pts = *img2_pts;
    klt_flags = cv::OPTFLOW_USE_INITIAL_FLOW;
  }

  // forwards track
  trackImpl(img1_pts, img1_pyr_, img2_pyr_, klt_flags, workspace_);

  // if we used OPTFLOW_USE_INITIAL_FLOW check that we actually got good flow
  if (klt_flags == cv::OPTFLOW_USE_INITIAL_FLOW) {
    static constexpr int kMinSuccessTracks = 10;
    int succ_num = 0;
    for (size_t i = 0; i < workspace_.status.size(); i++) {
      if (workspace_.status[i]) succ_num++;
    }
    if (succ_num < kMinSuccessTracks) {
      LOG(WARNING) << "Using initial flow for KLT tracking failed: only "
                   << succ_num << " tracked!";

      // run again but with klt_flags=0 (ie. default)
      trackImpl(img1_pts, img1_pyr_, img2_pyr_, 0, workspace_);
    }
  }

  // backwards track
  const std::vector<cv::Point2f>& curr_pts = workspace_.pts;
  // use initial flow for reverse check
  // prepare workspace
  // should not allocate memory if workspace is already at correct size
  reverse_workspace_.reserve(curr_pts.size());
  reverse_workspace_.pts = curr_pts;
  trackImpl(curr_pts, img2_pyr_, img1_pyr_, cv::OPTFLOW_USE_INITIAL_FLOW,
            reverse_workspace_);

  // modified output
  auto& forwards_status = workspace_.status;
  const auto& forward_error = workspace_.error;

  const auto& reverse_status = reverse_workspace_.status;
  const auto& reverse_pts = reverse_workspace_.pts;
  const auto& reverse_error = reverse_workspace_.error;

  CHECK_EQ(img1_pts.size(), curr_pts.size());
  CHECK_EQ(forwards_status.size(), curr_pts.size());
  CHECK_EQ(reverse_status.size(), curr_pts.size());

  static constexpr float kMaxErr = 20.0f;
  // update klt status based on result from flow
  for (size_t i = 0; i < forwards_status.size(); i++) {
    const bool both_status_good = forwards_status.at(i) && reverse_status.at(i);
    const bool within_distance =
        utils::distance(img1_pts.at(i), reverse_pts.at(i)) <= 0.5;
    const bool within_error =
        reverse_error[i] < kMaxErr && forward_error[i] < kMaxErr;

    // update output status
    if (both_status_good && within_distance && within_error) {
      forwards_status.at(i) = 1;
    } else {
      forwards_status.at(i) = 0;
    }
  }

  img1_pyr_ = std::move(img2_pyr_);

  return workspace_;
}

void SparseLKTracker::trackImpl(const std::vector<cv::Point2f>& img1_pts,
                                const OpticalFlowPyramid& img1_pyr,
                                const OpticalFlowPyramid& img2_pyr,
                                int klt_flags, LKWorkspace& workspace) const {
  // the workspace must be correctly allocated with memory prior
  cv::calcOpticalFlowPyrLK(img1_pyr.levels, img2_pyr.levels, img1_pts,
                           workspace.pts, workspace.status, workspace.error,
                           pyr_builder_.winSize(), pyr_builder_.maxLevel(),
                           criteria_, klt_flags,
                           1e-4  // minEigThreshold
  );
}

bool ImageTracksParams::showFrameInfo() const {
  return isDebug() && show_frame_info;
}

bool ImageTracksParams::showIntermediateTracking() const {
  return isDebug() && show_intermediate_tracking;
}

bool ImageTracksParams::drawObjectBoundingBox() const {
  return isDebug() && draw_object_bounding_box;
}
bool ImageTracksParams::drawObjectMask() const {
  return isDebug() && draw_object_mask;
}

int ImageTracksParams::bboxThickness() const {
  return isDebug() ? bbox_thickness_debug : bbox_thickness;
}
int ImageTracksParams::featureThickness() const {
  return isDebug() ? feature_thickness_debug : feature_thickness;
}

// doesnt make any sense for this function to be here?
// Debug could be part of a global config singleton?
cv::Mat FeatureTrackerBase::computeFeatureTracks(
    const Frame& frame_km1, const Frame& frame_k,
    const ImageTracksParams& config) const {
  const ImageWrapper<ImageType::RGBMono>& img_wrapper =
      frame_k.image_container_.rgb();
  cv::Mat img_rgb = img_wrapper.toRGB().clone();
  const cv::Mat& object_mask = frame_k.image_container_.objectMotionMask();

  const bool& debug = config.isDebug();
  const bool& show_intermediate_tracking = config.showIntermediateTracking();

  const int static_point_thickness = config.featureThickness();

  static const cv::Scalar red(Color::red().bgra());
  static const cv::Scalar green(Color::green().bgra());
  static const cv::Scalar blue(Color::blue().bgra());

  int num_static_tracks = 0;
  // Add all keypoints in cur_frame with the tracks.
  for (const Feature::Ptr& feature : frame_k.static_features_) {
    const Keypoint& px_cur = feature->keypoint();
    const auto pc_cur = utils::gtsamPointToCv(px_cur);
    if (!feature->usable() &&
        show_intermediate_tracking) {  // Untracked landmarks are red.
      cv::circle(img_rgb, pc_cur, static_point_thickness, red, 2, cv::LINE_AA);
    } else {
      const Feature::Ptr& prev_feature =
          frame_km1.static_features_.getByTrackletId(feature->trackletId());
      if (prev_feature) {
        // If feature was in previous frame, display tracked feature with
        // green circle/line:
        cv::circle(img_rgb, pc_cur, static_point_thickness, green, 1);

        // draw the optical flow arrow
        const auto pc_prev = utils::gtsamPointToCv(prev_feature->keypoint());
        cv::arrowedLine(img_rgb, pc_prev, pc_cur, green, 1);

        num_static_tracks++;

      } else if (debug &&
                 show_intermediate_tracking) {  // New feature tracks are blue.
        cv::circle(img_rgb, pc_cur, 6, blue, 1);
      }
    }
  }

  for (const Feature::Ptr& feature : frame_k.dynamic_features_) {
    const Keypoint& px_cur = feature->keypoint();
    if (!feature->usable()) {  // Untracked landmarks are red.
      // cv::circle(img_rgb,  utils::gtsamPointToCv(px_cur), 1, red, 2);
    } else {
      const Feature::Ptr& prev_feature =
          frame_km1.dynamic_features_.getByTrackletId(feature->trackletId());
      if (prev_feature) {
        const Keypoint& px_prev = prev_feature->keypoint();
        const cv::Scalar colour = Color::uniqueId(feature->objectId()).bgra();
        cv::arrowedLine(img_rgb, utils::gtsamPointToCv(px_prev),
                        utils::gtsamPointToCv(px_cur), colour, 1, 8, 0, 0.1);
        cv::circle(img_rgb, utils::gtsamPointToCv(px_cur), 2, colour, -1);
      }
    }
  }

  const int bbox_thickness = config.bboxThickness();

  std::vector<ObjectId> objects_to_print;
  for (const auto& object_observation_pair : frame_k.object_observations_) {
    const ObjectId object_id = object_observation_pair.first;
    const cv::Rect& bb = object_observation_pair.second.bounding_box;

    // TODO: if its marked as moving!!
    if (bb.empty()) {
      continue;
    }

    objects_to_print.push_back(object_id);

    if (config.drawObjectBoundingBox()) {
      const cv::Scalar colour = Color::uniqueId(object_id).bgra();
      const std::string label = "object " + std::to_string(object_id);
      utils::drawLabeledBoundingBox(img_rgb, label, colour, bb, bbox_thickness);
    }
  }

  if (config.drawObjectMask()) {
    constexpr static float kAlpha = 0.7;
    utils::labelMaskToRGB(object_mask, img_rgb, img_rgb, kAlpha);
  }

  // draw text info
  std::stringstream ss;
  ss << "Frame ID: " << frame_k.getFrameId() << " | ";
  ss << "VO tracks: " << num_static_tracks << " | ";
  ss << "Objects: ";

  if (objects_to_print.empty()) {
    ss << "None";
  } else {
    ss << "[";
    for (size_t i = 0; i < objects_to_print.size(); ++i) {
      ss << objects_to_print[i];
      if (i != objects_to_print.size() - 1) {
        ss << ", ";  // Add comma between elements
      }
    }
    ss << "]";
  }

  constexpr static double kFontScale = 0.6;
  constexpr static int kFontFace = cv::FONT_HERSHEY_SIMPLEX;
  constexpr static int kThickness = 1;

  if (config.showFrameInfo()) {
    // taken from ORB-SLAM2 ;)
    int base_line;
    cv::Size text_size = cv::getTextSize(ss.str(), kFontFace, kFontScale,
                                         kThickness, &base_line);
    cv::Mat image_text = cv::Mat(img_rgb.rows + text_size.height + 10,
                                 img_rgb.cols, img_rgb.type());
    img_rgb.copyTo(
        image_text.rowRange(0, img_rgb.rows).colRange(0, img_rgb.cols));
    image_text.rowRange(img_rgb.rows, image_text.rows) =
        cv::Mat::zeros(text_size.height + 10, img_rgb.cols, img_rgb.type());
    cv::putText(image_text, ss.str(), cv::Point(5, image_text.rows - 5),
                kFontFace, kFontScale, cv::Scalar(255, 255, 255), kThickness);
    return image_text;
  } else {
    return img_rgb;
  }
}

bool FeatureTrackerBase::drawStereoMatches(cv::Mat& output_image,
                                           const Frame& current_frame) const {
  // for now only static tracks
  if (!current_frame.image_container_.hasRightRgb()) {
    return false;
  }

  const ImageWrapper<ImageType::RGBMono>& left_img_wrapper =
      current_frame.image_container_.rgb();
  cv::Mat img_rgb_left = left_img_wrapper.toRGB().clone();

  const ImageWrapper<ImageType::RGBMono>& right_img_wrapper =
      current_frame.image_container_.rightRgb();
  cv::Mat img_rgb_right = right_img_wrapper.toRGB().clone();

  // Stack side by side
  cv::Mat canvas;
  cv::hconcat(img_rgb_left, img_rgb_right, canvas);

  int w1 = img_rgb_left.cols;

  auto itr = current_frame.static_features_.usableIterator();
  for (const auto& feature : itr) {
    if (!feature->hasRightKeypoint()) {
      continue;
    }

    const auto kp_left = utils::gtsamPointToCv(feature->keypoint());
    const auto kp_right = utils::gtsamPointToCv(feature->rightKeypoint()) +
                          cv::Point2f((float)w1, 0.0f);

    cv::circle(canvas, kp_left, 4, cv::Scalar(0, 0, 255), cv::FILLED,
               cv::LINE_AA);
    cv::circle(canvas, kp_right, 4, cv::Scalar(0, 0, 255), cv::FILLED,
               cv::LINE_AA);
    cv::line(canvas, kp_left, kp_right, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
  }

  output_image = canvas;
  return true;
}

bool FeatureTrackerBase::isWithinShrunkenImage(const Keypoint& kp) const {
  // involves double casting (keypoint to a cv::Pointf)
  // which then casts to a cv::Pointi.
  // We do this to take advantage of the cv::Point casting implementation
  // which handles rounding of floating point x/y values to ensure
  // that pixel locations on the edge of images (ie. 99.85 for an image edge of
  // 100) does not exceed the max image region this is important becase
  // accessing an image (ie cv::mat<>::at) uses discrete (ie integer) pixel
  // location and therefore accessing from a floating point type may be invalid
  return isWithinShrunkenImage(utils::gtsamPointToCv<float>(kp));
}

bool FeatureTrackerBase::isWithinShrunkenImage(const cv::Point2f& kp) const {
  return isWithinShrunkenImage(static_cast<cv::Point2i>(kp));
}

bool FeatureTrackerBase::isWithinShrunkenImage(const cv::Point2i& kp) const {
  const int r = kp.y;
  const int c = kp.x;
  return (r >= shrunken_row_min_ && r < shrunken_row_max_ &&
          c >= shrunken_col_min_ && c < shrunken_col_max_);
}

void declare_config(ImageTracksParams& config) {
  using namespace config;

  name("ImageTracksParams");

  field(config.feature_thickness_debug, "feature_thickness_debug");
  field(config.feature_thickness, "feature_thickness");

  field(config.bbox_thickness_debug, "bbox_thickness_debug");
  field(config.bbox_thickness, "bbox_thickness");

  field(config.show_frame_info, "show_frame_info");
  field(config.show_intermediate_tracking, "show_intermediate_tracking");

  field(config.draw_object_bounding_box, "draw_object_bounding_box");
  field(config.draw_object_mask, "draw_object_mask");

  field(config.is_debug, "is_debug");
}

}  // namespace dyno
