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
#include "dynosam/utils/GtsamUtils.hpp"
#include "dynosam/utils/OpenCVUtils.hpp"
#include "dynosam/visualizer/ColourMap.hpp"

namespace dyno {

decltype(TrackletIdManager::instance_) TrackletIdManager::instance_;

FeatureTrackerBase::FeatureTrackerBase(const TrackerParams& params,
                                       Camera::Ptr camera,
                                       ImageDisplayQueue* display_queue)
    : params_(params),
      img_size_(camera->getParams().imageSize()),
      camera_(camera),
      display_queue_(display_queue) {}

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
cv::Mat FeatureTrackerBase::computeImageTracks(
    const Frame& previous_frame, const Frame& current_frame,
    const ImageTracksParams& config) const {
  cv::Mat img_rgb;

  const cv::Mat& rgb = ImageType::RGBMono::toRGB(
      current_frame.image_container_.get<ImageType::RGBMono>());;
  const cv::Mat& object_mask =
      current_frame.image_container_.get<ImageType::MotionMask>();
  rgb.copyTo(img_rgb);

  const bool& debug = config.isDebug();
  const bool& show_intermediate_tracking = config.showIntermediateTracking();

  const int static_point_thickness = config.featureThickness();

  static const cv::Scalar red(Color::red().bgra());
  static const cv::Scalar green(Color::green().bgra());
  static const cv::Scalar blue(Color::blue().bgra());

  int num_static_tracks = 0;
  // Add all keypoints in cur_frame with the tracks.
  for (const Feature::Ptr& feature : current_frame.static_features_) {
    const Keypoint& px_cur = feature->keypoint();
    const auto pc_cur = utils::gtsamPointToCv(px_cur);
    if (!feature->usable() &&
        show_intermediate_tracking) {  // Untracked landmarks are red.
      cv::circle(img_rgb, pc_cur, static_point_thickness, red, 2);
    } else {
      const Feature::Ptr& prev_feature =
          previous_frame.static_features_.getByTrackletId(
              feature->trackletId());
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

  for (const Feature::Ptr& feature : current_frame.dynamic_features_) {
    const Keypoint& px_cur = feature->keypoint();
    if (!feature->usable()) {  // Untracked landmarks are red.
      // cv::circle(img_rgb,  utils::gtsamPointToCv(px_cur), 1, red, 2);
    } else {
      const Feature::Ptr& prev_feature =
          previous_frame.dynamic_features_.getByTrackletId(
              feature->trackletId());
      if (prev_feature) {
        // If feature was in previous frame, display tracked feature with
        // green circle/line:
        // cv::circle(img_rgb,  utils::gtsamPointToCv(px_cur), 6, green, 1);
        const Keypoint& px_prev = prev_feature->keypoint();
        const cv::Scalar colour = Color::uniqueId(feature->objectId()).bgra();
        cv::arrowedLine(img_rgb, utils::gtsamPointToCv(px_prev),
                        utils::gtsamPointToCv(px_cur), colour, 1);
      } else {  // New feature tracks are blue.
        // cv::circle(img_rgb, utils::gtsamPointToCv(px_cur), 1, blue, 1);
      }
    }
  }

  const int bbox_thickness = config.bboxThickness();

  std::vector<ObjectId> objects_to_print;
  for (const auto& object_observation_pair :
       current_frame.object_observations_) {
    const ObjectId object_id = object_observation_pair.first;
    const cv::Rect& bb = object_observation_pair.second.bounding_box_;

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
  ss << "Frame ID: " << current_frame.getFrameId() << " | ";
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

bool FeatureTrackerBase::isWithinShrunkenImage(const Keypoint& kp) const {
  const auto shrunken_row = params_.shrink_row;
  const auto shrunken_col = params_.shrink_col;

  const int predicted_col = functional_keypoint::u(kp);
  const int predicted_row = functional_keypoint::v(kp);

  const auto image_rows = img_size_.height;
  const auto image_cols = img_size_.width;
  return (predicted_row > shrunken_row &&
          predicted_row < (image_rows - shrunken_row) &&
          predicted_col > shrunken_col &&
          predicted_col < (image_cols - shrunken_col));
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
