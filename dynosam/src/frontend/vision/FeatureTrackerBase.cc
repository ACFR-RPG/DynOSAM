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

bool FeatureTrackerBase::predictKeypointsGivenRotation(
    std::vector<cv::Point2f>& predicted_pts_k,
    const std::vector<cv::Point2f>& pts_km1, const gtsam::Rot3& R_km1_k) const {
  // Handle case when rotation is small: just copy prev_kps
  // Removed bcs even small rotations lead to huge optical flow at the borders
  // of the image.
  // Keep because then you save a lot of computation.
  static constexpr double kSmallRotationTol = 1e-4;
  if (std::abs(1.0 - std::abs(R_km1_k.toQuaternion().w())) <
      kSmallRotationTol) {
    predicted_pts_k = pts_km1;
    return true;
  }

  const gtsam::Matrix K = camera_->getParams().getCameraMatrixEigen();
  const gtsam::Matrix K_inv = K.inverse();

  const cv::Matx33f K_cv = utils::gtsamMatrix3ToCvMat(K);
  const cv::Matx33f K_inv_cv = utils::gtsamMatrix3ToCvMat(K_inv);
  // R is a relative rotation which takes a vector from the last frame to
  // the current frame.
  const cv::Matx33f R = utils::gtsamMatrix3ToCvMat(R_km1_k.matrix());
  // Get bearing vector for kpt, rotate knowing frame to frame rotation,
  // get keypoints again
  const cv::Matx33f H = K_cv * R * K_inv_cv;

  const size_t& n_kps = pts_km1.size();
  predicted_pts_k.reserve(n_kps);
  for (size_t i = 0u; i < n_kps; ++i) {
    // Create homogeneous keypoints.
    const auto& prev_kpt = pts_km1[i];
    cv::Vec3f p1(prev_kpt.x, prev_kpt.y, 1.0f);

    cv::Vec3f p2 = H * p1;

    // Project predicted bearing vectors to 2D again and re-homogenize.
    cv::Point2f new_kpt;
    if (p2[2] > 0.0f) {
      new_kpt = cv::Point2f(p2[0] / p2[2], p2[1] / p2[2]);
    } else {
      LOG(WARNING) << "Landmark behind the camera:\n"
                   << "- p1: " << p1 << '\n'
                   << "- p2: " << p2;
      new_kpt = prev_kpt;
    }
    // Check that keypoints remain inside the image boundaries!
    if (isWithinShrunkenImage(new_kpt)) {
      predicted_pts_k.push_back(new_kpt);
    } else {
      // Otw copy-paste previous keypoint.
      predicted_pts_k.push_back(prev_kpt);
    }
  }

  return true;
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
cv::Mat FeatureTrackerBase::computeImageTracks(
    const Frame& previous_frame, const Frame& current_frame,
    const ImageTracksParams& config) const {
  const ImageWrapper<ImageType::RGBMono>& img_wrapper =
      current_frame.image_container_.rgb();
  cv::Mat img_rgb = img_wrapper.toRGB().clone();
  const cv::Mat& object_mask =
      current_frame.image_container_.objectMotionMask();

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
                        utils::gtsamPointToCv(px_cur), colour, 1, 8, 0, 0.1);
        cv::circle(img_rgb, utils::gtsamPointToCv(px_cur), 2, colour, -1);
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

  auto itr = current_frame.static_features_.beginUsable();
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

bool FeatureTrackerBase::isWithinShrunkenImage(const cv::Point2f& kp) const {
  return isWithinShrunkenImage(utils::cvPointToGtsam(kp));
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
