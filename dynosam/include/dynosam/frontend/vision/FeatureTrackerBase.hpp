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

#pragma once

#include <config_utilities/config_utilities.h>

#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam/frontend/vision/TrackerParams.hpp"
#include "dynosam/visualizer/Visualizer-Definitions.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_cv/Camera.hpp"

namespace dyno {

/**
 * @brief Singleton class to manage a global tracklet id for all trackers to
 * ensure they are unique.
 *
 * Accessors and modifiers are thread-safe.
 *
 */
class TrackletIdManager {
 public:
  DYNO_POINTER_TYPEDEFS(TrackletIdManager)

  static TrackletIdManager& instance() {
    if (!instance_) {
      instance_.reset(new TrackletIdManager());
    }
    return *instance_;
  }

  inline TrackletId getTrackletIdCount() const {
    const std::lock_guard<std::mutex> l(mutex_);
    return tracklet_count_;
  }
  inline void incrementTrackletIdCount() {
    const std::lock_guard<std::mutex> l(mutex_);
    tracklet_count_++;
  }

  inline TrackletId getAndIncrementTrackletId() {
    const std::lock_guard<std::mutex> l(mutex_);
    auto tracklet = tracklet_count_;
    tracklet_count_++;
    return tracklet;
  }

 private:
  TrackletIdManager() = default;
  TrackletId tracklet_count_{0};  //! Global TrackletId

  mutable std::mutex mutex_;

  static std::unique_ptr<TrackletIdManager> instance_;
};

/**
 * @brief Parameter struct to control the visualisation for
 * FeatureTrackerBase::computeImageTracks
 *
 */
class ImageTracksParams {
 public:
  constexpr static int kFeatureThicknessDebug = 5;
  constexpr static int kFeatureThickness = 4;

  constexpr static int kBBoxThicknessDebug = 4;
  constexpr static int kBBoxThickness = 2;

  ImageTracksParams(bool debug) : is_debug(debug) {}
  ImageTracksParams() {}

  friend void declare_config(ImageTracksParams& config);

  inline bool isDebug() const { return is_debug; }
  bool showFrameInfo() const;
  bool showIntermediateTracking() const;
  bool drawObjectBoundingBox() const;
  bool drawObjectMask() const;
  int bboxThickness() const;
  int featureThickness() const;

 private:
  //! High-level control over viz. If is_debug is set to false, no debug level
  //! viz will be used, otherwise, the fine-grained control
  // flags will be used to determine what to show.
  //! No debug (ie. is_debug == false) means only feature inlier feature
  //! tracks and object bounding boxes will be shown
  bool is_debug{false};

  int feature_thickness_debug{kFeatureThicknessDebug};
  int feature_thickness{kFeatureThickness};

  int bbox_thickness_debug{kBBoxThicknessDebug};
  int bbox_thickness{kBBoxThickness};

 public:
  //! Fine-grained control
  //! To show current frame info as text
  bool show_frame_info{true};
  //! To show outliers and new feature tracks (red and blue)
  bool show_intermediate_tracking{false};
  //! Draw bbox over each object and the object id label
  bool draw_object_bounding_box{true};
  //! Draw the detection mask of the whole object
  bool draw_object_mask{false};
};

class FeatureTrackerBase {
 public:
  FeatureTrackerBase(const TrackerParams& params, Camera::Ptr camera,
                     ImageDisplayQueue* display_queue);

  cv::Mat computeImageTracks(const Frame& previous_frame,
                             const Frame& current_frame,
                             const ImageTracksParams& config = false) const;

  bool drawStereoMatches(cv::Mat& output_image,
                         const Frame& current_frame) const;

  bool predictKeypointsGivenRotation(std::vector<cv::Point2f>& predicted_pts_k,
                                     const std::vector<cv::Point2f>& pts_km1,
                                     const gtsam::Rot3& R_km1_k) const;

  // bool predictSparseFlow(std::vector<cv::Point2f>& predicted_pts_k, const
  // std::vector<cv::Point2f>& pts_km1, const gtsam::Rot3& R_km1_k )

 protected:
  /**
   * @brief Checks if a keypoint is within an image, taking into account the
   * shrink row/col values in the params. If these values are zero, it just
   * checks that the keypoint is within the image size, as given by the camera
   * parameters.
   *
   * @param kp const Keypoint&
   * @return true
   * @return false
   */
  bool isWithinShrunkenImage(const Keypoint& kp) const;

  bool isWithinShrunkenImage(const cv::Point2f& kp) const;

 protected:
  const TrackerParams params_;
  const cv::Size img_size_;  //! Expected image size from the camera

  Camera::Ptr camera_;
  ImageDisplayQueue* display_queue_;
};

}  // namespace dyno
