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
#include "dynosam/visualizer/VisualizerPipelines.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_sensors/Camera.hpp"

namespace dyno {

/**
 * @brief Singleton class to manage a global tracklet id for all trackers to
 * ensure they are unique.
 *
 * Accessors and modifiers are thread-safe.
 *
 */
// class TrackletIdManager {
//  public:
//   DYNO_POINTER_TYPEDEFS(TrackletIdManager)

//   static TrackletIdManager& instance();

//   TrackletId getTrackletIdCount() const noexcept;
//   void incrementTrackletIdCount() noexcept;
//   TrackletId getAndIncrementTrackletId() noexcept;

//  private:
//   TrackletIdManager() = default;
//   //! GLobal thread safe tracked id count
//   std::atomic<TrackletId> tracklet_count_{0};
// };

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

struct OpticalFlowPyramid {
  // Constructed as frameId + image key
  std::string name{};
  std::vector<cv::Mat> levels;

  void reserve(int maxLevel) { levels.resize(maxLevel + 1); }

  void clear() {
    for (auto& lvl : levels) {
      lvl.release();  // keeps capacity, frees data if needed
    }
  }
};

struct LKWorkspace {
  std::vector<uchar> status;
  std::vector<float> error;
  // result from tracking img1 -> img2
  std::vector<cv::Point2f> pts;

  void reserve(size_t N) {
    status.reserve(N);
    error.reserve(N);
    pts.reserve(N);
  }

  void resize(size_t N) {
    status.resize(N);
    error.resize(N);
    pts.resize(N);
  }
};

class PyramidBuilder {
 public:
  PyramidBuilder(const cv::Size& win_size, int max_level);
  bool build(const ImageContainer& container, const std::string& img_key,
             OpticalFlowPyramid& pyr) const;

  const cv::Size& winSize() const { return win_size_; }
  int maxLevel() const { return max_level_; }

 private:
  cv::Size win_size_;
  int max_level_;
};

class SparseLKTracker {
 public:
  DYNO_POINTER_TYPEDEFS(SparseLKTracker)

  SparseLKTracker(const cv::Size& win_size, int max_level,
                  int expected_max_features,
                  const std::string& img1_key = "rgb",
                  const std::string& img2_key = "rgb");

  // current_points if provided is an initial guess of the flow
  // we track img1 -> img2 so in the VO case img1 == image @ k-1
  // and img2 = img @ k, while in the stereo case im1 = left imag @ k
  // and img2 = right img @ k
  const LKWorkspace& track(const ImageContainer& image_container_1,
                           const ImageContainer& image_container_2,
                           const std::vector<cv::Point2f>& img1_pts,
                           const std::vector<cv::Point2f>* img2_pts = nullptr);

 private:
  void trackImpl(const std::vector<cv::Point2f>& img1_pts,
                 const OpticalFlowPyramid& img1_pyr,
                 const OpticalFlowPyramid& img2_pyr, int klt_flags,
                 LKWorkspace& workspace) const;

 private:
  std::string img1_key_;
  std::string img2_key_;

  PyramidBuilder pyr_builder_;
  cv::TermCriteria criteria_;

  OpticalFlowPyramid img1_pyr_;
  OpticalFlowPyramid img2_pyr_;

  LKWorkspace workspace_;
  LKWorkspace reverse_workspace_;
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

  /** Computes debug image for frame-to-frame image tracks  */
  cv::Mat computeFeatureTracks(const Frame& frame_km1, const Frame& frame_k,
                               const ImageTracksParams& config = false) const;

  bool drawStereoMatches(cv::Mat& output_image,
                         const Frame& current_frame) const;

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

  /* All isWithinShrunkenImage variants eventually use this function. Where it
    is important to correctly construct the cv::Point2i from a floating point
    cv::Point type. The internal cv casting does some rounding to ensure the
    point is actually within the image bounds.
  */
  bool isWithinShrunkenImage(const cv::Point2i& kp) const;

 protected:
  const TrackerParams params_;
  const cv::Size img_size_;  //! Expected image size from the camera

  Camera::Ptr camera_;
  ImageDisplayQueue* display_queue_;

 private:
  /* From the set params and img_size, set the min/max rows and cols for use in
   * isWithinShrunkenImage*/
  void setImageBounds();
  // min/max of image size taking int account the
  // shrunkin row/col params
  // cached for use in the isWithinShrunkenImage
  // and set on init
  int shrunken_row_min_;
  int shrunken_row_max_;
  int shrunken_col_min_;
  int shrunken_col_max_;
};

}  // namespace dyno
