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

#include <eigen3/Eigen/Dense>
#include <opencv4/opencv2/opencv.hpp>

#include "dynosam/frontend/anms/NonMaximumSuppression.h"

namespace dyno {

/**
 * @brief Params for the feature tracker but also the feature detector
 *
 */
struct TrackerParams {
  // GFTT is goodFeaturesToTrack detector.
  // ORB_SLAM_ORB is the ORB implementation from OrbSLAM
  enum class FeatureDetectorType : unsigned int { GFTT = 0, ORB_SLAM_ORB = 1 };

  struct AnmsParams {
    AnmsAlgorithmType non_max_suppression_type = AnmsAlgorithmType::RangeTree;
    //! Number of horizontal bins for feature binning
    int nr_horizontal_bins = 5;
    //! Number of vertical bins for feature binning
    int nr_vertical_bins = 5;
    //! Binary mask by the user to control which bins to use
    Eigen::MatrixXd binning_mask;
  };
  struct SubPixelCornerRefinementParams {
    cv::Size window_size = cv::Size(5, 5);
    cv::Size zero_zone = cv::Size(-1, -1);
    cv::TermCriteria criteria = cv::TermCriteria(
        cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 40, 0.001);
  };

  //! Good features to track params
  //! See https://docs.opencv.org/4.x/df/d21/classcv_1_1GFTTDetector.html
  struct GFFTParams {
    double quality_level = 0.001;
    int block_size = 3;
    //! From my experience on the datasets we've tested on this should almost
    //! always be false...
    bool use_harris_corner_detector = false;
    double k = 0.04;
  };

  //! Orb features to track params
  //! Used for both cv::ORB and dyno::ORBextractor
  //! See https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html for details
  // TODO: right now just copy-pasted from FrontendParams and only includes
  // dyno::ORBextractor
  struct OrbParams {
    float scale_factor = 1.2;
    int n_levels = 8;
    int init_threshold_fast = 20;
    int min_threshold_fast = 7;
  };

  FeatureDetectorType feature_detector_type = FeatureDetectorType::GFTT;

  //! To use adaptvie non-maximum supression in the feature detector
  bool use_anms{true};
  //! To use subpixel refinement on the static pixels
  bool use_subpixel_corner_refinement{true};
  //! To use CLAHE filter on the imput image before processing
  bool use_clahe_filter{true};

  AnmsParams anms_params = AnmsParams();
  SubPixelCornerRefinementParams subpixel_corner_refinement_params =
      SubPixelCornerRefinementParams();

  //! Number features to detect - may be used across many detectors
  int max_nr_keypoints_before_anms = 2000;
  //! Used to grid the newly detected features and also used for the detection
  //! mask
  int min_distance_btw_tracked_and_detected_static_features = 8;
  int min_distance_btw_tracked_and_detected_dynamic_features = 2;
  //! Maximum number of features to be detected after ANMS
  int max_features_per_frame = 400;
  //! Threshold for the number of features to keep tracking - if num tracks drop
  //! below this number, new features are detected
  int min_features_per_frame = 200;
  //! We relabel the track as a new track for any features longer that this
  size_t max_feature_track_age = 25;
  //! Number of rows to shrink the input image by
  int shrink_row = 0;
  //! Number of cols to shrink the input image by
  int shrink_col = 0;

  bool prefer_provided_optical_flow = true;

  //! Good features to track params
  GFFTParams gfft_params = GFFTParams();
  OrbParams orb_params = OrbParams();

  // Dynamic tracking specific
  size_t max_dynamic_features_per_frame = 50u;
  size_t max_dynamic_feature_age = 25u;

  // Dynamic 'keyframing' criteria
  //! How early we want to retrack points based on their expiry age
  int dynamic_feature_age_buffer = 3;
  //! Smallest number of good tracks on an object before re-sampling
  int min_dynamic_tracks = 20;
  //! Minimum IOU percetage between currently tracked area and actual mask
  //! before re-sampling
  double min_dynamic_mask_iou = 0.3;

  bool use_propogate_mask = false;
};

void declare_config(TrackerParams::AnmsParams& config);
void declare_config(TrackerParams::SubPixelCornerRefinementParams& config);
void declare_config(TrackerParams::GFFTParams& config);
void declare_config(TrackerParams::OrbParams& config);
void declare_config(TrackerParams& config);

}  // namespace dyno
