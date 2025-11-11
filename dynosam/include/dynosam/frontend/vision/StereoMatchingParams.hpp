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

#include <opencv4/opencv2/calib3d.hpp>

namespace dyno {

// TODO: make parsable
struct StereoMatchingParams {
  double tolerance_template_matching_ = 0.15;
  double nominal_baseline_ = 0.11;
  int templ_cols_ = 101;       // must be odd
  int templ_rows_ = 11;        // must be odd
  int stripe_extra_rows_ = 0;  // must be even
  // stereo points triangulated below this distance are discarded.
  double min_point_dist_ = 0.1;
  // stereo points triangulated beyond this distance are discarded.
  double max_point_dist_ = 15.0;
  // check best match left->right and right->left
  bool bidirectional_matching_ = false;
  // refine stereo matches with subpixel accuracy
  bool subpixel_refinement_ = false;
  // do equalize image before processing options to use RGB-D vs. stereo.
  bool equalize_image_ = false;
};

// TODO: make parsable
struct DenseStereoParams {
  bool use_sgbm_ = true;
  bool post_filter_disparity_ = false;
  bool median_blur_disparity_ = false;
  int pre_filter_cap_ = 31;
  int sad_window_size_ = 11;
  int min_disparity_ = 1;
  int num_disparities_ = 64;
  int uniqueness_ratio_ = 0;
  int speckle_range_ = 3;
  int speckle_window_size_ = 500;
  // bm parameters
  int texture_threshold_ = 0;
  int pre_filter_type_ = cv::StereoBM::PREFILTER_XSOBEL;
  int pre_filter_size_ = 9;
  // sgbm parameters
  int p1_ = 120;
  int p2_ = 240;
  int disp_12_max_diff_ = -1;
  bool use_mode_HH_ = true;
};

};  // namespace dyno
