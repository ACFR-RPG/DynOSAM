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

#include "dynosam/frontend/vision/TrackerParams.hpp"

#include <config_utilities/config_utilities.h>
#include <config_utilities/types/eigen_matrix.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "dynosam_common/utils/OpenCVUtils.hpp"

DEFINE_int32(shrink_row, 0, "Number of rows to shrink the tracking image by");
DEFINE_int32(shrink_col, 0, "Number of cols to shrink the tracking image by");

DEFINE_int32(max_dynamic_features_per_frame, 50,
             "Maximum dynamic features to detect per object per frame");
DEFINE_bool(use_propogate_mask, true,
            "If true, the semantic mask will be propogated with optical flow");

namespace dyno {

void declare_config(TrackerParams::AnmsParams& config) {
  using namespace config;

  name("AnmsParams");
  enum_field(
      config.non_max_suppression_type, "non_max_suppression_type",
      {"TopN", "BrownANMS", "SDC", "KdTree", "RangeTree", "Ssc", "Binning"});

  field(config.nr_horizontal_bins, "nr_horizontal_bins");
  field(config.nr_vertical_bins, "nr_vertical_bins");
  field(config.binning_mask, "binning_mask");
}

void declare_config(TrackerParams::SubPixelCornerRefinementParams& config) {
  using namespace config;
  name("SubPixelCornerRefinementParams");
  field(config.window_size, "window_size");
  field(config.zero_zone, "zero_zone");
}

void declare_config(TrackerParams::GFFTParams& config) {
  using namespace config;
  name("GFFTParams");
  field(config.quality_level, "quality_level");
  field(config.block_size, "block_size");
  field(config.use_harris_corner_detector, "use_harris_corner_detector");
  field(config.k, "k");
}

void declare_config(TrackerParams::OrbParams& config) {
  using namespace config;
  name("OrbParams");
  field(config.scale_factor, "scale_factor");
  field(config.n_levels, "n_levels");
  field(config.init_threshold_fast, "init_threshold_fast");
  field(config.min_threshold_fast, "min_threshold_fast");
}

void declare_config(TrackerParams& config) {
  using namespace config;
  name("TrackerParams");

  enum_field(config.feature_detector_type, "feature_detector_type",
             std::vector<std::string>({"GFTT", "ORB_SLAM_ORB", "GFFT_CUDA"}));

  field(config.use_anms, "use_anms");
  field(config.use_subpixel_corner_refinement,
        "use_subpixel_corner_refinement");
  field(config.use_clahe_filter, "use_clahe_filter");
  field(config.max_nr_keypoints_before_anms, "max_nr_keypoints_before_anms");
  field(config.min_distance_btw_tracked_and_detected_static_features,
        "min_distance_btw_tracked_and_detected_static_features");
  field(config.min_distance_btw_tracked_and_detected_dynamic_features,
        "min_distance_btw_tracked_and_detected_dynamic_features");
  field(config.max_features_per_frame, "max_features_per_frame");
  field(config.min_features_per_frame, "min_features_per_frame");
  field(config.max_feature_track_age, "max_feature_track_age");

  field(config.shrink_row, "shrink_row");
  field(config.shrink_col, "shrink_col");

  // update with FLAGS
  config.shrink_row = FLAGS_shrink_row;
  config.shrink_col = FLAGS_shrink_col;

  field(config.anms_params, "anms_params");
  field(config.subpixel_corner_refinement_params,
        "subpixel_corner_refinement_params");
  field(config.gfft_params, "gfft_params");
  field(config.orb_params, "orb_params");

  field(config.max_dynamic_features_per_frame,
        "max_dynamic_features_per_frame");
  field(config.max_dynamic_feature_age, "max_dynamic_feature_age");

  field(config.dynamic_feature_age_buffer, "dynamic_feature_age_buffer");
  field(config.min_dynamic_tracks, "min_dynamic_tracks");
  field(config.min_dynamic_mask_iou, "min_dynamic_mask_iou");

  field(config.use_propogate_mask, "use_propogate_mask");

  field(config.prefer_provided_optical_flow, "prefer_provided_optical_flow");
  field(config.prefer_provided_object_detection,
        "prefer_provided_object_detection");

  // update with FLAGS
  // config.semantic_mask_step_size = FLAGS_semantic_mask_step_size;
  config.use_propogate_mask = FLAGS_use_propogate_mask;
}

}  // namespace dyno
