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

#include <cstddef>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "dynosam_common/byte_tracker/Detection.hpp"
#include "dynosam_common/byte_tracker/Track.hpp"

namespace dyno {
namespace byte_track {

struct ByteTrackerParams {
  int starting_frame{0};
  int frame_rate{30};
  int track_buffer{30};
  float track_thresh{0.5};
  float high_thresh{0.6};
  float match_thresh{0.8};
};

class ByteTracker {
 public:
  ByteTracker(const ByteTrackerParams &params = ByteTrackerParams());

  std::vector<Track::Ptr> update(const std::vector<DetectionBase::Ptr> &objects,
                                 size_t frame_id);

  void clear();

 private:
  std::tuple<std::vector<Track::Ptr>, std::vector<Track::Ptr>,
             std::vector<DetectionBase::Ptr>>
  iou_association(const std::vector<Track::Ptr> &track_pool,
                  const std::vector<DetectionBase::Ptr> &detections);

  std::vector<Track::Ptr> low_score_association(
      std::vector<Track::Ptr> &matched_tracks,
      const std::vector<DetectionBase::Ptr> &low_score_detections,
      const std::vector<Track::Ptr> &unmatched_tracked_tracks);

  std::vector<Track::Ptr> init_new_tracks(
      std::vector<Track::Ptr> &matched_tracks,
      const std::vector<Track::Ptr> &inactive_tracks,
      const std::vector<DetectionBase::Ptr> &unmatched_detections);

  std::vector<Track::Ptr> joint_tracks(
      const std::vector<Track::Ptr> &a_tlist,
      const std::vector<Track::Ptr> &b_tlist) const;

  std::vector<Track::Ptr> sub_tracks(
      const std::vector<Track::Ptr> &a_tlist,
      const std::vector<Track::Ptr> &b_tlist) const;

  std::tuple<std::vector<Track::Ptr>, std::vector<Track::Ptr>>
  remove_duplicate_tracks(const std::vector<Track::Ptr> &a_tracks,
                          const std::vector<Track::Ptr> &b_tracks) const;

  std::tuple<std::vector<std::pair<Track::Ptr, DetectionBase::Ptr>>,
             std::vector<Track::Ptr>, std::vector<DetectionBase::Ptr>>
  linear_assignment(const std::vector<Track::Ptr> &tracks,
                    const std::vector<DetectionBase::Ptr> &detections,
                    float thresh) const;

  std::tuple<std::vector<int>, std::vector<int>, double> exec_lapjv(
      std::vector<float> &&cost, size_t n_rows, size_t n_cols,
      bool extend_cost = false,
      float cost_limit = std::numeric_limits<float>::max(),
      bool return_cost = true) const;

 private:
  const float track_thresh_;
  const float high_thresh_;
  const float match_thresh_;
  const size_t max_time_lost_;

  size_t frame_id_;
  size_t track_id_count_;
  size_t count_;  //! Number of times update has been called

  std::vector<Track::Ptr> tracked_tracks_;
  std::vector<Track::Ptr> lost_tracks_;
};

}  // namespace byte_track
}  // namespace dyno
