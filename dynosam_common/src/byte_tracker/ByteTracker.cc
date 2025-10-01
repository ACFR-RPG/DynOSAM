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

#include "dynosam_common/byte_tracker/ByteTracker.hpp"

#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "dynosam_common/byte_tracker/Detection.hpp"
#include "dynosam_common/byte_tracker/Lapjv.hpp"
#include "dynosam_common/byte_tracker/Rect.hpp"
#include "dynosam_common/byte_tracker/Track.hpp"

namespace dyno {
namespace byte_track {

ByteTracker::ByteTracker(const ByteTrackerParams &params)
    : track_thresh_(params.track_thresh),
      high_thresh_(params.high_thresh),
      match_thresh_(params.match_thresh),
      max_time_lost_(
          static_cast<size_t>(params.frame_rate / 30.0 * params.track_buffer)),
      frame_id_(params.starting_frame),
      track_id_count_(0),
      count_(0) {}

std::vector<Track::Ptr> ByteTracker::update(
    const std::vector<DetectionBase::Ptr> &input_detections, size_t frame_id) {
  CHECK_GE(frame_id, frame_id_);
  // if first call of update, check incoming frame is the same as the starting
  // frame
  // TODO: (jesse) this is super weird logic - there is a dependancy on
  // ByteTrackerParams::params::starting_frame but nothing inside this class
  // enforces consistnecy, yet we check that the frame calls incrementing!
  // if(count_ == 0) CHECK_EQ(frame_id, frame_id_);
  // // if not first call of update, (when frame id should be set t)
  // if(count_ > 0) CHECK_EQ(frame_id, frame_id_ + 1u);

  // update frame count
  frame_id_ = frame_id;
  count_++;

  ////////// Step 1: Get detections                                   //////////

  // Sort new tracks from detection by score
  std::vector<DetectionBase::Ptr> detections;
  std::vector<DetectionBase::Ptr> low_score_detections;
  for (const auto &detection : input_detections) {
    if (detection->score() >= track_thresh_)
      detections.push_back(detection);
    else
      low_score_detections.push_back(detection);
  }

  // Sort existing tracks by confirmed status
  std::vector<Track::Ptr> confirmed_tracks;
  std::vector<Track::Ptr> unconfirmed_tracks;

  for (const auto &track : tracked_tracks_) {
    if (!track->is_confirmed())
      unconfirmed_tracks.push_back(track);
    else
      confirmed_tracks.push_back(track);
  }

  std::vector<Track::Ptr> track_pool;
  track_pool = joint_tracks(confirmed_tracks, lost_tracks_);

  // Predict current pose by KF
  for (auto &track : track_pool) track->predict();

  ////////// Step 2: Find matches between tracks and detections       //////////
  ////////// Step 2: First association, with IoU                      //////////
  auto [matched_tracks, unmatched_tracked_tracks, unmatched_detections] =
      iou_association(track_pool, detections);

  ////////// Step 3: Second association, using low score dets         //////////
  auto new_lost_tracks = low_score_association(
      matched_tracks, low_score_detections, unmatched_tracked_tracks);

  ////////// Step 4: Init new tracks                                  //////////
  auto removed_tracks =
      init_new_tracks(matched_tracks, unconfirmed_tracks, unmatched_detections);

  ////////// Step 5: Update state                                     //////////
  for (auto &lost_track : lost_tracks_) {
    if (frame_id_ - lost_track->get_frame_id() > max_time_lost_) {
      removed_tracks.push_back(lost_track);
    }
  }

  lost_tracks_ = sub_tracks(
      joint_tracks(sub_tracks(lost_tracks_, matched_tracks), new_lost_tracks),
      removed_tracks);

  std::tie(tracked_tracks_, lost_tracks_) =
      remove_duplicate_tracks(matched_tracks, lost_tracks_);

  std::vector<Track::Ptr> output_tracks;
  for (const auto &track : tracked_tracks_) {
    if (track->is_confirmed()) output_tracks.push_back(track);
  }

  return output_tracks;
}

void ByteTracker::clear() {
  tracked_tracks_.clear();
  lost_tracks_.clear();
  frame_id_ = 0;
  track_id_count_ = 0;
}

std::tuple<std::vector<Track::Ptr>, std::vector<Track::Ptr>,
           std::vector<DetectionBase::Ptr>>
ByteTracker::iou_association(
    const std::vector<Track::Ptr> &track_pool,
    const std::vector<DetectionBase::Ptr> &detections) {
  auto [matches, unmatched_tracks, unmatched_detections] =
      linear_assignment(track_pool, detections, match_thresh_);

  std::vector<Track::Ptr> matched_tracks;
  for (const auto &match : matches) {
    const auto track = match.first;
    const auto detection = match.second;
    track->update(detection, frame_id_);
    matched_tracks.push_back(track);
  }

  std::vector<Track::Ptr> unmatched_tracked_tracks;
  for (const auto &unmatch : unmatched_tracks) {
    if (unmatch->get_track_state() == TrackState::Tracked) {
      unmatched_tracked_tracks.push_back(unmatch);
    }
  }
  return {std::move(matched_tracks), std::move(unmatched_tracked_tracks),
          std::move(unmatched_detections)};
}

std::vector<Track::Ptr> ByteTracker::low_score_association(
    std::vector<Track::Ptr> &matched_tracks,
    const std::vector<DetectionBase::Ptr> &low_score_detections,
    const std::vector<Track::Ptr> &unmatched_tracked_tracks) {
  auto [matches, unmatched_tracks, unmatch_detection] =
      linear_assignment(unmatched_tracked_tracks, low_score_detections, 0.5);

  for (const auto &match : matches) {
    const auto track = match.first;
    const auto detection = match.second;
    track->update(detection, frame_id_);
    matched_tracks.push_back(track);
  }

  std::vector<Track::Ptr> new_lost_tracks;
  for (const auto &track : unmatched_tracks) {
    if (track->get_track_state() != TrackState::Lost) {
      track->mark_as_lost();
      new_lost_tracks.push_back(track);
    }
  }
  return new_lost_tracks;
}

std::vector<Track::Ptr> ByteTracker::init_new_tracks(
    std::vector<Track::Ptr> &matched_tracks,
    const std::vector<Track::Ptr> &unconfirmed_tracks,
    const std::vector<DetectionBase::Ptr> &unmatched_detections) {
  // Deal with unconfirmed tracks, usually tracks with only one beginning frame
  auto [matches, unmatched_unconfirmed_tracks, new_detections] =
      linear_assignment(unconfirmed_tracks, unmatched_detections, 0.7);

  for (const auto &match : matches) {
    match.first->update(match.second, frame_id_);
    matched_tracks.push_back(match.first);
  }

  std::vector<Track::Ptr> new_removed_tracks;
  for (const auto &track : unmatched_unconfirmed_tracks) {
    new_removed_tracks.push_back(track);
  }

  // Add new tracks
  for (const auto &detection : new_detections) {
    if (detection->score() < track_thresh_) continue;
    track_id_count_++;
    Track::Ptr new_track =
        std::make_shared<Track>(detection, frame_id_, track_id_count_);
    if (detection->score() >= high_thresh_) {
      new_track->mark_as_confirmed();
    }
    matched_tracks.push_back(new_track);
  }
  return new_removed_tracks;
}

std::vector<Track::Ptr> ByteTracker::joint_tracks(
    const std::vector<Track::Ptr> &a_tlist,
    const std::vector<Track::Ptr> &b_tlist) const {
  std::set<int> exists;
  std::vector<Track::Ptr> res;
  for (auto &track : a_tlist) {
    exists.emplace(track->get_track_id());
    res.push_back(track);
  }
  for (auto &track : b_tlist) {
    if (exists.count(track->get_track_id()) == 0) res.push_back(track);
  }
  return res;
}

std::vector<Track::Ptr> ByteTracker::sub_tracks(
    const std::vector<Track::Ptr> &a_tlist,
    const std::vector<Track::Ptr> &b_tlist) const {
  std::map<int, Track::Ptr> tracks;
  for (auto &track : a_tlist) tracks.emplace(track->get_track_id(), track);
  for (auto &track : b_tlist) tracks.erase(track->get_track_id());

  std::vector<Track::Ptr> res;
  for (auto &[_, track] : tracks) res.push_back(track);
  return res;
}

std::tuple<std::vector<Track::Ptr>, std::vector<Track::Ptr>>
ByteTracker::remove_duplicate_tracks(
    const std::vector<Track::Ptr> &a_tracks,
    const std::vector<Track::Ptr> &b_tracks) const {
  if (a_tracks.empty() || b_tracks.empty()) return {a_tracks, b_tracks};

  std::vector<std::vector<float>> ious;
  ious.resize(a_tracks.size());
  for (size_t i = 0; i < ious.size(); i++) ious[i].resize(b_tracks.size());
  for (size_t ai = 0; ai < a_tracks.size(); ai++) {
    for (size_t bi = 0; bi < b_tracks.size(); bi++) {
      ious[ai][bi] = 1 - calc_iou(b_tracks[bi]->predicted_rect_,
                                  a_tracks[ai]->predicted_rect_);
    }
  }

  std::vector<bool> a_overlapping(a_tracks.size(), false),
      b_overlapping(b_tracks.size(), false);
  for (size_t ai = 0; ai < ious.size(); ai++) {
    for (size_t bi = 0; bi < ious[ai].size(); bi++) {
      if (ious[ai][bi] < 0.15) {
        const int timep =
            a_tracks[ai]->get_frame_id() - a_tracks[ai]->get_start_frame_id();
        const int timeq =
            b_tracks[bi]->get_frame_id() - b_tracks[bi]->get_start_frame_id();
        if (timep > timeq) {
          b_overlapping[bi] = true;
        } else {
          a_overlapping[ai] = true;
        }
      }
    }
  }

  std::vector<Track::Ptr> a_tracks_out;
  for (size_t ai = 0; ai < a_tracks.size(); ai++) {
    if (!a_overlapping[ai]) a_tracks_out.push_back(a_tracks[ai]);
  }

  std::vector<Track::Ptr> b_tracks_out;
  for (size_t bi = 0; bi < b_tracks.size(); bi++) {
    if (!b_overlapping[bi]) b_tracks_out.push_back(b_tracks[bi]);
  }
  return {std::move(a_tracks_out), std::move(b_tracks_out)};
}

std::tuple<std::vector<std::pair<Track::Ptr, DetectionBase::Ptr>>,
           std::vector<Track::Ptr>, std::vector<DetectionBase::Ptr>>
ByteTracker::linear_assignment(
    const std::vector<Track::Ptr> &tracks,
    const std::vector<DetectionBase::Ptr> &detections, float thresh) const {
  if (tracks.empty() || detections.empty()) return {{}, tracks, detections};

  size_t n_rows = tracks.size();
  size_t n_cols = detections.size();
  std::vector<float> cost_matrix(n_rows * n_cols);
  for (size_t i = 0; i < n_rows; i++) {
    for (size_t j = 0; j < n_cols; j++) {
      cost_matrix[i * n_cols + j] =
          1 - calc_iou(detections[j]->rect(), tracks[i]->predicted_rect_);
    }
  }

  std::vector<std::pair<Track::Ptr, DetectionBase::Ptr>> matches;
  std::vector<Track::Ptr> a_unmatched;
  std::vector<DetectionBase::Ptr> b_unmatched;

  auto [rowsol, colsol, _] =
      exec_lapjv(std::move(cost_matrix), n_rows, n_cols, true, thresh);
  for (size_t i = 0; i < rowsol.size(); i++) {
    if (rowsol[i] >= 0)
      matches.push_back({tracks[i], detections[rowsol[i]]});
    else
      a_unmatched.push_back(tracks[i]);
  }

  for (size_t i = 0; i < colsol.size(); i++) {
    if (colsol[i] < 0) b_unmatched.push_back(detections[i]);
  }
  return {std::move(matches), std::move(a_unmatched), std::move(b_unmatched)};
}

std::tuple<std::vector<int>, std::vector<int>, double> ByteTracker::exec_lapjv(
    std::vector<float> &&cost, size_t n_rows, size_t n_cols, bool extend_cost,
    float cost_limit, bool return_cost) const {
  std::vector<int> rowsol(n_rows);
  std::vector<int> colsol(n_cols);

  if (n_rows != n_cols && !extend_cost) {
    throw std::runtime_error("The `extend_cost` variable should set True");
  }

  size_t n = 0;
  std::vector<float> cost_c;
  if (extend_cost || cost_limit < std::numeric_limits<float>::max()) {
    n = n_rows + n_cols;
    cost_c.resize(n * n);
    if (cost_limit < std::numeric_limits<float>::max()) {
      cost_c.assign(cost_c.size(), cost_limit / 2.0);
    } else {
      cost_c.assign(cost_c.size(),
                    *std::max_element(cost.begin(), cost.end()) + 1);
    }
    // Assign cost to top-left corner
    for (size_t i = 0; i < n_rows; i++) {
      for (size_t j = 0; j < n_cols; j++) {
        cost_c[i * n + j] = cost[i * n_cols + j];
      }
    }
    // Set bottom-right corner to 0
    for (size_t i = n_rows; i < n; i++) {
      for (size_t j = n_cols; j < n; j++) {
        cost_c[i * n + j] = 0;
      }
    }
  } else {
    n = n_rows;
    cost_c = std::move(cost);
  }

  std::vector<int> x_c(n), y_c(n);

  int ret = lapjv_internal(n, cost_c, x_c, y_c);
  if (ret != 0) {
    throw std::runtime_error("The result of lapjv_internal() is invalid.");
  }

  double opt = 0.0;

  if (n != n_rows) {
    for (size_t i = 0; i < n; i++) {
      if (x_c[i] >= static_cast<int>(n_cols)) x_c[i] = -1;
      if (y_c[i] >= static_cast<int>(n_rows)) y_c[i] = -1;
    }
    for (size_t i = 0; i < n_rows; i++) {
      rowsol[i] = x_c[i];
    }
    for (size_t i = 0; i < n_cols; i++) {
      colsol[i] = y_c[i];
    }

    if (return_cost) {
      for (size_t i = 0; i < rowsol.size(); i++) {
        if (rowsol[i] != -1) {
          opt += cost_c[i * n_cols + rowsol[i]];
        }
      }
    }
  } else if (return_cost) {
    for (size_t i = 0; i < rowsol.size(); i++) {
      opt += cost_c[i * n_cols + rowsol[i]];
    }
  }

  return {std::move(rowsol), std::move(colsol), opt};
}
}  // namespace byte_track
}  // namespace dyno
