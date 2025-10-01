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

#include "dynosam_common/byte_tracker/Track.hpp"

#include <cstddef>

#include "dynosam_common/byte_tracker/Detection.hpp"

namespace dyno {
namespace byte_track {

Track::Track(DetectionBase::Ptr detection, size_t start_frame_id,
             size_t track_id)
    : detection_(detection),
      predicted_rect_(detection->rect()),
      kalman_filter_(),
      state_(TrackState::Tracked),
      // Detections registered on first frame are considered as confirmed
      is_confirmed_(start_frame_id == 1 ? true : false),
      track_id_(track_id),
      frame_id_(start_frame_id),
      start_frame_id_(start_frame_id),
      tracklet_len_(0) {
  kalman_filter_.initiate(detection->rect());
}

const TrackState& Track::get_track_state() const { return state_; }

bool Track::is_confirmed() const { return is_confirmed_; }

size_t Track::get_track_id() const { return track_id_; }

size_t Track::get_frame_id() const { return frame_id_; }

size_t Track::get_start_frame_id() const { return start_frame_id_; }

size_t Track::get_tracklet_length() const { return tracklet_len_; }

const DetectionBase::ConstPtr Track::get_detection() const {
  return detection_;
}
TlwhRect Track::get_prediction() const { return predicted_rect_; }

void Track::predict() {
  predicted_rect_ = kalman_filter_.predict(state_ != TrackState::Tracked);
}

void Track::update(const DetectionBase::Ptr& matched_detection,
                   size_t frame_id) {
  detection_->set_rect(matched_detection->rect());
  detection_->set_score(matched_detection->score());
  predicted_rect_ = kalman_filter_.update(matched_detection->rect());

  // If the track was actively tracked, just increment the tracklet length
  // Otherwise, mark the track as tracked again and reset the tracklet length
  if (state_ == TrackState::Tracked) {
    tracklet_len_++;
  } else {
    state_ = TrackState::Tracked;
    tracklet_len_ = 0;
  }
  is_confirmed_ = true;
  frame_id_ = frame_id;
}

void Track::mark_as_lost() { state_ = TrackState::Lost; }

void Track::mark_as_confirmed() { is_confirmed_ = true; }

}  // namespace byte_track
}  // namespace dyno
