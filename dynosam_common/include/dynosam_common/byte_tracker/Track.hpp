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
#include <memory>

#include "dynosam_common/byte_tracker/Detection.hpp"
#include "dynosam_common/byte_tracker/KalmanFilter.hpp"
#include "dynosam_common/byte_tracker/Rect.hpp"
#include "dynosam_common/utils/Macros.hpp"

namespace dyno {
namespace byte_track {

enum class TrackState {
  Tracked = 0,
  Lost = 1,
};

class Track {
 public:
  DYNO_POINTER_TYPEDEFS(Track)

  Track() = delete;
  Track(DetectionBase::Ptr detection, size_t start_frame_id, size_t track_id);

  const TrackState& get_track_state() const;
  bool is_confirmed() const;
  size_t get_track_id() const;
  size_t get_frame_id() const;
  size_t get_start_frame_id() const;
  size_t get_tracklet_length() const;

  const DetectionBase::ConstPtr get_detection() const;
  TlwhRect get_prediction() const;

 protected:
  friend class ByteTracker;

  DetectionBase::Ptr detection_;
  TlwhRect predicted_rect_;

  void predict();
  void update(const DetectionBase::Ptr& new_track, size_t frame_id);

  void mark_as_lost();
  void mark_as_confirmed();

 private:
  KalmanFilter kalman_filter_;

  TrackState state_;
  bool is_confirmed_;
  size_t track_id_;
  size_t frame_id_;
  size_t start_frame_id_;
  size_t tracklet_len_;
};
}  // namespace byte_track
}  // namespace dyno
