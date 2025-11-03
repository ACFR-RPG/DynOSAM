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

#include "dynosam_nn/trackers/ObjectTracker.hpp"

namespace dyno {

std::vector<SingleDetectionResult> ByteObjectTracker::track(
    const std::vector<ObjectDetection>& detections, FrameId frame_id) {
  std::vector<byte_track::STrackPtr> object_tracks =
      impl_tracker_.update(detections);

  std::vector<SingleDetectionResult> output;
  for (size_t i = 0; i < object_tracks.size(); i++) {
    const auto& object_track = object_tracks.at(i);
    const byte_track::Rect<float>& filtered_bb = object_track->getRect();

    SingleDetectionResult single_result;

    cv::Rect bb(static_cast<int>(filtered_bb.x()),
                static_cast<int>(filtered_bb.y()),
                static_cast<int>(filtered_bb.width()),
                static_cast<int>(filtered_bb.height()));
    single_result.bounding_box = bb;
    single_result.mask = object_track->getMask();
    //   single_result.class_name = detection.class_name;
    single_result.confidence = object_track->getScore();

    single_result.object_id = static_cast<ObjectId>(object_track->getTrackId());
    //   //is this the only indicater we should be using!?
    single_result.well_tracked = object_track->isActivated();

    output.push_back(single_result);
  }

  return output;
}

}  // namespace dyno
