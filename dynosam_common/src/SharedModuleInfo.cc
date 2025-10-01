/*
 *   Copyright (c) 2025 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam_common/SharedModuleInfo.hpp"

#include <glog/logging.h>

namespace dyno {

decltype(SharedModuleInfo::instance_) SharedModuleInfo::instance_;

SharedModuleInfo& SharedModuleInfo::instance() {
  if (!instance_) {
    instance_.reset(new SharedModuleInfo());
  }
  return *instance_;
}

const FrameIdTimestampMap& SharedModuleInfo::getTimestampMap() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return frame_id_to_timestamp_map_;
}

std::optional<GroundTruthPacketMap> SharedModuleInfo::getGroundTruthPackets()
    const {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (gt_packet_map_.empty()) {
    return {};
  }
  return gt_packet_map_;
}

bool SharedModuleInfo::getTimestamp(FrameId frame_id,
                                    Timestamp& timestamp) const {
  const FrameIdTimestampMap& timestamp_map = getTimestampMap();

  const std::lock_guard<std::mutex> lock(mutex_);
  if (timestamp_map.exists(frame_id)) {
    timestamp = timestamp_map.at(frame_id);
    return true;
  } else {
    return false;
  }
}

SharedModuleInfo& SharedModuleInfo::updateGroundTruthPacket(
    FrameId frame_id, const GroundTruthInputPacket& ground_truth_packet) {
  const std::lock_guard<std::mutex> lock(mutex_);
  gt_packet_map_.insert2(frame_id, ground_truth_packet);
  return *this;
}

SharedModuleInfo& SharedModuleInfo::updateTimestampMapping(
    FrameId frame_id, Timestamp timestamp) {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (frame_id_to_timestamp_map_.exists(frame_id)) {
    CHECK_EQ(frame_id_to_timestamp_map_.at(frame_id), timestamp);
  } else {
    frame_id_to_timestamp_map_.insert2(frame_id, timestamp);
  }
  return *this;
}

}  // namespace dyno
