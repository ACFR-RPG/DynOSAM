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

#pragma once

#include <mutex>

#include "dynosam_common/GroundTruthPacket.hpp"
#include "dynosam_common/Types.hpp"

namespace dyno {

class SharedModuleInfo {
 public:
  static SharedModuleInfo& instance();

  std::optional<GroundTruthPacketMap> getGroundTruthPackets() const;
  const FrameIdTimestampMap& getTimestampMap() const;

  bool getTimestamp(FrameId frame_id, Timestamp& timestamp) const;

  SharedModuleInfo& updateGroundTruthPacket(
      FrameId frame_id, const GroundTruthInputPacket& ground_truth_packet);
  SharedModuleInfo& updateTimestampMapping(FrameId frame_id,
                                           Timestamp timestamp);

 private:
  static std::unique_ptr<SharedModuleInfo> instance_;

 private:
  mutable std::mutex mutex_;
  GroundTruthPacketMap gt_packet_map_;
  FrameIdTimestampMap frame_id_to_timestamp_map_;
};

struct SharedModuleInterface {
  SharedModuleInfo& shared_module_info = SharedModuleInfo::instance();
};

struct ConstSharedModuleInterface {
  const SharedModuleInfo& shared_module_info = SharedModuleInfo::instance();
};

}  // namespace dyno
