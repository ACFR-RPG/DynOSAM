/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam/backend/BackendOutputPacket.hpp"
#include "dynosam/common/SharedModuleInfo.hpp"
#include "dynosam/common/Types.hpp"
#include "dynosam/frontend/VisionImuOutputPacket.hpp"

namespace dyno {

namespace internal {
template <typename INPUT>
std::pair<Timestamp, FrameId> collectTemporalData(const INPUT&);

template <>
inline std::pair<Timestamp, FrameId> collectTemporalData(
    const VisionImuPacket& input) {
  return {input.timestamp(), input.frameId()};
}

template <>
inline std::pair<Timestamp, FrameId> collectTemporalData(
    const BackendOutputPacket& input) {
  return {input.timestamp, input.frame_id};
}
}  // namespace internal

template <typename INPUT>
class DisplayBase : public SharedModuleInterface {
 public:
  using Input = INPUT;
  using This = DisplayBase<INPUT>;
  DYNO_POINTER_TYPEDEFS(This)

  using InputConstPtr = typename INPUT::ConstPtr;

  DisplayBase() {}
  virtual ~DisplayBase() {}

  void spinOnce(const InputConstPtr& input) {
    CHECK(input);
    auto [timestamp, frame_id] = internal::collectTemporalData<INPUT>(*input);
    this->shared_module_info.updateTimestampMapping(frame_id, timestamp);
    spinOnceImpl(input);
  }

 protected:
  virtual void spinOnceImpl(const InputConstPtr& input) = 0;
};

using FrontendDisplay = DisplayBase<VisionImuPacket>;
using BackendDisplay = DisplayBase<BackendOutputPacket>;

}  // namespace dyno
