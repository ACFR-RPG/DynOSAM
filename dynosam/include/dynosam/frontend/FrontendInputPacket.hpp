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

#include <glog/logging.h>

#include <exception>
#include <opencv4/opencv2/opencv.hpp>
#include <type_traits>

#include "dynosam/frontend/imu/ImuMeasurements.hpp"
#include "dynosam_common/GroundTruthPacket.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/OpenCVUtils.hpp"
#include "dynosam_common/utils/Tuple.hpp"
#include "dynosam_cv/ImageContainer.hpp"

namespace dyno {

// inherit to add more sensor data (eg imu)
// can be of any InputImagePacketBase with this as the default. We use this so
// we can create a FrontendInputPacketBase type with the InputPacketType as the
// actual derived type and not the base class so different modules do not have
// to do their own casting to get image data
struct FrontendInputPacketBase {
  DYNO_POINTER_TYPEDEFS(FrontendInputPacketBase)

  ImageContainer::Ptr image_container_;
  GroundTruthInputPacket::Optional optional_gt_;
  ImuMeasurements::Optional imu_measurements;

  FrontendInputPacketBase()
      : image_container_(nullptr), optional_gt_(std::nullopt) {}

  FrontendInputPacketBase(
      ImageContainer::Ptr image_container,
      GroundTruthInputPacket::Optional optional_gt = std::nullopt)
      : image_container_(image_container), optional_gt_(optional_gt) {
    CHECK(image_container_);

    if (optional_gt) {
      CHECK_EQ(optional_gt_->frame_id_, image_container_->frameId());
    }
  }

  inline FrameId getFrameId() const {
    return CHECK_NOTNULL(image_container_)->frameId();
  }

  inline Timestamp getTimestamp() const {
    return CHECK_NOTNULL(image_container_)->timestamp();
  }

  virtual ~FrontendInputPacketBase() = default;
};

}  // namespace dyno
