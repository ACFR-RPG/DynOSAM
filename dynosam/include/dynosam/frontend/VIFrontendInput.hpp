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

#include "dynosam_common/GroundTruthPacket.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/OpenCVUtils.hpp"
#include "dynosam_common/utils/Tuple.hpp"
#include "dynosam_sensors/ImageContainer.hpp"
#include "dynosam_sensors/ImuMeasurements.hpp"

namespace dyno {

struct VIFrontendInput {
  DYNO_POINTER_TYPEDEFS(VIFrontendInput)

  ImageContainer::Ptr image_container_{nullptr};
  GroundTruthInputPacket::Optional ground_truth_packet{std::nullopt};
  ImuMeasurements::Optional imu_measurements{std::nullopt};

  VIFrontendInput() = default;

  VIFrontendInput(
      ImageContainer::Ptr image_container,
      GroundTruthInputPacket::Optional ground_truth_packet_ = std::nullopt,
      ImuMeasurements::Optional imu_measurements_ = std::nullopt)
      : image_container_(CHECK_NOTNULL(image_container)),
        ground_truth_packet(ground_truth_packet_),
        imu_measurements(imu_measurements_) {
    if (ground_truth_packet) {
      CHECK_EQ(ground_truth_packet->frame_id_, image_container_->frameId());
    }

    if (imu_measurements && imu_measurements->synchronised_frame_id) {
      CHECK_EQ(*imu_measurements->synchronised_frame_id,
               image_container_->frameId());
    }
  }

  inline FrameId getFrameId() const {
    return CHECK_NOTNULL(image_container_)->frameId();
  }

  inline Timestamp getTimestamp() const {
    return CHECK_NOTNULL(image_container_)->timestamp();
  }
};

}  // namespace dyno
