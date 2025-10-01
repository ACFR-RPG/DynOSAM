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

#include "dynosam/frontend/imu/Imu-Definitions.hpp"
#include "dynosam_common/Types.hpp"  //for timestamp

namespace dyno {

struct ImuMeasurement {
  ImuMeasurement() = default;
  ImuMeasurement(const Timestamp& timestamp, const ImuAccGyr& imu_data)
      : timestamp_(timestamp), acc_gyr_(imu_data) {}
  ImuMeasurement(Timestamp&& timestamp, ImuAccGyr&& imu_data)
      : timestamp_(std::move(timestamp)), acc_gyr_(std::move(imu_data)) {}

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Timestamp timestamp_;
  ImuAccGyr acc_gyr_;
};

// Multiple Imu measurements, bundled in dynamic matrices.
struct ImuMeasurements {
 public:
  DYNO_POINTER_TYPEDEFS(ImuMeasurements)

  ImuMeasurements() = default;
  ImuMeasurements(const Timestamps& timestamps, const ImuAccGyrs& measurements)
      : timestamps_(timestamps), acc_gyr_(measurements) {}
  ImuMeasurements(Timestamps&& timestamps, ImuAccGyrs&& measurements)
      : timestamps_(std::move(timestamps)), acc_gyr_(std::move(measurements)) {}

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // assume timestamps are in seconds
  Timestamps timestamps_;
  ImuAccGyrs acc_gyr_;

  //! Optional synchronised frame id to indicate that this imu bundle has
  //! already been associated with some frame
  std::optional<FrameId> synchronised_frame_id{};
};

}  // namespace dyno
