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

#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/ImuFactor.h>

#include "dynosam_common/Types.hpp"
#include "dynosam_sensors/ImuMeasurements.hpp"
#include "dynosam_sensors/ImuParams.hpp"

// TODO: eventually replace with Kimera or otherwise
namespace dyno {

class ImuFrontend {
 public:
  using PimPtr = std::shared_ptr<gtsam::PreintegrationType>;
  using PimUniquePtr = std::unique_ptr<gtsam::PreintegrationType>;

  DYNO_POINTER_TYPEDEFS(ImuFrontend)

  ImuFrontend(const ImuCalibration& imu_calib);
  ImuFrontend& operator=(const ImuFrontend&) = delete;
  ImuFrontend(ImuFrontend&&) noexcept = default;
  ImuFrontend& operator=(ImuFrontend&&) noexcept = default;

  ImuFrontend(const ImuFrontend& other);

  PimPtr preintegrateImuMeasurements(const ImuMeasurements& imu_measurements);

  inline void resetIntegration() { pim_->resetIntegration(); }

  // Returns a copy of the interal pim
  PimPtr getPim() const { return copyPimShared(); }

  static PimPtr copyPim(const PimPtr& pim);

 private:
  PimPtr copyPimShared() const;
  PimUniquePtr copyPimUnique() const;

 private:
  ImuCalibration params_;
  PimUniquePtr pim_ = nullptr;
};

}  // namespace dyno
