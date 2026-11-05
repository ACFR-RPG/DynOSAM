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

#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/ImuFactor.h>

#include "dynosam_common/Types.hpp"
#include "dynosam_sensors/ImuMeasurements.hpp"
#include "dynosam_sensors/ImuParams.hpp"

namespace {
template <typename R, typename... A>
struct FunctionSignature;

template <typename R, typename... A>
struct FunctionSignature<auto(A...)->R> {
  using ReturnType = R;
  using ArgTuple = std::tuple<A...>;
};
}  // namespace

namespace dyno {

using CombinedPIMParams = gtsam::PreintegratedCombinedMeasurements::Params;
using CombinedParamSignature =
    ::FunctionSignature<decltype(CombinedPIMParams::MakeSharedD)>;
using CombinedPIMParamPtr = CombinedParamSignature::ReturnType;

//! Alias to the predefined GTSAM Preintegrated Imu Measurement (PIM)
using Pim = gtsam::PreintegrationType;
using PimPtr = std::shared_ptr<Pim>;
using PimUniquePtr = std::unique_ptr<Pim>;

class ImuFrontend {
 public:
  DYNO_POINTER_TYPEDEFS(ImuFrontend)

  ImuFrontend(const ImuCalibration& imu_calib);
  ImuFrontend& operator=(const ImuFrontend&) = delete;
  ImuFrontend(ImuFrontend&&) noexcept = default;
  ImuFrontend& operator=(ImuFrontend&&) noexcept = default;

  ImuFrontend(const ImuFrontend& other);

  static CombinedPIMParamPtr makeCombinedParams(
      const ImuCalibration& imu_calib);

  static void preintegrateImuMeasurements(
      gtsam::PreintegrationType& pim, const ImuMeasurements& imu_measurements);
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
