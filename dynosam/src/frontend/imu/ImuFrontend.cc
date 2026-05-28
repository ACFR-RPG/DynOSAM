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

#include "dynosam/frontend/imu/ImuFrontend.hpp"

#include <gtsam/navigation/CombinedImuFactor.h>

namespace dyno {

template <typename R, typename... A>
struct FunctionSignature;

template <typename R, typename... A>
struct FunctionSignature<auto(A...)->R> {
  using ReturnType = R;
  using ArgTuple = std::tuple<A...>;
};

const gtsam::PreintegratedCombinedMeasurements&
safeCastToPreintegratedCombinedImuMeasurements(
    const gtsam::PreintegrationType& pim) {
  try {
    return dynamic_cast<const gtsam::PreintegratedCombinedMeasurements&>(pim);
  } catch (const std::bad_cast& e) {
    LOG(ERROR) << "Seems that you are casting PreintegratedType to "
                  "PreintegratedCombinedMeasurements, but this object is not "
                  "a PreintegratedCombinedMeasurements!";
    LOG(FATAL) << e.what();
    throw e;
  }
}

using CombinedParams = gtsam::PreintegratedCombinedMeasurements::Params;
using CombinedParamSignature =
    FunctionSignature<decltype(CombinedParams::MakeSharedD)>;
using CombinedParamPtr = CombinedParamSignature::ReturnType;

ImuFrontend::ImuFrontend(const ImuCalibration& imu_calib) : params_(imu_calib) {
  CombinedParamPtr pim_params;
  pim_params.reset(new CombinedParams(params_.n_gravity));

  pim_params->body_P_sensor = params_.T_CI;

  gtsam::Matrix33 gyroscopeCovariance =
      std::pow(params_.gyro_noise_density, 2.0) * Eigen::Matrix3d::Identity();
  pim_params->setGyroscopeCovariance(gyroscopeCovariance);
  gtsam::Matrix33 accelerometerCovariance =
      std::pow(params_.acc_noise_density, 2.0) * Eigen::Matrix3d::Identity();
  pim_params->setAccelerometerCovariance(accelerometerCovariance);

  pim_params->biasAccCovariance =
      std::pow(params_.gyro_random_walk, 2.0) * Eigen::Matrix3d::Identity();
  pim_params->biasOmegaCovariance =
      std::pow(params_.acc_random_walk, 2.0) * Eigen::Matrix3d::Identity();

  pim_params->use2ndOrderCoriolis = false;

  pim_params->integrationCovariance =
      std::pow(params_.imu_integration_sigma, 2.0) *
      Eigen::Matrix3d::Identity();

  pim_ = std::make_unique<gtsam::PreintegratedCombinedMeasurements>(
      pim_params, gtsam::imuBias::ConstantBias{});
}

ImuFrontend::ImuFrontend(const ImuFrontend& other)
    : ImuFrontend(other.params_) {
  if (other.pim_) {
    pim_ = std::move(other.copyPimUnique());
  }
}

void ImuFrontend::preintegrateImuMeasurements(
    gtsam::PreintegrationType& pim, const ImuMeasurements& imu_measurements) {
  const Timestamps& stamps = imu_measurements.timestamps_;
  const ImuAccGyrs& accgyr = imu_measurements.acc_gyr_;
  CHECK(stamps.cols() >= 2) << "No Imu data found.";
  CHECK(accgyr.cols() >= 2) << "No Imu data found.";

  for (int i = 0; i < stamps.cols() - 1; ++i) {
    const gtsam::Vector3& measured_acc = accgyr.block<3, 1>(0, i);
    const gtsam::Vector3& measured_omega = accgyr.block<3, 1>(3, i);
    // assume stamps are in seconds
    const double& delta_t = stamps(i + 1) - stamps(i);
    CHECK_GT(delta_t, 0.0) << "Imu delta is 0!";
    // TODO Shouldn't we use pim_->integrateMeasurements(); for less code
    // and efficiency??
    pim.integrateMeasurement(measured_acc, measured_omega, delta_t);
  }
}

ImuFrontend::PimPtr ImuFrontend::preintegrateImuMeasurements(
    const ImuMeasurements& imu_measurements) {
  preintegrateImuMeasurements(*pim_, imu_measurements);

  return std::make_unique<gtsam::PreintegratedCombinedMeasurements>(
      safeCastToPreintegratedCombinedImuMeasurements(*pim_));
  // return copyPimShared();
}

ImuFrontend::PimPtr ImuFrontend::copyPim(const ImuFrontend::PimPtr& pim) {
  auto combined_pim =
      std::dynamic_pointer_cast<gtsam::PreintegratedCombinedMeasurements>(pim);
  if (!combined_pim) {
    throw DynosamException(
        "Cannot copy PimPtr becuase derived type was not "
        "PreintegratedCombinedMeasurements!");
  }
  return std::make_shared<gtsam::PreintegratedCombinedMeasurements>(
      *combined_pim);
}

ImuFrontend::PimPtr ImuFrontend::copyPimShared() const {
  return std::make_shared<gtsam::PreintegratedCombinedMeasurements>(
      safeCastToPreintegratedCombinedImuMeasurements(*pim_));
}

ImuFrontend::PimUniquePtr ImuFrontend::copyPimUnique() const {
  return std::make_unique<gtsam::PreintegratedCombinedMeasurements>(
      safeCastToPreintegratedCombinedImuMeasurements(*pim_));
}

}  // namespace dyno
