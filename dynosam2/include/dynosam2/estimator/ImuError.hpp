#pragma once

#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/NavState.h>

#include "dynosam2/common/Types.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_sensors/ImuFrontend.hpp"
#include "dynosam_sensors/ImuMeasurements.hpp"

namespace dyno2 {

class RelativePoseErrorBase {
 public:
  DYNO_POINTER_TYPEDEFS(RelativePoseErrorBase)
  virtual ~RelativePoseErrorBase() {}
  virtual dyno::FrameId from() const = 0;
  virtual dyno::FrameId to() const = 0;

  /// @brief Relative pose measurement from i to j in i
  /// @return
  virtual gtsam::Pose3 deltaXij() const = 0;

  /// @brief Relative uncertainty from i to j in i
  /// @return
  virtual gtsam::Matrix66 covariance() const = 0;

  /// @brief Get the relative pose as a Pose3Measurement with bundled
  /// uncertainty
  /// @return
  virtual dyno::Pose3Measurement asMeasurement() const;
};

/**
 * @brief A wrapper around a Printegration IMU measurement containing extra
 * information, the PIM itself, and the raw measurements.
 *
 * Loosly inspired by okvis::ceres::ImuError
 *
 */
class ImuError : public RelativePoseErrorBase {
 public:
  typedef gtsam::imuBias::ConstantBias Bias;
  ImuError(const dyno::ImuMeasurements& measurements,
           dyno::CombinedPIMParamPtr params, dyno::FrameId k0, dyno::FrameId k1,
           const Bias& bias = {});

  DYNO_POINTER_TYPEDEFS(ImuError)

  inline dyno::FrameId from() const override { return k0(); }
  inline dyno::FrameId to() const override { return k1(); }

  // TODO: must provide delta after bias correction!?
  gtsam::Pose3 deltaXij() const override;
  gtsam::Matrix66 covariance() const override;

  /// @brief The starting (from) frame.
  /// @return
  inline dyno::FrameId k0() const { return imu_measurements_.from(); }
  /// @brief The ending (to) frame.
  /// @return
  inline dyno::FrameId k1() const { return imu_measurements_.to(); }

  /// @brief The starting timestamp.
  /// @return
  inline dyno::Timestamp t0() const {
    return imu_measurements_.estimate().timestamps_(0);
  }

  /// @brief The ending timestamp.
  /// @return
  inline dyno::Timestamp t1() const {
    return imu_measurements_.estimate().timestamps_(Eigen::last);
  }

  inline const gtsam::PreintegratedCombinedMeasurements& getPim() const {
    return pim_;
  }

  void append(const ImuError& error);

  /// @brief Append with the same bias terms
  /// @param measurements
  void append(const dyno::ImuMeasurements& measurements, dyno::FrameId k1);

  gtsam::NavState predict(const gtsam::NavState& nav_state_k0,
                          const Bias& bias) const;

 private:
  bool mergeMeasurements(const dyno::ImuMeasurements& z1);

 private:
  struct HeavyImuMeasurements
      : public dyno::HeavyReferenceFrameValue<dyno::ImuMeasurements> {
    typedef dyno::HeavyReferenceFrameValue<dyno::ImuMeasurements> Base;
    using Base::Base;
    void setTo(dyno::FrameId fid) { to_ = fid; }
  };

  HeavyImuMeasurements imu_measurements_;
  //! Preintegrated measurements
  gtsam::PreintegratedCombinedMeasurements pim_;
};

/**
 * @brief Relative pose constructed directly from a relative pose measurement
 *
 */
class RelativePoseError : public RelativePoseErrorBase {
 public:
  DYNO_POINTER_TYPEDEFS(RelativePoseError)

  RelativePoseError(const dyno::Pose3Measurement& T_ij, dyno::FrameId from,
                    dyno::FrameId to);

  inline dyno::FrameId from() const override { return impl_.from(); }
  inline dyno::FrameId to() const override { return impl_.to(); }

  gtsam::Pose3 deltaXij() const override;
  gtsam::Matrix66 covariance() const override;
  dyno::Pose3Measurement asMeasurement() const override;

 private:
  HeavyPose3Measurement impl_;
};

}  // namespace dyno2
