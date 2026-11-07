#include "dynosam2/estimator/ImuError.hpp"

namespace dyno2 {

dyno::Pose3Measurement RelativePoseErrorBase::asMeasurement() const {
  return dyno::Pose3Measurement(this->deltaXij(), this->covariance());
}

ImuError::ImuError(const dyno::ImuMeasurements& measurements,
                   dyno::CombinedPIMParamPtr params, dyno::FrameId k0,
                   dyno::FrameId k1, const ImuError::Bias& bias)
    : imu_measurements_(HeavyImuMeasurements(
          measurements, dyno::ReferenceFrame::LOCAL, k0, k1)),
      pim_(params, bias) {
  // immediately integrated new IMU measurements
  dyno::ImuFrontend::preintegrateImuMeasurements(pim_,
                                                 imu_measurements_.estimate());
}

gtsam::Pose3 ImuError::deltaXij() const { return pim_.deltaXij().pose(); }
gtsam::Matrix66 ImuError::covariance() const {
  return pim_.preintMeasCov().block<6, 6>(0, 0);
}

void ImuError::append(const dyno::ImuMeasurements& measurements,
                      dyno::FrameId k1) {
  // append to internal imu measurements
  if (!mergeMeasurements(measurements)) {
    // nothing to merge
    return;
  }
  // add to preintegration
  dyno::ImuFrontend::preintegrateImuMeasurements(pim_, measurements);
  imu_measurements_.setTo(k1);
}

gtsam::NavState ImuError::predict(const gtsam::NavState& nav_state_k0,
                                  const Bias& bias) const {
  return pim_.predict(nav_state_k0, bias);
}

bool ImuError::mergeMeasurements(const dyno::ImuMeasurements& z1) {
  const size_t z1_size = z1.acc_gyr_.cols();
  if (z1_size == 0) {
    return false;
  }

  // measurements to merge to
  dyno::ImuMeasurements& z0 = imu_measurements_.estimate();
  dyno::Timestamp z0_t1 = z0.timestamps_(Eigen::last);
  dyno::Timestamp z1_t0 = z1.timestamps_(0);

  CHECK_GT(z1_t0, z0_t1) << "Cannot merge IMU";

  // grow existing measurement vector
  const size_t z0_size = z0.acc_gyr_.cols();
  z0.acc_gyr_.conservativeResize(Eigen::NoChange, z0_size + z1_size);
  z0.acc_gyr_.rightCols(z1_size) = z1.acc_gyr_;

  z0.timestamps_.conservativeResize(Eigen::NoChange, z0_size + z1_size);
  z0.timestamps_.rightCols(z1_size) = z1.timestamps_;

  CHECK_EQ(t1(), z1.timestamps_(Eigen::last));
  return true;
}

RelativePoseError::RelativePoseError(const dyno::Pose3Measurement& T_ij,
                                     dyno::FrameId from, dyno::FrameId to)
    : impl_(T_ij, dyno::ReferenceFrame::LOCAL, from, to) {}

gtsam::Pose3 RelativePoseError::deltaXij() const {
  return impl_.estimate().measurement();
}
gtsam::Matrix66 RelativePoseError::covariance() const {
  return impl_.estimate().covariance();
}
dyno::Pose3Measurement RelativePoseError::asMeasurement() const {
  return impl_.estimate();
}

}  // namespace dyno2
