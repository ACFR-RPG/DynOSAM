#include "dynosam2/frontend/Frame.hpp"

namespace dyno2 {

void ViFrame::setRelativePose(const Motion3ReferenceFrame& T_ij) {
  CHECK_EQ(T_ij.to(), this->getFrameId());
  T_ij_ = T_ij;
}

void ViFrame::setImuMeasurements(const dyno::ImuMeasurements& imu) {
  imu_measurements_ = imu;
}

bool ViFrame::hasImuMeasurements() const {
  return imu_measurements_.acc_gyr_.cols() > 0;
}

}  // namespace dyno2
