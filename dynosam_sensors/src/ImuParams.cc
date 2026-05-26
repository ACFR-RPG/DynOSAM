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

#include "dynosam_sensors/ImuParams.hpp"

#include <config_utilities/config_utilities.h>
#include <config_utilities/parsing/yaml.h>

namespace dyno {

std::ostream& operator<<(std::ostream& os, const ImuCalibration& calibration) {
  os << "init_bias_sigma: " << calibration.init_bias_sigma << "\n";
  os << "gyro_noise_density: " << calibration.gyro_noise_density << "\n";
  os << "gyro_random_walk: " << calibration.gyro_random_walk << "\n";
  os << "acc_noise_density: " << calibration.acc_noise_density << "\n";
  os << "acc_random_walk: " << calibration.acc_random_walk << "\n";
  os << "imu_integration_sigma: " << calibration.imu_integration_sigma << "\n";
  os << "n_gravity: " << calibration.n_gravity << "\n";
  os << "T_CI: " << calibration.T_CI << "\n";
  os << "reference_frame: " << calibration.reference_frame << "\n";
  return os;
}

void declare_config(ImuParams& config) {
  using namespace config;

  name("ImuParams");
  field(config.init_bias_sigma, "init_bias_sigma");
  field(config.gyro_noise_density, "gyro_noise_density");
  field(config.gyro_random_walk, "gyro_random_walk");
  field(config.acc_noise_density, "acc_noise_density");
  field(config.acc_random_walk, "acc_random_walk");
  field(config.imu_integration_sigma, "imu_integration_sigma");

  std::vector<double> gravity_v;
  field(gravity_v, "n_gravity");
  checkCondition(gravity_v.size() == 3,
                 "param 'n_gravity' must be a 3 length vector.");

  config.n_gravity(0) = gravity_v.at(0);
  config.n_gravity(1) = gravity_v.at(1);
  config.n_gravity(2) = gravity_v.at(2);
}

}  // namespace dyno
