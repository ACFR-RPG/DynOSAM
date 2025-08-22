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

#include "dynosam/frontend/imu/ImuParams.hpp"

#include <config_utilities/config_utilities.h>
#include <config_utilities/parsing/yaml.h>
#include <config_utilities/types/eigen_matrix.h>

#include "dynosam/utils/GtsamUtils.hpp"   //for cv equals and pose convernsion
#include "dynosam/utils/Numerical.hpp"    //for equals
#include "dynosam/utils/OpenCVUtils.hpp"  //for cv equals

namespace dyno {

void declare_config(ImuParams& config) {
  using namespace config;

  name("ImuParams");

  // // camera to robot pose
  // std::vector<double> vector_pose;
  // field(vector_pose, "T_BS");
  // checkCondition(
  //     vector_pose.size() == 16u,
  //     "param 'T_BS' must be a 16 length vector in homogenous matrix form");
  // config.body_P_sensor = utils::poseVectorToGtsamPose3(vector_pose);
  field<utils::Pose3Converter>(config.body_P_sensor, "T_BS");

  field(config.gyro_noise_density, "gyroscope_noise_density");
  field(config.gyro_random_walk, "gyroscope_random_walk");
  field(config.acc_noise_density, "accelerometer_noise_density");
  field(config.acc_random_walk, "accelerometer_random_walk");
  field(config.imu_integration_sigma, "imu_integration_sigma");

  field(config.n_gravity, "n_gravity");
}

}  // namespace dyno
