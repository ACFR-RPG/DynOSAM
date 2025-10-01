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

#include <config_utilities/config_utilities.h>
#include <gtsam/base/Vector.h>

#include "dynosam_common/Types.hpp"

namespace dyno {

struct ImuParams {
  DYNO_POINTER_TYPEDEFS(ImuParams)

  double init_bias_sigma = 0.0;
  //! sigma for gyro covariance [rad/s/sqrt(Hz)] (gyro "white noise")
  double gyro_noise_density = 0.0;
  //! sigma for gyro bias covariance [rad/s^2/sqrt(Hz)] (gyro bias diffusion)
  double gyro_random_walk = 0.0;
  //! sigma for accel covariance [m/s^2/sqrt(Hz)] (accel "white noise")
  double acc_noise_density = 0.0;
  //! sigma for accel bias covariance [m/s^3/sqrt(Hz)] (accel bias diffusion)
  double acc_random_walk = 0.0;

  double imu_integration_sigma = 0.0;

  // In our case this is actually going to be the transform into the camera
  // frame (which we consider as the body frame!!!)
  gtsam::Pose3 body_P_sensor;

  gtsam::Vector3 n_gravity = gtsam::Vector3::Zero();
};

void declare_config(ImuParams& config);

}  // namespace dyno
