/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "simulator.hpp"

#include <random>
#include <variant>

std::mt19937 gen(42);  // Fixed seed for reproducibility

namespace dyno_testing {

using namespace dyno;

Point3Measurement RGBDScenario::addNoiseStaticPoint(
    const Point3Measurement& p_local) const {
  return addNoisePoint(p_local, noise_params_.static_point_noise,
                       params_.static_outlier_ratio);
}
Point3Measurement RGBDScenario::addNoiseDynamicPoint(
    const Point3Measurement& p_local) const {
  return addNoisePoint(p_local, noise_params_.dynamic_point_noise,
                       params_.dynamic_outlier_ratio);
}

Point3Measurement RGBDScenario::addNoisePoint(const Point3Measurement& p_local,
                                              const PointNoise& options,
                                              double outlier_ratio) const {
  gtsam::Point3 noisy_p_local = p_local.measurement();
  gtsam::SharedGaussian model = p_local.model();

  if (std::holds_alternative<NaivePoint3dNoiseParams>(options)) {
    NaivePoint3dNoiseParams point_noise_params =
        std::get<NaivePoint3dNoiseParams>(options);
    noisy_p_local = dyno::utils::perturbWithNoise(p_local.measurement(),
                                                  point_noise_params.sigma);

    model = gtsam::noiseModel::Isotropic::Sigma(3, point_noise_params.sigma);

  } else if (std::holds_alternative<Point3NoiseParams>(options)) {
    Point3NoiseParams point_noise_params = std::get<Point3NoiseParams>(options);
    std::tie(noisy_p_local, model) = addAnisotropicNoiseToPoint(
        noisy_p_local, point_noise_params.sigma_xy, point_noise_params.sigma_z);
  } else if (std::holds_alternative<RGBDNoiseParams>(options)) {
    LOG(FATAL) << "Not implemented!";
  }

  if (outlier_ratio > 0 && outlier_dist(gen) < outlier_ratio) {
    // simulate out of distribution noise
    noisy_p_local =
        dyno::utils::perturbWithUniformNoise(noisy_p_local, -30, 30);
  }

  CHECK(model);

  Point3Measurement noisy_model(noisy_p_local, model);
  return noisy_model;
}

std::pair<gtsam::Point3, gtsam::SharedGaussian>
RGBDScenario::addAnisotropicNoiseToPoint(const gtsam::Point3& p,
                                         double sigma_xy,
                                         double sigma_z) const {
  double z = p.z();

  // Standard deviations for noise
  double s_x = sigma_xy * z;
  double s_y = sigma_xy * z;
  double s_z = sigma_z * z * z;

  std::normal_distribution<double> dist_x(0.0, s_x);
  std::normal_distribution<double> dist_y(0.0, s_y);
  std::normal_distribution<double> dist_z(0.0, s_z);

  Eigen::Vector3d noise(dist_x(gen), dist_y(gen), dist_z(gen));

  gtsam::Vector3 sigmas;
  sigmas << s_x, s_y, s_z;

  return {p + noise, gtsam::noiseModel::Isotropic::Sigmas(sigmas)};
}

}  // namespace dyno_testing
