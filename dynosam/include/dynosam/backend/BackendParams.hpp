/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
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

#include <gflags/gflags.h>

#include <string>

// defined in BackendParams.cc

DECLARE_double(constant_object_motion_rotation_sigma);
DECLARE_double(constant_object_motion_translation_sigma);

DECLARE_double(motion_ternary_factor_noise_sigma);

DECLARE_double(odometry_rotation_sigma);
DECLARE_double(odometry_translation_sigma);

DECLARE_double(static_point_noise_sigma);
DECLARE_double(dynamic_point_noise_sigma);

DECLARE_double(static_pixel_noise_sigma);
DECLARE_double(dynamic_pixel_noise_sigma);

DECLARE_bool(use_smoothing_factor);
DECLARE_bool(use_robust_kernals);

DECLARE_bool(use_vo_factor);

DECLARE_bool(dynamic_point_noise_as_robust);

DECLARE_int32(optimization_mode);
DECLARE_int32(static_formulation_type);

DECLARE_string(updater_suffix);

DECLARE_int32(min_static_observations);
DECLARE_int32(min_dynamic_observations);

namespace dyno {

// TODO:(jesse) why is this here...?
enum RegularOptimizationType : int {
  FULL_BATCH = 0,
  SLIDING_WINDOW = 1,
  INCREMENTAL = 2
};

struct BackendParams {
  enum StaticFormulationType {
    PTP = 0,  //! Pose-to-point factor with 3D projection residual in the camera
              //! frame
    GENERIC_PROJECTION = 1,  //! Regular projection factor with 2D projection
                             //! residual on the image plane
    STEREO_PROJECTION = 2,
    SMART_PROJECTION = 3
  };

  //! RGBD/Stereo
  bool use_robust_kernals_ = FLAGS_use_robust_kernals;
  bool static_point_noise_as_robust = true;
  bool dynamic_point_noise_as_robust = FLAGS_dynamic_point_noise_as_robust;

  double k_huber_3d_points_ =
      0.0001;  //! Huber constant used for robust kernal on dynamic points
  double static_point_noise_sigma =
      FLAGS_static_point_noise_sigma;  //! Isotropic noise used on
                                       //! PoseToPointFactor for static points
  // TODO: make param!!! and really should come from covariance on image plane
  // and then projection!!!
  double dynamic_point_noise_sigma =
      FLAGS_dynamic_point_noise_sigma;  //! Isotropic noise used on
                                        //! PoseToPointFactor for dynamic points
                                        //! //0.0125

  double static_pixel_noise_sigma = FLAGS_static_pixel_noise_sigma;
  double dynamic_pixel_noise_sigma = FLAGS_dynamic_pixel_noise_sigma;

  double odometry_rotation_sigma_ =
      FLAGS_odometry_rotation_sigma;  //! sigma used to construct the noise
                                      //! model on the rotation component of the
                                      //! odomety (between factor)
  double odometry_translation_sigma_ =
      FLAGS_odometry_translation_sigma;  //! sigma used to construct the noise
                                         //! model on the translation component
                                         //! of the odomety (between factor)

  double constant_object_motion_rotation_sigma_ =
      FLAGS_constant_object_motion_rotation_sigma;
  double constant_object_motion_translation_sigma_ =
      FLAGS_constant_object_motion_translation_sigma;

  double motion_ternary_factor_noise_sigma_ =
      FLAGS_motion_ternary_factor_noise_sigma;

  bool use_logger_ = true;  // TODO: make param!?
  bool use_vo = FLAGS_use_vo_factor;

  RegularOptimizationType optimization_mode =
      static_cast<RegularOptimizationType>(FLAGS_optimization_mode);

  bool use_smoothing_factor = FLAGS_use_smoothing_factor;

  std::string updater_suffix = FLAGS_updater_suffix;

  size_t min_static_observations = FLAGS_min_static_observations;
  size_t min_dynamic_observations = FLAGS_min_dynamic_observations;

  StaticFormulationType static_formulation =
      static_cast<StaticFormulationType>(FLAGS_static_formulation_type);

  static BackendParams fromYaml(const std::string& file_path);

  /**
   * @brief True if robust kernals are on and the static point noise should also
   * be robust.
   *
   * @return true
   * @return false
   */
  bool makeStaticMeasurementsRobust() const {
    return use_robust_kernals_ && static_point_noise_as_robust;
  };

  /**
   * @brief True if robust kernals are on and the dynamic point noise should
   * also be robust.
   *
   * @return true
   * @return false
   */
  bool makeDynamicMeasurementsRobust() const {
    return use_robust_kernals_ && dynamic_point_noise_as_robust;
  };

  BackendParams& useLogger(bool value) {
    this->use_logger_ = value;
    return *this;
  }

  // for testing only
  mutable int full_batch_frame = -1;
};

}  // namespace dyno
