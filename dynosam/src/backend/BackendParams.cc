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

#include "dynosam/backend/BackendParams.hpp"

DEFINE_double(constant_object_motion_rotation_sigma, 0.01,
              "Noise used on rotation componenent of smoothing factor");
DEFINE_double(constant_object_motion_translation_sigma, 0.1,
              "Noise used on translation componenent of smoothing factor");

DEFINE_double(motion_ternary_factor_noise_sigma, 0.01,
              "Noise used on motion ternary factor");

DEFINE_double(odometry_rotation_sigma, 0.02,
              "Noise used on rotation component of odometry");
DEFINE_double(odometry_translation_sigma, 0.01,
              "Noise used on translation component of odometry");

DEFINE_int32(optimization_mode, 0,
             "0: Full-batch, 1: sliding-window, 2: incremental");

DEFINE_double(static_point_noise_sigma, 0.2,
              "Point (depth) noise for static points");
DEFINE_double(dynamic_point_noise_sigma, 0.2,
              "Point (depth) noise for dynamic points");

DEFINE_double(static_pixel_noise_sigma, 2.0,
              "Pixel noise used on static points");
DEFINE_double(dynamic_pixel_noise_sigma, 2.0,
              "Pixel noise used on dynamic points");

DEFINE_bool(use_smoothing_factor, true,
            "If the backend should use the smoothing factor between motions");

DEFINE_bool(use_backend_logger, true, "If the backend should use a logger");

DEFINE_bool(
    use_vo_factor, true,
    "If true, use visual odometry measurement as factor from the frontend");

DEFINE_bool(use_robust_kernals, true,
            "If the backend should use the robust noise kernals");
DEFINE_bool(
    dynamic_point_noise_as_robust, true,
    "If the backend should use the robust noise kernals on the Dynamic Points");

DEFINE_int32(min_static_observations, 2u,
             "Min number of observations of a point before it is added to the "
             "optimisation");
DEFINE_int32(min_dynamic_observations, 3u,
             "Min number of observations of a point before it is added to the "
             "optimisation");
