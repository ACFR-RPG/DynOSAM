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

#pragma once

#include <gtsam/geometry/Pose3.h>

#include <vector>

#include "dynosam_common/utils/Numerical.hpp"

namespace dyno {
namespace utils {

/**
 * @brief Translation-Rotation error pair
 *
 */
struct TRErrorPair {
  double translation_ = 0;  //! meters
  double rot_ = 0;          //! rads

  TRErrorPair() = default;

  /**
   * @brief Construct a new TRErrorPair object
   *
   * @param translation Translation error in meters
   * @param rot Rotation error in radians
   */
  TRErrorPair(double translation, double rot);

  /**
   * @brief Get the rotation error in degrees
   *
   * @return double
   */
  double getRotationErrorDegrees() const;

  /**
   * @brief Calculates the pose error between two poses in the same frame
   *
   * E = M^{-1} M_hat
   *
   * t_e = L2 norm of the translation component of E
   * r_e = axis angle of E (trace of the rotation matrix of E)
   *
   * @param M const gtsam::Pose3& Estimated pose
   * @param M_hat const gtsam::Pose3& Ground Truth (comparison) pose
   * @return TRErrorPair
   */
  static TRErrorPair CalculatePoseError(const gtsam::Pose3& M,
                                        const gtsam::Pose3& M_hat);

  /**
   * @brief Calculates the relative pose error between poses.
   *
   * First calcualte the relative pose change between the reference and current
   * frames ie. ref_T_curr and ref_T_hat_curr and then take the pose error
   * between these two relative poses.
   *
   * The ref frame is usually the previous frame when analysing frame to frame
   * motion but is just used to calcualte the relative transformation
   *
   *
   * @param M_ref const gtsam::Pose3& Estimated pose at the reference frame
   * @param M_curr const gtsam::Pose3& Estimated pose at the current frame
   * @param M_hat_ref const gtsam::Pose3& Ground truth (comparison) pose at the
   * reference frame
   * @param M_hat_curr const gtsam::Pose3& Ground truth (comparison) pose at the
   * current frame
   * @return TRErrorPair
   */
  static TRErrorPair CalculateRelativePoseError(const gtsam::Pose3& M_ref,
                                                const gtsam::Pose3& M_curr,
                                                const gtsam::Pose3& M_hat_ref,
                                                const gtsam::Pose3& M_hat_curr);
};

struct TRErrorPairVector : public std::vector<TRErrorPair> {
  using Base = std::vector<TRErrorPair>;
  using Base::push_back;

  void push_back(double translation, double rot);

  // https://en.wikipedia.org/wiki/Circular_mean
  TRErrorPair average() const;
};

}  // namespace utils
}  // namespace dyno
