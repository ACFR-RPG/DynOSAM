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

#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/StereoCamera.h>

#include "dynosam_common/Types.hpp"
#include "dynosam_cv/Camera.hpp"
#include "dynosam_cv/Feature.hpp"

namespace dyno {

using StereoCalibPtr = gtsam::Cal3_S2Stereo::shared_ptr;

class InvalidRGBDCameraParams : public DynosamException {
 public:
  InvalidRGBDCameraParams()
      : DynosamException(
            "Cannot construct RGBDCamera from a CameraParams that is missing "
            "depth "
            "information!") {}
};

class RGBDCamera : public Camera {
 public:
  RGBDCamera(const CameraParams& camera_params);

  double depthFromDisparity(double disparity) const;

  /**
   * @brief  Projects a feature with valid depth and a left keypoint (uL) into
   * the right keypoint (uR) of the feature data-structure
   *
   * @param feature Feature::Ptr
   * @return true
   * @return false
   */
  bool projectRight(Feature::Ptr feature) const;
  Keypoint rightKeypoint(double depth, const Keypoint& left_keypoint) const;
  double rightKeypoint(double depth, double uL) const;
  /**
   * @brief Get the stereo measurement for this feature.
   *
   * If the feature already has a right keypoint then the corresponding stereo
   * measurement is returned.
   *
   * If it does not, RGBDCamera#projectRight is used to construct the right
   * feature from depth and the resulting StereoPoint2 is returned. This will
   * also update the right keypoint in feature.
   *
   * If force_recalculation is true, the right keypoint will be recalculated
   * regardless
   *
   * @param feature
   * @param force_recalculation
   * @return std::pair<bool, gtsam::StereoPoint2>
   */
  std::pair<bool, gtsam::StereoPoint2> getStereo(
      Feature::Ptr feature, const bool force_recalculation = false);

  /**
   * @brief Get the stereo measurement for this feature.
   *
   * If the feature already has a right keypoint then the corresponding stereo
   * measurement is returned.
   *
   * If it does not, RGBDCamera#projectRight is used to construct the right
   * feature from depth and the resulting StereoPoint2 is returned. This will
   * also update the right keypoint in feature.
   *
   * If force_recalculation is true, the right keypoint will be recalculated
   * regardless
   *
   * @param feature
   * @param force_recalculation
   * @return std::pair<bool, gtsam::StereoPoint2>
   */
  std::pair<bool, gtsam::StereoPoint2> getStereo(
      Feature::Ptr feature, const bool force_recalculation = false);

  double fxb() const;
  Baseline baseline() const;

  /**
   * @brief Get gtsam::Cal3_S2Stereo from rgbd camera and virtual baseline
   */
  StereoCalibPtr getFakeStereoCalib() const;
  gtsam::StereoCamera getFakeStereoCamera() const;

 private:
  double fx_b_;
};

}  // namespace dyno
