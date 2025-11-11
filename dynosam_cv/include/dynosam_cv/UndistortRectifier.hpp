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

/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   UndistorterRectifier.h
 * @brief  Class to undistort (and rectify) images.
 * @author Antoni Rosinol
 * @author Marcus Abate
 */

#pragma once

#include <gtsam/geometry/Point3.h>

#include <opencv4/opencv2/calib3d.hpp>
#include <opencv4/opencv2/core.hpp>
#include <optional>

#include "dynosam_common/Types.hpp"
#include "dynosam_cv/CameraParams.hpp"

namespace dyno {

/**
 * @brief The UndistorterRectifier class Computes undistortion maps and
 * undistorts on a per-image basis. Optionally, one can also apply image
 * rectification by providing a corresponding rotation matrix R.
 */
class UndistorterRectifier {
 public:
  DYNO_POINTER_TYPEDEFS(UndistorterRectifier)
  DYNO_DELETE_COPY_CONSTRUCTORS(UndistorterRectifier)

  /**
   * @brief UndistorterRectifier
   * @param P new projection matrix after stereo rectification (identity if no
   * stereo rectification needed). In the case of the monocular camera, P (new
   * camera matrix) is usually equal to K (the camera matrix). See:
   * https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga7dfb72c9cf9780a347fbe3d1c47e5d5a
   * P is normally set to P1 or P2 computed by cv::stereoRectify.
   * @param cam_params Camera Parameters.
   * @param R optional rotation matrix if you want to rectify image (aka apply
   * a rotation matrix) typically computed by cv::stereoRectify. If you have
   * a mono camera, you typically don't set this matrix.
   */
  UndistorterRectifier(const cv::Mat& P, const CameraParams& cam_params,
                       const cv::Mat& R = cv::Mat());
  virtual ~UndistorterRectifier() = default;

  /**
   * @brief undistortRectifyKeypoints undistorts and rectifies
   */
  static void UndistortRectifyKeypoints(const Keypoints& keypoints,
                                        Keypoints& undistorted_keypoints,
                                        const CameraParams& cam_params,
                                        std::optional<cv::Mat> P = {},
                                        std::optional<cv::Mat> R = {});

  /**
   * @brief UndistortKeypointAndGetVersor undistort a single pixel,
   * and return the corresponding versor.
   * (unit norm vector corresponding to bearing).
   * @param cv_px keypoint
   * @param cam_params CameraParams instance
   * @param R std::optional<cv::Mat> not needed in the monocular case
   */
  static gtsam::Vector3 UndistortKeypointAndGetVersor(
      const Keypoint& keypoint, const CameraParams& cam_params,
      std::optional<cv::Mat> R = {});

  /**
   * @brief Undistorts a single pixel and returns the corresponding versor (unit
   * norm vector corresponding to bearing). The rectifciation and new camera
   * matrix (R and P) are used from construction
   *
   * @param keypoint
   * @return gtsam::Vector3
   */
  gtsam::Vector3 undistortKeypointAndGetVersor(const Keypoint& keypoint) const;

  /**
   * @brief UndistortKeypointAndGetVersor undistort a single pixel,
   * and return the corresponding projected versor using the new camera matrix P
   * If P is not provided, the camera matrix, K is used.
   *
   * The projected version is calculated as K^{-1} * [u, v, 1] and then
   * normalized
   *
   * @param keypoint const Keypoints&
   * @param cam_params const CameraParams&
   * @param P std::optional<cv::Mat> new camera matrix used for the projections
   * @param R std::optional<cv::Mat> not needed in the monocular case
   * @return gtsam::Vector3
   */
  static gtsam::Vector3 UndistortKeypointAndGetProjectedVersor(
      const Keypoint& keypoint, const CameraParams& cam_params,
      std::optional<cv::Mat> P = {}, std::optional<cv::Mat> R = {});

  gtsam::Vector3 undistortKeypointAndGetProjectedVersor(
      const Keypoint& keypoint) const;

  /**
   * @brief undistortRectifyImage Given distorted (and optionally non-rectified)
   * image, returns a distortion-free rectified one.
   * @param img Distorted non-rectified input image
   * @param undistorted_img Undistorted Rectified output image
   */
  void undistortRectifyImage(const cv::Mat& img,
                             cv::Mat& undistorted_img) const;

  /**
   * @brief undistortRectifyKeypoints Undistorts and rectifies a sparse set of
   * keypoints (instead of a whole image), using OpenCV undistortPoints.
   *
   * OpenCV undistortPoints (this does not use the remap function, but an
   * iterative approach to find the undistorted keypoints...).
   * Check OpenCV documentation for details about this function.
   *
   * It uses the internal camera parameters, and the optional R_ and P_ matrices
   * as provided at construction (by cv::stereoRectify). If R_/P_ are identity
   * no rectification is performed and the resulting keypoints are in normalized
   * coordinates.
   * @param keypoints Distorted and unrectified keypoints
   * @param undistorted_keypoints Undistorted and rectified keypoints.
   */
  void undistortRectifyKeypoints(const Keypoints& keypoints,
                                 Keypoints& undistorted_keypoints) const;

  // void undistortRectifyKeypoints(FeatureContainer* features) const;

  //   void checkUndistortedRectifiedLeftKeypoints(
  //       const KeypointsCV& distorted_kpts,
  //       const KeypointsCV& undistorted_kpts,
  //       StatusKeypointsCV* status_kpts,
  //       // This tolerance is huge...
  //       const float& pixel_tolerance = 2.0f) const;

  //   void distortUnrectifyKeypoints(const StatusKeypointsCV&
  //   keypoints_rectified,
  //                                  KeypointsCV* keypoints_unrectified) const;

 protected:
  /**
   * @brief initUndistortRectifyMaps Initialize pixel to pixel maps for
   * undistortion and rectification. If rectification is not needed, as is the
   * case with a monocular camera, the identity matrix should be passed as R.
   *
   * The function computes the joint undistortion and rectification
   * transformation and represents the
   * result in the form of maps for remap. The undistorted image looks like
   * original, as if it is
   * captured with a camera using the camera matrix = P and zero
   * distortion. In case of a
   * monocular camera, P is usually equal to cameraMatrix, or it
   * can be computed by
   * cv::getOptimalNewCameraMatrix for a better control over scaling. In case of
   * a stereo camera, it is the output of stereo rectification
   * (cv::stereoRectify)
   *
   * NOTE: for stereo cameras, cam_params.P_ should be already computed using
   * cv::stereoRectify().
   *
   * @param cam_params Camera Parameters.
   * @param R optional rotation matrix if you want to rectify image (aka apply
   * a rotation matrix typically computed by cv::stereoRectify).
   * @param P new projection matrix after stereo rectification (identity if no
   * stereo rectification needed).
   * P is normally set to P1 or P2 computed by cv::stereoRectify.
   */
  void initUndistortRectifyMaps(const CameraParams& cam_params,
                                const cv::Mat& R, const cv::Mat& P,
                                cv::Mat* map_x, cv::Mat* map_y);

 protected:
  cv::Mat map_x_;
  cv::Mat map_y_;

  cv::Mat P_;
  cv::Mat R_;

  CameraParams cam_params_;

  static constexpr int kRectifyImageType = CV_32FC1;

  // Replicate instead of constant is more efficient for GPUs to calculate.
  bool remap_use_constant_border_type_ = false;
  int remap_interpolation_type_ = cv::INTER_LINEAR;
};

}  // namespace dyno
