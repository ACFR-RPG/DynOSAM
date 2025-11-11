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

#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/StereoCamera.h>

#include <opencv4/opencv2/opencv.hpp>

#include "dynosam_cv/Camera.hpp"
#include "dynosam_cv/CameraParams.hpp"
#include "dynosam_cv/UndistortRectifier.hpp"

namespace dyno {

class StereoCamera {
 public:
  DYNO_POINTER_TYPEDEFS(StereoCamera)
  DYNO_DELETE_COPY_CONSTRUCTORS(StereoCamera)

  StereoCamera(Camera::ConstPtr left_camera, Camera::ConstPtr right_camera);

  StereoCamera(const CameraParams& left_cam_params,
               const CameraParams& right_cam_params);

  inline const Camera::ConstPtr& getOriginalLeftCamera() const {
    return original_left_camera_;
  }

  inline const Camera::ConstPtr& getOriginalRightCamera() const {
    return original_right_camera_;
  }

  inline gtsam::StereoCamera getUndistortedRectifiedStereoCamera() const {
    return undistorted_rectified_stereo_camera_impl_;
  }

  inline const gtsam::Cal3_S2Stereo& getStereoCameraCalibration() const {
    return undistorted_rectified_stereo_camera_impl_.calibration();
  }

  inline cv::Mat getP1() const { return P1_; }

  inline cv::Mat getP2() const { return P2_; }

  inline cv::Mat getR1() const { return R1_; }

  inline cv::Mat getR2() const { return R2_; }

  inline cv::Mat getQ() const { return Q_; }

  inline cv::Rect getROI1() const { return ROI1_; }

  inline cv::Rect getROI2() const { return ROI2_; }

  inline Baseline getBaseline() const { return stereo_baseline_; }

  void undistortRectifyImages(cv::Mat& rectify_left, cv::Mat& rectify_right,
                              const cv::Mat& left, const cv::Mat& right) const;

 private:
  /**
   * @brief computeRectificationParameters
   *
   * Outputs new rotation matrices R1,R2
   * so that the image planes of the stereo camera are parallel.
   * It also outputs new projection matrices P1, P2, and a disparity to depth
   * matrix for stereo pointcloud reconstruction.
   *
   * @param left_cam_params Left camera parameters
   * @param right_cam_params Right camera parameters
   *
   * @param R1 Output 3x3 rectification transform (rotation matrix) for the
   * first camera.
   * @param R2 Output 3x3 rectification transform (rotation matrix) for the
   * second camera.
   * @param P1 Output 3x4 projection matrix in the new (rectified) coordinate
   * systems for the first camera.
   * @param P2 Output 3x4 projection matrix in the new (rectified) coordinate
   * systems for the second camera.
   * @param Q Output \f$4 \times 4\f$ disparity-to-depth mapping matrix (see
   * reprojectImageTo3D ).
   * @param ROI1 Region of interest in image 1.
   * @param ROI2 Region of interest in image 2.
   */
  static void computeRectificationParameters(
      const CameraParams& left_cam_params, const CameraParams& right_cam_params,
      cv::Mat& R1, cv::Mat& R2, cv::Mat& P1, cv::Mat& P2, cv::Mat& Q,
      cv::Rect& ROI1, cv::Rect& ROI2);

  static void calculateBaseLine(Baseline& base_line, const cv::Mat& Q);

 private:
  Camera::ConstPtr original_left_camera_;
  Camera::ConstPtr original_right_camera_;

  //! Stereo camera implementation
  gtsam::StereoCamera undistorted_rectified_stereo_camera_impl_;

  //! Stereo camera calibration
  gtsam::Cal3_S2Stereo::shared_ptr stereo_calibration_;

  //! Undistortion rectification pre-computed maps for cv::remap
  UndistorterRectifier::UniquePtr left_cam_undistort_rectifier_;
  UndistorterRectifier::UniquePtr right_cam_undistort_rectifier_;

  // TODO(Toni): perhaps wrap these params in a struct instead.
  /// Projection matrices after rectification
  /// P1,P2 Output 3x4 projection matrix in the new (rectified) coordinate
  /// systems for the left and right camera (see cv::stereoRectify).
  cv::Mat P1_, P2_;

  /// R1,R2 Output 3x3 rectification transform (rotation matrix) for the left
  /// and for the right camera.
  cv::Mat R1_, R2_;

  /// Q Output 4x4 disparity-to-depth mapping matrix (see
  /// cv::reprojectImageTo3D or cv::stereoRectify).
  cv::Mat Q_;

  //! Regions of interest in the left/right image.
  cv::Rect ROI1_, ROI2_;

  //! Stereo baseline
  Baseline stereo_baseline_;
};

}  // namespace dyno
