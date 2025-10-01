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

#include "dynosam_vision_common/StereoCamera.hpp"

#include <glog/logging.h>

#include <Eigen/Core>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include "dynosam_common/utils/GtsamUtils.hpp"
#include "dynosam_common/utils/OpenCVUtils.hpp"

namespace dyno {

StereoCamera::StereoCamera(Camera::ConstPtr left_camera,
                           Camera::ConstPtr right_camera)
    : original_left_camera_(CHECK_NOTNULL(left_camera)),
      original_right_camera_(CHECK_NOTNULL(right_camera)),
      undistorted_rectified_stereo_camera_impl_(),
      stereo_calibration_(nullptr),
      left_cam_undistort_rectifier_(nullptr),
      right_cam_undistort_rectifier_(nullptr),
      stereo_baseline_(0.0) {
  computeRectificationParameters(left_camera->getParams(),
                                 right_camera->getParams(), R1_, R2_, P1_, P2_,
                                 Q_, ROI1_, ROI2_);
  calculateBaseLine(stereo_baseline_, Q_);

  //! Create stereo camera calibration after rectification and undistortion.
  const auto left_undist_rect_cam_mat = utils::Cvmat2Cal3_S2(P1_);
  stereo_calibration_.reset(new gtsam::Cal3_S2Stereo(
      left_undist_rect_cam_mat.fx(), left_undist_rect_cam_mat.fy(),
      left_undist_rect_cam_mat.skew(), left_undist_rect_cam_mat.px(),
      left_undist_rect_cam_mat.py(), stereo_baseline_));

  //! Create undistort rectifiers: these should be called after
  //! computeRectificationParameters.
  left_cam_undistort_rectifier_ = std::make_unique<UndistorterRectifier>(
      P1_, left_camera->getParams(), R1_);
  right_cam_undistort_rectifier_ = std::make_unique<UndistorterRectifier>(
      P2_, right_camera->getParams(), R2_);

  //! Create stereo camera implementation. As per all dynosam cameras, make
  //! camera in the local frame
  // TODO: is this right? In the original implementation (Kimera) we use the
  // rectified pose of the left image
  //  which SHOULD still be local!!
  undistorted_rectified_stereo_camera_impl_ =
      gtsam::StereoCamera(gtsam::Pose3::Identity(), stereo_calibration_);
}

StereoCamera::StereoCamera(const CameraParams& left_cam_params,
                           const CameraParams& right_cam_params)
    : StereoCamera(std::make_shared<Camera>(left_cam_params),
                   std::make_shared<Camera>(right_cam_params)) {}

void StereoCamera::undistortRectifyImages(cv::Mat& rectify_left,
                                          cv::Mat& rectify_right,
                                          const cv::Mat& left,
                                          const cv::Mat& right) const {
  left_cam_undistort_rectifier_->undistortRectifyImage(left, rectify_left);
  right_cam_undistort_rectifier_->undistortRectifyImage(right, rectify_right);
}

void StereoCamera::computeRectificationParameters(
    const CameraParams& left_cam_params, const CameraParams& right_cam_params,
    cv::Mat& R1, cv::Mat& R2, cv::Mat& P1, cv::Mat& P2, cv::Mat& Q,
    cv::Rect& ROI1, cv::Rect& ROI2) {
  // ! Extrinsics of the stereo (not rectified) relative pose between cameras
  gtsam::Pose3 camL_Pose_camR = (left_cam_params.getExtrinsics())
                                    .between(right_cam_params.getExtrinsics());

  LOG(INFO) << camL_Pose_camR;

  //! check extrinsics are not identity!
  CHECK(!camL_Pose_camR.equals(gtsam::Pose3::Identity()))
      << "Relative pose between cameras cannot be identity - camL_Pose_camR = "
      << camL_Pose_camR;

  // Get extrinsics in open CV format.
  // NOTE: openCV pose convention is the opposite, that's why we have to
  // invert
  cv::Mat camL_Rot_camR, camL_Tran_camR;
  std::tie(camL_Rot_camR, camL_Tran_camR) =
      utils::Pose2cvmats(camL_Pose_camR.inverse());

  LOG(INFO) << camL_Rot_camR;
  LOG(INFO) << camL_Tran_camR;

  const auto left_k = left_cam_params.getCameraMatrix();
  const auto left_d = left_cam_params.getDistortionCoeffs();

  const auto right_k = right_cam_params.getCameraMatrix();
  const auto right_d = right_cam_params.getDistortionCoeffs();

  const auto size = left_cam_params.imageSize();

  // kAlpha is -1 by default, but that introduces invalid keypoints!
  // here we should use kAlpha = 0 so we get only valid pixels...
  // But that has an issue that it removes large part of the image, check:
  // https://github.com/opencv/opencv/issues/7240 for this issue with kAlpha
  // Setting to -1 to make it easy, but it should NOT be -1!
  static constexpr int kAlpha = 0;
  switch (left_cam_params.getDistortionModel()) {
    case DistortionModel::RADTAN: {
      cv::stereoRectify(
          // Input
          left_k, left_d, right_k, right_d, size, camL_Rot_camR, camL_Tran_camR,
          // Output
          R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, kAlpha, cv::Size(),
          &ROI1, &ROI2);
    } break;
    case DistortionModel::EQUIDISTANT: {
      cv::fisheye::stereoRectify(
          // Input
          left_k, left_d, right_k, right_d, size, camL_Rot_camR, camL_Tran_camR,
          // Output
          R1, R2, P1, P2, Q,
          // TODO: Flag to maximise area???
          cv::CALIB_ZERO_DISPARITY);
    } break;
    default: {
      LOG(FATAL) << "Unknown DistortionModel: "
                 << to_string(left_cam_params.getDistortionModel());
    }
  }

  LOG(INFO) << Q;
}

void StereoCamera::calculateBaseLine(Baseline& base_line, const cv::Mat& Q) {
  // Calc baseline (see L.2700 and L.2616 in
  // https://github.com/opencv/opencv/blob/master/modules/calib3d/src/calibration.cpp
  // NOTE: OpenCV pose convention is the opposite, therefore the missing -1.0
  CHECK_NE(Q.at<double>(3, 2), 0.0);
  base_line = 1.0 / Q.at<double>(3, 2);
  CHECK_GT(base_line, 0.0);
}

}  // namespace dyno
