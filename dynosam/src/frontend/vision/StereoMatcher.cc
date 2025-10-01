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

#include "dynosam/frontend/vision/StereoMatcher.hpp"

#include "dynosam_vision_common/ImageTypes.hpp"  //for depth image

namespace dyno {

StereoMatcher::StereoMatcher(StereoCamera::Ptr stereo_camera,
                             const StereoMatchingParams& stereo_param,
                             const DenseStereoParams& dense_stereo_params)
    : stereo_camera_(stereo_camera),
      stereo_matching_params_(stereo_param),
      dense_stereo_params_(dense_stereo_params) {
  CHECK_NOTNULL(stereo_camera);

  cv_stereo_matcher_ = constructStereoMatcher(
      stereo_camera_, stereo_matching_params_, dense_stereo_params_);
  CHECK_NOTNULL(cv_stereo_matcher_);
}

void StereoMatcher::denseStereoReconstruction(const cv::Mat& left_img,
                                              const cv::Mat& right_img,
                                              cv::Mat& depth_image) const {
  CHECK_EQ(right_img.cols, left_img.cols);
  CHECK_EQ(right_img.rows, left_img.rows);
  CHECK_EQ(right_img.type(), left_img.type());

  cv::Mat left_rectified, right_rectified;
  stereo_camera_->undistortRectifyImages(left_rectified, right_rectified,
                                         left_img, right_img);

  cv::Mat disparity_img;
  disparityRectifiedReconstruction(left_rectified, right_rectified,
                                   disparity_img);

  constexpr auto depth_type = ImageType::Depth::OpenCVType;
  depth_image = cv::Mat::zeros(disparity_img.size(), depth_type);

  // Check
  // https://github.com/opencv/opencv/blob/master/samples/cpp/stereo_match.cpp
  // only if output is CV_16S
  cv::Mat disparity_float;
  disparity_img.convertTo(disparity_float, CV_32F, 1.0f / 16.0f);
  // disp_img.convertTo(floatDisp, CV_32F, 1.0f / 16.0f);
  disparity_img = disparity_float;

  const auto baseline = stereo_camera_->getBaseline();
  const auto fx = stereo_camera_->getStereoCameraCalibration().fx();

  // Get depth from disparity
  for (int i = 0u; i < disparity_img.rows; i++) {
    // Loop over rows
    const float* disp_ptr = disparity_img.ptr<float>(i);
    double* depth_ptr = depth_image.ptr<double>(i);

    for (int j = 0u; j < disparity_img.cols; j++) {
      // Loop over cols
      const double depth = (baseline * fx) / static_cast<double>(disp_ptr[j]);
      *(depth_ptr + j) = depth;
    }
  }
}

void StereoMatcher::disparityRectifiedReconstruction(
    const cv::Mat& rectified_left_img, const cv::Mat& rectified_right_img,
    cv::Mat& disparity_image, cv::Mat* disparity_viz) const {
  CHECK_EQ(rectified_right_img.cols, rectified_left_img.cols);
  CHECK_EQ(rectified_right_img.rows, rectified_left_img.rows);
  CHECK_EQ(rectified_right_img.type(), rectified_left_img.type());

  // input expects 8bit single channel
  cv::Mat rectified_left_mono = ImageType::RGBMono::toMono(rectified_left_img);
  cv::Mat rectified_right_mono =
      ImageType::RGBMono::toMono(rectified_right_img);

  CHECK_EQ(rectified_left_mono.channels(), 1);
  CHECK_EQ(rectified_right_mono.channels(), 1);

  // initially computed as 16SC1
  cv_stereo_matcher_->compute(rectified_left_mono, rectified_right_mono,
                              disparity_image);
  CHECK_EQ(disparity_image.type(), CV_16S);

  if (dense_stereo_params_.median_blur_disparity_) {
    cv::medianBlur(disparity_image, disparity_image, 5);
  }

  if (disparity_viz) {
    utils::getDisparityVis(disparity_image, *disparity_viz);
  }
}

cv::Ptr<cv::StereoMatcher> StereoMatcher::constructStereoMatcher(
    StereoCamera::ConstPtr stereo_camera,
    const StereoMatchingParams& /*stereo_param*/,
    const DenseStereoParams& dense_stereo_params) {
  // Setup stereo matcher
  cv::Ptr<cv::StereoMatcher> cv_stereo_matcher;
  if (dense_stereo_params.use_sgbm_) {
    int mode;
    if (dense_stereo_params.use_mode_HH_) {
      LOG(INFO) << "Using mode = cv::StereoSGBM::MODE_HH for dense stereo "
                   "reconstruction";
      // mode = cv::StereoSGBM::MODE_HH;
      mode = cv::StereoSGBM::MODE_SGBM_3WAY;
    } else {
      LOG(INFO) << "Using mode = cv::StereoSGBM::MODE_SGBM for dense stereo "
                   "reconstruction";
      mode = cv::StereoSGBM::MODE_SGBM;
    }
    cv_stereo_matcher = cv::StereoSGBM::create(
        dense_stereo_params.min_disparity_,
        dense_stereo_params.num_disparities_,
        dense_stereo_params.sad_window_size_, dense_stereo_params.p1_,
        dense_stereo_params.p2_, dense_stereo_params.disp_12_max_diff_,
        dense_stereo_params.pre_filter_cap_,
        dense_stereo_params.uniqueness_ratio_,
        dense_stereo_params.speckle_window_size_,
        dense_stereo_params.speckle_range_, mode);
  } else {
    LOG(INFO) << "Using StereoBM for dense stereo reconstruction";
    cv::Ptr<cv::StereoBM> sbm =
        cv::StereoBM::create(dense_stereo_params.num_disparities_,
                             dense_stereo_params.sad_window_size_);

    sbm->setPreFilterType(dense_stereo_params.pre_filter_type_);
    sbm->setPreFilterSize(dense_stereo_params.pre_filter_size_);
    sbm->setPreFilterCap(dense_stereo_params.pre_filter_cap_);
    sbm->setMinDisparity(dense_stereo_params.min_disparity_);
    sbm->setTextureThreshold(dense_stereo_params.texture_threshold_);
    sbm->setUniquenessRatio(dense_stereo_params.uniqueness_ratio_);
    sbm->setSpeckleRange(dense_stereo_params.speckle_range_);
    sbm->setSpeckleWindowSize(dense_stereo_params.speckle_window_size_);
    const cv::Rect& roi1 = stereo_camera->getROI1();
    const cv::Rect& roi2 = stereo_camera->getROI2();
    if (!roi1.empty() && !roi2.empty()) {
      sbm->setROI1(roi1);
      sbm->setROI2(roi2);
    } else {
      LOG(WARNING) << "ROIs are empty.";
    }

    cv_stereo_matcher = sbm;
  }

  CHECK_NOTNULL(cv_stereo_matcher);
  return cv_stereo_matcher;
}

}  // namespace dyno
