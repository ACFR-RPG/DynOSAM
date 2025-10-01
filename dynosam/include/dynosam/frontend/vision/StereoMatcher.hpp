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

#include <glog/logging.h>

#include <opencv2/calib3d.hpp>

#include "dynosam/frontend/vision/StereoMatchingParams.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_vision_common/StereoCamera.hpp"

namespace dyno {

class StereoMatcher {
 public:
  DYNO_POINTER_TYPEDEFS(StereoMatcher)
  DYNO_DELETE_COPY_CONSTRUCTORS(StereoMatcher)

  StereoMatcher(
      StereoCamera::Ptr stereo_camera,
      const StereoMatchingParams& stereo_params = StereoMatchingParams(),
      const DenseStereoParams& dense_stereo_params = DenseStereoParams());

  virtual ~StereoMatcher() = default;

  /**
   * @brief Dense stereo reconstruction from two (distored, non-rectified)
   * images and produces a dense depth map.
   *
   * The left/right images must match the specifics from the L/R cameras as part
   * of the input stereo camera and the depth is is non-inverted and is in
   * meters. The depth mat uses CV_64F (double) data-type.
   *
   * @param left_img const cv::Mat&
   * @param right_img const cv::Mat&
   * @param depth_image cv::Mat&
   */
  void denseStereoReconstruction(const cv::Mat& left_img,
                                 const cv::Mat& right_img,
                                 cv::Mat& depth_image) const;

 protected:
  /**
   * @brief Dense stereo reconstruction from two (undistored, rectified) images
   * and produces a disparity map.
   *
   * This is the direct result computing the disparity from the
   * cv_stereo_matcher_. The disparity_image is computed as 16SC1.
   *
   * If a disparity_viz is provided (non-null), this will be set with a viz
   * image from utils::getDisparityVis
   *
   * @param left_img const cv::Mat&
   * @param right_img const cv::Mat&
   * @param disparity_image cv::Mat&
   * @param disparity_viz cv::Mat*
   */
  void disparityRectifiedReconstruction(const cv::Mat& rectified_left_img,
                                        const cv::Mat& rectified_right_img,
                                        cv::Mat& disparity_image,
                                        cv::Mat* disparity_viz = nullptr) const;

 private:
  static cv::Ptr<cv::StereoMatcher> constructStereoMatcher(
      StereoCamera::ConstPtr stereo_camera,
      const StereoMatchingParams& stereo_param,
      const DenseStereoParams& dense_stereo_params);

 protected:
  //! Stereo camera shared that might be shared across modules
  StereoCamera::ConstPtr stereo_camera_;

  //! Parameters for sparse stereo matching
  const StereoMatchingParams stereo_matching_params_;

  //! Parameters for dense stereo matching
  const DenseStereoParams dense_stereo_params_;

  //! Stereo matcher. Maybe unused
  cv::Ptr<cv::StereoMatcher> cv_stereo_matcher_;
};

}  // namespace dyno
