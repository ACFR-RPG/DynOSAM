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
#include <gtest/gtest.h>
#include <gtsam/geometry/StereoCamera.h>

#include <cmath>

#include "dynosam/frontend/vision/VisionTools.hpp"
#include "dynosam_common/Types.hpp"
#include "internal/helpers.hpp"

using namespace dyno;

TEST(VisionTools, determineOutlierIdsBasic) {
  TrackletIds tracklets = {1, 2, 3, 4, 5};
  TrackletIds inliers = {1, 2};

  TrackletIds expected_outliers = {3, 4, 5};
  TrackletIds outliers;
  determineOutlierIds(inliers, tracklets, outliers);
  EXPECT_EQ(expected_outliers, outliers);
}

TEST(VisionTools, determineOutlierIdsUnorderd) {
  TrackletIds tracklets = {12, 45, 1, 85, 3, 100};
  TrackletIds inliers = {3, 1, 100};

  TrackletIds expected_outliers = {12, 45, 85};
  TrackletIds outliers;
  determineOutlierIds(inliers, tracklets, outliers);
  EXPECT_EQ(expected_outliers, outliers);
}

TEST(VisionTools, determineOutlierIdsNoSubset) {
  TrackletIds tracklets = {12, 45, 1, 85, 3, 100};
  TrackletIds inliers = {12, 45, 1, 85, 3, 100};

  TrackletIds outliers = {4, 5, 6};  // also add a test that outliers is cleared
  determineOutlierIds(inliers, tracklets, outliers);
  EXPECT_TRUE(outliers.empty());
}

TEST(VisionTools, testMacVOUncertaintyPropogation) {
  Camera camera = dyno_testing::makeDefaultCamera();
  CameraParams params = camera.getParams();
  auto camera_impl = camera.getImplCamera();

  // make stereo camera
  const double base_line = 0.5;
  gtsam::Cal3_S2Stereo::shared_ptr stereo_params =
      boost::make_shared<gtsam::Cal3_S2Stereo>(
          params.fx(), params.fy(), 0, params.cu(), params.cv(), base_line);
  gtsam::StereoCamera stereo_camera(gtsam::Pose3::Identity(), stereo_params);

  Feature feature;
  Keypoint kp(params.cu(), params.cv() + 20);
  feature.keypoint(kp);
  feature.depth(1.0);

  // sigmas squared
  double kp_sigma_2 = 0.1;
  double depth_sigma_2 = 0.000005;

  // first check diagonal components of proposed macv matrix
  gtsam::Matrix32 J_keypoint;
  gtsam::Matrix31 J_depth;
  gtsam::Point3 landmark =
      camera_impl->backproject(feature.keypoint(), feature.depth(), boost::none,
                               J_keypoint, J_depth, boost::none);

  gtsam::StereoPoint2 stereo_kp = stereo_camera.project(landmark);
  EXPECT_EQ(kp(0), stereo_kp.uL());
  EXPECT_EQ(kp(1), stereo_kp.v());

  gtsam::Matrix33 J_stereo_point;
  stereo_camera.backproject2(stereo_kp, boost::none, J_stereo_point);

  gtsam::Point3 calc_landmark(
      ((feature.keypoint()(0) - params.cu()) * feature.depth()) / params.fx(),
      ((feature.keypoint()(1) - params.cv()) * feature.depth()) / params.fy(),
      feature.depth());
  EXPECT_TRUE(gtsam::assert_equal(calc_landmark, landmark));

  // form measurement covariance matrices
  gtsam::Matrix22 pixel_covariance_matrix;
  pixel_covariance_matrix << kp_sigma_2, 0.0, 0.0, kp_sigma_2;

  gtsam::Matrix33 stereo_pixel_covariance_matrix;
  stereo_pixel_covariance_matrix << kp_sigma_2, 0.0, 0.0, 0, kp_sigma_2, 0, 0,
      0, kp_sigma_2;

  // for depth uncertainty, we model it as a quadratic increase with distnace
  // double depth_covariance = depth_sigma * std::pow(depth, 2);
  double depth_covariance = depth_sigma_2;
  LOG(INFO) << "J_keypoint " << J_keypoint;
  // calcualte 3x3 covairance matrix
  gtsam::Matrix33 covariance =
      J_keypoint * pixel_covariance_matrix * J_keypoint.transpose();
  // J_depth * depth_covariance * J_depth.transpose();

  gtsam::Matrix33 stereo_covariance = J_stereo_point *
                                      stereo_pixel_covariance_matrix *
                                      J_stereo_point.transpose();

  LOG(INFO) << "Jacobian cov " << covariance;
  LOG(INFO) << "Stereo Jacobian cov " << stereo_covariance;

  double d_2 = std::pow(feature.depth(), 2);
  double u_2 = std::pow(feature.keypoint()(0), 2);
  double v_2 = std::pow(feature.keypoint()(1), 2);
  double fx_2 = std::pow(params.fx(), 2);
  double fy_2 = std::pow(params.fy(), 2);
  double cx_2 = std::pow(params.cu(), 2);
  double cy_2 = std::pow(params.cv(), 2);

  double mac_v_sigma_x = ((kp_sigma_2 + d_2) * (depth_sigma_2 + u_2) -
                          u_2 * d_2 + cx_2 * depth_sigma_2) /
                         fx_2;
  double mac_v_sigma_y = ((kp_sigma_2 + d_2) * (depth_sigma_2 + v_2) -
                          v_2 * d_2 + cy_2 * depth_sigma_2) /
                         fy_2;
  double mac_v_sigma_z = depth_sigma_2;

  gtsam::Matrix33 macvo_covariance;
  macvo_covariance << mac_v_sigma_x, 0, 0, 0, mac_v_sigma_y, 0, 0, 0,
      mac_v_sigma_z;

  LOG(INFO) << "macvo cov " << macvo_covariance;
}
