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

#include "dynosam_common/utils/GtsamUtils.hpp"

#include <glog/logging.h>

#include <eigen3/Eigen/Dense>
#include <opencv4/opencv2/core/eigen.hpp>

namespace dyno {
namespace utils {

// TODO: unit test
gtsam::Pose3 cvMatToGtsamPose3(const cv::Mat& H) {
  CHECK_EQ(H.rows, 4);
  CHECK_EQ(H.cols, 4);

  cv::Mat R(3, 3, H.type());
  cv::Mat T(3, 1, H.type());

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      R.at<double>(i, j) = H.at<double>(i, j);
    }
  }

  for (int i = 0; i < 3; i++) {
    T.at<double>(i, 0) = H.at<double>(i, 3);
  }

  return cvMatsToGtsamPose3(R, T);
}

gtsam::Pose3 cvMatsToGtsamPose3(const cv::Mat& R, const cv::Mat& T) {
  return gtsam::Pose3(cvMatToGtsamRot3(R), cvMatToGtsamPoint3(T));
}

cv::Mat gtsamPose3ToCvMat(const gtsam::Pose3& pose) {
  cv::Mat RT(4, 4, CV_64F);
  cv::eigen2cv(pose.matrix(), RT);
  RT.convertTo(RT, CV_64F);
  return RT;
}

gtsam::Rot3 cvMatToGtsamRot3(const cv::Mat& R) {
  CHECK_EQ(R.rows, 3);
  CHECK_EQ(R.cols, 3);
  gtsam::Matrix rot_mat = gtsam::Matrix::Identity(3, 3);
  cv::cv2eigen(R, rot_mat);
  return gtsam::Rot3(rot_mat);
}

gtsam::Point3 cvMatToGtsamPoint3(const cv::Mat& cv_t) {
  CHECK_EQ(cv_t.rows, 3);
  CHECK_EQ(cv_t.cols, 1);
  gtsam::Point3 gtsam_t;
  gtsam_t << cv_t.at<double>(0, 0), cv_t.at<double>(1, 0),
      cv_t.at<double>(2, 0);
  return gtsam_t;
}

cv::Mat gtsamPoint3ToCvMat(const gtsam::Point3& point) {
  cv::Mat T(3, 1, CV_32F);
  cv::eigen2cv(point, T);
  T.convertTo(T, CV_32F);
  return T.clone();
}

gtsam::Pose3 poseVectorToGtsamPose3(const std::vector<double>& vector_pose) {
  CHECK_EQ(vector_pose.size(), 16u);
  CHECK_EQ(vector_pose[12], 0.0);
  CHECK_EQ(vector_pose[13], 0.0);
  CHECK_EQ(vector_pose[14], 0.0);
  CHECK_EQ(vector_pose[15], 1.0);
  return gtsam::Pose3(Eigen::Matrix4d(vector_pose.data()).transpose());
}

gtsam::Cal3_S2 Cvmat2Cal3_S2(const cv::Mat& M) {
  CHECK_EQ(M.rows, 3);  // We expect homogeneous camera matrix.
  CHECK_GE(M.cols, 3);  // We accept extra columns (which we do not use).
  const double& fx = M.at<double>(0, 0);
  const double& fy = M.at<double>(1, 1);
  const double& s = M.at<double>(0, 1);
  const double& u0 = M.at<double>(0, 2);
  const double& v0 = M.at<double>(1, 2);
  return gtsam::Cal3_S2(fx, fy, s, u0, v0);
}

gtsam::Pose3 openGvTfToGtsamPose3(const opengv::transformation_t& RT) {
  gtsam::Matrix poseMat = gtsam::Matrix::Identity(4, 4);
  poseMat.block<3, 4>(0, 0) = RT;
  return gtsam::Pose3(poseMat);
}

std::pair<cv::Mat, cv::Mat> Pose2cvmats(const gtsam::Pose3& pose) {
  const gtsam::Matrix3& rot = pose.rotation().matrix();
  const gtsam::Vector3& tran = pose.translation();
  return std::make_pair(gtsamMatrix3ToCvMat(rot), gtsamVector3ToCvMat(tran));
}

// TODO(Toni): template this on type double, float etc.
cv::Mat gtsamMatrix3ToCvMat(const gtsam::Matrix3& rot) {
  cv::Mat R = cv::Mat(3, 3, CV_64F);
  cv::eigen2cv(rot, R);
  return R;
}

cv::Mat gtsamVector3ToCvMat(const gtsam::Vector3& tran) {
  cv::Mat T = cv::Mat(3, 1, CV_64F);
  cv::eigen2cv(tran, T);
  return T;
}

cv::Point3d gtsamVector3ToCvPoint3(const gtsam::Vector3& tran) {
  return cv::Point3d(tran[0], tran[1], tran[2]);
}

}  // namespace utils
}  // namespace dyno
