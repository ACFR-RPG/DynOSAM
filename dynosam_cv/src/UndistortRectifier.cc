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

#include "dynosam_cv/UndistortRectifier.hpp"

#include <eigen3/Eigen/Dense>
#include <opencv4/opencv2/calib3d.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/core/eigen.hpp>

#include "dynosam_common/utils/GtsamUtils.hpp"

namespace dyno {

UndistorterRectifier::UndistorterRectifier(const cv::Mat& P,
                                           const CameraParams& cam_params,
                                           const cv::Mat& R)
    : map_x_(), map_y_(), P_(P), R_(R), cam_params_(cam_params) {
  initUndistortRectifyMaps(cam_params, R, P, &map_x_, &map_y_);
}

void UndistorterRectifier::UndistortRectifyKeypoints(
    const Keypoints& keypoints, Keypoints& undistorted_keypoints,
    const CameraParams& cam_params, std::optional<cv::Mat> P,
    std::optional<cv::Mat> R) {
  const auto distortion_model = cam_params.getDistortionModel();
  const cv::Mat& distortion_matrix = cam_params.getDistortionCoeffs();
  const cv::Mat& camera_matrix = cam_params.getCameraMatrix();

  // but slow as we have to do all conversion back and forth
  // if the Eigen matrices are aligned we might not even need to do this...
  std::vector<cv::Point2f> keypoints_cv = utils::gtsamPointsToCv(keypoints);
  std::vector<cv::Point2f> undistorted_keypoints_cv;

  switch (distortion_model) {
    case DistortionModel::RADTAN: {
      cv::undistortPoints(keypoints_cv, undistorted_keypoints_cv, camera_matrix,
                          distortion_matrix, R ? R.value() : cv::noArray(),
                          P ? P.value() : cv::noArray());
    } break;
    case DistortionModel::EQUIDISTANT: {
      // TODO: Create unit test for fisheye / equidistant model
      cv::fisheye::undistortPoints(keypoints_cv, undistorted_keypoints_cv,
                                   camera_matrix, distortion_matrix,
                                   R ? R.value() : cv::noArray(),
                                   P ? P.value() : cv::noArray());
    } break;
    default: {
      LOG(FATAL) << "Unknown distortion model " << to_string(distortion_model)
                 << " when rectifying keypoints";
    }
  }

  undistorted_keypoints.clear();
  undistorted_keypoints = utils::cvPointsToGtsam(undistorted_keypoints_cv);
}

// NOTE: we don't pass P because we want normalized/canonical pixel
// coordinates (3D bearing vectors with last element = 1) for versors.
// If we were to pass P, it would convert back to pixel coordinates.
gtsam::Vector3 UndistorterRectifier::UndistortKeypointAndGetVersor(
    const Keypoint& keypoint, const CameraParams& cam_params,
    std::optional<cv::Mat> R) {
  Keypoints distorted_keypoint;
  distorted_keypoint.push_back(keypoint);

  Keypoints undistorted_keypoint;
  UndistorterRectifier::UndistortRectifyKeypoints(
      distorted_keypoint, undistorted_keypoint, cam_params, std::nullopt, R);

  // Transform to unit vector.
  gtsam::Vector3 versor(undistorted_keypoint.at(0)(0),
                        undistorted_keypoint.at(0)(1), 1.0);
  return versor.normalized();
}

gtsam::Vector3 UndistorterRectifier::undistortKeypointAndGetVersor(
    const Keypoint& keypoint) const {
  return UndistortKeypointAndGetVersor(keypoint, cam_params_, R_);
}

gtsam::Vector3 UndistorterRectifier::UndistortKeypointAndGetProjectedVersor(
    const Keypoint& keypoint, const CameraParams& cam_params,
    std::optional<cv::Mat> P, std::optional<cv::Mat> R) {
  Keypoints distorted_keypoint;
  distorted_keypoint.push_back(keypoint);

  Keypoints undistorted_keypoint;
  UndistorterRectifier::UndistortRectifyKeypoints(
      distorted_keypoint, undistorted_keypoint, cam_params, std::nullopt, R);

  gtsam::Vector3 versor(undistorted_keypoint.at(0)(0),
                        undistorted_keypoint.at(0)(1), 1.0);
  gtsam::Matrix K = gtsam::Matrix::Identity(3, 3);

  if (P) {
    // construct K from new camera matrix P
    cv::cv2eigen(*P, K);
  } else {
    cv::cv2eigen(cam_params.getCameraMatrix(), K);
  }

  return gtsam::Vector3(K.inverse() * versor).normalized();
}

gtsam::Vector3 UndistorterRectifier::undistortKeypointAndGetProjectedVersor(
    const Keypoint& keypoint) const {
  return UndistortKeypointAndGetProjectedVersor(keypoint, cam_params_, P_, R_);
}

void UndistorterRectifier::undistortRectifyImage(
    const cv::Mat& img, cv::Mat& undistorted_img) const {
  CHECK_EQ(map_x_.size, img.size);
  CHECK_EQ(map_y_.size, img.size);

  const auto original_image_type = img.type();
  // convert the input image into a CV type compatible with the rectify maps
  cv::Mat converted_img;
  img.copyTo(converted_img);
  converted_img.convertTo(converted_img, kRectifyImageType);

  cv::remap(converted_img, undistorted_img, map_x_, map_y_,
            remap_interpolation_type_,
            remap_use_constant_border_type_ ? cv::BORDER_CONSTANT
                                            : cv::BORDER_REPLICATE);

  // convert back to the original input image type
  undistorted_img.convertTo(undistorted_img, original_image_type);
}

void UndistorterRectifier::undistortRectifyKeypoints(
    const Keypoints& keypoints, Keypoints& undistorted_keypoints) const {
  undistorted_keypoints.clear();
  UndistorterRectifier::UndistortRectifyKeypoints(
      keypoints, undistorted_keypoints, cam_params_, R_, P_);
}

// void UndistorterRectifier::undistortRectifyKeypoints(FeatureContainer*
// features) const
// {
//   CHECK_NOTNULL(features);
//   Keypoints keypoints, undistorted_keypoints;

//   //fill keypoints
//   for (const auto& feature : *features) {
//     keypoints.push_back(feature->keypoint_);
//   }

//   UndistorterRectifier::UndistortRectifyKeypoints(
//       keypoints, undistorted_keypoints, cam_params_, R_, P_);

//   CHECK_EQ(undistorted_keypoints.size(), features->size());
//   //udpate keypoints
//   for
// }

void UndistorterRectifier::initUndistortRectifyMaps(
    const CameraParams& cam_params, const cv::Mat& R, const cv::Mat& P,
    cv::Mat* map_x, cv::Mat* map_y) {
  CHECK_NOTNULL(map_x);
  CHECK_NOTNULL(map_y);
  // static constexpr int kImageType = CV_32FC1;
  // static constexpr int kImageType = CV_16SC2;

  const cv::Size image_size = cam_params.imageSize();
  const cv::Mat& distortion_matrix = cam_params.getDistortionCoeffs();
  const cv::Mat& camera_matrix = cam_params.getCameraMatrix();
  const DistortionModel distortion_model = cam_params.getDistortionModel();

  cv::Mat map_x_float, map_y_float;
  switch (distortion_model) {
    case DistortionModel::NONE: {
      map_x_float.create(image_size, kRectifyImageType);
      map_y_float.create(image_size, kRectifyImageType);
    } break;
    case DistortionModel::RADTAN: {
      cv::initUndistortRectifyMap(
          // Input
          camera_matrix, distortion_matrix, R, P, image_size, kRectifyImageType,
          // Output:
          map_x_float, map_y_float);
    } break;
    // TODO: Jesse should this not also be for FISH_EYE model?
    case DistortionModel::EQUIDISTANT: {
      cv::fisheye::initUndistortRectifyMap(
          // Input,
          camera_matrix, distortion_matrix, R, P, image_size, kRectifyImageType,
          // Output:
          map_x_float, map_y_float);
    } break;
    default: {
      LOG(FATAL) << "Unknown distortion model: " << to_string(distortion_model)
                 << " when constructing undistortion maps";
    }
  }

  // TODO: (Jesse) experiment with what marcus was talking about
  //  TODO(marcus): can we add this in without causing errors like before?
  //  The reason we convert from floating to fixed-point representations
  //  of a map is that they can yield much faster (~2x) remapping operations.
  //  cv::convertMaps(map_x_float, map_y_float, *map_x, *map_y, CV_16SC2,
  //  false);

  *map_x = map_x_float;
  *map_y = map_y_float;
}

}  // namespace dyno
