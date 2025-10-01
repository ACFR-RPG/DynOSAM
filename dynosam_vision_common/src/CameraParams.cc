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

#include "dynosam_vision_common/CameraParams.hpp"

#include <glog/logging.h>

#include <memory>

#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/GtsamUtils.hpp"   //for cv equals
#include "dynosam_common/utils/Numerical.hpp"    //for equals
#include "dynosam_common/utils/OpenCVUtils.hpp"  //for cv equals

namespace dyno {

template <>
std::string to_string(const DistortionModel& distortion) {
  switch (distortion) {
    case DistortionModel::NONE:
      return "None";
      break;
    case DistortionModel::RADTAN:
      return "Radtan";
      break;
    case DistortionModel::EQUIDISTANT:
      return "Equidistant";
      break;
    case DistortionModel::FISH_EYE:
      return "Fish-eye";
      break;
    default:
      return "Unknown distortion model";
      break;
  }
}

void declare_config(CameraParams& config) {
  using namespace config;
  name("Camera Params");

  cv::Size image_size;
  field(image_size, "resolution");

  // camera to robot pose
  std::vector<double> vector_pose;
  field(vector_pose, "T_BS");
  checkCondition(
      vector_pose.size() == 16u,
      "param 'T_BS' must be a 16 length vector in homogenous matrix form");
  gtsam::Pose3 T_robot_camera = utils::poseVectorToGtsamPose3(vector_pose);

  std::vector<double> intrinsics_v;
  field(intrinsics_v, "intrinsics");
  checkCondition(intrinsics_v.size() == 4,
                 "param 'intrinsics' must be a 4 length vector.");
  CameraParams::IntrinsicsCoeffs intrinsics;
  intrinsics.resize(4u);
  // Move elements from one to the other.
  std::copy_n(std::make_move_iterator(intrinsics_v.begin()), intrinsics.size(),
              intrinsics.begin());

  CameraParams::DistortionCoeffs distortion;
  field(distortion, "distortion_coefficients");

  std::string distortion_model, camera_model;
  field(distortion_model, "distortion_model");
  field(camera_model, "camera_model");

  DistortionModel model =
      CameraParams::stringToDistortion(distortion_model, camera_model);
  if (model == DistortionModel::NONE) {
    throw InvalidCameraCalibration(
        "Invalid distortion model/camera model combination when constructing "
        "camera params from yaml file - "
        " distortion model: " +
        distortion_model + ", camera model: " + camera_model);
  }

  config =
      CameraParams(intrinsics, distortion, image_size, model, T_robot_camera);
}

CameraParams::CameraParams(const IntrinsicsCoeffs& intrinsics,
                           const DistortionCoeffs& distortion,
                           const cv::Size& image_size,
                           const std::string& distortion_model,
                           const gtsam::Pose3& T_robot_camera)
    : CameraParams(
          intrinsics, distortion, image_size,
          CameraParams::stringToDistortion(distortion_model, "pinhole"),
          T_robot_camera) {}

CameraParams::CameraParams(const IntrinsicsCoeffs& intrinsics,
                           const DistortionCoeffs& distortion,
                           const cv::Size& image_size,
                           const DistortionModel& distortion_model,
                           const gtsam::Pose3& T_robot_camera)
    : intrinsics_(intrinsics),
      distortion_coeff_(distortion),
      image_size_(image_size),
      distortion_model_(distortion_model),
      T_robot_camera_(T_robot_camera) {
  CHECK_EQ(intrinsics_.size(), 4u)
      << "Intrinsics must be of length 4 - [fx fy cu cv]";
  CHECK_GT(distortion_coeff_.size(), 0u);

  CameraParams::convertDistortionVectorToMatrix(distortion_coeff_, &D_);
  CameraParams::convertIntrinsicsVectorToMatrix(intrinsics_, &K_);

  cv::cv2eigen(K_, K_eigen_);
  K_.copyTo(P_);
}

CameraParams& CameraParams::setDepthParams(const DepthParams& depth_params) {
  depth_ = depth_params;
  return *this;
}

CameraParams& CameraParams::setDepthParams(double baseline) {
  depth_ = DepthParams(baseline);
  return *this;
}

bool CameraParams::hasDepthParams() const { return depth_.has_value(); }

const CameraParams::DepthParams& CameraParams::depthParams() const {
  return depth_.value();
}

void CameraParams::convertDistortionVectorToMatrix(
    const DistortionCoeffs& distortion_coeffs, cv::Mat* distortion_coeffs_mat) {
  *distortion_coeffs_mat = cv::Mat::zeros(1, distortion_coeffs.size(), CV_64F);
  for (int k = 0; k < distortion_coeffs_mat->cols; k++) {
    distortion_coeffs_mat->at<double>(0, k) = distortion_coeffs[k];
  }
}

void CameraParams::convertIntrinsicsVectorToMatrix(
    const IntrinsicsCoeffs& intrinsics, cv::Mat* camera_matrix) {
  CHECK_GT(intrinsics.at(0), 0.0) << "fx cannot be zero";
  CHECK_GT(intrinsics.at(1), 0.0) << "fy cannot be zero";
  CHECK_GT(intrinsics.at(2), 0.0) << "cx cannot be zero";
  CHECK_GT(intrinsics.at(3), 0.0) << "cy cannot be zero";

  *camera_matrix = cv::Mat::eye(3, 3, CV_64F);
  camera_matrix->at<double>(0, 0) = intrinsics[0];
  camera_matrix->at<double>(1, 1) = intrinsics[1];
  camera_matrix->at<double>(0, 2) = intrinsics[2];
  camera_matrix->at<double>(1, 2) = intrinsics[3];
}

void CameraParams::convertKMatrixToIntrinsicsCoeffs(
    const cv::Mat& K, IntrinsicsCoeffs& intrinsics) {
  CHECK_EQ(K.type(), CV_64F) << "Cannot convert: Camera Matrix (K) is expected "
                                "to be of type CV_64F but is "
                             << utils::cvTypeToString(K.type());
  CHECK(K.rows == 3 && K.cols) << "Cannot convert: Camera Matrix (K) is "
                                  "expected to be of size (3 x 3) but is ["
                               << K.rows << " x " << K.cols << "]";
  ;

  intrinsics.resize(4);
  intrinsics[0] = K.at<double>(0, 0);
  intrinsics[1] = K.at<double>(1, 1);
  intrinsics[2] = K.at<double>(0, 2);
  intrinsics[3] = K.at<double>(1, 2);
}

DistortionModel CameraParams::stringToDistortion(
    const std::string& distortion_model, const std::string& camera_model) {
  std::string lower_case_distortion_model = distortion_model;
  std::string lower_case_camera_model = camera_model;

  std::transform(lower_case_distortion_model.begin(),
                 lower_case_distortion_model.end(),
                 lower_case_distortion_model.begin(), ::tolower);
  std::transform(lower_case_camera_model.begin(), lower_case_camera_model.end(),
                 lower_case_camera_model.begin(), ::tolower);

  if (lower_case_camera_model == "pinhole") {
    if (lower_case_distortion_model == "none") {
      return DistortionModel::NONE;
    } else if ((lower_case_distortion_model == "plumb_bob") ||
               (lower_case_distortion_model == "radial_tangential") ||
               (lower_case_distortion_model == "radtan")) {
      return DistortionModel::RADTAN;
    } else if (lower_case_distortion_model == "equidistant") {
      return DistortionModel::EQUIDISTANT;
    } else if (lower_case_distortion_model == "kannala_brandt") {
      return DistortionModel::FISH_EYE;
    } else {
      LOG(ERROR) << "Unrecognized distortion model for pinhole camera. Valid "
                    "pinhole distortion model options are 'none', 'radtan', "
                    "'equidistant', 'fish eye'.";
    }
  } else {
    LOG(ERROR)
        << "Unrecognized camera model. Valid camera models are 'pinhole'";
  }
  // Return no distortion model if invalid model is provided
  return DistortionModel::NONE;
}

bool CameraParams::distortionModelToString(const DistortionModel& model,
                                           std::string& distortion_model,
                                           std::string& camera_model) {
  switch (model) {
    case DistortionModel::NONE:
      return false;
    case DistortionModel::RADTAN:
      // in ROS only plumb_bob is defined so we use this
      distortion_model = "plumb_bob";
      camera_model = "pinhole";
      return true;
    case DistortionModel::EQUIDISTANT:
      distortion_model = "equidistant";
      camera_model = "pinhole";
      return true;
    case DistortionModel::FISH_EYE:
      distortion_model = "kannala_brandt";
      camera_model = "pinhole";
      return true;
    default:
      LOG(WARNING) << "Cannot convert distortion model to string: "
                   << to_string(model);
      return false;
  }
}

bool CameraParams::equals(const CameraParams& other, double tol) const {
  return dyno::equals_with_abs_tol(intrinsics_, other.intrinsics_, tol) &&
         dyno::equals_with_abs_tol(distortion_coeff_, other.distortion_coeff_,
                                   tol) &&
         (image_size_.width == other.image_size_.width &&
          image_size_.height == other.image_size_.height) &&
         distortion_model_ == other.distortion_model_ &&
         T_robot_camera_.equals(other.T_robot_camera_, tol) &&
         utils::compareCvMatsUpToTol(K_, other.K_, tol) &&
         utils::compareCvMatsUpToTol(
             D_, other.D_);  // NOTE: do not comapre with P matrix as currently
                             // not calculated
}

const std::string CameraParams::toString() const {
  std::stringstream out;
  out << "\nIntrinsics: \n- fx " << fx() << "\n- fy " << fy() << "\n- cu "
      << cu() << "\n- cv " << cv()
      << "\nimage_size: \n- width: " << ImageWidth()
      << "\n- height: " << ImageHeight() << "\n- K: " << K_ << '\n'
      << "- Distortion Model: " << to_string(distortion_model_) << '\n'
      << "- D: " << D_ << '\n'
      << "- P: " << P_ << '\n';

  return out.str();
}

template <>
gtsam::Cal3_S2 CameraParams::constructGtsamCalibration<gtsam::Cal3_S2>() const {
  static const auto requested_calibration_name =
      type_name<gtsam::Cal3_S2>();  // only used for debug so seems waste to
                                    // allocate everytime

  if (distortion_model_ != DistortionModel::RADTAN) {
    throw InvalidCameraCalibration(
        "Requested gtsam calibration was " + requested_calibration_name +
        " which is unsupported by this camera model: " +
        to_string(distortion_model_));
  }

  // if(distortion_coeff_.size() < 4u) {
  //   throw InvalidCameraCalibration("Distortion coefficients have size <4 for
  //   camera params with distortion model"
  //     + to_string(distortion_model_) + " and requested gtsam calibration of
  //     type " + requested_calibration_name);
  // }

  constexpr static double skew = 0.0;
  return gtsam::Cal3_S2(fx(), fy(), skew, cu(), cv());
}

template <>
gtsam::Cal3DS2 CameraParams::constructGtsamCalibration<gtsam::Cal3DS2>() const {
  static const auto requested_calibration_name =
      type_name<gtsam::Cal3DS2>();  // only used for debug so seems waste to
                                    // allocate everytime

  if (distortion_model_ != DistortionModel::RADTAN) {
    throw InvalidCameraCalibration(
        "Requested gtsam calibration was " + requested_calibration_name +
        " which is unsupported by this camera model: " +
        to_string(distortion_model_));
  }

  if (distortion_coeff_.size() < 4u) {
    throw InvalidCameraCalibration(
        "Distortion coefficients have size <4 for camera params with "
        "distortion model" +
        to_string(distortion_model_) +
        " and requested gtsam calibration of type " +
        requested_calibration_name);
  }

  constexpr static double skew = 0.0;
  return gtsam::Cal3DS2(fx(), fy(), skew, cu(), cv(), distortion_coeff_.at(0),
                        distortion_coeff_.at(1), distortion_coeff_.at(2),
                        distortion_coeff_.at(3));
}

template <>
gtsam::Cal3Fisheye CameraParams::constructGtsamCalibration<gtsam::Cal3Fisheye>()
    const {
  static const auto requested_calibration_name =
      type_name<gtsam::Cal3Fisheye>();  // only used for debug so seems waste to
                                        // allocate everytime

  // jesse: not sure if gtsam well supports equidistant models!
  if (distortion_model_ != DistortionModel::FISH_EYE &&
      distortion_model_ != DistortionModel::EQUIDISTANT) {
    throw InvalidCameraCalibration(
        "Requested gtsam calibration was " + requested_calibration_name +
        " which is unsupported by this camera model: " +
        to_string(distortion_model_));
  }

  if (distortion_coeff_.size() < 4u) {
    throw InvalidCameraCalibration(
        "Distortion coefficients have size <4 for camera params with "
        "distortion model" +
        to_string(distortion_model_) +
        " and requested gtsam calibration of type " +
        requested_calibration_name);
  }

  constexpr static double skew = 0.0;
  return gtsam::Cal3Fisheye(fx(), fy(), skew, cu(), cv(),
                            distortion_coeff_.at(0), distortion_coeff_.at(1),
                            distortion_coeff_.at(2), distortion_coeff_.at(3));
}

}  // namespace dyno

// specalising the printing function in the config utils namespace
namespace config {

template <>
std::string toString(const dyno::CameraParams& config, bool) {
  return config.toString();
}
}  // namespace config
