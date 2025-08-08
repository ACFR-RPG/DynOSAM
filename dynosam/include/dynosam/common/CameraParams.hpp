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

#pragma once

#include <config_utilities/config_utilities.h>
#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/geometry/Cal3Fisheye.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Pose3.h>

#include <eigen3/Eigen/Core>
#include <exception>
#include <opencv2/core/eigen.hpp>  //for cv::cv2eigen

#include "dynosam/common/Types.hpp"
#include "dynosam/utils/YamlParser.hpp"

namespace dyno {

class InvalidCameraCalibration : public std::runtime_error {
 public:
  InvalidCameraCalibration(const std::string& what)
      : std::runtime_error(what) {}
};

enum class DistortionModel { NONE, RADTAN, EQUIDISTANT, FISH_EYE };

class CameraParams {
 public:
  DYNO_POINTER_TYPEDEFS(CameraParams)

  using DistortionCoeffs = std::vector<double>;
  // fu, fv, cu, cv
  using IntrinsicsCoeffs = std::vector<double>;
  // eg pinhole
  using CameraModel = std::string;

  //! RGBD only parameters
  //! Modified from:
  //! https://github.com/MIT-SPARK/Kimera-VIO/blob/master/include/kimera-vio/frontend/CameraParams.h
  struct DepthParams {
    //! Whether or not the parameters were read
    bool valid = false;

    //! Virtual depth baseline: smaller means less disparity
    double virtual_baseline = 1.0e-2f;

    //! Conversion factor between raw depth measurements and meters
    double depth_to_meters = 1.0f;

    //! Minimum depth to convert
    double min_depth = 0.0f;

    //! Maximum depth to convert
    double max_depth = 10.0f;

    //! Whether or not the image is registered
    bool is_registered = true;

    //! Camera matrix for the depth image
    cv::Mat K;

    //! Extrinsic transform between the depth and rgb cameras
    cv::Mat T_color_depth;

    DepthParams(double baseline) : valid(true), virtual_baseline(baseline) {}
  };

  /**
   * @brief Needed for IO construction
   *
   */
  CameraParams() {}

  /**
   * @brief Constructs the parameters for a specific camera.
   *
   * Distortion model expects a string as either "none", "plumb_bob",
   * "radial-tangential", "radtan", "equidistant" or "kanna_brandt". The
   * corresponding DistortionModel (enum) will then be assigned.
   *
   * @param intrinsics const IntrinsicsCoeffs& Coefficients for the intrinsics
   * matrix. Should be in the form [fu fv cu cv]
   * @param distortion const DistortionCoeffs& Coefficients for the camera
   * distortion.
   * @param image_size const cv::Size& Image width and height
   * @param distortion_model_ const std::string& Expected distortion model to be
   * used for this camera.
   * @param T_robot_camera const gtsam::Pose3& The camera extrinsics describing
   * the transformation of the camera in the robot frame
   */
  CameraParams(const IntrinsicsCoeffs& intrinsics,
               const DistortionCoeffs& distortion, const cv::Size& image_size,
               const std::string& distortion_model,
               const gtsam::Pose3& T_robot_camera = gtsam::Pose3::Identity());

  CameraParams(const IntrinsicsCoeffs& intrinsics,
               const DistortionCoeffs& distortion, const cv::Size& image_size,
               const DistortionModel& distortion_model,
               const gtsam::Pose3& T_robot_camera = gtsam::Pose3::Identity());

  virtual ~CameraParams() = default;

  inline double fx() const { return intrinsics_[0]; }
  inline double fy() const { return intrinsics_[1]; }
  inline double cu() const { return intrinsics_[2]; }
  inline double cv() const { return intrinsics_[3]; }
  inline int ImageWidth() const { return image_size_.width; }
  inline int ImageHeight() const { return image_size_.height; }

  inline const cv::Size& imageSize() const { return image_size_; }

  inline cv::Mat getCameraMatrix() const { return K_; }

  inline gtsam::Matrix3 getCameraMatrixEigen() const { return K_eigen_; }

  inline cv::Mat getDistortionCoeffs() const { return D_; }

  /**
   * @brief Get the camera extrinsics - the transform of the camera in the body
   * frame (e.g T_robot_camera or ^RT_C)
   *
   * @return gtsam::Pose3
   */
  inline gtsam::Pose3 getExtrinsics() const { return T_robot_camera_; }

  inline DistortionModel getDistortionModel() const {
    return distortion_model_;
  }

  CameraParams& setDepthParams(const DepthParams& depth_params);

  CameraParams& setDepthParams(double baseline);
  bool hasDepthParams() const;
  const DepthParams& depthParams() const;

  static void convertDistortionVectorToMatrix(
      const DistortionCoeffs& distortion_coeffs,
      cv::Mat* distortion_coeffs_mat);

  /**
   * @brief Converts the IntrinsicsCoeffs (in the form: fu, fy, cu, cv) to a 3x3
   * camera matrix in the form: [fx  0 cu] K = [ 0 fy cv] [ 0  0  1]
   *
   * The camera matrix type is CV_64F
   *
   * @param intrinsics const IntrinsicsCoeffs&
   * @param camera_matrix cv::Mat*
   */
  static void convertIntrinsicsVectorToMatrix(
      const IntrinsicsCoeffs& intrinsics, cv::Mat* camera_matrix);

  /**
   * @brief Converts a 3x3 camera calibration matrix to an IntrinsicsCoeffs in
   * the form fu, fy, cu, cv. K is expected to a be of type CV_64F;
   *
   * @param K const cv::Mat&
   * @param intrinsics IntrinsicsCoeffs&
   */
  static void convertKMatrixToIntrinsicsCoeffs(const cv::Mat& K,
                                               IntrinsicsCoeffs& intrinsics);

  /** Taken from: https://github.com/ethz-asl/image_undistort
   * @brief stringToDistortion
   * @param distortion_model
   * @param camera_model
   * @return actual distortion model enum class
   */
  static DistortionModel stringToDistortion(const std::string& distortion_model,
                                            const std::string& camera_model);

  /**
   * @brief Converts the DistortionModel enum to a distortion model string and
   * associated camera model.
   *
   * The returned distortion model (e.g. "plumb_bob", "radtan") will match
   * distortion model's used in the ROS ecosystem which are defined in
   * sensor_msgs/distortion_models.hpp.
   *
   * Only pinhole camera models are currently supported.
   *
   * Out-args are only set when the function returns true.
   *
   * @param model const DistortionModel&
   * @param distortion_model  std::string&
   * @param camera_model std::string&
   * @return true
   * @return false
   */
  static bool distortionModelToString(const DistortionModel& model,
                                      std::string& distortion_model,
                                      std::string& camera_model);

  bool equals(const CameraParams& other, double tol = 1e-9) const;

  const std::string toString() const;

  // specalisations for gtsam::Cal3Fisheye and gtsam::Cal3DS2 are provided
  template <typename CALIBRATION>
  CALIBRATION constructGtsamCalibration() const;

 private:
  //! fu, fv, cu, cv
  IntrinsicsCoeffs intrinsics_;
  DistortionCoeffs distortion_coeff_;
  cv::Size image_size_;

  //! Distortion parameters
  DistortionModel distortion_model_;
  gtsam::Pose3
      T_robot_camera_;  //! Transform of the camera frame to the robot frame

  //! OpenCV structures: needed to compute the undistortion map.
  //! 3x3 camera matrix K (last row is {0,0,1})
  //! stored as a CV_64F (double)
  cv::Mat K_;
  gtsam::Matrix3 K_eigen_;

  //! New camera matrix constructed from
  //! estimateNewCameraMatrixForUndistortRectify
  //! for mono case not touched and initalised as K_
  cv::Mat P_;

  cv::Mat D_;

  //! Optional params for an RGB-D
  std::optional<DepthParams> depth_ = {};
};

void declare_config(CameraParams& config);

}  // namespace dyno
