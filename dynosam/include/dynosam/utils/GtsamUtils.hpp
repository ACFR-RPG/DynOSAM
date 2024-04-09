/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 */

#pragma once

#include "dynosam/utils/Numerical.hpp"

#include <optional>

#include <eigen3/Eigen/Core> //must be included before opencv
#include <gtsam/geometry/Unit3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/linear/Sampler.h>

#include <opencv4/opencv2/opencv.hpp>


template <typename T>
struct is_gtsam_factor : std::is_base_of<gtsam::Factor, T>
{
};

template <class T>
static constexpr bool is_gtsam_factor_v = is_gtsam_factor<T>::value;

template <class T>
using enable_if_gtsam_factor = std::enable_if_t<is_gtsam_factor_v<T>>;



namespace opengv {
typedef Eigen::Matrix<double, 3, 4> transformation_t;
}


namespace dyno
{
namespace utils
{

/**
 * @brief Check if two objects wrapped in std::optional are equal.
 * If both are std::nullopt, function will return true
 *
 * T MUST be a gtsam::Value type (ie. have gtsam::traits<T>::Equals) defined
 *
 * @tparam T a gtsam::Value concept type
 * @param a const std::optional<T>&
 * @param b const std::optional<T>&
 * @return true
 * @return false
 */
template<typename T>
bool equateGtsamOptionalValues(const std::optional<T>& a, const std::optional<T>& b) {
  if(a && b) {
    return gtsam::traits<T>::Equals(*a, *b);
  }

  if(!a && !b) {
    return true;
  }

  return false;
}

gtsam::Pose3 cvMatToGtsamPose3(const cv::Mat& H);
// Converts a rotation matrix and translation vector from opencv to gtsam
// pose3
gtsam::Pose3 cvMatsToGtsamPose3(const cv::Mat& R, const cv::Mat& T);

cv::Mat gtsamPose3ToCvMat(const gtsam::Pose3& pose);

/* ------------------------------------------------------------------------ */
// Converts a 3x3 rotation matrix from opencv to gtsam Rot3
gtsam::Rot3 cvMatToGtsamRot3(const cv::Mat& R);

// Converts a 3x1 OpenCV matrix to gtsam Point3
gtsam::Point3 cvMatToGtsamPoint3(const cv::Mat& cv_t);
cv::Mat gtsamPoint3ToCvMat(const gtsam::Point3& point);

/**
 * @brief Converts a vector of 16 elements listing the elements of a 4x4 3D pose
*  matrix by rows into a pose3 in gtsam
 *
 */
gtsam::Pose3 poseVectorToGtsamPose3(const std::vector<double>& vector_pose);

/**
 * @brief Converts a 3x3 (or 3xM, M > 3) camera matrix from opencv to gtsam::Cal3_S2
 *
 * @param M
 * @return gtsam::Cal3_S2
 */
gtsam::Cal3_S2 Cvmat2Cal3_S2(const cv::Mat& M);


/**
 * @brief  Converts a gtsam pose3 to a 3x3 rotation matrix and translation vector
 * in opencv format (note: the function only extracts R and t, without
 * changing them)
 *
 * @param pose const gtsam::Pose3&
 * @return std::pair<cv::Mat, cv::Mat>
 */
std::pair<cv::Mat, cv::Mat> Pose2cvmats(const gtsam::Pose3& pose);

cv::Mat gtsamMatrix3ToCvMat(const gtsam::Matrix3& rot);
cv::Mat gtsamVector3ToCvMat(const gtsam::Vector3& tran);
cv::Point3d gtsamVector3ToCvPoint3(const gtsam::Vector3& tran);

/**
 * @brief Perturbs a gtsam::Value type object via sampling a normal distribution using
 * the sigmas provided. The length of the sigmas must match the dimension of the value
 * as given by gtsam::traits<T>::dimension.
 *
 * @tparam T
 * @param t
 * @param sigmas
 * @param seed
 * @return T
 */
template<typename T>
T perturbWithNoise(const T& t, const gtsam::Vector& sigmas, int32_t seed = 42) {
  CHECK_EQ(gtsam::traits<T>::dimension, sigmas.size());
  //! Make static so that the generator (internal to the sampler) remains in memory during calls
  //! and that we actually get a random distribution
  static gtsam::Sampler sample(sigmas, seed);

  gtsam::Vector delta(sample.sample()); //delta should be the tangent vector
  return gtsam::traits<T>::Retract(t, delta);
}

template<typename T>
T perturbWithNoise(const T& t, double sigma, int32_t seed = 42) {
  gtsam::Vector sigmas = gtsam::Vector::Constant(gtsam::traits<T>::dimension, sigma);
  return perturbWithNoise<T>(t, sigmas, seed);
}


template <typename T=double>
inline gtsam::Point2 cvPointToGtsam(const cv::Point_<T>& point)
{
  return gtsam::Point2(static_cast<double>(point.x), static_cast<double>(point.y));
}

template <typename T=float>
gtsam::Point2Vector cvPointsToGtsam(const std::vector<cv::Point_<T>>& points) {
  gtsam::Point2Vector gtsam_points;
  for (const auto& p : points)
  {
    gtsam_points.push_back(cvPointToGtsam<T>(p));
  }
  return gtsam_points;
}

template<typename T=float>
inline cv::Point_<T> gtsamPointToCv(const gtsam::Point2& point) {
  return cv::Point_<T>(static_cast<T>(point(0)), static_cast<T>(point(1)));
}

template<typename T=float>
std::vector<cv::Point_<T>> gtsamPointsToCv(const gtsam::Point2Vector& points) {
  std::vector<cv::Point_<T>> cv_points;
  for(const auto& p : points) {
    cv_points.push_back(gtsamPointToCv<T>(p));
  }
  return cv_points;
}

//  converts an opengv transformation (3x4 [R t] matrix) to a gtsam::Pose3
gtsam::Pose3 openGvTfToGtsamPose3(const opengv::transformation_t& RT);


template <class T>
static bool getEstimateOfKey(const gtsam::Values& state, const gtsam::Key& key, T* estimate)
{
  if (state.exists(key))
  {
    *CHECK_NOTNULL(estimate) = state.at<T>(key);
    return true;
  }
  else
  {
    return false;
  }
}

inline bool saveNoiseModelAsUpperTriangular(std::ostream& os, const gtsam::noiseModel::Gaussian& noise_model) {
  const gtsam::Matrix info = noise_model.information();
  return saveMatrixAsUpperTriangular(os, info);
}


} //utils
} //dyno

#include "dynosam/utils/GtsamUtils-inl.hpp"
