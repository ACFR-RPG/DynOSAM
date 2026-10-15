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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>

#include <exception>

#include "dynosam_common/utils/GtsamUtils.hpp"
#include "dynosam_opt/FactorGraphTools.hpp"
#include "dynosam_opt/factors/LandmarkMotionPoseFactor.hpp"
#include "dynosam_opt/factors/LandmarkMotionTernaryFactor.hpp"
#include "dynosam_opt/factors/MotionBetweenFactor.hpp"
#include "dynosam_opt/factors/Pose3FlowProjectionFactor.h"
#include "dynosam_sensors/CameraParams.hpp"

// taking from dyno/test/internal/helpers.hpp
namespace dyno_testing {

static dyno::KeypointStatus makeStatusKeypointMeasurement(
    dyno::TrackletId tracklet_id, dyno::ObjectId object_id,
    dyno::FrameId frame_id, const dyno::Keypoint& keypoint = dyno::Keypoint(),
    double sigma = 2.0) {
  gtsam::Vector2 kp_sigmas;
  kp_sigmas << sigma, sigma;
  dyno::MeasurementWithCovariance<dyno::Keypoint> kp_measurement =
      dyno::MeasurementWithCovariance<dyno::Keypoint>::FromSigmas(keypoint,
                                                                  kp_sigmas);
  return dyno::KeypointStatus(kp_measurement, frame_id, 0.0, tracklet_id,
                              object_id, dyno::ReferenceFrame::LOCAL);
}

static dyno::CameraParams makeDefaultCameraParams() {
  dyno::CameraParams::IntrinsicsCoeffs intrinsics(4);
  dyno::CameraParams::DistortionCoeffs distortion(4);
  intrinsics.at(0) = 554.256;  // fx
  intrinsics.at(1) = 554.256;  // fy
  intrinsics.at(2) = 640 / 2;  // u0
  intrinsics.at(3) = 480 / 2;  // v0
  return dyno::CameraParams(intrinsics, distortion, cv::Size(640, 480),
                            "radtan");
}

}  // namespace dyno_testing

using namespace dyno;

TEST(LandmarkMotionPoseFactor, visualiseJacobiansWithNonZeros) {
  gtsam::Pose3 L1(gtsam::Rot3::Rodrigues(-0.1, 0.2, 0.25),
                  gtsam::Point3(0.05, -0.10, 0.20));

  gtsam::Pose3 L2(gtsam::Rot3::Rodrigues(0.3, 0.2, -0.5),
                  gtsam::Point3(0.5, -0.15, 0.1));

  gtsam::Point3 p1(0.1, 2, 4);
  gtsam::Point3 p2(0.2, 3, 2);

  auto object_pose_k_1_key = ObjectPoseSymbol(0, 0);
  auto object_pose_k_key = ObjectPoseSymbol(0, 1);

  auto object_point_key_k_1 = DynamicLandmarkSymbol(0, 1);
  auto object_point_key_k = DynamicLandmarkSymbol(1, 1);

  LOG(INFO) << (std::string)object_point_key_k_1;

  gtsam::Values values;
  values.insert(object_pose_k_1_key, L1);
  values.insert(object_pose_k_key, L2);
  values.insert(object_point_key_k_1, p1);
  values.insert(object_point_key_k, p2);

  auto landmark_motion_noise = gtsam::noiseModel::Isotropic::Sigma(3u, 0.1);

  gtsam::NonlinearFactorGraph graph;
  graph.emplace_shared<LandmarkMotionPoseFactor>(
      object_point_key_k_1, object_point_key_k, object_pose_k_1_key,
      object_pose_k_key, landmark_motion_noise);

  NonlinearFactorGraphManager nlfgm(graph, values);

  cv::Mat block_jacobians = nlfgm.drawBlockJacobian(
      gtsam::Ordering::OrderingType::COLAMD,
      factor_graph_tools::DrawBlockJacobiansOptions::makeDynoSamOptions());

  // cv::imshow("LandmarkMotionPoseFactor block jacobians", block_jacobians);
  // cv::waitKey(0);
}

TEST(Pose3FlowProjectionFactor, testJacobians) {
  gtsam::Pose3 previous_pose =
      utils::createRandomAroundIdentity<gtsam::Pose3>(0.4);
  static gtsam::Pose3 kDeltaPose(gtsam::Rot3::Rodrigues(-0.1, 0.2, 0.25),
                                 gtsam::Point3(0.05, -0.10, 0.20));
  gtsam::Pose3 current_pose = previous_pose * kDeltaPose;

  gtsam::Point2 kp(1.2, 2.4);
  double depth = 0.5;
  gtsam::Point2 flow(0.1, -0.3);

  auto noise = gtsam::noiseModel::Isotropic::Sigma(2u, 0.1);

  auto camera_params = dyno_testing::makeDefaultCameraParams();
  gtsam::Cal3_S2 calibration =
      camera_params.constructGtsamCalibration<gtsam::Cal3_S2>();

  Pose3FlowProjectionFactor<gtsam::Cal3_S2> factor(
      0, 1, kp, depth, previous_pose, calibration, noise);

  gtsam::Matrix H1, H2;
  gtsam::Vector error = factor.evaluateError(flow, current_pose, H1, H2);

  // now do numerical jacobians
  gtsam::Matrix numerical_H1 =
      gtsam::numericalDerivative21<gtsam::Vector2, gtsam::Point2, gtsam::Pose3>(
          std::bind(&Pose3FlowProjectionFactor<gtsam::Cal3_S2>::evaluateError,
                    &factor, std::placeholders::_1, std::placeholders::_2,
                    boost::none, boost::none),
          flow, current_pose);

  gtsam::Matrix numerical_H2 =
      gtsam::numericalDerivative22<gtsam::Vector2, gtsam::Point2, gtsam::Pose3>(
          std::bind(&Pose3FlowProjectionFactor<gtsam::Cal3_S2>::evaluateError,
                    &factor, std::placeholders::_1, std::placeholders::_2,
                    boost::none, boost::none),
          flow, current_pose);

  EXPECT_TRUE(gtsam::assert_equal(H1, numerical_H1, 1e-4));
  EXPECT_TRUE(gtsam::assert_equal(H2, numerical_H2, 1e-4));
}

TEST(LandmarkMotionTernaryFactor, testJacobians) {
  gtsam::Pose3 H(gtsam::Rot3::Rodrigues(-0.1, 0.2, 0.25),
                 gtsam::Point3(0.05, -0.10, 0.20));
  gtsam::Pose3 HPerturbed = utils::perturbWithNoise<gtsam::Pose3>(H, 0.3);

  gtsam::Point3 P1(0.4, 1.0, 0.8);
  gtsam::Point3 P2 = H * P1;

  auto noise = gtsam::noiseModel::Isotropic::Sigma(3u, 0.1);

  LandmarkMotionTernaryFactor factor(0, 1, 2, noise);

  gtsam::Matrix H1, H2, H3;
  gtsam::Vector error = factor.evaluateError(P1, P2, HPerturbed, H1, H2, H3);

  // now do numerical jacobians
  gtsam::Matrix numerical_H1 =
      gtsam::numericalDerivative31<gtsam::Vector3, gtsam::Point3, gtsam::Point3,
                                   gtsam::Pose3>(
          std::bind(&LandmarkMotionTernaryFactor::evaluateError, &factor,
                    std::placeholders::_1, std::placeholders::_2,
                    std::placeholders::_3, boost::none, boost::none,
                    boost::none),
          P1, P2, HPerturbed);

  gtsam::Matrix numerical_H2 =
      gtsam::numericalDerivative32<gtsam::Vector3, gtsam::Point3, gtsam::Point3,
                                   gtsam::Pose3>(
          std::bind(&LandmarkMotionTernaryFactor::evaluateError, &factor,
                    std::placeholders::_1, std::placeholders::_2,
                    std::placeholders::_3, boost::none, boost::none,
                    boost::none),
          P1, P2, HPerturbed);

  gtsam::Matrix numerical_H3 =
      gtsam::numericalDerivative33<gtsam::Vector3, gtsam::Point3, gtsam::Point3,
                                   gtsam::Pose3>(
          std::bind(&LandmarkMotionTernaryFactor::evaluateError, &factor,
                    std::placeholders::_1, std::placeholders::_2,
                    std::placeholders::_3, boost::none, boost::none,
                    boost::none),
          P1, P2, HPerturbed);

  EXPECT_TRUE(gtsam::assert_equal(H1, numerical_H1));
  EXPECT_TRUE(gtsam::assert_equal(H2, numerical_H2));
  EXPECT_TRUE(gtsam::assert_equal(H3, numerical_H3));
}

TEST(LandmarkMotionTernaryFactor, testZeroError) {
  gtsam::Pose3 H(gtsam::Rot3::Rodrigues(-0.1, 0.2, 0.25),
                 gtsam::Point3(0.05, -0.10, 0.20));

  gtsam::Point3 P1(0.4, 1.0, 0.8);
  gtsam::Point3 P2 = H * P1;

  auto noise = gtsam::noiseModel::Isotropic::Sigma(3u, 0.1);

  LandmarkMotionTernaryFactor factor(0, 1, 2, noise);

  gtsam::Matrix H1, H2, H3;
  gtsam::Vector error = factor.evaluateError(P1, P2, H, H1, H2, H3);
  EXPECT_TRUE(gtsam::assert_equal(gtsam::Point3(0, 0, 0), error, 1e-4));
}

gtsam::Point3 perturbCameraAndMotion(const gtsam::Point3& point_l,
                                     const gtsam::Pose3& L_e,
                                     gtsam::Pose3& motion, gtsam::Pose3& pose,
                                     double sigma = 0.2) {
  motion = utils::perturbWithNoise(motion, sigma);
  pose = utils::perturbWithNoise(pose, sigma);
  auto result = pose.inverse() * motion * L_e * (point_l);
  return result;
}

TEST(MotionBetweenFactor, ZeroError) {
  using namespace gtsam;
  Pose3 Xi = utils::createRandomAroundIdentity<gtsam::Pose3>(2.0);
  Pose3 Z = utils::createRandomAroundIdentity<gtsam::Pose3>(3.0);

  // Construct Xj so constraint is EXACTLY satisfied
  Pose3 Xj = Z.compose(Xi);

  auto model = noiseModel::Isotropic::Sigma(6, 1e-6);
  MotionBetweenFactor<Pose3> factor(0, 1, Z, model);

  Vector error = factor.evaluateError(Xi, Xj);

  EXPECT_TRUE(gtsam::assert_equal(error, Z_6x1, 1e-6));
}

TEST(MotionBetweenFactor, NotSameAsStandardBetween) {
  using namespace gtsam;
  Pose3 Xi = utils::createRandomAroundIdentity<gtsam::Pose3>(1.0);
  Pose3 Z = utils::createRandomAroundIdentity<gtsam::Pose3>(2.5);

  // For RIGHT factor: Xj = Z * Xi
  Pose3 Xj = Z.compose(Xi);

  auto model = noiseModel::Isotropic::Sigma(6, 1.0);

  MotionBetweenFactor<Pose3> rightFactor(0, 1, Z, model);
  BetweenFactor<Pose3> leftFactor(0, 1, Z, model);

  Vector err_right = rightFactor.evaluateError(Xi, Xj);
  Vector err_left = leftFactor.evaluateError(Xi, Xj);

  // Right factor should be zero
  EXPECT_TRUE(gtsam::assert_equal(err_right, Z_6x1, 1e-6));

  // Standard between SHOULD NOT be zero
  EXPECT_GT(err_left.norm(), 1e-3);
}

TEST(MotionBetweenFactor, InverseConsistency) {
  using namespace gtsam;
  Pose3 Xi = utils::createRandomAroundIdentity<gtsam::Pose3>(4.20);
  Pose3 Z = utils::createRandomAroundIdentity<gtsam::Pose3>(8.1);

  Pose3 Xj = Z.compose(Xi);

  auto model = noiseModel::Unit::Create(6);

  MotionBetweenFactor<Pose3> factor(0, 1, Z, model);

  Vector err = factor.evaluateError(Xi, Xj);
  EXPECT_NEAR(err.norm(), 0.0, 1e-9);

  // Flip roles
  MotionBetweenFactor<Pose3> factor_inv(1, 0, Z.inverse(), model);

  Vector err_inv = factor_inv.evaluateError(Xj, Xi);
  EXPECT_TRUE(gtsam::assert_equal(err_inv, Z_6x1, 1e-6));
}

TEST(MotionBetweenFactor, Jacobians) {
  using namespace gtsam;
  Pose3 Xi = utils::createRandomAroundIdentity<gtsam::Pose3>(4.20);
  Pose3 Z = utils::createRandomAroundIdentity<gtsam::Pose3>(8.1);
  Pose3 Xj = utils::createRandomAroundIdentity<gtsam::Pose3>(2.1);

  auto model = noiseModel::Unit::Create(6);
  MotionBetweenFactor<Pose3> factor(0, 1, Z, model);

  Matrix H_i, H_j;
  Vector err = factor.evaluateError(Xi, Xj, H_i, H_j);

  // Numerical derivatives
  Matrix H_i_num = numericalDerivative21<Vector, Pose3, Pose3>(
      [&](const Pose3& Xi_, const Pose3& Xj_) {
        return factor.evaluateError(Xi_, Xj_);
      },
      Xi, Xj);

  Matrix H_j_num = numericalDerivative22<Vector, Pose3, Pose3>(
      [&](const Pose3& Xi_, const Pose3& Xj_) {
        return factor.evaluateError(Xi_, Xj_);
      },
      Xi, Xj);

  EXPECT_TRUE(assert_equal(H_i, H_i_num, 1e-6));
  EXPECT_TRUE(assert_equal(H_j, H_j_num, 1e-6));
}

TEST(MotionBetweenFactor, MatchesManualError) {
  using namespace gtsam;
  Pose3 Xi = utils::createRandomAroundIdentity<gtsam::Pose3>(1.20);
  Pose3 Z = utils::createRandomAroundIdentity<gtsam::Pose3>(0.2);
  Pose3 Xj = utils::createRandomAroundIdentity<gtsam::Pose3>(6);

  auto model = noiseModel::Unit::Create(6);
  MotionBetweenFactor<Pose3> factor(0, 1, Z, model);

  Vector err = factor.evaluateError(Xi, Xj);

  // Manual computation
  Pose3 predicted = Xj.compose(Xi.inverse());
  Vector err_manual = Z.localCoordinates(predicted);

  EXPECT_TRUE(assert_equal(err, err_manual, 1e-9));
}

// TEST(SmartMotionFactor, computeTriangulationBasic) {
//    using SmartFactor = dyno::SmartMotionFactor<3, gtsam::Pose3,
//    gtsam::Pose3>; gtsam::Pose3 pose1 = gtsam::Pose3::Identity();
//   gtsam::Pose3 pose2(gtsam::Rot3::Rodrigues(-0.1, 0.2, 0.25),
//                     gtsam::Point3(0.05, -0.10, 0.20));
//   gtsam::Pose3 L_e = utils::createRandomAroundIdentity<gtsam::Pose3>(2.0);
//   // gtsam::Pose3 motion1 = gtsam::Pose3::Identity();
//   // gtsam::Pose3 motion2 =
//   utils::createRandomAroundIdentity<gtsam::Pose3>(3.0); gtsam::Point3
//   point(3.0, 0, 1.2); gtsam::Values values; gtsam::Values values_perturb;

//   double sigma = 0.001;

//    SmartFactor::shared_ptr factor(new SmartFactor(
//       L_e, gtsam::noiseModel::Isotropic::Sigma(3, sigma)));

//   size_t NumObs = 20;
//   for(size_t i =0; i < NumObs; i++) {
//     gtsam::Key pose_key(2 * i);
//    gtsam::Key motion_key(2*i+1);
//    gtsam::Pose3 cam_pose;
//    gtsam::Pose3 object_motion;
//     if(i == 0) {
//         cam_pose = gtsam::Pose3::Identity();
//         object_motion = gtsam::Pose3::Identity();
//     }
//     else {
//       gtsam::Key last_pose_key(2 * (i-1));
//       gtsam::Pose3 last_cam_pose = values.at<gtsam::Pose3>(last_pose_key);

//       gtsam::Pose3 relative_cam_motion =
//       utils::createRandomAroundIdentity<gtsam::Pose3>(3.0); cam_pose =
//       last_cam_pose * relative_cam_motion;

//       object_motion = utils::createRandomAroundIdentity<gtsam::Pose3>(6.0);
//     }

//     values.insert(pose_key, cam_pose);
//     values.insert(motion_key, object_motion);

//     //this will pereturb the motion and pose
//     gtsam::Point3 measurement =
//       perturbCameraAndMotion(point, L_e, object_motion, cam_pose, sigma);

//     values_perturb.insert(pose_key, cam_pose);
//     values_perturb.insert(motion_key, object_motion);

//     factor->add(measurement, motion_key, pose_key);

//   }
// perturbCameraAndMotion
//   // gtsam::Point3 measurement1 =
//   //     perturbCameraAndMotion(point, L_e, motion1, pose1, sigma);
//   // gtsam::Point3 measurement2 =
//   //     perturbCameraAndMotion(point, L_e, motion2, pose2, sigma);

//   // SmartFactor::shared_ptr factor(new SmartFactor(
//   //     L_e, gtsam::noiseModel::Isotropic::Sigma(3, sigma)));
//   // EXPECT_FALSE(factor->point());

//   //  gtsam::Key pose_key1(1);
//   // gtsam::Key motion_key1(2);

//   // gtsam::Key pose_key2(3);
//   // gtsam::Key motion_key2(4);

//   // gtsam::Values values;
//   // values.insert(pose_key1, pose1);
//   // values.insert(motion_key1, motion1);
//   // values.insert(pose_key2, pose2);
//   // values.insert(motion_key2, motion2);

//   //should now be degenerate
//   // factor->add(measurement1, motion_key1, pose_key1);
//   // EXPECT_FALSE(factor->point(values));
//   // EXPECT_TRUE(factor->point(values).degenerate());

//   // factor->add(measurement2, motion_key2, pose_key2);

//   //given erroneous measurements but accurayte motion/pose
//   EXPECT_TRUE(factor->point(values));

//   LOG(INFO) << "Triangulated point = " << factor->point();
//   EXPECT_TRUE(gtsam::assert_equal(point, factor->point().value(), 1E-5));
// }

// TEST(SmartMotionFactor, LostTriangulation3D) {

//   gtsam::Pose3 pose1 = gtsam::Pose3::Identity();
//   gtsam::Pose3 pose2(gtsam::Rot3::Rodrigues(-0.1, 0.2, 0.25),
//                      gtsam::Point3(0.05, -0.10, 0.20));

//   gtsam::Point3 point(3.0, 0, 1.2);
//   gtsam::Point3 measurement1 = pose1.inverse() * point;
//   gtsam::Point3 measurement2 = pose2.inverse() * point;

// }
