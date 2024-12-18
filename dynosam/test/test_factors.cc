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

#include <exception>

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/FactorGraphTools.hpp"
#include "dynosam/backend/rgbd/ObjectCentricEstimator.hpp"
#include "dynosam/factors/LandmarkMotionPoseFactor.hpp"
#include "dynosam/factors/LandmarkMotionTernaryFactor.hpp"
#include "dynosam/factors/Pose3FlowProjectionFactor.h"
#include "dynosam/utils/GtsamUtils.hpp"
#include "internal/helpers.hpp"

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

TEST(SmartMotionFactor, testZeroErrorWithIdenties) {
  using SmartFactor = dyno::SmartMotionFactor<3, gtsam::Pose3, gtsam::Pose3>;

  gtsam::Pose3 pose = gtsam::Pose3::Identity();
  gtsam::Pose3 motion = gtsam::Pose3::Identity();
  gtsam::Pose3 L_s = gtsam::Pose3::Identity();
  gtsam::Point3 point(1.0, 2.0, 3.0);
  gtsam::Point3 noise(0.0, 0.0, 0.0);
  gtsam::Point3 measured = point + noise;

  gtsam::Key pose_key(1);
  gtsam::Key motion_key(2);
  gtsam::Values values;
  values.insert(pose_key, pose);
  values.insert(motion_key, motion);

  SmartFactor factor(L_s, gtsam::noiseModel::Isotropic::Sigma(3, 0.05), point);
  factor.add(measured, motion_key, pose_key);

  gtsam::Vector expectedError = gtsam::Vector3(0.0, 0.0, 0);
  gtsam::Vector actualReproError = factor.reprojectionError(values);
  // Vector actualError = factor.unwhitenedError(values);
  EXPECT_TRUE(gtsam::assert_equal(expectedError, actualReproError, 1E-5));
}

TEST(SmartMotionFactor, testZeroErrorWithL0AndCamera) {
  using SmartFactor = dyno::SmartMotionFactor<3, gtsam::Pose3, gtsam::Pose3>;

  gtsam::Pose3 pose(gtsam::Rot3::Rodrigues(-0.1, 0.2, 0.25),
                    gtsam::Point3(0.05, -0.10, 0.20));
  gtsam::Pose3 motion = gtsam::Pose3::Identity();
  gtsam::Pose3 L_s = gtsam::Pose3(gtsam::Rot3::Ypr(-M_PI / 10, 0., -M_PI / 10),
                                  gtsam::Point3(0.5, 0.1, 0.3));
  gtsam::Point3 point_l(1.0, 2.0, 3.0);
  gtsam::Point3 noise(0.0, 0.0, 0.0);
  gtsam::Point3 measured_c = pose.inverse() * L_s * (point_l + noise);

  gtsam::Key pose_key(1);
  gtsam::Key motion_key(2);
  gtsam::Values values;
  values.insert(pose_key, pose);
  values.insert(motion_key, motion);

  SmartFactor factor(L_s, gtsam::noiseModel::Isotropic::Sigma(3, 0.05),
                     point_l);
  factor.add(measured_c, motion_key, pose_key);

  gtsam::Vector expectedError = gtsam::Vector3(0.0, 0.0, 0);
  gtsam::Vector actualReproError = factor.reprojectionError(values);
  // Vector actualError = factor.unwhitenedError(values);
  EXPECT_TRUE(gtsam::assert_equal(expectedError, actualReproError, 1E-5));
}

TEST(SmartMotionFactor, testNoiseWithIdentity) {
  using SmartFactor = dyno::SmartMotionFactor<3, gtsam::Pose3, gtsam::Pose3>;

  gtsam::Pose3 pose = gtsam::Pose3::Identity();
  gtsam::Pose3 motion = gtsam::Pose3::Identity();
  gtsam::Pose3 L_s = gtsam::Pose3::Identity();
  gtsam::Point3 point_l(1.0, 2.0, 3.0);
  gtsam::Point3 noise(0.4, 1.0, 2.0);
  gtsam::Point3 measured_c = pose.inverse() * L_s * (point_l + noise);

  gtsam::Key pose_key(1);
  gtsam::Key motion_key(2);
  gtsam::Values values;
  values.insert(pose_key, pose);
  values.insert(motion_key, motion);

  SmartFactor factor(L_s, gtsam::noiseModel::Isotropic::Sigma(3, 0.05),
                     point_l);
  factor.add(measured_c, motion_key, pose_key);

  gtsam::Vector expectedError = -noise;
  gtsam::Vector actualReproError = factor.reprojectionError(values);
  // Vector actualError = factor.unwhitenedError(values);
  EXPECT_TRUE(gtsam::assert_equal(expectedError, actualReproError, 1E-5));
}

TEST(SmartMotionFactor, testBasicSchurCompliment) {
  using SmartFactor = dyno::SmartMotionFactor<3, gtsam::Pose3, gtsam::Pose3>;

  gtsam::Pose3 pose1 = gtsam::Pose3::Identity();
  gtsam::Pose3 pose2(gtsam::Rot3::Rodrigues(-0.1, 0.2, 0.25),
                     gtsam::Point3(0.05, -0.10, 0.20));
  // gtsam::Pose3 pose(gtsam::Rot3::Identity(), gtsam::Point3(1.4, 0, 0));
  gtsam::Pose3 motion = gtsam::Pose3::Identity();
  gtsam::Pose3 L_s = gtsam::Pose3::Identity();
  gtsam::Point3 point(1.0, 0, 0);
  gtsam::Point3 noise(10.0, 0.0, 7.0);
  gtsam::Point3 measured = point + noise;

  gtsam::Key pose_key(1);
  gtsam::Key motion_key(2);

  gtsam::Key pose_key1(3);
  gtsam::Key motion_key1(4);
  gtsam::Values values;
  values.insert(pose_key, pose1);
  values.insert(motion_key, motion);
  values.insert(pose_key1, pose2);
  values.insert(motion_key1, motion);

  SmartFactor factor(L_s, gtsam::noiseModel::Isotropic::Sigma(3, 0.05), point);
  factor.add(measured, motion_key, pose_key);
  factor.add(measured + 2 * noise, motion_key1, pose_key1);

  LOG(INFO) << "createReducedMatrix";

  gtsam::SymmetricBlockMatrix actualReduced =
      factor.createReducedMatrix(values);

  LOG(INFO) << "N blocks " << actualReduced.nBlocks();

  gtsam::Matrix adjoint_view = actualReduced.selfadjointView();
  LOG(INFO) << "adjoint_view " << adjoint_view;

  SmartFactor::GBlocks Gs;
  SmartFactor::FBlocks Fs;
  SmartFactor::EBlocks Es;

  // not negative...?
  gtsam::Vector b = -factor.reprojectionError(values, Gs, Fs, Es);
  // factor.whitenJacobians(Gs, Fs, Es, b);

  gtsam::Matrix E;
  SmartFactor::EVectorToMatrix(Es, E);
  EXPECT_EQ(E.rows(), 6);
  EXPECT_EQ(E.cols(), 3);

  gtsam::Matrix Et = E.transpose();
  const Eigen::Matrix<double, 3, 3> P = (Et * E).inverse();

  SmartFactor::GFBlocks GFs;
  SmartFactor::GFVectorsToGFBlocks(Gs, Fs, GFs);

  /**
   * for one measurement we should have 1 motion and 1 pose (in that order)
      the A matrix should then iniially look like
          key(h_1)  key(x_1) m
      A = [G1       F1       E]

      with the regular schur compliment we expect
      g = Ft * (b - E * P * Et * b);
      G = Ft * F - Ft * E * P * Et * F
      Schur = G, g, g.transpose(), b.squaredNorm()

      we treat the GF blocks like the F block in the original schur compliment
   *
   */
  EXPECT_EQ(GFs.size(), 2u);
  Eigen::Matrix<double, 3, 12> F1 = GFs.at(0);
  Eigen::Matrix<double, 3, 12> F2 = GFs.at(1);
  Eigen::Matrix<double, 6, 24> F;

  // F is diagonal
  F << F1, Eigen::Matrix<double, 3, 12>::Zero(),
      Eigen::Matrix<double, 3, 12>::Zero(), F2;
  gtsam::Matrix Ft = F.transpose();

  Eigen::Matrix<double, 24, 1> g = Ft * (b - E * P * Et * b);
  Eigen::Matrix<double, 24, 24> G = Ft * F - Ft * E * P * Et * F;

  // LOG(INFO) << "g Matrix size: " << g.rows() << " x " << g.cols();
  // LOG(INFO) << "G Matrix size: " << G.rows() << " x " << G.cols();
  // LOG(INFO) << "adjoint_view Matrix size: " << adjoint_view.rows() << " x "
  // << adjoint_view.cols();

  // LOG(INFO) << "g:\n " << g;
  // LOG(INFO) << "G:\n " << G;

  gtsam::Matrix schur(25, 25);
  schur << G, g, g.transpose(), b.squaredNorm();
  // LOG(INFO) << "schur:\n" << schur;
  // LOG(INFO) << "adjoint_view:\n" << adjoint_view;
  EXPECT_TRUE(gtsam::assert_equal(schur, adjoint_view, 1E-5));
}

gtsam::Point3 perturbCameraAndMotion(const gtsam::Point3& point_l,
                                     const gtsam::Pose3& L_s,
                                     gtsam::Pose3& motion, gtsam::Pose3& pose,
                                     double sigma = 0.2) {
  motion = utils::perturbWithNoise(motion, sigma);
  pose = utils::perturbWithNoise(motion, sigma);
  return pose.inverse() * motion * L_s * (point_l);
}

TEST(SmartMotionFactor, testSimpleOptimise) {
  using SmartFactor = dyno::SmartMotionFactor<3, gtsam::Pose3, gtsam::Pose3>;

  gtsam::Pose3 pose1 = gtsam::Pose3::Identity();
  gtsam::Pose3 pose2(gtsam::Rot3::Rodrigues(-0.1, 0.2, 0.25),
                     gtsam::Point3(0.05, -0.10, 0.20));
  gtsam::Pose3 L_s = utils::createRandomAroundIdentity<gtsam::Pose3>(2.0);
  gtsam::Pose3 motion1 = gtsam::Pose3::Identity();
  gtsam::Pose3 motion2 = utils::createRandomAroundIdentity<gtsam::Pose3>(3.0);
  gtsam::Point3 point(3.0, 0, 1.2);

  double sigma = 0.2;

  // measurements have explicity noisy since we puerturb both motions and poses
  // however the noisy is not propogated correctly since we do not add sigma
  // noisy to the measurements
  gtsam::Point3 measurement1 =
      perturbCameraAndMotion(point, L_s, motion1, pose1, sigma);
  gtsam::Point3 measurement2 =
      perturbCameraAndMotion(point, L_s, motion2, pose2, sigma);

  gtsam::Key pose_key1(1);
  gtsam::Key motion_key1(2);

  gtsam::Key pose_key2(3);
  gtsam::Key motion_key2(4);
  gtsam::Values values;
  values.insert(pose_key1, pose1);
  values.insert(motion_key1, motion1);
  values.insert(pose_key2, pose2);
  values.insert(motion_key2, motion2);

  SmartFactor::shared_ptr factor(new SmartFactor(
      L_s, gtsam::noiseModel::Isotropic::Sigma(3, sigma), point));

  factor->add(measurement1, motion_key1, pose_key1);
  factor->add(measurement2, motion_key2, pose_key2);

  gtsam::NonlinearFactorGraph graph;
  graph.addPrior<gtsam::Pose3>(motion_key1, gtsam::Pose3::Identity(),
                               gtsam::noiseModel::Isotropic::Sigma(6, 1e-5));
  graph.add(factor);

  gtsam::LevenbergMarquardtParams lmParams;
  lmParams.relativeErrorTol = 1e-8;
  lmParams.absoluteErrorTol = 0;
  lmParams.maxIterations = 20;

  gtsam::Values result;
  gtsam::LevenbergMarquardtOptimizer optimizer(graph, values, lmParams);
  result = optimizer.optimize();
}

// TEST(SmartMotionFactor, LostTriangulation3D) {

//   gtsam::Pose3 pose1 = gtsam::Pose3::Identity();
//   gtsam::Pose3 pose2(gtsam::Rot3::Rodrigues(-0.1, 0.2, 0.25),
//                      gtsam::Point3(0.05, -0.10, 0.20));

//   gtsam::Point3 point(3.0, 0, 1.2);
//   gtsam::Point3 measurement1 = pose1.inverse() * point;
//   gtsam::Point3 measurement2 = pose2.inverse() * point;

// }
