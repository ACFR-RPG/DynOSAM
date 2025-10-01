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
#include "dynosam/backend/rgbd/HybridEstimator.hpp"  //TODO: move implementation to factors?
#include "dynosam/factors/LandmarkMotionPoseFactor.hpp"
#include "dynosam/factors/LandmarkMotionTernaryFactor.hpp"
#include "dynosam/factors/Pose3FlowProjectionFactor.h"
#include "dynosam_common/utils/GtsamUtils.hpp"
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
  gtsam::Pose3 L_e = gtsam::Pose3::Identity();
  gtsam::Point3 point(1.0, 2.0, 3.0);
  gtsam::Point3 noise(0.0, 0.0, 0.0);
  gtsam::Point3 measured = point + noise;

  gtsam::Key pose_key(1);
  gtsam::Key motion_key(2);
  gtsam::Values values;
  values.insert(pose_key, pose);
  values.insert(motion_key, motion);

  SmartFactor factor(L_e, gtsam::noiseModel::Isotropic::Sigma(3, 0.05),
                     SmartMotionFactorParams{}, point);
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
  gtsam::Pose3 L_e = gtsam::Pose3(gtsam::Rot3::Ypr(-M_PI / 10, 0., -M_PI / 10),
                                  gtsam::Point3(0.5, 0.1, 0.3));
  gtsam::Point3 point_l(1.0, 2.0, 3.0);
  gtsam::Point3 noise(0.0, 0.0, 0.0);
  gtsam::Point3 measured_c = pose.inverse() * L_e * (point_l + noise);

  gtsam::Key pose_key(1);
  gtsam::Key motion_key(2);
  gtsam::Values values;
  values.insert(pose_key, pose);
  values.insert(motion_key, motion);

  SmartFactor factor(L_e, gtsam::noiseModel::Isotropic::Sigma(3, 0.05),
                     SmartMotionFactorParams{}, point_l);
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
  gtsam::Pose3 L_e = gtsam::Pose3::Identity();
  gtsam::Point3 point_l(1.0, 2.0, 3.0);
  gtsam::Point3 noise(0.4, 1.0, 2.0);
  gtsam::Point3 measured_c = pose.inverse() * L_e * (point_l + noise);

  gtsam::Key pose_key(1);
  gtsam::Key motion_key(2);
  gtsam::Values values;
  values.insert(pose_key, pose);
  values.insert(motion_key, motion);

  SmartFactor factor(L_e, gtsam::noiseModel::Isotropic::Sigma(3, 0.05),
                     SmartMotionFactorParams{}, point_l);
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
  gtsam::Pose3 L_e = gtsam::Pose3::Identity();
  gtsam::Point3 point(1.0, 1.4, 2);
  gtsam::Point3 noise(10.0, 0.0, 7.0);
  gtsam::Point3 measured = point + noise;

  // with two sets of emasurements (2 * (1 pose + 1 motion) the number of blocks
  // is 3
  // motion + pose form a single block currently + 1 block for error

  gtsam::Key pose_key(1);
  gtsam::Key motion_key(2);

  gtsam::Key pose_key1(3);
  gtsam::Key motion_key1(4);

  gtsam::Key pose_key2(5);
  gtsam::Key motion_key2(6);
  gtsam::Values values;
  values.insert(pose_key, pose1);
  values.insert(motion_key, motion);
  values.insert(pose_key1, pose2);
  values.insert(motion_key1, motion);
  // values.insert(pose_key2, pose2);
  // values.insert(motion_key2, motion);

  SmartFactor factor(L_e, gtsam::noiseModel::Isotropic::Sigma(3, 0.05),
                     SmartMotionFactorParams{}, point);
  factor.add(measured, motion_key, pose_key);
  factor.add(measured + 2 * noise, motion_key1, pose_key1);
  // factor.add(measured + 3 * noise, motion_key2, pose_key2);

  LOG(INFO) << "createReducedMatrix";

  gtsam::SymmetricBlockMatrix actualReduced =
      factor.createReducedMatrix(values);

  LOG(INFO) << "N blocks " << actualReduced.nBlocks()
            << " rows=" << actualReduced.rows()
            << " cols=" << actualReduced.cols();

  gtsam::Matrix adjoint_view = actualReduced.selfadjointView();
  // LOG(INFO) << "adjoint_view " << adjoint_view;

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

  LOG(INFO) << "F=" << F;

  Eigen::Matrix<double, 24, 1> g = Ft * (b - E * P * Et * b);
  Eigen::Matrix<double, 24, 24> G = Ft * F - Ft * E * P * Et * F;

  gtsam::Matrix schur(25, 25);
  schur << G, g, g.transpose(), b.squaredNorm();

  // test programatic construction
  {
    constexpr static auto HDim = 6;
    constexpr static auto XDim = 6;
    constexpr static auto HessianDim = 12;
    size_t m = 2;  // measurements
    std::vector<Eigen::DenseIndex> f_block_dims(2 * m);
    std::fill(f_block_dims.begin(), f_block_dims.end(),
              HDim);  // assuming HDim and Xdim are the same size
    // dims.back() = 1;

    // Make full stock block matrix
    gtsam::Matrix F_block_matrix(m * 3, m * HessianDim);
    F_block_matrix.setZero();
    // gtsam::SymmetricBlockMatrix F_block_matrix(f_block_dims);
    // size_t block_idx = 0;
    for (size_t i = 0; i < m; i++) {
      const Eigen::Matrix<double, 3, HessianDim>& GFblock = GFs.at(i);
      LOG(INFO) << GFblock;

      // Eigen::Matrix<double, 3, HDim> gblock = GFblock.leftCols(HDim);
      // Eigen::Matrix<double, 3, HDim> fblock = GFblock.rightCols(XDim);

      // Eigen::Matrix<double, 3, HessianDim> GF
      // set along diagonals i, j, p,q
      LOG(INFO) << 3 * i << " " << HessianDim * i;
      // F_block_matrix.block( 3*i, HessianDim*i, 3, HessianDim) = GFblock;
      F_block_matrix.block<3, HessianDim>(3 * i, HessianDim * i) = GFblock;

      // F_block_matrix.setDiagonalBlock(2*i, gblock);
      // F_block_matrix.setDiagonalBlock(2*i+1, fblock);
    }

    gtsam::Matrix F = F_block_matrix;
    gtsam::Matrix Ft = F.transpose();
    // LOG(INFO) << "F=" << F;

    gtsam::Matrix g = Ft * (b - E * P * Et * b);
    gtsam::Matrix G = Ft * F - Ft * E * P * Et * F;

    // size of schur = num measurements * Hessian size + 1
    size_t aug_hessian_size = m * HessianDim + 1;
    gtsam::Matrix other_schur(aug_hessian_size, aug_hessian_size);

    other_schur << G, g, g.transpose(), b.squaredNorm();

    EXPECT_TRUE(gtsam::assert_equal(schur, other_schur, 1E-5));

    std::vector<Eigen::DenseIndex> dims(2 * m + 1);  // includes b term
    std::fill(dims.begin(), dims.end() - 1,
              HDim);  // assuming HDim and Xdim are the same size
    dims.back() = 1;

    gtsam::SymmetricBlockMatrix augmented_hessian(dims, other_schur);
    gtsam::RegularHessianFactor<XDim> hessian_factor(factor.keys(),
                                                     augmented_hessian);

    // boost::make_shared<gtsam::RegularHessianFactor<HessianDim>>(
    // this->keys_, augmented_hessian
  }
  // LOG(INFO) << "g Matrix size: " << g.rows() << " x " << g.cols();
  // LOG(INFO) << "G Matrix size: " << G.rows() << " x " << G.cols();
  // LOG(INFO) << "adjoint_view Matrix size: " << adjoint_view.rows() << " x "
  // << adjoint_view.cols();

  // LOG(INFO) << "g:\n " << g;
  // LOG(INFO) << "G:\n " << G;
  // LOG(INFO) << "schur:\n" << schur;
  // LOG(INFO) << "adjoint_view:\n" << adjoint_view;
  EXPECT_TRUE(gtsam::assert_equal(schur, adjoint_view, 1E-5));
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

TEST(SmartMotionFactor, testSimpleOptimise) {
  using SmartFactor = dyno::SmartMotionFactor<3, gtsam::Pose3, gtsam::Pose3>;

  gtsam::Pose3 pose1 = gtsam::Pose3::Identity();
  gtsam::Pose3 pose2(gtsam::Rot3::Rodrigues(-0.1, 0.2, 0.25),
                     gtsam::Point3(0.05, -0.10, 0.20));
  gtsam::Pose3 L_e = utils::createRandomAroundIdentity<gtsam::Pose3>(2.0);
  gtsam::Pose3 motion1 = gtsam::Pose3::Identity();
  gtsam::Pose3 motion2 = utils::createRandomAroundIdentity<gtsam::Pose3>(3.0);
  gtsam::Point3 point1(3.0, 0, 1.2);
  gtsam::Point3 point2(2.0, -5, 3.2);

  double sigma = 0.001;

  gtsam::Key pose_key1(1);
  gtsam::Key motion_key1(2);

  gtsam::Key pose_key2(3);
  gtsam::Key motion_key2(4);

  auto create_smart_factor = [=](const gtsam::Point3& point,
                                 double sigma) -> SmartFactor::shared_ptr {
    // start with actual point ie no linearization error here
    SmartFactor::shared_ptr factor(
        new SmartFactor(L_e, gtsam::noiseModel::Isotropic::Sigma(3, sigma),
                        SmartMotionFactorParams{}, point));

    auto Z1 = HybridObjectMotion::projectToCamera3(pose1, motion1, L_e, point);
    Z1 = utils::perturbWithNoise(Z1, sigma);

    auto Z2 = HybridObjectMotion::projectToCamera3(pose2, motion2, L_e, point);
    Z2 = utils::perturbWithNoise(Z2, sigma);

    factor->add(Z1, motion_key1, pose_key1);
    factor->add(Z2, motion_key2, pose_key2);

    return factor;
  };

  // measurements have explicity noisy since we puerturb both motions and poses
  // however the noisy is not propogated correctly since we do not add sigma
  // noisy to the measurements
  gtsam::Pose3 pose1_peturb = utils::perturbWithNoise(pose1, sigma);
  gtsam::Pose3 pose2_peturb = utils::perturbWithNoise(pose2, sigma);
  gtsam::Pose3 motion1_peturb = utils::perturbWithNoise(motion1, sigma);
  gtsam::Pose3 motion2_peturb = utils::perturbWithNoise(motion2, sigma);

  // gtsam::Point3 measurement1 =
  //     perturbCameraAndMotion(point, L_e, motion1_peturb, pose1_peturb,
  //     sigma);
  // gtsam::Point3 measurement2 =
  //     perturbCameraAndMotion(point, L_e, motion2_peturb, pose2_peturb,
  //     sigma);

  gtsam::Values values;
  values.insert(pose_key1, pose1_peturb);
  values.insert(motion_key1, motion1_peturb);
  values.insert(pose_key2, pose2_peturb);
  values.insert(motion_key2, motion2_peturb);

  // SmartFactor::shared_ptr factor(new SmartFactor(
  //     L_e, gtsam::noiseModel::Isotropic::Sigma(3, sigma), point));

  // factor->add(measurement1, motion_key1, pose_key1);
  // factor->add(measurement2, motion_key2, pose_key2);

  gtsam::NonlinearFactorGraph graph;
  graph.addPrior<gtsam::Pose3>(motion_key1, gtsam::Pose3::Identity(),
                               gtsam::noiseModel::Isotropic::Sigma(6, 1e-5));
  graph.add(create_smart_factor(point1, sigma));
  graph.add(create_smart_factor(point2, sigma));

  gtsam::LevenbergMarquardtParams lmParams;
  lmParams.relativeErrorTol = 1e-8;
  lmParams.absoluteErrorTol = 0;
  lmParams.maxIterations = 20;

  gtsam::Values result;
  values.print("Before ");
  gtsam::LevenbergMarquardtOptimizer optimizer(graph, values, lmParams);
  result = optimizer.optimize();
  result.print("After ");

  EXPECT_TRUE(
      gtsam::assert_equal(result.at<gtsam::Pose3>(pose_key1), pose1, 1E-5));
  EXPECT_TRUE(
      gtsam::assert_equal(result.at<gtsam::Pose3>(pose_key2), pose2, 1E-5));
  EXPECT_TRUE(
      gtsam::assert_equal(result.at<gtsam::Pose3>(motion_key1), motion1, 1E-5));
  EXPECT_TRUE(
      gtsam::assert_equal(result.at<gtsam::Pose3>(motion_key2), motion2, 1E-5));

  LOG(INFO) << graph.error(values);
  LOG(INFO) << graph.error(result);
}
TEST(SmartMotionFactor, computeTriangulationBasic) {
  using SmartFactor = dyno::SmartMotionFactor<3, gtsam::Pose3, gtsam::Pose3>;
  gtsam::Pose3 pose1 = gtsam::Pose3::Identity();
  gtsam::Pose3 pose2(gtsam::Rot3::Rodrigues(-0.1, 0.2, 0.25),
                     gtsam::Point3(0.05, -0.10, 0.20));
  gtsam::Pose3 L_e = utils::createRandomAroundIdentity<gtsam::Pose3>(2.0);
  gtsam::Pose3 motion1 = gtsam::Pose3::Identity();
  gtsam::Pose3 motion2 = utils::createRandomAroundIdentity<gtsam::Pose3>(3.0);
  gtsam::Point3 point(3.0, 0, 1.2);

  // no noise!!!
  const double sigma = 0.00;

  gtsam::Values values;
  gtsam::Key pose_key1(1);
  gtsam::Key motion_key1(2);

  gtsam::Key pose_key2(3);
  gtsam::Key motion_key2(4);

  values.insert(pose_key1, pose1);
  values.insert(motion_key1, motion1);
  values.insert(pose_key2, pose2);
  values.insert(motion_key2, motion2);

  SmartFactor::shared_ptr factor(
      new SmartFactor(L_e, gtsam::noiseModel::Isotropic::Sigma(3, sigma),
                      SmartMotionFactorParams{}));

  auto Z1 = HybridObjectMotion::projectToCamera3(pose1, motion1, L_e, point);
  Z1 = utils::perturbWithNoise(Z1, sigma);

  auto Z2 = HybridObjectMotion::projectToCamera3(pose2, motion2, L_e, point);
  Z2 = utils::perturbWithNoise(Z2, sigma);

  EXPECT_FALSE(factor->point(values));
  EXPECT_FALSE(factor->point());

  // should now be degenerate
  factor->add(Z1, motion_key1, pose_key1);
  EXPECT_FALSE(factor->point(values));
  EXPECT_TRUE(factor->point(values).degenerate());

  factor->add(Z2, motion_key2, pose_key2);

  EXPECT_EQ(factor->numMeasurements(), 2u);

  // given erroneous measurements but accurayte motion/pose
  // actually should only be okay given the params!!
  EXPECT_TRUE(factor->point(values));

  LOG(INFO) << "Triangulated point = " << factor->point();
  EXPECT_TRUE(gtsam::assert_equal(point, factor->point().value(), 1E-5));
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
