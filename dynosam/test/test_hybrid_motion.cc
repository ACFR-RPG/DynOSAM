#include <glog/logging.h>
#include <gtest/gtest.h>
#include <gtsam/base/debug.h>

#include "dynosam/backend/rgbd/impl/test_HybridFormulations.hpp"
#include "dynosam/factors/HybridFormulationFactors.hpp"
#include "dynosam/utils/GtsamUtils.hpp"
#include "internal/helpers.hpp"
#include "internal/simulator.hpp"

using namespace dyno;

TEST(HybridObjectMotion, testProjections) {
  gtsam::Pose3 e_H_k_world =
      dyno::utils::perturbWithNoise(gtsam::Pose3::Identity(), 0.5, 21);
  gtsam::Pose3 cam_pose =
      dyno::utils::perturbWithNoise(gtsam::Pose3::Identity(), 0.5, 19);
  gtsam::Pose3 L0 =
      dyno::utils::perturbWithNoise(gtsam::Pose3::Identity(), 0.5, 12);

  gtsam::Point3 m_object =
      dyno::utils::perturbWithNoise(gtsam::Point3(0, 0, 0), 1.5, 42);

  // put point in Xk
  gtsam::Point3 m_camera = cam_pose.inverse() * e_H_k_world * L0 * m_object;

  gtsam::Point3 projected_m_object = dyno::HybridObjectMotion::projectToObject3(
      cam_pose, e_H_k_world, L0, m_camera);

  // basically checking the inverse operation
  gtsam::Point3 projected_m_camera = dyno::HybridObjectMotion::projectToCamera3(
      cam_pose, e_H_k_world, L0, projected_m_object);

  EXPECT_TRUE(gtsam::assert_equal(projected_m_object, m_object));
  EXPECT_TRUE(gtsam::assert_equal(projected_m_camera, m_camera));
}

// TEST(HybridObjectMotion, ProjectToCamera3Jacobian) {
//   using namespace gtsam;

//   // Create random inputs
//   Pose3 X_k = Pose3(Rot3::RzRyRx(0.1, 0.2, 0.3), Point3(1.0, 2.0, 3.0));
//   Pose3 e_H_k_world = Pose3(Rot3::RzRyRx(-0.2, 0.1, 0.05), Point3(-1.0,
//   0.5, 2.0)); Pose3 L_e = Pose3(Rot3::RzRyRx(0.05, -0.1, 0.2),
//   Point3(0.0, 1.0, -1.0)); Point3 m_L(0.5, -0.4, 1.2);

//   // Storage for analytical Jacobians
//   Matrix36 H_Xk, H_eHk, H_Le;
//   Matrix33 H_mL;

//   // Evaluate function with analytical Jacobians
//   Point3 result = HybridObjectMotion::projectToCamera3(X_k, e_H_k_world, L_e,
//   m_L,
//                                        &H_Xk, &H_eHk, &H_Le, &H_mL);

//   // Numerical Jacobians
//   auto f_Xk = [&](const Pose3& X) { return
//   HybridObjectMotion::projectToCamera3(X, e_H_k_world, L_e, m_L); }; auto
//   f_eHk = [&](const Pose3& E) { return
//   HybridObjectMotion::projectToCamera3(X_k, E, L_e, m_L); }; auto f_Le =
//   [&](const Pose3& L) { return HybridObjectMotion::projectToCamera3(X_k,
//   e_H_k_world, L, m_L); }; auto f_mL = [&](const Point3& p) { return
//   HybridObjectMotion::projectToCamera3(X_k, e_H_k_world, L_e, p); };

//   Matrix H_Xk_num = numericalDerivative11<Point3, Pose3>(f_Xk, X_k);
//   Matrix H_eHk_num = numericalDerivative11<Point3, Pose3>(f_eHk,
//   e_H_k_world); Matrix H_Le_num = numericalDerivative11<Point3, Pose3>(f_Le,
//   L_e); Matrix H_mL_num = numericalDerivative11<Point3, Point3>(f_mL, m_L);

//   // Compare
//   EXPECT_TRUE(assert_equal(H_Xk_num, H_Xk, 1e-7));
//   EXPECT_TRUE(assert_equal(H_eHk_num, H_eHk, 1e-7));
//   EXPECT_TRUE(assert_equal(H_Le_num, H_Le, 1e-7));
//   EXPECT_TRUE(assert_equal(H_mL_num, H_mL, 1e-7));
// }

// TEST(StructurelessObjectCentricMotionFactor2, testZeroError) {
//   // construct point in L and then move it
//   using namespace dyno::test_hybrid;

//   gtsam::Pose3 L_e = utils::createRandomAroundIdentity<gtsam::Pose3>(0.4);
//   gtsam::Pose3 e_H_k_world(gtsam::Rot3::Rodrigues(-0.1, 0.2, 0.25),
//                            gtsam::Point3(0.05, -0.10, 0.20));
//   gtsam::Pose3 s_H_k_1 =
//       utils::perturbWithNoise<gtsam::Pose3>(e_H_k_world, 0.3);

//   // observing poses
//   gtsam::Pose3 X_k = utils::createRandomAroundIdentity<gtsam::Pose3>(0.4);
//   gtsam::Pose3 X_k_1 = utils::createRandomAroundIdentity<gtsam::Pose3>(0.4);

//   gtsam::Point3 m_object(0.4, 1.0, 0.8);

//   // measurements in camera at k-1 and k
//   gtsam::Point3 Z_k = X_k.inverse() * e_H_k_world * L_e * m_object;
//   gtsam::Point3 Z_k_1 = X_k_1.inverse() * s_H_k_1 * L_e * m_object;

//   auto noise = gtsam::noiseModel::Isotropic::Sigma(3u, 0.1);

//   StructurelessObjectCentricMotionFactor2 factor(0, 1, 2, 3, Z_k_1, Z_k, L_e,
//                                                  noise);
//   gtsam::Vector error = factor.evaluateError(X_k_1, s_H_k_1, X_k,
//   e_H_k_world); EXPECT_TRUE(gtsam::assert_equal(gtsam::Point3(0, 0, 0),
//   error, 1e-4));
// }
