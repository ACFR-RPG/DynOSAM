#include <glog/logging.h>
#include <gtest/gtest.h>
#include <gtsam/base/debug.h>

#include "dynosam/factors/HybridFormulationFactors.hpp"
#include "dynosam_common/utils/GtsamUtils.hpp"
#include "internal/helpers.hpp"
#include "internal/simulator.hpp"

using namespace dyno;
using namespace gtsam;

namespace Original {

// --- Original Implementations for Comparison ---
gtsam::Point3 projectToObject3(const gtsam::Pose3& X_k,
                               const gtsam::Pose3& e_H_k_world,
                               const gtsam::Pose3& L_s0,
                               const gtsam::Point3& Z_k) {
  gtsam::Pose3 k_H_s0_k = (L_s0.inverse() * e_H_k_world * L_s0).inverse();
  gtsam::Pose3 L_k = e_H_k_world * L_s0;
  gtsam::Pose3 k_H_s0_W = L_k * k_H_s0_k * L_k.inverse();
  gtsam::Point3 projected_m_object = L_s0.inverse() * k_H_s0_W * X_k * Z_k;
  return projected_m_object;
}

gtsam::Pose3 projectToCamera3Transform(const gtsam::Pose3& X_k,
                                       const gtsam::Pose3& e_H_k_world,
                                       const gtsam::Pose3& L_e) {
  return X_k.inverse() * e_H_k_world * L_e;
}

gtsam::Point3 projectToCamera3(const gtsam::Pose3& X_k,
                               const gtsam::Pose3& e_H_k_world,
                               const gtsam::Pose3& L_e,
                               const gtsam::Point3& m_L) {
  // apply transform to put map point into world via its motion
  const auto A = projectToCamera3Transform(X_k, e_H_k_world, L_e);
  gtsam::Point3 m_camera_k = A * m_L;
  return m_camera_k;
}

}  // namespace Original

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

class HybridMotionTest : public ::testing::Test {
 protected:
  Pose3 X_k, e_H_k_world, L_e;
  Point3 m_L, Z_k;
  SharedNoiseModel model;

  void SetUp() override {
    // Initialize with non-identity values to avoid trivial derivatives
    // X_k = Pose3(Rot3::Ypr(0.1, 0.2, 0.3), Point3(1, 2, 3));
    // e_H_k_world = Pose3(Rot3::Ypr(0.4, 0.1, -0.2), Point3(-1, 0.5, 2));
    // L = Pose3(Rot3::Ypr(-0.3, 0.2, 0.1), Point3(0.5, -1, 1));
    // L_s0 = Pose3(Rot3::Ypr(0.1, -0.1, 0.4), Point3(2, 1, -1));
    // m_L = Point3(1.5, -0.5, 2.0);
    // Z_k = Point3(-0.5, 1.2, 3.0);
    X_k = Pose3(Rot3::Ypr(0.1, 0.2, 0.3), Point3(1, 2, 3));
    e_H_k_world = Pose3(Rot3::Ypr(0.4, 0.1, -0.2), Point3(-1, 0.5, 2));
    L_e = Pose3(Rot3::Ypr(-0.1, 0.0, 0.1), Point3(0.1, 0.0, 0.0));
    m_L = Point3(1.5, -0.5, 2.0);

    // Generate a "measured" point based on the current state (perfect
    // prediction) This ensures the error is zero, but derivatives are still
    // valid.
    Z_k = HybridObjectMotion::projectToCamera3(X_k, e_H_k_world, L_e, m_L);

    // Isotropic noise model
    model = noiseModel::Isotropic::Sigma(3, 1.0);
  }
};

// --- Logic Comparison Tests ---
TEST_F(HybridMotionTest, JacobianEvaluation) {
  // Keys for the factor
  Key key_X = Symbol('x', 1);
  Key key_E = Symbol('e', 1);
  Key key_M = Symbol('m', 1);

  // Instantiate the factor
  HybridMotionFactor factor(key_X, key_E, key_M, Z_k, L_e, model);

  // 1. Analytical Jacobians
  Matrix H1_act, H2_act, H3_act;
  Vector error =
      factor.evaluateError(X_k, e_H_k_world, m_L, H1_act, H2_act, H3_act);

  // 2. Numerical Jacobians
  // Wrap the evaluateError function for use with numericalDerivative
  auto funcX = [&](const Pose3& x) {
    return factor.evaluateError(x, e_H_k_world, m_L);
  };
  auto funcE = [&](const Pose3& e) {
    return factor.evaluateError(X_k, e, m_L);
  };
  auto funcM = [&](const Point3& m) {
    return factor.evaluateError(X_k, e_H_k_world, m);
  };

  Matrix H1_num = numericalDerivative11<Vector, Pose3>(funcX, X_k);
  Matrix H2_num = numericalDerivative11<Vector, Pose3>(funcE, e_H_k_world);
  Matrix H3_num = numericalDerivative11<Vector, Point3>(funcM, m_L);

  // 3. Verify Dimensions
  EXPECT_EQ(H1_act.rows(), 3);
  EXPECT_EQ(H1_act.cols(), 6);
  EXPECT_EQ(H2_act.rows(), 3);
  EXPECT_EQ(H2_act.cols(), 6);
  EXPECT_EQ(H3_act.rows(), 3);
  EXPECT_EQ(H3_act.cols(), 3);

  // 4. Verify Values
  EXPECT_TRUE(assert_equal(H1_num, H1_act, 1e-5));
  EXPECT_TRUE(assert_equal(H2_num, H2_act, 1e-5));
  EXPECT_TRUE(assert_equal(H3_num, H3_act, 1e-5));
}

TEST_F(HybridMotionTest, CompareOriginal_ProjectToObject3) {
  // Compute using the new simplified implementation
  Point3 result_new =
      HybridObjectMotion::projectToObject3(X_k, e_H_k_world, L_e, Z_k);

  // Compute using the original implementation
  Point3 result_orig = Original::projectToObject3(X_k, e_H_k_world, L_e, Z_k);

  // Assert equality
  EXPECT_TRUE(assert_equal(result_orig, result_new, 1e-9));
}

TEST_F(HybridMotionTest, CompareOriginal_ProjectToCamera3Transform) {
  Pose3 result_new =
      HybridObjectMotion::projectToCamera3Transform(X_k, e_H_k_world, L_e);
  Pose3 result_orig =
      Original::projectToCamera3Transform(X_k, e_H_k_world, L_e);

  EXPECT_TRUE(assert_equal(result_orig, result_new, 1e-9));
}

TEST_F(HybridMotionTest, CompareOriginal_ProjectToCamera3) {
  Point3 result_new =
      HybridObjectMotion::projectToCamera3(X_k, e_H_k_world, L_e, m_L);
  Point3 result_orig = Original::projectToCamera3(X_k, e_H_k_world, L_e, m_L);

  EXPECT_TRUE(assert_equal(result_orig, result_new, 1e-9));
}

// --- Jacobian Tests ---

TEST_F(HybridMotionTest, ProjectToObject3_Jacobians) {
  Matrix36 H_X, H_E, H_L;
  Point3 result = HybridObjectMotion::projectToObject3(X_k, e_H_k_world, L_e,
                                                       Z_k, H_X, H_E, H_L);

  auto func = [=](const Pose3& x, const Pose3& e, const Pose3& l,
                  const Point3& z) -> Point3 {
    return HybridObjectMotion::projectToObject3(x, e, l, z);
  };

  Matrix36 num_H_X = numericalDerivative41<Point3, Pose3, Pose3, Pose3, Point3>(
      func, X_k, e_H_k_world, L_e, Z_k);

  Matrix36 num_H_E = numericalDerivative42<Point3, Pose3, Pose3, Pose3, Point3>(
      func, X_k, e_H_k_world, L_e, Z_k);

  Matrix36 num_H_L = numericalDerivative43<Point3, Pose3, Pose3, Pose3, Point3>(
      func, X_k, e_H_k_world, L_e, Z_k);

  //   Matrix33 num_H_Z = numericalDerivative44<Point3, Pose3, Pose3, Pose3,
  //   Point3>(
  //       func, X_k, e_H_k_world, L_s0, Z_k);

  EXPECT_TRUE(assert_equal(num_H_X, H_X, 1e-5));
  EXPECT_TRUE(assert_equal(num_H_E, H_E, 1e-5));
  EXPECT_TRUE(assert_equal(num_H_L, H_L, 1e-5));
  //   EXPECT_TRUE(assert_equal(num_H_Z, H_Z, 1e-5));
}

TEST_F(HybridMotionTest, ProjectToCamera3Transform_Jacobians) {
  Matrix66 H_X, H_E, H_L;
  Pose3 result = HybridObjectMotion::projectToCamera3Transform(
      X_k, e_H_k_world, L_e, H_X, H_E, H_L);

  auto func = [=](const Pose3& x, const Pose3& e,
                  const Pose3& l) -> gtsam::Pose3 {
    return HybridObjectMotion::projectToCamera3Transform(x, e, l);
  };

  // Numerical Derivatives
  Matrix66 num_H_X = numericalDerivative31<Pose3, Pose3, Pose3, Pose3>(
      func, X_k, e_H_k_world, L_e);

  Matrix66 num_H_E = numericalDerivative32<Pose3, Pose3, Pose3, Pose3>(
      func, X_k, e_H_k_world, L_e);

  Matrix66 num_H_L = numericalDerivative33<Pose3, Pose3, Pose3, Pose3>(
      func, X_k, e_H_k_world, L_e);

  EXPECT_TRUE(assert_equal(num_H_X, H_X, 1e-5));
  EXPECT_TRUE(assert_equal(num_H_E, H_E, 1e-5));
  EXPECT_TRUE(assert_equal(num_H_L, H_L, 1e-5));
}

TEST_F(HybridMotionTest, ProjectToCamera3_Jacobians) {
  Matrix36 H_X, H_E, H_L;
  Matrix33 H_m;
  Point3 result = HybridObjectMotion::projectToCamera3(X_k, e_H_k_world, L_e,
                                                       m_L, H_X, H_E, H_L, H_m);

  // Wrapper for numerical derivative because m_L is 4th argument
  auto func = [=](const Pose3& x, const Pose3& e, const Pose3& l,
                  const Point3& m) {
    return HybridObjectMotion::projectToCamera3(x, e, l, m);
  };

  Matrix36 num_H_X = numericalDerivative41<Point3, Pose3, Pose3, Pose3, Point3>(
      func, X_k, e_H_k_world, L_e, m_L);

  Matrix36 num_H_E = numericalDerivative42<Point3, Pose3, Pose3, Pose3, Point3>(
      func, X_k, e_H_k_world, L_e, m_L);

  Matrix36 num_H_L = numericalDerivative43<Point3, Pose3, Pose3, Pose3, Point3>(
      func, X_k, e_H_k_world, L_e, m_L);

  Matrix33 num_H_m = numericalDerivative44<Point3, Pose3, Pose3, Pose3, Point3>(
      func, X_k, e_H_k_world, L_e, m_L);

  EXPECT_TRUE(assert_equal(num_H_X, H_X, 1e-5));
  EXPECT_TRUE(assert_equal(num_H_E, H_E, 1e-5));
  EXPECT_TRUE(assert_equal(num_H_L, H_L, 1e-5));
  EXPECT_TRUE(assert_equal(num_H_m, H_m, 1e-5));
}

class StereoHybridFactorTest : public ::testing::Test {
 protected:
  Pose3 X_k, e_H_k_world, L_e;
  Point3 m_L;
  Cal3_S2Stereo::shared_ptr K;
  SharedNoiseModel model;
  StereoPoint2 measured;

  void SetUp() override {
    // Setup Arbitrary Poses (Non-identity to test rotations)
    X_k = Pose3(Rot3::Ypr(0.1, 0.2, 0.3), Point3(1, 2, 3));
    e_H_k_world = Pose3(Rot3::Ypr(0.4, 0.1, -0.2), Point3(-1, 0.5, 2));
    L_e = Pose3(Rot3::Ypr(-0.1, 0.0, 0.1), Point3(0.1, 0.0, 0.0));
    m_L = Point3(1.5, -0.5, 5.0);  // Point needs to be in front of camera

    // Setup Calibration (fx, fy, s, u0, v0, b)
    K = boost::make_shared<Cal3_S2Stereo>(1000, 1000, 0, 320, 240, 0.5);

    // Setup Noise Model
    model = noiseModel::Isotropic::Sigma(3, 1.0);

    // Create a "Perfect" measurement using the exact same logic as the Factor
    // 1. Project to camera frame
    Point3 p_cam =
        HybridObjectMotion::projectToCamera3(X_k, e_H_k_world, L_e, m_L);

    // 2. Project using StereoCamera at Identity
    StereoCamera cam(Pose3::Identity(), K);
    StereoPoint2 perfect = cam.project2(p_cam);

    // Perturb slightly for test
    measured =
        StereoPoint2(perfect.uL() + 2.0, perfect.uR() - 1.0, perfect.v() + 0.5);
  }
};

TEST_F(StereoHybridFactorTest, JacobianEvaluation) {
  // Keys
  Key key_X = Symbol('x', 1);
  Key key_E = Symbol('e', 1);
  Key key_M = Symbol('m', 1);

  // Create Factor
  StereoHybridMotionFactor factor(measured, L_e, model, K, key_X, key_E, key_M);

  // 1. Compute Analytical Jacobians via evaluateError
  Matrix H1_act, H2_act, H3_act;
  Vector error =
      factor.evaluateError(X_k, e_H_k_world, m_L, H1_act, H2_act, H3_act);

  // 2. Compute Numerical Jacobians
  // We use a lambda to bind the factor's evaluateError function for numerical
  // derivative

  auto funcX = [&](const Pose3& x) {
    return factor.evaluateError(x, e_H_k_world, m_L);
  };

  auto funcE = [&](const Pose3& e) {
    return factor.evaluateError(X_k, e, m_L);
  };

  auto funcM = [&](const Point3& m) {
    return factor.evaluateError(X_k, e_H_k_world, m);
  };

  Matrix H1_num = numericalDerivative11<Vector, Pose3>(funcX, X_k);
  Matrix H2_num = numericalDerivative11<Vector, Pose3>(funcE, e_H_k_world);
  Matrix H3_num = numericalDerivative11<Vector, Point3>(funcM, m_L);

  // 3. Compare
  // Dimensions check
  EXPECT_EQ(H1_act.rows(), 3);
  EXPECT_EQ(H1_act.cols(), 6);
  EXPECT_EQ(H2_act.rows(), 3);
  EXPECT_EQ(H2_act.cols(), 6);
  EXPECT_EQ(H3_act.rows(), 3);
  EXPECT_EQ(H3_act.cols(), 3);

  // Value check (using 1e-5 tolerance for chain rule complexity)
  EXPECT_TRUE(assert_equal(H1_num, H1_act, 1e-5));
  EXPECT_TRUE(assert_equal(H2_num, H2_act, 1e-5));
  EXPECT_TRUE(assert_equal(H3_num, H3_act, 1e-5));
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
