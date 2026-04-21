/*
 *   Copyright (c) 2025 ACFR-RPG, University of Sydney, Jesse Morris
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
#include "dynosam/factors/HybridFormulationFactors.hpp"

#include <gtsam/base/numericalDerivative.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace dyno {

gtsam::Point3 HybridObjectMotion::projectToObject3(
    const gtsam::Pose3& X_k, const gtsam::Pose3& e_H_k_world,
    const gtsam::Pose3& L_s0, const gtsam::Point3& Z_k,
    gtsam::OptionalJacobian<3, 6> J1, gtsam::OptionalJacobian<3, 6> J2,
    gtsam::OptionalJacobian<3, 6> J3) {
  // gtsam::Pose3 k_H_s0_k = (L_s0.inverse() * e_H_k_world * L_s0).inverse();
  // gtsam::Pose3 L_k = e_H_k_world * L_s0;
  // gtsam::Pose3 k_H_s0_W = L_k * k_H_s0_k * L_k.inverse();
  // gtsam::Point3 projected_m_object = L_s0.inverse() * k_H_s0_W * X_k * Z_k;
  // return projected_m_object;
  // Mathematical Simplification:
  // The original kinematic chain simplifies algebraically to:
  // P = L_s0.inverse() * e_H_k_world.inverse() * X_k * Z_k

  // 1. Invert L_s0
  gtsam::Matrix66 H_invL_L;
  const gtsam::Pose3 invL = L_s0.inverse(J3 ? &H_invL_L : 0);

  // 2. Invert e_H_k_world
  gtsam::Matrix66 H_invE_E;
  const gtsam::Pose3 invE = e_H_k_world.inverse(J2 ? &H_invE_E : 0);

  // 3. Compose (invL * invE)
  gtsam::Matrix66 H_comb1_invL, H_comb1_invE;
  const gtsam::Pose3 comb1 = invL.compose(invE, J3 ? &H_comb1_invL : 0,
                                          (J2 || J3 || J1) ? &H_comb1_invE : 0);

  // 4. Compose (comb1 * X_k)
  gtsam::Matrix66 H_comb2_comb1, H_comb2_X;
  const gtsam::Pose3 comb2 =
      comb1.compose(X_k, (J3 || J2) ? &H_comb2_comb1 : 0, J1 ? &H_comb2_X : 0);

  // 5. Transform point (comb2 * Z_k)
  gtsam::Matrix36 H_res_comb2;
  gtsam::Matrix33 H_res_Z;
  const gtsam::Point3 result = comb2.transformFrom(
      Z_k, (J3 || J2 || J1) ? &H_res_comb2 : 0, boost::none);

  // Chain Rules

  // // dRes/dZ
  // if (H_Z_k) *H_Z_k = H_res_Z;

  // dRes/dX = dRes/dComb2 * dComb2/dX
  if (J1) *J1 = H_res_comb2 * H_comb2_X;

  // dRes/dE = dRes/dComb2 * dComb2/dComb1 * dComb1/dInvE * dInvE/dE
  if (J2) {
    *J2 = H_res_comb2 * H_comb2_comb1 * H_comb1_invE * H_invE_E;
  }

  // dRes/dL = dRes/dComb2 * dComb2/dComb1 * dComb1/dInvL * dInvL/dL
  if (J3) {
    *J3 = H_res_comb2 * H_comb2_comb1 * H_comb1_invL * H_invL_L;
  }

  return result;
}

gtsam::Point3 HybridObjectMotion::projectToCamera3(
    const gtsam::Pose3& X_k, const gtsam::Pose3& e_H_k_world,
    const gtsam::Pose3& L_e, const gtsam::Point3& m_L,
    gtsam::OptionalJacobian<3, 6> J1, gtsam::OptionalJacobian<3, 6> J2,
    gtsam::OptionalJacobian<3, 6> J3, gtsam::OptionalJacobian<3, 3> J4) {
  // // apply transform to put map point into world via its motion
  // const auto A = projectToCamera3Transform(X_k, e_H_k_world, L_e);
  // gtsam::Point3 m_camera_k = A * m_L;
  // return m_camera_k;

  // 1. Get the transform T = X_k^-1 * E * L_e
  gtsam::Matrix66 H_T_X, H_T_E, H_T_L;
  const gtsam::Pose3 T = projectToCamera3Transform(
      X_k, e_H_k_world, L_e, J1 ? &H_T_X : 0, J2 ? &H_T_E : 0, J3 ? &H_T_L : 0);

  // 2. Transform the point: P = T * m_L
  gtsam::Matrix36 H_P_T;
  gtsam::Matrix33 H_P_m;
  const gtsam::Point3 m_camera_k =
      T.transformFrom(m_L, (J1 || J2 || J3) ? &H_P_T : 0, J4 ? &H_P_m : 0);

  // Chain Rules
  if (J1) *J1 = H_P_T * H_T_X;
  if (J2) *J2 = H_P_T * H_T_E;
  if (J3) *J3 = H_P_T * H_T_L;
  if (J4) *J4 = H_P_m;

  return m_camera_k;
}

gtsam::Pose3 HybridObjectMotion::projectToCamera3Transform(
    const gtsam::Pose3& X_k, const gtsam::Pose3& e_H_k_world,
    const gtsam::Pose3& L_e, gtsam::OptionalJacobian<6, 6> J1,
    gtsam::OptionalJacobian<6, 6> J2, gtsam::OptionalJacobian<6, 6> J3) {
  // return X_k.inverse() * e_H_k_world * L_e;

  // Formula: T = X_k.inverse() * e_H_k_world * L_e

  // 1. Calculate X_k inverse
  gtsam::Matrix66 H_invX_Xk;
  gtsam::Pose3 invX = X_k.inverse(J1 ? &H_invX_Xk : 0);

  // 2. Compose e_H_k_world and L_e
  gtsam::Matrix66 H_comb1_E, H_comb1_L;
  gtsam::Pose3 comb1 =
      e_H_k_world.compose(L_e, (J2 || J1 || J3) ? &H_comb1_E : 0,
                          (J3 || J1 || J2) ? &H_comb1_L : 0);

  // 3. Compose result
  gtsam::Matrix66 H_res_invX, H_res_comb1;
  const gtsam::Pose3 result =
      invX.compose(comb1, J1 ? &H_res_invX : 0, (J2 || J3) ? &H_res_comb1 : 0);

  // Chain Rules
  if (J1) {
    // dRes/dX = dRes/dInvX * dInvX/dX
    *J1 = H_res_invX * H_invX_Xk;
  }

  if (J2) {
    // dRes/dE = dRes/dComb1 * dComb1/dE
    *J2 = H_res_comb1 * H_comb1_E;
  }

  if (J3) {
    // dRes/dL = dRes/dComb1 * dComb1/dL
    *J3 = H_res_comb1 * H_comb1_L;
  }

  return result;
}
gtsam::Vector3 HybridObjectMotion::residual(const gtsam::Pose3& X_k,
                                            const gtsam::Pose3& e_H_k_world,
                                            const gtsam::Point3& m_L,
                                            const gtsam::Point3& Z_k,
                                            const gtsam::Pose3& L_e) {
  return projectToCamera3(X_k, e_H_k_world, L_e, m_L) - Z_k;
}

gtsam::Vector6 HybridObjectMotion::constantMotionResidual(
    const gtsam::Pose3& H_W_KF_km2, const gtsam::Pose3& L_W_KFkm2,
    const gtsam::Pose3& H_W_KF_km1, const gtsam::Pose3& L_W_KFkm1,
    const gtsam::Pose3& H_W_KF_k, const gtsam::Pose3& L_W_KFk,
    boost::optional<gtsam::Matrix&> J1, boost::optional<gtsam::Matrix&> J2,
    boost::optional<gtsam::Matrix&> J3) {
  // note argument ordering is different!!
  auto residual =
      [](const gtsam::Pose3& H_W_KF_km2, const gtsam::Pose3& L_W_KFkm2,
         const gtsam::Pose3& H_W_KF_km1, const gtsam::Pose3& L_W_KFkm1,
         const gtsam::Pose3& H_W_KF_k,
         const gtsam::Pose3& L_W_KFk) -> gtsam::Vector6 {
    const gtsam::Pose3 L_k_2 = H_W_KF_km2 * L_W_KFkm2;
    const gtsam::Pose3 L_k_1 = H_W_KF_km1 * L_W_KFkm1;
    const gtsam::Pose3 L_k = H_W_KF_k * L_W_KFk;

    gtsam::Pose3 k_2_H_k_1 = L_k_2.inverse() * L_k_1;
    gtsam::Pose3 k_1_H_k = L_k_1.inverse() * L_k;

    gtsam::Pose3 relative_motion = k_2_H_k_1.inverse() * k_1_H_k;

    return gtsam::traits<gtsam::Pose3>::Local(gtsam::Pose3::Identity(),
                                              relative_motion);
  };

  // Proper wrapper: explicit ordering, no bind
  auto f = [L_W_KFkm2, L_W_KFkm1, L_W_KFk, &residual](
               const gtsam::Pose3& H_km2, const gtsam::Pose3& H_km1,
               const gtsam::Pose3& H_k) -> gtsam::Vector6 {
    return residual(H_km2, L_W_KFkm2, H_km1, L_W_KFkm1, H_k, L_W_KFk);
  };

  // Numerical Jacobians
  if (J1) {
    *J1 = gtsam::numericalDerivative31<gtsam::Vector6, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Pose3>(
        f, H_W_KF_km2, H_W_KF_km1, H_W_KF_k);
  }

  if (J2) {
    *J2 = gtsam::numericalDerivative32<gtsam::Vector6, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Pose3>(
        f, H_W_KF_km2, H_W_KF_km1, H_W_KF_k);
  }

  if (J3) {
    *J3 = gtsam::numericalDerivative33<gtsam::Vector6, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Pose3>(
        f, H_W_KF_km2, H_W_KF_km1, H_W_KF_k);
  }

  // Return residual
  return f(H_W_KF_km2, H_W_KF_km1, H_W_KF_k);
}

gtsam::Vector HybridMotionFactor::evaluateError(
    const gtsam::Pose3& X_k, const gtsam::Pose3& e_H_k_world,
    const gtsam::Point3& m_L, boost::optional<gtsam::Matrix&> J1,
    boost::optional<gtsam::Matrix&> J2,
    boost::optional<gtsam::Matrix&> J3) const {
  return HybridObjectMotion::projectToCamera3(
             X_k, e_H_k_world, L_e_, m_L,
             J1,           // J w.r.t X_k
             J2,           // J w.r.t e_H_k_world
             boost::none,  // J w.r.t L_e (fixed, not optimized here)
             J3            // J w.r.t m_L
             ) -
         z_k_;
}

StereoHybridMotionFactorBase::StereoHybridMotionFactorBase(
    const gtsam::StereoPoint2& measured, const gtsam::Pose3& L_KF,
    gtsam::Cal3_S2Stereo::shared_ptr K, bool throw_cheirality)
    : measured_(measured),
      L_KF_(L_KF),
      K_(K),
      camera_(gtsam::Pose3::Identity(), K),
      throw_cheirality_(throw_cheirality) {}

gtsam::Vector StereoHybridMotionFactorBase::evaluateError(
    const gtsam::Pose3& X_k, const gtsam::Pose3& e_H_k_world,
    const gtsam::Point3& m_L, boost::optional<gtsam::Matrix&> J1,
    boost::optional<gtsam::Matrix&> J2,
    boost::optional<gtsam::Matrix&> J3) const {
  // 1. Project map point to camera frame using HybridObjectMotion
  // We need Jacobians w.r.t inputs, but L_e is constant here so we pass none
  // for it.
  try {
    gtsam::Matrix36 H_cam_X, H_cam_E;
    gtsam::Matrix33 H_cam_m;
    const gtsam::Point3 m_camera = HybridObjectMotion::projectToCamera3(
        X_k, e_H_k_world, L_KF_, m_L, J1 ? &H_cam_X : 0, J2 ? &H_cam_E : 0,
        boost::none,  // L_e is fixed
        J3 ? &H_cam_m : 0);

    gtsam::Matrix36
        H_stereo_pose;  // J1: Jacobian w.r.t the Camera Pose (Identity) - 3x6
    gtsam::Matrix33
        H_stereo_point;  // J2: Jacobian w.r.t the Point (p_camera) - 3x3
                         // We request both Jacobians as requested, though only
                         // J2 is needed for the chain rule since the camera
                         // pose (Identity) is constant and not part of the
                         // optimization.
    const gtsam::StereoPoint2 projected_stereo =
        camera_.project2(m_camera, (J1 || J2 || J3) ? &H_stereo_pose : 0,
                         (J1 || J2 || J3) ? &H_stereo_point : 0);
    // dErr/dX
    if (J1) *J1 = H_stereo_point * H_cam_X;
    // dErr/dE
    if (J2) *J2 = H_stereo_point * H_cam_E;
    // dErr/dm
    if (J3) *J3 = H_stereo_point * H_cam_m;

    // 3. Compute Error
    const gtsam::Vector3 error = (projected_stereo - measured_).vector();
    return error;
  } catch (gtsam::StereoCheiralityException& e) {
    if (J1) *J1 = gtsam::Matrix36::Zero();
    if (J2) *J2 = gtsam::Matrix36::Zero();
    if (J3) *J3 = gtsam::Matrix33::Zero();

    if (throw_cheirality_) {
      // only inherting classes (or using noise models) know which
      // key the point is associated with
      // throw custom exception and let using class catch and throw
      // gtsam::StereoCheiralityException once gtsam::Key is known!
      throw CheiralityException();
    }
  }
  return gtsam::Vector3::Constant(2.0 * K_->fx());
}

void StereoHybridMotionFactorBase::print(const std::string& s,
                                         const gtsam::KeyFormatter&) const {
  std::cout << s << "StereoHybridMotionFactorBase z = " << measured_ << "\n";
  std::cout << "L_KF= " << L_KF_ << "\n";
}

bool StereoHybridMotionFactorBase::equals(const StereoHybridMotionFactorBase& f,
                                          double tol) const {
  if (!gtsam::traits<gtsam::Pose3>::Equals(f.L_KF_, this->L_KF_, tol)) {
    return false;
  }

  if (!gtsam::traits<gtsam::StereoPoint2>::Equals(f.measured_, this->measured_,
                                                  tol)) {
    return false;
  }

  if (!gtsam::traits<gtsam::Cal3_S2Stereo>::Equals(*f.K_, *this->K_, tol)) {
    return false;
  }

  return true;
}

const gtsam::StereoPoint2& StereoHybridMotionFactorBase::measured() const {
  return measured_;
}
const gtsam::Cal3_S2Stereo::shared_ptr
StereoHybridMotionFactorBase::calibration() const {
  return K_;
}

const gtsam::Pose3& StereoHybridMotionFactorBase::referencePose() const {
  return L_KF_;
}

StereoHybridMotionFactor::StereoHybridMotionFactor(
    const gtsam::StereoPoint2& measured, const gtsam::Pose3& L_e,
    const gtsam::SharedNoiseModel& model, gtsam::Cal3_S2Stereo::shared_ptr K,
    gtsam::Key X_k_key, gtsam::Key e_H_k_world_key, gtsam::Key m_L_key,
    bool throw_cheirality)
    : Base(model, X_k_key, e_H_k_world_key, m_L_key),
      StereoHybridMotionFactorBase(measured, L_e, K, throw_cheirality) {}

gtsam::NonlinearFactor::shared_ptr StereoHybridMotionFactor::clone() const {
  return boost::static_pointer_cast<gtsam::NonlinearFactor>(
      gtsam::NonlinearFactor::shared_ptr(new This(*this)));
}

void StereoHybridMotionFactor::print(
    const std::string& s, const gtsam::KeyFormatter& keyFormatter) const {
  Base::print(s, keyFormatter);
  measured_.print(s + ".z");
}

gtsam::Vector StereoHybridMotionFactor::evaluateError(
    const gtsam::Pose3& X_k, const gtsam::Pose3& e_H_k_world,
    const gtsam::Point3& m_L, boost::optional<gtsam::Matrix&> J1,
    boost::optional<gtsam::Matrix&> J2,
    boost::optional<gtsam::Matrix&> J3) const {
  try {
    return StereoHybridMotionFactorBase::evaluateError(X_k, e_H_k_world, m_L,
                                                       J1, J2, J3);
  } catch (const CheiralityException&) {
    // only derived class knows about the key so throw here not in base
    // which throws CheiralityException
    // CheiralityException is only thrown if throw_cheirality true
    throw gtsam::StereoCheiralityException(this->key3());
  }
}

StereoHybridMotionFactor2::StereoHybridMotionFactor2(
    const gtsam::StereoPoint2& measured, const gtsam::Pose3& L_e,
    const gtsam::Pose3& X_W_k, const gtsam::SharedNoiseModel& model,
    gtsam::Cal3_S2Stereo::shared_ptr K, gtsam::Key e_H_k_world_key,
    gtsam::Key m_L_key, bool throw_cheirality)
    : Base(model, e_H_k_world_key, m_L_key),
      StereoHybridMotionFactorBase(measured, L_e, K, throw_cheirality),
      X_W_k_(X_W_k) {}

gtsam::Vector StereoHybridMotionFactor2::evaluateError(
    const gtsam::Pose3& e_H_k_world, const gtsam::Point3& m_L,
    boost::optional<gtsam::Matrix&> J1,
    boost::optional<gtsam::Matrix&> J2) const {
  try {
    // J1 corresponds with second entry in Base (ie. motion)
    // J2 corresponds with third entry in Base (ie. point)
    return StereoHybridMotionFactorBase::evaluateError(X_W_k_, e_H_k_world, m_L,
                                                       {}, J1, J2);
  } catch (const CheiralityException&) {
    // only derived class knows about the key so throw here not in base
    // which throws CheiralityException
    // CheiralityException is only thrown if throw_cheirality true
    throw gtsam::StereoCheiralityException(this->key2());
  }
}

StereoHybridMotionFactor3::StereoHybridMotionFactor3(
    const gtsam::StereoPoint2& measured, const gtsam::Pose3& L_KF,
    const gtsam::Pose3& X_W_k, const gtsam::Point3& m_L,
    const gtsam::SharedNoiseModel& model, gtsam::Cal3_S2Stereo::shared_ptr K,
    gtsam::Key H_W_K_k_key, bool throw_cheirality)
    : Base(model, H_W_K_k_key),
      StereoHybridMotionFactorBase(measured, L_KF, K, throw_cheirality),
      X_W_k_(X_W_k),
      m_L_(m_L) {}

void StereoHybridMotionFactor3::print(
    const std::string& s, const gtsam::KeyFormatter& keyFormatter) const {
  std::cout << s << "StereoHybridMotionFactor3 ";
  StereoHybridMotionFactorBase::print(s);
  std::cout << "X=" << X_W_k_ << "\n";
  std::cout << "point=" << m_L_ << "\n";
  Base::print(s, keyFormatter);
}

bool StereoHybridMotionFactor3::equals(const gtsam::NonlinearFactor& f,
                                       double tol) const {
  if (const This* e = dynamic_cast<const This*>(&f)) {
    if (!gtsam::traits<gtsam::Point3>::Equals(e->m_L_, this->m_L_, tol)) {
      return false;
    }

    if (!gtsam::traits<gtsam::Pose3>::Equals(e->X_W_k_, this->X_W_k_, tol)) {
      return false;
    }

    return StereoHybridMotionFactorBase::equals(*e, tol);

  } else {
    return false;
  }
}

gtsam::Vector StereoHybridMotionFactor3::evaluateError(
    const gtsam::Pose3& H_W_KF_k, boost::optional<gtsam::Matrix&> J1) const {
  try {
    // J1 corresponds with second entry in Base (ie. motion)
    return StereoHybridMotionFactorBase::evaluateError(X_W_k_, H_W_KF_k, m_L_,
                                                       {}, J1, {});
  } catch (const CheiralityException&) {
    // only derived class knows about the key so throw here not in base
    // which throws CheiralityException
    // CheiralityException is only thrown if throw_cheirality true
    throw gtsam::StereoCheiralityException(this->key1());
  }
}

BatchStereoHybridMotionFactor3::BatchStereoHybridMotionFactor3(
    const gtsam::Point3& m_L, const gtsam::Pose3& L_KF,
    const gtsam::SharedNoiseModel& model, gtsam::Cal3_S2Stereo::shared_ptr K,
    bool use_hessian_factor)
    : m_L_(m_L),
      L_KF_(L_KF),
      noise_model_(model),
      K_(K),
      useHessianFactor_(use_hessian_factor) {}

double BatchStereoHybridMotionFactor3::error(const gtsam::Values& c) const {
  double total_error = 0.0;
  for (const auto& f : factors_) {
    total_error += f.error(c);
  }
  return total_error;
}

void BatchStereoHybridMotionFactor3::print(
    const std::string& s, const gtsam::KeyFormatter& keyFormatter) const {
  gtsam::NonlinearFactor::print(s, keyFormatter);
  std::cout << "BatchStereoHybridMotionFactor3 with " << factors_.size()
            << " factors:" << std::endl;
  for (const auto& f : factors_) {
    f.print("", keyFormatter);
  }
}

bool BatchStereoHybridMotionFactor3::equals(const gtsam::NonlinearFactor& f,
                                            double tol) const {
  const This* p = dynamic_cast<const This*>(&f);
  if (!p || factors_.size() != p->factors_.size()) return false;
  for (size_t i = 0; i < factors_.size(); ++i) {
    if (!factors_[i].equals(p->factors_[i], tol)) return false;
  }
  return true;
}

size_t BatchStereoHybridMotionFactor3::dim() const {
  return factors_.size() * 3;
}

boost::shared_ptr<gtsam::GaussianFactor>
BatchStereoHybridMotionFactor3::linearize(const gtsam::Values& values) const {
  if (factors_.empty()) {
    return boost::make_shared<gtsam::JacobianFactor>();
  }

  constexpr size_t ErrorDim = 3;

  const size_t total_rows = factors_.size() * ErrorDim;

  std::vector<size_t> dims(keys_.size(), 6);
  gtsam::VerticalBlockMatrix Ab(dims, total_rows, true);
  Ab.matrix().setZero();

  // std::vector<gtsam::Matrix> H(BatchStereoHybridMotionFactor3::N);
  std::vector<gtsam::Matrix> H(1);

  for (size_t i = 0; i < factors_.size(); ++i) {
    const auto& factor = factors_[i];
    const size_t row = i * ErrorDim;
    const auto index = indices_[i];

    gtsam::Vector raw_error = factor.unwhitenedError(values, H);

    if (factor.noiseModel()) {
      factor.noiseModel()->WhitenSystem(H, raw_error);
    }

    Ab(index).block(row, 0, ErrorDim, H[0].cols()) = H[0];
    Ab(keys_.size()).block(row, 0, ErrorDim, 1) = -raw_error;
  }

  auto jacobian = boost::make_shared<gtsam::JacobianFactor>(
      keys_, std::move(Ab), gtsam::noiseModel::Unit::Create(total_rows));

  if (useHessianFactor_) {
    return boost::make_shared<gtsam::HessianFactor>(*jacobian);
  }

  return jacobian;
}

void BatchStereoHybridMotionFactor3::add(const gtsam::StereoPoint2& measured,
                                         const gtsam::Pose3& X_W_k,
                                         gtsam::Key H_W_K_k_key) {
  // Reject duplicate keys
  auto it = std::find(keys_.begin(), keys_.end(), H_W_K_k_key);
  if (it != keys_.end()) {
    throw std::runtime_error(
        "BatchStereoHybridMotionFactor3::add(): duplicate key added");
  }

  // Register key
  keys_.push_back(H_W_K_k_key);
  indices_.push_back(keys_.size() - 1);

  // Create and store factor
  factors_.emplace_back(measured, L_KF_, X_W_k, m_L_, noise_model_, K_,
                        H_W_K_k_key, true);
}

gtsam::Vector HybridSmoothingFactorBase::evaluateError(
    const gtsam::Pose3& H_W_KF_km2, const gtsam::Pose3& H_W_KF_km1,
    const gtsam::Pose3& H_W_KF_k, boost::optional<gtsam::Matrix&> J1,
    boost::optional<gtsam::Matrix&> J2,
    boost::optional<gtsam::Matrix&> J3) const {
  return HybridObjectMotion::constantMotionResidual(
      H_W_KF_km2, this->keyframePosekm2(), H_W_KF_km1, this->keyframePosekm1(),
      H_W_KF_k, this->keyframePosek(), J1, J2, J3);
}

}  // namespace dyno
