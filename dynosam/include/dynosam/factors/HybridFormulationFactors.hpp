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
#pragma once
#include <gtsam/base/numericalDerivative.h>  //only needed for factors

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam_common/Types.hpp"  //only needed for factors

namespace dyno {

/**
 * @brief Definitions for common Hybrid formulation functions
 *
 */
struct HybridObjectMotion {
  /**
   * @brief Project a point in the camera frame (at time k) into the object
   * frame given the key-framed motion and the embedded pose.
   *
   * Implements: ^L_m = ^WL_e^{-1} ^W_eH_k ^WX_k ^{X_k}m_k
   *
   * @param X_k const gtsam::Pose3& X_k Observing camera pose.
   * @param e_H_k_world const gtsam::Pose3& object motion from s to k in the
   * world frame.
   * @param L_s0 const gtsam::Pose3& embedded object frame.
   * @param Z_k const gtsam::Point3& measured 3D point in the camera (X_k)
   * frame.
   * @return gtsam::Point3 point in the object frame (m_L).
   */
  static gtsam::Point3 projectToObject3(
      const gtsam::Pose3& X_k, const gtsam::Pose3& e_H_k_world,
      const gtsam::Pose3& L_s0, const gtsam::Point3& Z_k,
      gtsam::OptionalJacobian<3, 6> J1 = boost::none,
      gtsam::OptionalJacobian<3, 6> J2 = boost::none,
      gtsam::OptionalJacobian<3, 6> J3 = boost::none);

  /**
   * @brief Project a point in the object frame to the camera frame (at time k)
   * given the key-framed motion and the embedded pose.
   *
   * Implements: z_k = ^{X_k}m_k =  ^WX_k^{-1} ^W_eH_k ^WL_e ^L_m
   *
   * @param X_k const gtsam::Pose3& X_k Observing camera pose.
   * @param e_H_k_world const gtsam::Pose3& object motion from e to k in the
   * world frame.
   * @param L_e const gtsam::Pose3& embedded object frame.
   * @param m_L gtsam::Point3 point in the object frame (m_L).
   * @return gtsam::Point3 measured 3D point in the camera frame (z_k).
   */
  static gtsam::Point3 projectToCamera3(
      const gtsam::Pose3& X_k, const gtsam::Pose3& e_H_k_world,
      const gtsam::Pose3& L_e, const gtsam::Point3& m_L,
      gtsam::OptionalJacobian<3, 6> J1 = boost::none,
      gtsam::OptionalJacobian<3, 6> J2 = boost::none,
      gtsam::OptionalJacobian<3, 6> J3 = boost::none,
      gtsam::OptionalJacobian<3, 3> J4 = boost::none);

  /**
   * @brief Constructs the transform that projects a point/pose in L_e into W.
   * This is also the design matrix (A in Ax=b) LHSfor the linear system.
   *
   * @param X_k  const gtsam::Pose3& X_k Observing camera pose.
   * @param e_H_k_world const gtsam::Pose3& object motion from eto k in the
   * world frame.
   * @param L_e const gtsam::Pose3& embedded object frame.
   * @return gtsam::Pose3
   */
  static gtsam::Pose3 projectToCamera3Transform(
      const gtsam::Pose3& X_k, const gtsam::Pose3& e_H_k_world,
      const gtsam::Pose3& L_e, gtsam::OptionalJacobian<6, 6> J1 = boost::none,
      gtsam::OptionalJacobian<6, 6> J2 = boost::none,
      gtsam::OptionalJacobian<6, 6> J3 = boost::none);

  /**
   * @brief Residual 3D error for a measured 3D point (z_k) and an estimated
   * point in the object frame (m_L) at time k, given the key-framed motion and
   * the embedded pose.
   *
   * Implements z_k - ^WX_k^{-1} ^W_eH_k ^WL_e ^L_m
   *
   * @param X_k const gtsam::Pose3& X_k Observing camera pose at k
   * @param e_H_k_world const gtsam::Pose3& object motion from s to k in the
   * world frame.
   * @param m_L gtsam::Point3 point in the object frame (m_L).
   * @param Z_k const gtsam::Point3 3D point measurement in the camera frame
   * (z_k).
   * @param L_e const gtsam::Pose3& embedded object frame.
   * @return gtsam::Vector3
   */
  static gtsam::Vector3 residual(const gtsam::Pose3& X_k,
                                 const gtsam::Pose3& e_H_k_world,
                                 const gtsam::Point3& m_L,
                                 const gtsam::Point3& Z_k,
                                 const gtsam::Pose3& L_e);

  static gtsam::Vector6 constantMotionResidual(
      const gtsam::Pose3& H_W_KF_km2, const gtsam::Pose3& L_W_KFkm2,
      const gtsam::Pose3& H_W_KF_km1, const gtsam::Pose3& L_W_KFkm1,
      const gtsam::Pose3& H_W_KF_k, const gtsam::Pose3& L_W_KFk,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none,
      boost::optional<gtsam::Matrix&> J3 = boost::none);
};

/**
 * @brief Motion factor connecting a point in the object frame (^L_m), the
 * key-framed object motion from e to k in W (^W_eH_k) and the observing camera
 * pose (^WX_k).
 *
 * Error residual is in the camera local frame.
 *
 */
class HybridMotionFactor
    : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3,
                                      gtsam::Point3>,
      public HybridObjectMotion {
 public:
  typedef boost::shared_ptr<HybridMotionFactor> shared_ptr;
  typedef HybridMotionFactor This;
  typedef gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Point3>
      Base;

  gtsam::Point3 z_k_;
  gtsam::Pose3 L_e_;

  HybridMotionFactor(gtsam::Key X_k_key, gtsam::Key e_H_k_world_key,
                     gtsam::Key m_L_key, const gtsam::Point3& z_k,
                     const gtsam::Pose3& L_e, gtsam::SharedNoiseModel model)
      : Base(model, X_k_key, e_H_k_world_key, m_L_key), z_k_(z_k), L_e_(L_e) {}

  gtsam::Vector evaluateError(
      const gtsam::Pose3& X_k, const gtsam::Pose3& e_H_k_world,
      const gtsam::Point3& m_L,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none,
      boost::optional<gtsam::Matrix&> J3 = boost::none) const override;
};

class StereoHybridMotionFactorBase {
 public:
  /**
   * @brief Exception thrown by evaluate error when stereo projection raises a
   * Cheirality Exception.
   *
   * Used instead of gtsam::StereoCheiralityException becuase the gtsam version
   * takes a gtsam::Key as input and this base class does not know about the
   * keys.
   *
   * Intendied so that the using class just catches this exception and then can
   * throw the gtsam version if necessary.
   *
   * Exception is only thrown if throw_cheirality_ is true so no need to check
   * the flag
   *
   */
  class CheiralityException : public std::runtime_error {
   public:
    CheiralityException()
        : std::runtime_error(
              "Cheirality occured in StereoHybridMotionFactorBase") {}
  };

  StereoHybridMotionFactorBase(const gtsam::StereoPoint2& measured,
                               const gtsam::Pose3& L_KF,
                               gtsam::Cal3_S2Stereo::shared_ptr K,
                               bool throw_cheirality = false);

  const gtsam::StereoPoint2& measured() const;
  const gtsam::Cal3_S2Stereo::shared_ptr calibration() const;
  const gtsam::Pose3& referencePose() const;

  void print(
      const std::string& s = "",
      const gtsam::KeyFormatter& keyFormatter = DynosamKeyFormatter) const;

  bool equals(const StereoHybridMotionFactorBase& f, double tol = 1e-9) const;

  gtsam::Vector evaluateError(
      const gtsam::Pose3& X_k, const gtsam::Pose3& e_H_k_world,
      const gtsam::Point3& m_L,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none,
      boost::optional<gtsam::Matrix&> J3 = boost::none) const;

 protected:
  gtsam::StereoPoint2 measured_;
  // fixed reference frame
  gtsam::Pose3 L_KF_;
  gtsam::Cal3_S2Stereo::shared_ptr K_;
  // with identity pose, acts as the refenence frame
  gtsam::StereoCamera camera_;

 private:
  bool throw_cheirality_;
};

class StereoHybridMotionFactor
    : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3,
                                      gtsam::Point3>,
      public StereoHybridMotionFactorBase {
 public:
  using This = StereoHybridMotionFactor;
  using Base =
      gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Point3>;

  StereoHybridMotionFactor(const gtsam::StereoPoint2& measured,
                           const gtsam::Pose3& L_e,
                           const gtsam::SharedNoiseModel& model,
                           gtsam::Cal3_S2Stereo::shared_ptr K,
                           gtsam::Key X_k_key, gtsam::Key e_H_k_world_key,
                           gtsam::Key m_L_key, bool throw_cheirality = false);

  gtsam::NonlinearFactor::shared_ptr clone() const override;
  void print(const std::string& s = "",
             const gtsam::KeyFormatter& keyFormatter =
                 DynosamKeyFormatter) const override;

  gtsam::Vector evaluateError(
      const gtsam::Pose3& X_k, const gtsam::Pose3& e_H_k_world,
      const gtsam::Point3& m_L,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none,
      boost::optional<gtsam::Matrix&> J3 = boost::none) const override;
};

/**
 * @brief Stereo Hybrid Motion factor \sigma(H, m) with a fixed camera pose X
 *
 */
class StereoHybridMotionFactor2
    : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3>,
      public StereoHybridMotionFactorBase {
 public:
  using This = StereoHybridMotionFactor2;
  using Base = gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3>;

  StereoHybridMotionFactor2(const gtsam::StereoPoint2& measured,
                            const gtsam::Pose3& L_e, const gtsam::Pose3& X_W_k,
                            const gtsam::SharedNoiseModel& model,
                            gtsam::Cal3_S2Stereo::shared_ptr K,
                            gtsam::Key e_H_k_world_key, gtsam::Key m_L_key,
                            bool throw_cheirality = false);

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }

  gtsam::Vector evaluateError(
      const gtsam::Pose3& e_H_k_world, const gtsam::Point3& m_L,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none) const override;

  const gtsam::Pose3& cameraPose() const { return X_W_k_; }

 private:
  //! Fixed camera pose
  gtsam::Pose3 X_W_k_;
};

/**
 * @brief Stereo Hybrid Motion factor \sigma(H) with a fixed camera pose X
 * and a fixed point ^Lm
 *
 */
class StereoHybridMotionFactor3 : public gtsam::NoiseModelFactor1<gtsam::Pose3>,
                                  public StereoHybridMotionFactorBase {
 public:
  using This = StereoHybridMotionFactor3;
  using Base = gtsam::NoiseModelFactor1<gtsam::Pose3>;

  using shared_ptr = boost::shared_ptr<This>;

  StereoHybridMotionFactor3(const gtsam::StereoPoint2& measured,
                            const gtsam::Pose3& L_KF, const gtsam::Pose3& X_W_k,
                            const gtsam::Point3& m_L,
                            const gtsam::SharedNoiseModel& model,
                            gtsam::Cal3_S2Stereo::shared_ptr K,
                            gtsam::Key H_W_K_k_key,
                            bool throw_cheirality = false);

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }

  void print(const std::string& s = "",
             const gtsam::KeyFormatter& keyFormatter =
                 DynosamKeyFormatter) const override;

  bool equals(const gtsam::NonlinearFactor& f,
              double tol = 1e-9) const override;

  gtsam::Vector evaluateError(
      const gtsam::Pose3& H_W_KF_k,
      boost::optional<gtsam::Matrix&> J1 = boost::none) const override;

  const gtsam::Pose3& cameraPose() const { return X_W_k_; }
  const gtsam::Point3& objectPoint() const { return m_L_; }

  // allows external updating of the internal point
  // it is the users responsability to ensure the factor
  // is correctly relinearized with the new point after update!
  void objectPoint(const gtsam::Point3& m_L) { m_L_ = m_L; }

 private:
  //! Fixed camera pose
  gtsam::Pose3 X_W_k_;
  //! Fixed observed point
  gtsam::Point3 m_L_;
};

/**
 * @brief Variant of the StereoHybridMotionFactor3 which captures multiple
 * measurements of a single object point (m_L) and estimates for the motion
 * H_W_KF_K.
 *
 * Like the StereoHybridMotionFactor3 this assumes the point (m_L) and the
 * observing camera pose (X_W_k) are known.
 *
 * A single noise model is used for all measurements.
 *
 * Like a smart factor, if used with iSAM2, the smoother must be alernted that
 * the factor needs to be rellinearized once the factor is added to the smoother
 * and new measurements added.
 *
 */
class BatchStereoHybridMotionFactor3 : public gtsam::NonlinearFactor {
 public:
  using shared_ptr = boost::shared_ptr<BatchStereoHybridMotionFactor3>;
  using This = BatchStereoHybridMotionFactor3;

 private:
  using Allocator = Eigen::aligned_allocator<StereoHybridMotionFactor3>;
  std::vector<StereoHybridMotionFactor3, Allocator> factors_;
  std::vector<gtsam::DenseIndex> indices_;
  bool useHessianFactor_{false};

  //! Fixed observed point
  gtsam::Point3 m_L_;
  gtsam::Pose3 L_KF_;

  gtsam::SharedNoiseModel noise_model_;
  gtsam::Cal3_S2Stereo::shared_ptr K_;

 public:
  BatchStereoHybridMotionFactor3(const gtsam::Point3& m_L,
                                 const gtsam::Pose3& L_KF,
                                 const gtsam::SharedNoiseModel& model,
                                 gtsam::Cal3_S2Stereo::shared_ptr K);

  void print(const std::string& s = "",
             const gtsam::KeyFormatter& keyFormatter =
                 DynosamKeyFormatter) const override;

  /// Check equality with another factor.
  bool equals(const gtsam::NonlinearFactor& f,
              double tol = 1e-9) const override;

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }

  double error(const gtsam::Values& c) const override;

  /// Get the dimension of the factor (number of rows on linearization)
  size_t dim() const override;

  /**
   * Linearize to a single JacobianFactor.
   *
   * Optimization:
   * - Pre-calculates the total size required for the JacobianFactor.
   * - Collects all unique Keys involved across all sub-factors.
   * - Iterates linearly over factors_ (cache-friendly) to compute Jacobians.
   * - Fills the pre-allocated JacobianFactor directly.
   */
  boost::shared_ptr<gtsam::GaussianFactor> linearize(
      const gtsam::Values& values) const override;

  void add(const gtsam::StereoPoint2& measured, const gtsam::Pose3& X_W_k,
           gtsam::Key H_W_K_k_key);

 private:
};

/**
 * @brief Implements a 3-way smoothing factor on the (key-framed) object motion.
 * This is analgous to a constant motion prior and minimises the change in
 * object motion in the body frame of the object.
 *
 */
class HybridSmoothingFactorBase
    : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3,
                                      gtsam::Pose3> {
 public:
  typedef boost::shared_ptr<HybridSmoothingFactorBase> shared_ptr;
  typedef HybridSmoothingFactorBase This;
  typedef gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>
      Base;

  HybridSmoothingFactorBase(gtsam::Key H_W_KF_km2_key,
                            gtsam::Key H_W_KF_km1_key, gtsam::Key H_W_KF_k_key,
                            gtsam::SharedNoiseModel model)
      : Base(model, H_W_KF_km2_key, H_W_KF_km1_key, H_W_KF_k_key) {}

  gtsam::Vector evaluateError(
      const gtsam::Pose3& H_W_KF_km2, const gtsam::Pose3& H_W_KF_km1,
      const gtsam::Pose3& H_W_KF_k,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none,
      boost::optional<gtsam::Matrix&> J3 = boost::none) const override;

  virtual const gtsam::Pose3& keyframePosekm2() const = 0;
  virtual const gtsam::Pose3& keyframePosekm1() const = 0;
  virtual const gtsam::Pose3& keyframePosek() const = 0;
};

/**
 * @brief Implements a 3-way smoothing factor on the (key-framed) object motion.
 * This is analgous to a constant motion prior and minimises the change in
 * object motion in the body frame of the object.
 *
 * Assumes all motions refer to the same keyframe pose
 *
 */
class HybridSmoothingFactor : public HybridSmoothingFactorBase {
 public:
  typedef boost::shared_ptr<HybridSmoothingFactor> shared_ptr;
  typedef HybridSmoothingFactor This;

  HybridSmoothingFactor(gtsam::Key e_H_km2_world_key,
                        gtsam::Key e_H_km1_world_key,
                        gtsam::Key e_H_k_world_key, const gtsam::Pose3& L_e,
                        gtsam::SharedNoiseModel model)
      : HybridSmoothingFactorBase(e_H_km2_world_key, e_H_km1_world_key,
                                  e_H_k_world_key, model),
        L_e_(L_e) {}

  const gtsam::Pose3& keyframePosekm2() const override { return L_e_; }
  const gtsam::Pose3& keyframePosekm1() const override { return L_e_; }
  const gtsam::Pose3& keyframePosek() const override { return L_e_; }

 private:
  gtsam::Pose3 L_e_;
};

/**
 * @brief Implements a 3-way smoothing factor on the (key-framed) object motion.
 * This is analgous to a constant motion prior and minimises the change in
 * object motion in the body frame of the object.
 *
 * Same as HybridSmoothingFactor except each motion can be associated with a
 * different pose
 *
 */
class HybridSmoothingFactor2 : public HybridSmoothingFactorBase {
 public:
  typedef boost::shared_ptr<HybridSmoothingFactor2> shared_ptr;
  typedef HybridSmoothingFactor2 This;

  HybridSmoothingFactor2(gtsam::Key H_W_KF_km2_key, gtsam::Key H_W_KF_km1_key,
                         gtsam::Key H_W_KF_k_key, const gtsam::Pose3& L_W_KFkm2,
                         const gtsam::Pose3& L_W_KFkm1,
                         const gtsam::Pose3& L_W_KFk,
                         gtsam::SharedNoiseModel model)
      : HybridSmoothingFactorBase(H_W_KF_km2_key, H_W_KF_km1_key, H_W_KF_k_key,
                                  model),
        L_W_KFkm2_(L_W_KFkm2),
        L_W_KFkm1_(L_W_KFkm1),
        L_W_KFk_(L_W_KFk) {}

  const gtsam::Pose3& keyframePosekm2() const override { return L_W_KFkm2_; }
  const gtsam::Pose3& keyframePosekm1() const override { return L_W_KFkm1_; }
  const gtsam::Pose3& keyframePosek() const override { return L_W_KFk_; }

 private:
  gtsam::Pose3 L_W_KFkm2_;
  gtsam::Pose3 L_W_KFkm1_;
  gtsam::Pose3 L_W_KFk_;
};

}  // namespace dyno
