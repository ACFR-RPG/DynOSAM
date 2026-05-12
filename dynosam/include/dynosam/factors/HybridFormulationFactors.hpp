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

  // allows external updating of the reference
  // it is the users responsability to ensure the factor
  // is correctly relinearized with the new point after update!
  void referencePose(const gtsam::Pose3& L_KF) { L_KF_ = L_KF; }

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
  using shared_ptr = boost::shared_ptr<This>;

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

  // allows external updating of the internal camera pose
  // it is the users responsability to ensure the factor
  // is correctly relinearized with the new point after update!
  void cameraPose(const gtsam::Pose3& X_W_k) { X_W_k_ = X_W_k; }

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

  // allows external updating of the internal camera pose
  // it is the users responsability to ensure the factor
  // is correctly relinearized with the new point after update!
  void cameraPose(const gtsam::Pose3& X_W_k) { X_W_k_ = X_W_k; }

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

  // for testing!
 public:
  using Allocator = Eigen::aligned_allocator<StereoHybridMotionFactor3>;
  std::vector<StereoHybridMotionFactor3, Allocator> factors_;
  std::vector<gtsam::DenseIndex> indices_;

  //! Fixed observed point
  gtsam::Point3 m_L_;
  gtsam::Pose3 L_KF_;

  gtsam::SharedNoiseModel noise_model_;
  gtsam::Cal3_S2Stereo::shared_ptr K_;

  bool useHessianFactor_{false};

 public:
  BatchStereoHybridMotionFactor3(const gtsam::Point3& m_L,
                                 const gtsam::Pose3& L_KF,
                                 const gtsam::SharedNoiseModel& model,
                                 gtsam::Cal3_S2Stereo::shared_ptr K,
                                 bool use_hessian_factor = false);

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

  const gtsam::Point3 point() const { return m_L_; }

 private:
};

template <size_t DIM = 3u, typename MOTION = gtsam::Pose3>
class SmartMotionFactor2 : public gtsam::NonlinearFactor {
 public:
  using Base = gtsam::NonlinearFactor;
  using Motion = MOTION;
  using This = SmartMotionFactor2<DIM, MOTION>;
  using shared_ptr = boost::shared_ptr<This>;

  // Dimensions for Point (3), Motion (6), and Measurement (3)
  static constexpr size_t ZDim = 3;
  static constexpr size_t MDim = 6;
  static constexpr size_t PDim = 3;  // Point3

  // Typedefs for GTSAM compatibility
  using GBlocks = std::vector<gtsam::Matrix>;  // Jacobians w.r.t Motion
  using EBlocks = std::vector<gtsam::Matrix>;  // Jacobians w.r.t Point

 private:
  gtsam::Pose3 L_e_;  // Embedded object frame
  mutable gtsam::TriangulationResult result_;
  gtsam::SharedNoiseModel noise_model_;
  gtsam::Cal3_S2Stereo::shared_ptr K_;
  //   SmartMotionFactorParams params_;

  std::vector<StereoHybridMotionFactorBase> measured_;
  //   std::vector<gtsam::Point3> measured_;      // Measurements
  std::vector<gtsam::Pose3> poses_;  // FIXED camera poses (not keys)

 public:
  SmartMotionFactor2(const gtsam::Pose3& L_e, const gtsam::Point3& m_L,
                     const gtsam::SharedNoiseModel& noise_model,
                     gtsam::Cal3_S2Stereo::shared_ptr K)
      : Base(), L_e_(L_e), result_(m_L), noise_model_(noise_model), K_(K) {}

  /**
   * @brief Add a measurement. Pose is passed as a constant value, not a Key.
   */
  void add(const gtsam::StereoPoint2& measured, const gtsam::Key& motion_key,
           const gtsam::Pose3& fixed_camera_pose) {
    this->measured_.emplace_back(measured, L_e_, K_);
    this->keys_.push_back(motion_key);
    this->poses_.push_back(fixed_camera_pose);
  }

  double error(const gtsam::Values& values) const override {
    if (this->active(values)) {
      std::vector<Motion> motions;
      for (const auto& k : keys_) motions.push_back(values.at<Motion>(k));

      triangulateSafe(motions);
      if (!result_) return 0.0;

      gtsam::Vector b = unwhitenedError(motions, *result_);
      if (noise_model_)
        return noise_model_->loss(noise_model_->squaredMahalanobisDistance(b));
      else
        return 0.5 * b.squaredNorm();
    }
    return 0.0;
  }

  boost::shared_ptr<gtsam::GaussianFactor> linearize(
      const gtsam::Values& values) const override {
    std::vector<Motion> motions;
    for (const auto& k : keys_) motions.push_back(values.at<Motion>(k));

    triangulateSafe(motions);
    if (!result_) return boost::make_shared<gtsam::JacobianFactor>();

    GBlocks Gs;  // W.R.T Motion
    EBlocks Es;  // W.R.T Point
    gtsam::Vector b;

    // 1. Compute Jacobians
    b = -unwhitenedError(motions, *result_, &Gs, &Es);

    // 2. Whiten
    if (noise_model_) {
      for (auto& G : Gs) G = noise_model_->Whiten(G);
      for (auto& E : Es) E = noise_model_->Whiten(E);
      b = noise_model_->whiten(b);
    }

    // 3. Schur Complement Elimination of Point
    // Matrix E is (3*m x 3), Matrix G is (3*m x 6*m) block diagonal
    gtsam::Matrix E_stacked(ZDim * measured_.size(), PDim);
    for (size_t i = 0; i < Es.size(); ++i)
      E_stacked.block<ZDim, PDim>(ZDim * i, 0) = Es[i];

    gtsam::Matrix EtE = E_stacked.transpose() * E_stacked;
    gtsam::Matrix P = EtE.inverse();  // Information inverse for the point

    // Construct the Reduced Hessian (Smart Factor logic)
    // H_reduced = G'G - G'E * (E'E)^-1 * E'G
    // b_reduced = G'b - G'E * (E'E)^-1 * E'b

    size_t m = keys_.size();
    std::vector<Eigen::DenseIndex> dims(m + 1);
    std::fill(dims.begin(), dims.end() - 1, MDim);
    dims.back() = 1;
    gtsam::SymmetricBlockMatrix augmentedHessian(dims);

    for (size_t i = 0; i < m; ++i) {
      for (size_t j = i; j < m; ++j) {
        // Hessian Block (i, j)
        gtsam::Matrix Hij = Gs[i].transpose() * Gs[j];
        gtsam::Matrix E_correction =
            (Gs[i].transpose() * Es[i]) * P * (Es[j].transpose() * Gs[j]);
        augmentedHessian.aboveDiagonalBlock(i, j) = Hij - E_correction;
      }
      // Info vector block (i, last)
      gtsam::Vector bi = Gs[i].transpose() * b.segment<ZDim>(i * ZDim);
      gtsam::Vector b_correction =
          (Gs[i].transpose() * Es[i]) * P * (E_stacked.transpose() * b);
      augmentedHessian.aboveDiagonalBlock(i, m) = bi - b_correction;
    }

    // Constant term (last, last)
    augmentedHessian.aboveDiagonalBlock(m, m) = gtsam::Matrix11(
        b.dot(b) - (b.transpose() * E_stacked * P * E_stacked.transpose() * b));

    return boost::make_shared<gtsam::RegularHessianFactor<MDim>>(
        keys_, augmentedHessian);
  }

 private:
  gtsam::Vector unwhitenedError(const std::vector<Motion>& motions,
                                const gtsam::Point3& point_l,
                                GBlocks* Gs = nullptr,
                                EBlocks* Es = nullptr) const {
    size_t m = measured_.size();
    gtsam::Vector b(ZDim * m);

    CHECK_EQ(motions.size(), this->poses_);

    if (Gs) Gs->resize(m);
    if (Es) Es->resize(m);

    for (size_t i = 0; i < m; ++i) {
      gtsam::Matrix G;
      gtsam::Matrix E;
      gtsam::Vector3 err = measured_[i].evaluateError(
          poses_.at(i), motions.at(i), point_l, {}, G, E);

      //   // h(x) = cam_T_w * (Motion_i * L_e * point_l)
      //   auto project = [&](const Motion& mi, const gtsam::Point3& pl) ->
      //   gtsam::Point3 {
      //     return cam_T_w * (mi * (L_e_ * pl));
      //   };

      if (Gs) (*Gs)[i] = G;
      if (Es) (*Es)[i] = E;

      //   if (Gs) (*Gs)[i] = gtsam::numericalDerivative21<gtsam::Point3,
      //   Motion, gtsam::Point3>(project, motions[i], point_l); if (Es)
      //   (*Es)[i] = gtsam::numericalDerivative22<gtsam::Point3, Motion,
      //   gtsam::Point3>(project, motions[i], point_l);

      b.segment<ZDim>(i * ZDim) = err;
    }
    return b;
  }

  void triangulateSafe(const std::vector<Motion>& motions) const {
    // Logic similar to your provided triangulateSafe,
    // but using this->poses_ (fixed) instead of keys.
    // ... Implementation of triangulation ...
  }
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
