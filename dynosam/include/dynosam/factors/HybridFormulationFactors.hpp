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

class StereoHybridMotionFactor
    : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3,
                                      gtsam::Point3> {
 private:
  gtsam::StereoPoint2 measured_;
  // fixed reference frame
  gtsam::Pose3 L_e_;
  gtsam::Cal3_S2Stereo::shared_ptr K_;

  // with identity pose, acts as the refenence frame
  gtsam::StereoCamera camera_;

  bool throw_cheirality_;

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

  const gtsam::StereoPoint2& measured() const;
  const gtsam::Cal3_S2Stereo::shared_ptr calibration() const;
  const gtsam::Pose3& embeddedPose() const;
};

/**
 * @brief Implements a 3-way smoothing factor on the (key-framed) object motion.
 * This is analgous to a constant motion prior and minimises the change in
 * object motion in the body frame of the object.
 *
 */
class HybridSmoothingFactor
    : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3,
                                      gtsam::Pose3> {
 public:
  typedef boost::shared_ptr<HybridSmoothingFactor> shared_ptr;
  typedef HybridSmoothingFactor This;
  typedef gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>
      Base;

  gtsam::Pose3 L_e_;

  HybridSmoothingFactor(gtsam::Key e_H_km2_world_key,
                        gtsam::Key e_H_km1_world_key,
                        gtsam::Key e_H_k_world_key, const gtsam::Pose3& L_e,
                        gtsam::SharedNoiseModel model)
      : Base(model, e_H_km2_world_key, e_H_km1_world_key, e_H_k_world_key),
        L_e_(L_e) {}

  gtsam::Vector evaluateError(
      const gtsam::Pose3& e_H_km2_world, const gtsam::Pose3& e_H_km1_world,
      const gtsam::Pose3& e_H_k_world,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none,
      boost::optional<gtsam::Matrix&> J3 = boost::none) const override;

  static gtsam::Vector residual(const gtsam::Pose3& e_H_km2_world,
                                const gtsam::Pose3& e_H_km1_world,
                                const gtsam::Pose3& e_H_k_world,
                                const gtsam::Pose3& L_e);
};

}  // namespace dyno
