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

#include <gtsam/base/numericalDerivative.h>  //only needed for factors

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/common/Types.hpp"  //only needed for factors

namespace dyno {

gtsam::Point3 projectToObject(const gtsam::Pose3& X_k,
                              const gtsam::Pose3& s0_H_k_world,
                              const gtsam::Pose3& L_s0,
                              const gtsam::Point3 Z_k);

struct ObjectCentricMotion {
  // error residual for a 3d point in the local frame given the KF
  // representation of motion
  static gtsam::Vector3 residual(const gtsam::Pose3& X_k,
                                 const gtsam::Pose3& s0_H_k_world,
                                 const gtsam::Point3& m_L,
                                 const gtsam::Point3& Z_k,
                                 const gtsam::Pose3& L_0);
};

// optimizes camera pose, object motion and object point using KF motion
// representation
class ObjectCentricMotionFactor
    : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3,
                                      gtsam::Point3>,
      public ObjectCentricMotion {
 public:
  typedef boost::shared_ptr<ObjectCentricMotionFactor> shared_ptr;
  typedef ObjectCentricMotionFactor This;
  typedef gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Point3>
      Base;

  gtsam::Point3 measurement_;
  gtsam::Pose3 L_0_;

  ObjectCentricMotionFactor(gtsam::Key camera_pose, gtsam::Key motion,
                            gtsam::Key point_object,
                            const gtsam::Point3& measurement,
                            const gtsam::Pose3& L_0,
                            gtsam::SharedNoiseModel model)
      : Base(model, camera_pose, motion, point_object),
        measurement_(measurement),
        L_0_(L_0) {}

  gtsam::Vector evaluateError(
      const gtsam::Pose3& camera_pose, const gtsam::Pose3& motion,
      const gtsam::Point3& point_object,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none,
      boost::optional<gtsam::Matrix&> J3 = boost::none) const override;
};

class ObjectCentricSmoothing
    : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3,
                                      gtsam::Pose3> {
 public:
  typedef boost::shared_ptr<ObjectCentricSmoothing> shared_ptr;
  typedef ObjectCentricSmoothing This;
  typedef gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>
      Base;

  gtsam::Pose3 L_0_;

  ObjectCentricSmoothing(gtsam::Key motion_k_2, gtsam::Key motion_k_1,
                         gtsam::Key motion_k, const gtsam::Pose3& L_0,
                         gtsam::SharedNoiseModel model)
      : Base(model, motion_k_2, motion_k_1, motion_k), L_0_(L_0) {}

  gtsam::Vector evaluateError(
      const gtsam::Pose3& motion_k_2, const gtsam::Pose3& motion_k_1,
      const gtsam::Pose3& motion_k,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none,
      boost::optional<gtsam::Matrix&> J3 = boost::none) const override;

  static gtsam::Vector residual(const gtsam::Pose3& motion_k_2,
                                const gtsam::Pose3& motion_k_1,
                                const gtsam::Pose3& motion_k,
                                const gtsam::Pose3& L_0);
};

// optimizes object motion and object point using KF motion representation
// assumes camera pose is known
class DecoupledObjectCentricMotionFactor
    : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3>,
      public ObjectCentricMotion {
 public:
  typedef boost::shared_ptr<DecoupledObjectCentricMotionFactor> shared_ptr;
  typedef DecoupledObjectCentricMotionFactor This;
  typedef gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3> Base;

  using ObjectCentricMotion::residual;

  gtsam::Point3 Z_k_;
  gtsam::Pose3 L_0_;
  gtsam::Pose3 X_k_;

  DecoupledObjectCentricMotionFactor(gtsam::Key motion_key,
                                     gtsam::Key point_object_key,
                                     const gtsam::Point3& Z_k,
                                     const gtsam::Pose3& L_0,
                                     const gtsam::Pose3& X_k,
                                     gtsam::SharedNoiseModel model)
      : Base(model, motion_key, point_object_key),
        Z_k_(Z_k),
        L_0_(L_0),
        X_k_(X_k) {}

  gtsam::Vector evaluateError(
      const gtsam::Pose3& s0_H_k_world, const gtsam::Point3& m_L,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none) const override;
};

struct StructurelessObjectCentricMotion2 {
  // error residual given 2 views of a single point on an object
  // given the camera poses, motions and measurements of that point at k-1 and k
  // using a KF representation of motion
  static gtsam::Vector residual(const gtsam::Pose3& X_k_1,
                                const gtsam::Pose3& H_k_1,
                                const gtsam::Pose3& X_k,
                                const gtsam::Pose3& H_k,
                                const gtsam::Point3& Z_k_1,
                                const gtsam::Point3& Z_k,
                                const gtsam::Pose3& L_0);
};

// structurless object motion factor between two H's and two observin X's
//  all other variables are fixed
// TODO: should add the 2 to the class name to indicate the 2-view constraint?
class StructurelessDecoupledObjectCentricMotion
    : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>,
      public StructurelessObjectCentricMotion2 {
 public:
  typedef boost::shared_ptr<StructurelessDecoupledObjectCentricMotion>
      shared_ptr;
  typedef StructurelessDecoupledObjectCentricMotion This;
  typedef gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> Base;

  using StructurelessObjectCentricMotion2::residual;

  gtsam::Point3 Z_k_1_;
  gtsam::Point3 Z_k_;
  gtsam::Pose3 L_0_;
  gtsam::Pose3 X_k_1_;
  gtsam::Pose3 X_k_;

  StructurelessDecoupledObjectCentricMotion(
      gtsam::Key H_k_1_key, gtsam::Key H_k_key, const gtsam::Pose3& X_k_1,
      const gtsam::Pose3& X_k, const gtsam::Point3& Z_k_1,
      const gtsam::Point3& Z_k, const gtsam::Pose3& L_0,
      gtsam::SharedNoiseModel model)
      : Base(model, H_k_1_key, H_k_key),
        Z_k_1_(Z_k_1),
        Z_k_(Z_k),
        L_0_(L_0),
        X_k_1_(X_k_1),
        X_k_(X_k) {}

  gtsam::Vector evaluateError(
      const gtsam::Pose3& H_k_1, const gtsam::Pose3& H_k,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none) const override;
};

// structurless object motion factor between two H's and two observin X's
// TODO: this is NOT decoupled as we optimize for X
class StructurelessObjectCentricMotionFactor2
    : public gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3,
                                      gtsam::Pose3>,
      public StructurelessObjectCentricMotion2 {
 public:
  typedef boost::shared_ptr<StructurelessObjectCentricMotionFactor2> shared_ptr;
  typedef StructurelessObjectCentricMotionFactor2 This;
  typedef gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3,
                                   gtsam::Pose3>
      Base;

  gtsam::Point3 Z_k_1_;
  gtsam::Point3 Z_k_;
  gtsam::Pose3 L_0_;

  StructurelessObjectCentricMotionFactor2(
      gtsam::Key X_k_1_key, gtsam::Key H_k_1_key, gtsam::Key X_k_key,
      gtsam::Key H_k_key, const gtsam::Point3& Z_k_1, const gtsam::Point3& Z_k,
      const gtsam::Pose3& L_0, gtsam::SharedNoiseModel model)
      : Base(model, X_k_1_key, H_k_1_key, X_k_key, H_k_key),
        Z_k_1_(Z_k_1),
        Z_k_(Z_k),
        L_0_(L_0) {}

  gtsam::Vector evaluateError(
      const gtsam::Pose3& X_k_1, const gtsam::Pose3& H_k_1,
      const gtsam::Pose3& X_k, const gtsam::Pose3& H_k,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none,
      boost::optional<gtsam::Matrix&> J3 = boost::none,
      boost::optional<gtsam::Matrix&> J4 = boost::none) const override;
};

}  // namespace dyno
