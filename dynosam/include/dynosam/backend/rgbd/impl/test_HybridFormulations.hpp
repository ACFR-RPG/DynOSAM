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

#include "dynosam/backend/rgbd/HybridEstimator.hpp"
#include "dynosam/factors/HybridFormulationFactors.hpp"

namespace dyno {
namespace test_hybrid {

// optimizes object motion and object point using KF motion representation
// assumes camera pose is known
class DecoupledObjectCentricMotionFactor
    : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3>,
      public HybridObjectMotion {
 public:
  typedef boost::shared_ptr<DecoupledObjectCentricMotionFactor> shared_ptr;
  typedef DecoupledObjectCentricMotionFactor This;
  typedef gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3> Base;

  using HybridObjectMotion::residual;

  gtsam::Point3 Z_k_;
  gtsam::Pose3 L_e_;
  gtsam::Pose3 X_k_;

  DecoupledObjectCentricMotionFactor(gtsam::Key motion_key,
                                     gtsam::Key point_object_key,
                                     const gtsam::Point3& Z_k,
                                     const gtsam::Pose3& L_e,
                                     const gtsam::Pose3& X_k,
                                     gtsam::SharedNoiseModel model)
      : Base(model, motion_key, point_object_key),
        Z_k_(Z_k),
        L_e_(L_e),
        X_k_(X_k) {}

  gtsam::Vector evaluateError(
      const gtsam::Pose3& e_H_k_world, const gtsam::Point3& m_L,
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
                                const gtsam::Pose3& L_e);
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
  gtsam::Pose3 L_e_;
  gtsam::Pose3 X_k_1_;
  gtsam::Pose3 X_k_;

  StructurelessDecoupledObjectCentricMotion(
      gtsam::Key H_k_1_key, gtsam::Key H_k_key, const gtsam::Pose3& X_k_1,
      const gtsam::Pose3& X_k, const gtsam::Point3& Z_k_1,
      const gtsam::Point3& Z_k, const gtsam::Pose3& L_e,
      gtsam::SharedNoiseModel model)
      : Base(model, H_k_1_key, H_k_key),
        Z_k_1_(Z_k_1),
        Z_k_(Z_k),
        L_e_(L_e),
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
  gtsam::Pose3 L_e_;

  StructurelessObjectCentricMotionFactor2(
      gtsam::Key X_k_1_key, gtsam::Key H_k_1_key, gtsam::Key X_k_key,
      gtsam::Key H_k_key, const gtsam::Point3& Z_k_1, const gtsam::Point3& Z_k,
      const gtsam::Pose3& L_e, gtsam::SharedNoiseModel model)
      : Base(model, X_k_1_key, H_k_1_key, X_k_key, H_k_key),
        Z_k_1_(Z_k_1),
        Z_k_(Z_k),
        L_e_(L_e) {}

  gtsam::Vector evaluateError(
      const gtsam::Pose3& X_k_1, const gtsam::Pose3& H_k_1,
      const gtsam::Pose3& X_k, const gtsam::Pose3& H_k,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none,
      boost::optional<gtsam::Matrix&> J3 = boost::none,
      boost::optional<gtsam::Matrix&> J4 = boost::none) const override;
};

class StructurelessDecoupledFormulation : public HybridFormulation {
 public:
  DYNO_POINTER_TYPEDEFS(StructurelessDecoupledFormulation)

  using Base = HybridFormulation;
  StructurelessDecoupledFormulation(const FormulationParams& params,
                                    typename Map::Ptr map,
                                    const NoiseModels& noise_models,
                                    const FormulationHooks& hooks)
      : Base(params, map, noise_models, hooks) {}

  void dynamicPointUpdateCallback(
      const PointUpdateContextType& context, UpdateObservationResult& result,
      gtsam::Values& new_values,
      gtsam::NonlinearFactorGraph& new_factors) override;

  std::string loggerPrefix() const override {
    return "hybrid_structureless_decoupled";
  }
};

class DecoupledFormulation : public HybridFormulation {
 public:
  DYNO_POINTER_TYPEDEFS(DecoupledFormulation)

  using Base = HybridFormulation;
  using Base::Map;

  DecoupledFormulation(const FormulationParams& params, typename Map::Ptr map,
                       const NoiseModels& noise_models,
                       const FormulationHooks& hooks)
      : Base(params, map, noise_models, hooks) {}

  void dynamicPointUpdateCallback(
      const PointUpdateContextType& context, UpdateObservationResult& result,
      gtsam::Values& new_values,
      gtsam::NonlinearFactorGraph& new_factors) override;

  std::string loggerPrefix() const override {
    return "hybrid_centric_decoupled";
  }
};

class StructurlessFormulation : public HybridFormulation {
 public:
  DYNO_POINTER_TYPEDEFS(StructurlessFormulation)

  using Base = HybridFormulation;
  StructurlessFormulation(const FormulationParams& params,
                          typename Map::Ptr map,
                          const NoiseModels& noise_models,
                          const FormulationHooks& hooks)
      : Base(params, map, noise_models, hooks) {}

  void dynamicPointUpdateCallback(
      const PointUpdateContextType& context, UpdateObservationResult& result,
      gtsam::Values& new_values,
      gtsam::NonlinearFactorGraph& new_factors) override;

  std::string loggerPrefix() const override { return "hybrid_structureless"; }
};

class SmartStructurlessAccessor : public HybridAccessor {
 public:
  SmartStructurlessAccessor(
      const SharedFormulationData& shared_data, Map3d2d::Ptr map,
      const SharedHybridFormulationData& shared_hybrid_formulation_data,
      const gtsam::FastMap<TrackletId, HybridSmartFactor::shared_ptr>*
          tracklet_id_to_smart_factor)
      : HybridAccessor(shared_data, map, shared_hybrid_formulation_data),
        tracklet_id_to_smart_factor_(tracklet_id_to_smart_factor) {}

  StateQuery<gtsam::Point3> queryPoint(gtsam::Key point_key,
                                       TrackletId tracklet_id) const override;

 private:
  const gtsam::FastMap<TrackletId, HybridSmartFactor::shared_ptr>*
      tracklet_id_to_smart_factor_;
};

class SmartStructurlessFormulation : public RegularHybridFormulation {
 public:
  DYNO_POINTER_TYPEDEFS(SmartStructurlessFormulation)

  using Base = RegularHybridFormulation;
  SmartStructurlessFormulation(const FormulationParams& params,
                               typename Map::Ptr map,
                               const NoiseModels& noise_models,
                               const FormulationHooks& hooks)
      : Base(params, map, noise_models, hooks) {}

  void dynamicPointUpdateCallback(
      const PointUpdateContextType& context, UpdateObservationResult& result,
      gtsam::Values& new_values,
      gtsam::NonlinearFactorGraph& new_factors) override;

  AccessorTypePointer createAccessor(
      const SharedFormulationData& shared_data) const override {
    SharedHybridFormulationData shared_hybrid_data;
    shared_hybrid_data.key_frame_data = &key_frame_data_;
    shared_hybrid_data.tracklet_id_to_keyframe = &all_dynamic_landmarks_;

    return std::make_shared<SmartStructurlessAccessor>(
        shared_data, this->map(), shared_hybrid_data,
        &tracklet_id_to_smart_factor_);
  }

  std::string loggerPrefix() const override {
    return "hybrid_smart_structureless";
  }

 private:
  void postUpdate(const PostUpdateData& data) override;

 private:
  gtsam::FastMap<TrackletId, HybridSmartFactor::shared_ptr>
      tracklet_id_to_smart_factor_;
  gtsam::FastMap<TrackletId, gtsam::FactorIndex>
      tracklet_id_to_smart_factor_index_;

  // this will get cleared on each postUpdate... what if smoother fails? Should
  // we reset this??!!
  std::vector<TrackletId> tracklet_ids_of_new_smart_factors_;
};

}  // namespace test_hybrid
}  // namespace dyno
