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

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/rgbd/HybridEstimator.hpp"

namespace dyno {

struct MPCFormulationProperties {
  static constexpr SymbolChar kControlCommandSymbolChar = 'c';
  static constexpr SymbolChar kAccelerationSymbolChar = 'a';
  inline gtsam::Symbol makeControlCommandKey(FrameId frame_k) const {
    return gtsam::Symbol(kControlCommandSymbolChar, frame_k);
  }

  inline gtsam::Symbol makeAccelerationKey(FrameId frame_k) const {
    return gtsam::Symbol(kAccelerationSymbolChar, frame_k);
  }
};

class MPCAccessor : public HybridAccessor, public MPCFormulationProperties {
 public:
  MPCAccessor(const SharedFormulationData& shared_data, MapVision::Ptr map,
              const SharedHybridFormulationData& shared_hybrid_formulation_data)
      : HybridAccessor(shared_data, map, shared_hybrid_formulation_data) {}

  StateQuery<gtsam::Vector2> getControlCommand(FrameId frame_k) const;

  template <typename VALUE>
  StateQuery<VALUE> queryWithTheta(gtsam::Key key,
                                   const gtsam::Values& new_values) {
    if (StateQuery<VALUE> theta = this->query<VALUE>(key); theta) {
      return theta;
    }

    if (new_values.exists(key)) {
      return StateQuery<VALUE>(key, new_values.at<VALUE>(key));
    } else {
      return StateQuery<VALUE>::NotInMap(key);
    }
  }
};

// forward
class MPCEstimationViz;

class MPCFormulation : public RegularHybridFormulation,
                       public MPCFormulationProperties {
 public:
  DYNO_POINTER_TYPEDEFS(MPCFormulation)

  using Base = RegularHybridFormulation;
  using typename Base::OtherUpdateContextType;

  MPCFormulation(const FormulationParams& params, typename Map::Ptr map,
                 const NoiseModels& noise_models, const Sensors& sensors,
                 const FormulationHooks& hooks);

  AccessorTypePointer createAccessor(
      const SharedFormulationData& shared_data) const override {
    SharedHybridFormulationData shared_hybrid_data;
    shared_hybrid_data.key_frame_data = &key_frame_data_;
    shared_hybrid_data.tracklet_id_to_keyframe = &all_dynamic_landmarks_;

    return std::make_shared<MPCAccessor>(shared_data, this->map(),
                                         shared_hybrid_data);
  }

  std::string loggerPrefix() const override { return "dyno_mpc"; }

  // TODO: for now!
  std::unique_ptr<MPCEstimationViz> viz_;

 private:
  void otherUpdatesContext(const OtherUpdateContextType& context,
                           UpdateObservationResult& result,
                           gtsam::Values& new_values,
                           gtsam::NonlinearFactorGraph& new_factors) override;

  void postUpdate(const PostUpdateData& data) override;

 private:
  size_t mpc_horizon{4};
  Timestamp dt_{0.1};

  // factors added/are relevant to each frame
  // will be used then delete the relevant factors at each update
  gtsam::FastMap<FrameId, gtsam::NonlinearFactorGraph> factors_per_frame_;
  ObjectId object_to_follow_{1};

  // prediction pose prior
  gtsam::SharedNoiseModel camera_pose_prior_noise_;
  gtsam::SharedNoiseModel vel2d_prior_noise_;
  gtsam::SharedNoiseModel accel2d_prior_noise_;

  gtsam::SharedNoiseModel vel2d_limit_noise_;
  gtsam::SharedNoiseModel accel2d_limit_noise_;

  gtsam::SharedNoiseModel accel2d_cost_noise_;
  gtsam::SharedNoiseModel accel2d_smoothing_noise_;

  gtsam::SharedNoiseModel dynamic_factor_noise_;

  struct Limits {
    double min;
    double max;
  };

  Limits lin_vel_;
  Limits ang_vel_;
  Limits lin_acc_;
  Limits ang_acc_;
};

// HACK for now to get ros stuff easily into this backend!!
class MPCEstimationViz {
 public:
  MPCEstimationViz() = default;
  virtual ~MPCEstimationViz() = default;

  virtual void spin(FrameId frame_id, const MPCFormulation* formulation) = 0;
};

}  // namespace dyno
