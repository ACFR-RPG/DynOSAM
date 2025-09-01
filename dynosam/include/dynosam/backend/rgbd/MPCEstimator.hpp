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

struct SharedMPCData {
  size_t mpc_horizon;
  ObjectId object_to_follow;
};

class MPCAccessor : public HybridAccessor, public MPCFormulationProperties {
 public:
  MPCAccessor(const SharedFormulationData& shared_data, MapVision::Ptr map,
              const SharedHybridFormulationData& shared_hybrid_formulation_data,
              const SharedMPCData* shared_mpc_data)
      : HybridAccessor(shared_data, map, shared_hybrid_formulation_data),
        mpc_data_(shared_mpc_data) {}

  StateQuery<gtsam::Vector2> getControlCommand(FrameId frame_k) const;

 private:
  const SharedMPCData* mpc_data_;
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
                                         shared_hybrid_data, &mpc_data_);
  }

  std::string loggerPrefix() const override { return "dyno_mpc"; }

  // TODO: as in Hybrid accessor these functions should be shared with the
  // accessor!!
  std::pair<ObjectMotionMap, ObjectPoseMap> getObjectPredictions(
      FrameId frame_k) const;

  gtsam::Pose3Vector getPredictedCameraPoses(FrameId frame_k) const;

  size_t horizon() const { return mpc_data_.mpc_horizon; }

  StateQuery<gtsam::Vector2> getControlCommand(FrameId frame_k) const;

  void updateGlobalPath(Timestamp timestamp,
                        const gtsam::Pose3Vector& global_path);

  std::optional<gtsam::Pose3> queryCurrentLocalGoal() const;

  // TODO: for now!
  std::unique_ptr<MPCEstimationViz> viz_;

  struct Limits {
    double min;
    double max;
  };

  Limits lin_vel_;
  Limits ang_vel_;
  Limits lin_acc_;
  Limits ang_acc_;

  // set usually by a call to getLocalGoalFromGlobalPath from within
  // otherUpdatesContext we only save a member copy so we can then visualise it!
  std::optional<gtsam::Pose3> local_goal_ = {};

 private:
  void otherUpdatesContext(const OtherUpdateContextType& context,
                           UpdateObservationResult& result,
                           gtsam::Values& new_values,
                           gtsam::NonlinearFactorGraph& new_factors) override;

  bool getLocalGoalFromGlobalPath(const gtsam::Pose3& X_k, size_t horizon,
                                  gtsam::Pose3& goal);

  void postUpdate(const PostUpdateData& data) override;

  // void addPredictionObjectFactors(
  //   ObjectId object_id,
  //   const OtherUpdateContextType& context,
  //   UpdateObservationResult& result,
  //   gtsam::Values& new_values,
  //   gtsam::NonlinearFactorGraph& new_factors
  // );

 private:
  SharedMPCData mpc_data_;
  Timestamp dt_{0.1};

  // factors added/are relevant to each frame
  // will be used then delete the relevant factors at each update
  gtsam::FastMap<FrameId, gtsam::NonlinearFactorGraph> factors_per_frame_;

  //! Datastructure to contain which objects had predictions added at which
  //! frame Mostly used to handle disappaering/reappearing objects The frame is
  //! not the timestep of the prediction (ie state H_k) But the real-time
  //! (current time k) that a prediction was made we then expect N predicted
  //! states to exist from k -> k+N
  gtsam::FastMap<FrameId, ObjectIds> predicted_objects_at_frame_;
  //! When an object disappears we need to add prior factors on the predicted
  //! variables To ensure the optimisation does not crash. This data-structure
  //! tracks which factors we have added and when so we can then remove them
  //! latter if the object reapears!
  gtsam::FastMap<ObjectId, gtsam::NonlinearFactorGraph>
      stabilising_object_factors_;

  // prediction pose prior
  gtsam::SharedNoiseModel vel2d_prior_noise_;

  gtsam::SharedNoiseModel vel2d_limit_noise_;
  gtsam::SharedNoiseModel accel2d_limit_noise_;

  gtsam::SharedNoiseModel accel2d_cost_noise_;
  gtsam::SharedNoiseModel accel2d_smoothing_noise_;

  gtsam::SharedNoiseModel dynamic_factor_noise_;

  gtsam::SharedNoiseModel object_prediction_constant_motion_noise_;

  gtsam::SharedNoiseModel follow_noise_;
  gtsam::SharedNoiseModel goal_noise_;

  double desired_follow_distance_;
  double desired_follow_heading_;

  // timestamp of k-1 where otherUpdatesContext is called each k
  Timestamp timestamp_km1_{0};

  // global path things
  Timestamp last_global_path_update_;
  std::optional<gtsam::Pose3Vector> global_path_ = {};

  enum MissionType { FOLLOW = 0, NAVIGATE = 1 };

  MissionType mission_type_;
};

// HACK for now to get ros stuff easily into this backend!!
class MPCEstimationViz {
 public:
  MPCEstimationViz() = default;
  virtual ~MPCEstimationViz() = default;

  virtual void spin(Timestamp timestamp, FrameId frame_id,
                    const MPCFormulation* formulation) = 0;
};

}  // namespace dyno
