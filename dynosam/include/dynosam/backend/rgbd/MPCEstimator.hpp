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
namespace mpc_factors {
/**
 * @brief Class connecting a ego-motion pose (X) with an object motion (H) that
 * can be used as a mission factor.
 *
 * We expect the motion to be in the Hybrid motion form (Morris RA-L 2025) and
 * so we also include the embedded frame L_e.
 *
 * Factor provides some additional functionality but does not implement any
 * factor specific implementations (e.g. evaluateError etc...)
 *
 */
class MissionFactorBase
    : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {
 public:
  using shared_ptr = boost::shared_ptr<MissionFactorBase>;
  using This = MissionFactorBase;
  using Base = gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>;
  MissionFactorBase(gtsam::Key X_k_key, gtsam::Key H_W_e_k_key,
                    const gtsam::Pose3& L_e, gtsam::SharedNoiseModel model)
      : Base(model, X_k_key, H_W_e_k_key), L_e_(L_e) {}

  gtsam::Key cameraPoseKey() const { return key1(); }
  gtsam::Key objectMotionKey() const { return key2(); }

  virtual void print(
      const std::string& s = "",
      const KeyFormatter& keyFormatter = DynoLikeKeyFormatter) const override;

  virtual bool isFuture(FrameId frame_k) const;

 protected:
  gtsam::Pose3 L_e_;
};
}  // namespace mpc_factors

class MissionFactorGraph
    : public gtsam::FactorGraph<mpc_factors::MissionFactorBase> {
 public:
  using Base = gtsam::FactorGraph<mpc_factors::MissionFactorBase>;
  MissionFactorGraph(){};
};

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
class SDFMap2D;

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

  void preUpdate(const PreUpdateData& data) override;
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

  //! Mission factors from previous frame
  //! Will be removed immediately if the object is seen as we're going to add
  //! new mission factors at this frame If the object is not seen that factors
  gtsam::FastMap<ObjectId, MissionFactorGraph> previous_mission_factors_;

  // prediction pose prior
  gtsam::SharedNoiseModel vel2d_prior_noise_;

  gtsam::SharedNoiseModel vel2d_limit_noise_;
  gtsam::SharedNoiseModel accel2d_limit_noise_;

  gtsam::SharedNoiseModel accel2d_cost_noise_;
  gtsam::SharedNoiseModel accel2d_smoothing_noise_;

  // this is really the factor on the camera dynamics (ie x(k+1) = f(x, v, a))
  // as opposed to the dynamic obstacle factor
  gtsam::SharedNoiseModel dynamic_factor_noise_;

  gtsam::SharedNoiseModel object_prediction_constant_motion_noise_;

  gtsam::SharedNoiseModel follow_noise_;
  gtsam::SharedNoiseModel goal_noise_;

  gtsam::SharedNoiseModel static_obstacle_X_noise_;
  gtsam::SharedNoiseModel static_obstacle_H_noise_;

  gtsam::SharedNoiseModel dynamic_obstacle_factor_;

  double desired_follow_distance_;
  double desired_follow_heading_;

  // timestamp of k-1 where otherUpdatesContext is called each k
  Timestamp timestamp_km1_{0};

  // global path things
  Timestamp last_global_path_update_;
  std::optional<gtsam::Pose3Vector> global_path_ = {};

  enum MissionType { FOLLOW = 0, NAVIGATE = 1 };

  MissionType mission_type_;

  // internal logic for object porediction (currently only works for single
  // object)
  //  bool object_reappeared_{false};
  //  bool object_disappeared_{false};

  // only shared becuase its forward declared because cbf to put it in the
  // header file for now! also shared between all obstacle factors...
  std::shared_ptr<SDFMap2D> sdf_map_{nullptr};
  // this transform takes us from the camera (opencv) to the map (robotic) frame
  // this is not simply the current pose to camera/odometry but the actual map
  // frame of the sdf map which was generated w.r.t the 'odom' frame N.B this
  // name is misleading it should really be called the world/map frame since
  // dynosam estimates w.r.t camera/odom this is static transform from
  // camera/odom to odom and allows us to put the current pose (which is in
  // camera) into the map frame and should include the conversion from
  // opencv-robotic
  std::optional<gtsam::Pose3> T_map_camera_;
};

// HACK for now to get ros stuff easily into this backend!!
class MPCEstimationViz {
 public:
  MPCEstimationViz() = default;
  virtual ~MPCEstimationViz() = default;

  // bascially queries odom to world_frame_id
  // in this setup dynosam does not start the the real world frame (ie the frame
  // of the map) and so we have a transform between the maps origin and the
  // starting point of the offset this transform is that static offset from
  // which the dynosam odometry is referenced gainnst
  virtual bool queryGlobalOffset(gtsam::Pose3& T_world_camera) = 0;
  virtual void inPreUpdate() {}
  virtual void inPostUpdate() {}

  virtual void spin(Timestamp timestamp, FrameId frame_id,
                    const MPCFormulation* formulation) = 0;
};

}  // namespace dyno
