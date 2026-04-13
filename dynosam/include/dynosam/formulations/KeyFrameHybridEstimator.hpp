#pragma once

#include "dynosam/formulations/HybridEstimator.hpp"

namespace dyno {

class HybridFormulationKeyFrameAccessor : public HybridAccessor<MapVision> {
 public:
  HybridFormulationKeyFrameAccessor(
      const SharedFormulationData::Ptr& shared_data, MapVision::Ptr map,
      const SharedHybridFormulationData& shared_hybrid_formulation_data)
      : HybridAccessor(shared_data, map, shared_hybrid_formulation_data) {}

  /**
   * @brief H_W_lKF_k as a  Motion3ReferenceFrame.
   *
   * Overwritten again from the base HybridAccessor
   * since this expects a motion to exist per frame and therefore
   * returns H_W_km1_k with a F2F representation style.
   *
   * However, when Keyframing we only have access to the KF states
   * and therefore the return Motion3ReferenceFrame must be KF style
   *
   * @param frame_id FrameId must be a KF!
   * @param object_id
   * @return StateQuery<Motion3ReferenceFrame>
   */
  StateQuery<Motion3ReferenceFrame> getObjectMotionReferenceFrame(
      FrameId frame_id, ObjectId object_id) const override;

  // /** Same logic as above, this is F2F motion! */
  // std::optional<Motion3ReferenceFrame> getRelativeLocalMotion(
  //     FrameId frame_id, ObjectId object_id) const; override;
};

class HybridFormulationKeyFrame : public HybridFormulation<MapVision> {
 public:
  using Base = HybridFormulation;
  using Base::MapTraitsType;
  using SharedObjectNode = typename MapTraitsType::SharedObjectNode;
  using SharedLandmarkNode = typename MapTraitsType::SharedLandmarkNode;
  using SharedFrameNode = typename MapTraitsType::SharedFrameNode;

  DYNO_POINTER_TYPEDEFS(HybridFormulationKeyFrame)

  HybridFormulationKeyFrame(const FormulationParams& params,
                            typename Map::Ptr map,
                            const NoiseModels& noise_models,
                            const Sensors& sensors,
                            const FormulationHooks& hooks)
      : Base(params, map, noise_models, sensors, hooks) {}

  AccessorTypePointer createAccessor(
      const SharedFormulationData::Ptr& shared_data) const override {
    SharedHybridFormulationData shared_hybrid_data;
    shared_hybrid_data.key_frame_data = &key_frame_data_;
    shared_hybrid_data.tracklet_id_to_keyframe = &all_dynamic_landmarks_;

    return std::make_shared<HybridFormulationKeyFrameAccessor>(
        shared_data, this->map(), shared_hybrid_data);
  }

  UpdateObservationResult updateDynamicObservations(
      FrameId frame_id_k, gtsam::Values& new_values,
      gtsam::NonlinearFactorGraph& new_factors,
      const UpdateObservationParams& update_params) override;

  void addObjects(FrameId frame_id,
                  const ObjectPoseChangeInfoMap& object_motion_info);

  const KeyFrameData& getRegularKeyFrames() const {
    return front_end_keyframes_;
  }
  const KeyFrameData& getAnchorKeyFrames() const { return key_frame_data_; }

  ObjectPoseMap getInitialObjectPoses() const;
  // object points in L for all objects in the state
  TrackedPointsPerObject getObjectPoints() const;
  // object points in L for objects observed at frame_id
  TrackedPointsPerObject getObjectPoints(FrameId frame_id) const;

  // uses last frame in state
  HybridKeyFrameUpdate generateUpdateInfo() const;

  MultiObjectTrajectories refinePerFrameMotionsPGO(
      const MultiObjectTrajectories& full_trajectories) const;

 private:
  struct Context {
    SharedObjectNode object_node;
    SharedFrameNode frame_node;
    gtsam::Pose3 X_k_measured;
    //! When an update starts only a subset of the factors are provided to the
    //! update This value indicates the factor slot offset (ie the total graph
    //! size before any update)
    Slot starting_factor_slot = -1;

    inline ObjectId getObjectId() const { return object_node->objectId(); }
    inline FrameId getFrameId() const { return frame_node->frameId(); }
  };

  struct KeyFrameMetaData {
    ObjectKeyFrameStatus keyframe_status;

    //! Measured object motion from the frontend
    //! Taking us from last RKF to most recent KF (ie. k)
    Motion3ReferenceFrame H_W_lRKF_KF;
  };

  void preUpdate(const PreUpdateData&) override {}
  void postUpdate(const PostUpdateData&) override {}

  // we may have object measurements at non-keyframes depending on how the
  // front-end is implemented...
  bool isObjectKeyFrame(ObjectId object_id, FrameId frame_id) const;

  void updateObject(const Context& context, UpdateObservationResult& result,
                    gtsam::Values& new_values,
                    gtsam::NonlinearFactorGraph& new_factors);

  void addHybridMotionFactor(gtsam::NonlinearFactorGraph& new_factors,
                             gtsam::Key pose_key, gtsam::Key object_motion_key,
                             gtsam::Key point_key, const gtsam::Pose3& KF_pose,
                             SharedLandmarkNode lmk_node,
                             SharedFrameNode frame_node);
  // helper function
  TrackedPointsPerObject getObjectPoints(const ObjectIds& objects) const;

 protected:
  // neither overridden update callback is used as we directly overwrite the
  // updateDynamicObservations and use with our specific callback function
  inline void dynamicPointUpdateCallback(
      const PointUpdateContextType& context, UpdateObservationResult& result,
      gtsam::Values& new_values,
      gtsam::NonlinearFactorGraph& new_factors) override {}

  inline void objectUpdateContext(
      const ObjectUpdateContextType& context, UpdateObservationResult& result,
      gtsam::Values& new_values,
      gtsam::NonlinearFactorGraph& new_factors) override {}

  inline IntermediateMotionInfo getIntermediateMotionInfo(ObjectId object_id,
                                                          FrameId frame_id) {}

 private:
  TemporalObjectCentricMap<Motion3ReferenceFrame> initial_H_W_AKF_k_;
  //! Bookkeeps the keyframing from the front-end so we manage the to/from
  //! frames provided by the front-end estimate This is not used to manage the
  //! keyframe pose or keyframe id in the backend Since this is DIFFERENT to the
  //! frontend The pose held in each KeyFrameRangeis the L_e_frontend (which is
  //! used to anchor) The frontend estimates

  // TODO: dont even think we need this!!
  KeyFrameData front_end_keyframes_;

  TemporalObjectCentricMap<KeyFrameMetaData> key_frames_per_object_;

  //! Initial estimate of points in object frame from frontend
  GenericObjectCentricMap<gtsam::Point3, TrackletId> m_L_initial_;

  //! Bookkeeping of landmarks to frames which have factors added
  gtsam::FastMap<TrackletId, std::set<FrameId>> factors_added_;
};

}  // namespace dyno
