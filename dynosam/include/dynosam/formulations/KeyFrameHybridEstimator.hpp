#pragma once

#include "dynosam/formulations/HybridEstimator.hpp"
#include "dynosam/formulations/KeyFrameHybridMap.hpp"
#include "dynosam/frontend/vision/Frame.hpp"

namespace dyno {

class HybridFormulationKeyFrameAccessor : public HybridAccessor<KeyFrameMap> {
 public:
  typedef HybridAccessor<KeyFrameMap> Base;

  HybridFormulationKeyFrameAccessor(
      const SharedFormulationData::Ptr& shared_data, KeyFrameMap::Ptr map,
      const SharedHybridFormulationData& shared_hybrid_formulation_data)
      : HybridAccessor(shared_data, map, shared_hybrid_formulation_data) {}

  /** Get camera pose trajectory. Overwirtten to only include keyframes
   * currently in the optimisation problem*/
  PoseTrajectory getCameraTrajectory() const override;
  // TODO: probably should do the same objects

  /** Get multi-object trajectories. Overwirtten to only include keyframes
   * currently in the optimisation problem*/
  MultiObjectTrajectories getMultiObjectTrajectories() const override;

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

class StereoHybridMotionExtrapolatedFactor
    : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3,
                                      gtsam::Point3>,
      public StereoHybridMotionFactorBase {
 public:
  using Base =
      gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Point3>;

  StereoHybridMotionExtrapolatedFactor(
      const gtsam::StereoPoint2& measured, const gtsam::Pose3& L_KF,
      const gtsam::Pose3& T_i_j,  // <-- fixed relative transform
      const gtsam::SharedNoiseModel& model, gtsam::Cal3_S2Stereo::shared_ptr K,
      gtsam::Key X_i_key, gtsam::Key H_W_KF_j_key, gtsam::Key m_L_key,
      bool throw_cheirality = false)
      : Base(model, X_i_key, H_W_KF_j_key, m_L_key),
        StereoHybridMotionFactorBase(measured, L_KF, K, throw_cheirality),
        T_i_j_(T_i_j) {}

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(
            new StereoHybridMotionExtrapolatedFactor(*this)));
  }

  gtsam::Vector evaluateError(
      const gtsam::Pose3& X_i, const gtsam::Pose3& H_W_KF_j,
      const gtsam::Point3& m_L,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none,
      boost::optional<gtsam::Matrix&> J3 = boost::none) const override {
    // --- 1. Compose pose ---
    gtsam::Matrix66 H_comp_Xi;
    const gtsam::Pose3 X_j = X_i.compose(T_i_j_, J1 ? &H_comp_Xi : nullptr);

    // --- 2. Evaluate base factor ---
    gtsam::Matrix J_Xj;  // 3x6
    gtsam::Matrix J_H;   // 3x6
    gtsam::Matrix J_m;   // 3x3

    try {
      const gtsam::Vector error = StereoHybridMotionFactorBase::evaluateError(
          X_j, H_W_KF_j, m_L, J_Xj, J_H, J_m);

      // --- 3. Chain rule ---
      if (J1) {
        *J1 = J_Xj * H_comp_Xi;  // (3x6)*(6x6) = 3x6
      }

      if (J2) {
        *J2 = J_H;  // unchanged
      }

      if (J3) {
        *J3 = J_m;  // unchanged
      }

      return error;
    } catch (const CheiralityException&) {
      // only derived class knows about the key so throw here not in base
      // which throws CheiralityException
      // CheiralityException is only thrown if throw_cheirality true
      throw gtsam::StereoCheiralityException(this->key3());
    }
  }

 private:
  gtsam::Pose3 T_i_j_;  // fixed transform
};

class HybridFormulationKeyFrame : public HybridFormulation<KeyFrameMap> {
 public:
  using Base = HybridFormulation;
  using Base::MapTraitsType;
  using SharedObjectNode = typename MapTraitsType::SharedObjectNode;
  using SharedLandmarkNode = typename MapTraitsType::SharedLandmarkNode;
  using SharedFrameNode = typename MapTraitsType::SharedFrameNode;

  DYNO_POINTER_TYPEDEFS(HybridFormulationKeyFrame)

  HybridFormulationKeyFrame(const FormulationParams& params,
                            KeyFrameMap::Ptr map,
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
  // helper function
  // points in local
  TrackedPointsPerObject getObjectPoints(const ObjectIds& objects) const;

  MultiObjectTrajectories refinePerFrameMotionsPGO(
      const MultiObjectTrajectories& full_trajectories) const;

  bool matchToStaticMap(Frame::Ptr frame, AbsolutePoseCorrespondences& matches,
                        double* tracking_quality = nullptr) const;

 private:
  struct Context {
    SharedObjectNode object_node;
    SharedFrameNode frame_node;
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

  /** How the camera pose was extracted */
  enum CameraPoseExtraction { Keyframe, Interpolated };

  void preUpdate(const PreUpdateData&) override {}
  void postUpdate(const PostUpdateData&) override {}

  // we may have object measurements at non-keyframes depending on how the
  // front-end is implemented...
  bool isObjectKeyFrame(ObjectId object_id, FrameId frame_id) const;

  void updateObject(const Context& context, UpdateObservationResult& result,
                    gtsam::Values& new_values,
                    gtsam::NonlinearFactorGraph& new_factors);

  // void addHybridMotionFactor(gtsam::NonlinearFactorGraph& new_factors,
  //                            gtsam::Key pose_key, gtsam::Key
  //                            object_motion_key, gtsam::Key point_key, const
  //                            gtsam::Pose3& KF_pose, SharedLandmarkNode
  //                            lmk_node, SharedFrameNode frame_node);

  // get a camera pose at possibly a non-ckf frame
  // TODo: better name (either from state or interpolated via VIO)
  std::pair<gtsam::Pose3, CameraPoseExtraction> getBestCameraPose(
      FrameId frame_id) const;

  void addHybridMotionFactor(gtsam::NonlinearFactorGraph& new_factors,
                             gtsam::Key point_key, ObjectId object_id,
                             const gtsam::Pose3& KF_pose,
                             SharedLandmarkNode lmk_node,
                             SharedFrameNode frame_node);

  void addHybridMotionFactorCameraKF(gtsam::NonlinearFactorGraph& new_factors,
                                     gtsam::Key point_key, ObjectId object_id,
                                     const gtsam::Pose3& KF_pose,
                                     const gtsam::StereoPoint2& z,
                                     const gtsam::SharedNoiseModel& z_model,
                                     SharedFrameNode frame_node_CKF);

  void addHybridMotionFactorNonCameraKF(
      gtsam::NonlinearFactorGraph& new_factors, gtsam::Key point_key,
      ObjectId object_id, const gtsam::Pose3& KF_pose,
      const gtsam::StereoPoint2& z, const gtsam::SharedNoiseModel& z_model,
      SharedFrameNode frame_node_nonCKF);

  void addNewObjectMotionVariable(gtsam::Values& new_values,
                                  SharedFrameNode frame_node,
                                  ObjectId object_id,
                                  const Motion3ReferenceFrame& motion);

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

  inline RelEgoPoseInfoInfo getRelEgoPoseInfoInfo(ObjectId object_id,
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
