#pragma once

#include <gtsam/nonlinear/ISAM2.h>

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/BackendModule.hpp"
#include "dynosam/backend/Formulation.hpp"
#include "dynosam/formulations/KeyFrameHybridEstimator.hpp"
#include "dynosam_opt/IncrementalOptimization.hpp"

namespace dyno {

// kinda hacky but the way the pipelines are set up we need a single type
// between the frontend and backend
// but for the PoseChange modules we need to convert the single input
// into a batch in the pipeline
// take advatange of polymorphism to do this
struct PoseChangeInput {
  DYNO_POINTER_TYPEDEFS(PoseChangeInput)
  virtual ~PoseChangeInput() = default;
};

struct SinglePoseChangeInput : public PoseChangeInput {
  DYNO_POINTER_TYPEDEFS(SinglePoseChangeInput)
  //! Frame id associated with the creation of this input
  FrameId frame_id;
  //! Timestamp associated with the creation of this input
  Timestamp timestamp;

  struct FGInput {
    gtsam::Values values;
    gtsam::NonlinearFactorGraph factors;
  };

  FGInput new_static_fg_input;
  FGInput new_dynamic_fg_input;

  //! Record keyframes (objects/camera) for this frame so we know the state of
  //! the map when this input was generated
  KeyframeInfo keyframe_info;

  // HACK FOR NOW
  // delay construction of dynamic object factors until we have the best
  // possible camera pose estimate
  ObjectPoseChangeInfoMap kf_pose_change_infos;
};

// assume construced in temporal order
class BatchPoseChangeInput : public PoseChangeInput {
 private:
  BatchPoseChangeInput() = default;
  // store as ConstPtr becuase less copying when we construct this input from
  // the pipeline since all pipelines construct ConstPtr
  std::vector<SinglePoseChangeInput::ConstPtr> batch_;

  FrameId starting_frame_;
  FrameId ending_frame_;

  ObjectIds involved_objects_;
  bool is_camera_involved_;

  gtsam::FastMap<ObjectId, FrameId> latest_keyframes_;

 public:
  DYNO_POINTER_TYPEDEFS(BatchPoseChangeInput)
  ~BatchPoseChangeInput() = default;

  class Builder {
   public:
    Builder() = default;
    Builder(size_t reserve) { batch_.reserve(reserve); }

    void add(const SinglePoseChangeInput::ConstPtr& entry);
    BatchPoseChangeInput::Ptr finalise();

   private:
    std::vector<SinglePoseChangeInput::ConstPtr> batch_;
  };

  FrameId startingFrame() const;
  FrameId endingFrame() const;

  decltype(auto) begin() { return batch_.begin(); }
  decltype(auto) end() { return batch_.end(); }

  decltype(auto) begin() const { return batch_.begin(); }
  decltype(auto) end() const { return batch_.end(); }

  const ObjectIds& involvedObjects() const;
  bool isCameraInvolved() const;

  /* Latest keyframe frame per object (includes camera j=0)*/
  const gtsam::FastMap<ObjectId, FrameId>& latestKeyframes() const;
};

class PoseChangeVIBackendModule : public BackendModule<PoseChangeInput> {
 public:
  using Base = BackendModule<PoseChangeInput>;
  DYNO_POINTER_TYPEDEFS(PoseChangeVIBackendModule)

  PoseChangeVIBackendModule(const BackendParams& params, Camera::Ptr camera,
                            HybridFormulationKeyFrame::Ptr formulation,
                            const SharedGroundTruth& shared_ground_truth = {});

  ~PoseChangeVIBackendModule();

  std::pair<gtsam::Values, gtsam::NonlinearFactorGraph> getActiveOptimisation()
      const override {
    LOG(FATAL) << "Not implemented!";
  }

  Accessor::Ptr getAccessor() const override {
    return formulation_->getAsVIOAccessor();
  }
  HybridFormulationKeyFrame::Ptr getFormulation() const { return formulation_; }

  void registerUpdateCallback(const PoseChangeUpdateCompleteCallback& callback);

 private:
  using SpinReturn = Base::SpinReturn;

  // just call spin once
  SpinReturn boostrapSpin(PoseChangeInput::ConstPtr input) override {
    return {State::Nominal, spinOnce(input)};
  }
  SpinReturn nominalSpin(PoseChangeInput::ConstPtr input) override {
    return {State::Nominal, spinOnce(input)};
  }

  DynoState::Ptr spinOnce(PoseChangeInput::ConstPtr input);

  /** Acts like the base class of a fixed lag smoother by storing keys to frames
   * but is designed specifcially becuase we need to handle the fixed lag for
   * objects and camera independantly. Instead of explicitly using timestamps we
   * compute the fixed lag based on the keyframe index
   */
  struct KeyframeIndexBookkeeping {
    //! really mapping of keys to keyframe INDEX not the keyframe itself
    typedef std::map<gtsam::Key, int> KeyFrameIndexMap;
    typedef std::multimap<int, gtsam::Key> FrameIndexKeyMap;

    int lag_;
    /** The current timestamp associated with each tracked key */
    FrameIndexKeyMap kf_index_key_map_;
    KeyFrameIndexMap key_kf_index_map_;

    KeyframeIndexBookkeeping(FrameId fixed_lag) : lag_(fixed_lag) {}

    /** Update the Keyframe indices associated with the keys */
    void updateKeyFrameIndexMap(const KeyFrameIndexMap& new_frames);

    /** Erase keys from the Key-Frameids database */
    void eraseKeyFrameIndexMap(const gtsam::KeyVector& keys);

    /** Find the most recent keyframe index of the system */
    int getCurrentFrameIndex() const;

    /** Find all of the keys associated with keyframes before the provided index
     */
    gtsam::KeyVector findKeysBefore(int kf_index) const;

    /** Find all of the keys associated with keyframes before the provided index
     */
    gtsam::KeyVector findKeysAfter(int kf_index) const;

    gtsam::KeyVector findMarginalizableKeys() const;

    void createOrderingConstraints(
        const gtsam::KeyVector& marginalizableKeys,
        gtsam::FastMap<gtsam::Key, int>& constrainedKeys) const;
  };

  using KeyFrameIndexMap = KeyframeIndexBookkeeping::KeyFrameIndexMap;

  /* new_kf_indices_per_object inclucdes camera and also represents set of
   * involved objects/camera*/
  void prepareArgumentsForUpdate(
      const BatchPoseChangeInput::ConstPtr& batch_input,
      gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors,
      gtsam::FastMap<ObjectId, KeyFrameIndexMap>& new_kf_indices_per_object,
      FrameKeyframeIndexMapping& frame_to_frameIndex);

  void updateKeyframeIndexBookkeeping(
      const gtsam::FastMap<ObjectId, KeyFrameIndexMap>&
          new_kf_indices_per_object);

  bool optimize(gtsam::ISAM2Result* result, const gtsam::Values& new_values,
                const gtsam::NonlinearFactorGraph& new_factors,
                const gtsam::FastMap<ObjectId, KeyFrameIndexMap>&
                    new_kf_indices_per_object,
                const gtsam::ISAM2UpdateParams& update_params =
                    gtsam::ISAM2UpdateParams());

  void createPoseChangeUpdateComplete(
      const BatchPoseChangeInput::ConstPtr& batch_input,
      const DynoState::Ptr& state, PoseChangeUpdateComplete& event) const;

 private:
  HybridFormulationKeyFrame::Ptr formulation_;
  HybridFormulationKeyFrameAccessor::Ptr hybrid_accessor_;
  ErrorHandlingHooks error_hooks_;

  // keyframe lag for fixed-lag marginalization
  const int fixed_lag_{7};

  //! Mapping of keyframe indices/keys per object. Used to manually manage
  //! fixed-lag smoothing When we do the fixed-lag we look at the current set of
  //! frame indixes for each object and use that to determine which variables
  //! should be marginalized
  gtsam::FastMap<ObjectId, KeyframeIndexBookkeeping> key_keyframe_indices_;

  using SmootherInterface = IncrementalInterface<gtsam::ISAM2>;
  SmootherInterface smoother_interface_;
  std::unique_ptr<gtsam::ISAM2> smoother_;

  //! Callback to asynchronously alert the frontend an update is complete
  PoseChangeUpdateCompleteCallback update_callback_;
};

}  // namespace dyno
