#pragma once

#include "dynosam/formulations/HybridEstimator.hpp"

namespace dyno {

/**
 * @brief The original Hybrid motion implementation (as presented in
 * "Online Dynamic SLAM with Incremental Smoothing and Mapping" from RA-L) -
 * this is independant from the frontend and ONLY accepts frame-to-frame motion
 * (i.e H_W_k_1_k) as input. It constructs object keyframes independantly of the
 * front-end and therefore has slightly different embedded poses.
 *
 * Used as is for the Parallal-Hybrid implementation.
 *
 */
class HybridFormulationV1 : public HybridFormulation {
 public:
  using Base = HybridFormulation;
  DYNO_POINTER_TYPEDEFS(HybridFormulationV1)

  HybridFormulationV1(const FormulationParams& params, typename Map::Ptr map,
                      const NoiseModels& noise_models, const Sensors& sensors,
                      const FormulationHooks& hooks)
      : Base(params, map, noise_models, sensors, hooks) {}

  // SHOULD be in variation of Hybrid that is independant of the front-end
  //  leave in for now!
  std::pair<FrameId, gtsam::Pose3> forceNewKeyFrame(FrameId frame_id,
                                                    ObjectId object_id);

 protected:
  IntermediateMotionInfo getIntermediateMotionInfo(ObjectId object_id,
                                                   FrameId frame_id) override;

  std::pair<FrameId, gtsam::Pose3> getOrConstructL0(ObjectId object_id,
                                                    FrameId frame_id);

  // hacky update solution for now!!
  gtsam::Pose3 computeInitialH(ObjectId object_id, FrameId frame_id,
                               bool* keyframe_updated = nullptr);

  gtsam::Pose3 calculateObjectCentroid(ObjectId object_id,
                                       FrameId frame_id) const;

  ErrorHandlingHooks getCustomErrorHooks() override;
};

// additional functionality when solved with the Regular Backend!
// differs only from Base class with pre and post update functions
// which are needed for use in the RegularBackend
class RegularHybridFormulation : public HybridFormulationV1 {
 public:
  using Base = HybridFormulationV1;

  RegularHybridFormulation(const FormulationParams& params,
                           typename Map::Ptr map,
                           const NoiseModels& noise_models,
                           const Sensors& sensors,
                           const FormulationHooks& hooks)
      : Base(params, map, noise_models, sensors, hooks) {}

  // using previous (postUdpate) check when the last time an object was updated
  // (in the estimator) if more than 1 frame ago, create new keyframe - this
  // will then take affect as this function is called prior to the graph
  // construction!
  void preUpdate(const PreUpdateData& data) override;
  // use post update information to set internal data about when objects were
  // last udpated!!
  virtual void postUpdate(const PostUpdateData& data) override;

 protected:
  struct ObjectUpdateData {
    FrameId frame_id{0};  //! Last time the object was updted in the estimator
    size_t count{0};  //! Number of (total) times the object has been updated.
                      //! If 1, then new

    inline bool isNew() const { return count == 1u; }
  };
  // The last frame
  gtsam::FastMap<ObjectId, ObjectUpdateData> objects_update_data_;
};

}  // namespace dyno
