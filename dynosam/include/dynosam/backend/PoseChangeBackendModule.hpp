#pragma once

#include <gtsam/nonlinear/ISAM2.h>

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/BackendModule.hpp"
#include "dynosam/backend/Formulation.hpp"
#include "dynosam/formulations/KeyFrameHybridEstimator.hpp"

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
  gtsam::Values new_values;
  gtsam::NonlinearFactorGraph new_factors;

  //! Record keyframes (objects/camera) for this frame so we know the state of
  //! the map when this input was generated
  KeyframeInfo keyframe_info;
};

// assume construced in temporal order
struct BatchPoseChangeInput : public PoseChangeInput {
  DYNO_POINTER_TYPEDEFS(BatchPoseChangeInput)
  // store as ConstPtr becuase less copying when we construct this input from
  // the pipeline since all pipelines construct ConstPtr
  std::vector<SinglePoseChangeInput::ConstPtr> batch;

  FrameId startingFrame() const;
  FrameId endingFrame() const;

  std::pair<gtsam::Values, gtsam::NonlinearFactorGraph> newValuesAndFactors()
      const;
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

 private:
  HybridFormulationKeyFrame::Ptr formulation_;
  HybridFormulationKeyFrameAccessor::Ptr hybrid_accessor_;
  ErrorHandlingHooks error_hooks_;

  std::unique_ptr<gtsam::ISAM2> smoother_;

  //! Callback to asynchronously alert the frontend an update is complete
  PoseChangeUpdateCompleteCallback update_callback_;
};

}  // namespace dyno
