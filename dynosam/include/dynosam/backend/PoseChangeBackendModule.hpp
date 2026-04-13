#pragma once

#include <gtsam/nonlinear/ISAM2.h>

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/BackendModule.hpp"
#include "dynosam/backend/Formulation.hpp"
#include "dynosam/formulations/KeyFrameHybridEstimator.hpp"

namespace dyno {

struct PoseChangeInput {
  DYNO_POINTER_TYPEDEFS(PoseChangeInput)

  //! Frame id associated with the creation of this input
  FrameId frame_id;
  //! Timestamp associated with the creation of this input
  Timestamp timestamp;
  gtsam::Values new_values;
  gtsam::NonlinearFactorGraph new_factors;

  //! Which objects (if any) were invoved in factors/values at this frame
  ObjectIds involved_objects;
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
