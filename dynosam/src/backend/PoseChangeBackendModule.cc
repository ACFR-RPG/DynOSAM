#include "dynosam/backend/PoseChangeBackendModule.hpp"

#include <gtsam/nonlinear/ISAM2Params.h>

#include "dynosam_common/PointCloudProcess.hpp"
#include "dynosam_opt/IncrementalOptimization.hpp"

namespace dyno {

FrameId BatchPoseChangeInput::startingFrame() const {
  return batch.front()->frame_id;
}

FrameId BatchPoseChangeInput::endingFrame() const {
  return batch.back()->frame_id;
}

std::pair<gtsam::Values, gtsam::NonlinearFactorGraph>
BatchPoseChangeInput::newValuesAndFactors() const {
  gtsam::Values values;
  gtsam::NonlinearFactorGraph factors;

  for (const auto& entry : batch) {
    values.insert_or_assign(entry->new_values);
    factors += entry->new_factors;
  }

  return {values, factors};
}

PoseChangeVIBackendModule::PoseChangeVIBackendModule(
    const BackendParams& params, Camera::Ptr camera,
    HybridFormulationKeyFrame::Ptr formulation,
    const SharedGroundTruth& shared_ground_truth)
    : Base(params, camera, shared_ground_truth),
      formulation_(CHECK_NOTNULL(formulation)) {
  hybrid_accessor_ =
      formulation_->derivedAccessor<HybridFormulationKeyFrameAccessor>();
  CHECK_NOTNULL(hybrid_accessor_);

  gtsam::ISAM2Params isam2_params;
  isam2_params.relinearizeThreshold = 0.01;
  isam2_params.relinearizeSkip = 1;
  // isam2_params.relinearizeSkip = FLAGS_regular_backend_relinearize_skip;
  isam2_params.keyFormatter = DynosamKeyFormatter;
  // isam2_params.enablePartialRelinearizationCheck = true;
  isam2_params.enablePartialRelinearizationCheck = false;
  isam2_params.evaluateNonlinearError = true;
  smoother_ = std::make_unique<gtsam::ISAM2>(isam2_params);

  error_hooks_ = formulation_->getCustomErrorHooks();
}

PoseChangeVIBackendModule::~PoseChangeVIBackendModule() {}

DynoState::Ptr PoseChangeVIBackendModule::spinOnce(
    PoseChangeInput::ConstPtr input) {
  LOG(INFO) << "In PoseChangeVIBackendModule";

  // expect pipeline to provide a batch input
  BatchPoseChangeInput::ConstPtr batch_input =
      safeCast<PoseChangeInput, BatchPoseChangeInput>(input);
  CHECK_NOTNULL(batch_input);

  utils::ChronoTimingStats timer(formulation_->getFullyQualifiedName() +
                                 ".update_incremental");
  using SmootherInterface = IncrementalInterface<gtsam::ISAM2>;
  SmootherInterface smoother_interface(smoother_.get());
  smoother_interface.setMaxExtraIterations(6);

  SharedModuleStates* shared_module_states =
      formulation_->map()->getSharedModuleStates();
  shared_module_states->is_backend_optimizing = true;

  const auto [new_values, new_factors] = batch_input->newValuesAndFactors();

  gtsam::ISAM2Result result;
  bool is_smoother_ok = smoother_interface.optimize(
      &result,
      [&](const gtsam::ISAM2&,
          SmootherInterface::UpdateArguments& update_arguments) {
        update_arguments.new_values = new_values;
        update_arguments.new_factors = new_factors;
      },
      error_hooks_);

  // update frame first so anything waiting on the is_backend_optimizing flag
  // will have the latest information about the frame
  shared_module_states->last_optimized_frame = batch_input->endingFrame();
  shared_module_states->is_backend_optimizing = false;

  if (!is_smoother_ok) {
    LOG(FATAL) << "Failed...";
  }

  LOG(INFO) << "ISAM2 result. Error before " << result.getErrorBefore()
            << " error after " << result.getErrorAfter();
  gtsam::Values optimised_values = smoother_interface.calculateEstimate();
  formulation_->updateTheta(optimised_values);

  // alert frontend
  if (update_callback_) {
    PoseChangeUpdateComplete event;
    event.starting_frame_id = batch_input->startingFrame();
    event.ending_frame_id = batch_input->endingFrame();

    for (const auto& entry : batch_input->batch) {
      event.keyframe_infos.insert2(entry->frame_id, entry->keyframe_info);
    }

    update_callback_(event);
  }

  return makeOutput();
}

void PoseChangeVIBackendModule::registerUpdateCallback(
    const PoseChangeUpdateCompleteCallback& callback) {
  update_callback_ = callback;
}

}  // namespace dyno
