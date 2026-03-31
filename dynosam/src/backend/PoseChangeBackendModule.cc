#include "dynosam/backend/PoseChangeBackendModule.hpp"

#include <gtsam/nonlinear/ISAM2Params.h>

#include "dynosam_common/PointCloudProcess.hpp"
#include "dynosam_opt/IncrementalOptimization.hpp"

namespace dyno {

PoseChangeVIBackendModule::PoseChangeVIBackendModule(
    const BackendParams& params, Camera::Ptr camera,
    HybridFormulationKeyFrame::Ptr formulation,
    const SharedGroundTruth& shared_ground_truth)
    : Base(params, camera, shared_ground_truth),
      formulation_(CHECK_NOTNULL(formulation)) {
  hybrid_accessor_ = formulation_->derivedAccessor<HybridAccessor>();
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

  utils::ChronoTimingStats timer(formulation_->getFullyQualifiedName() +
                                 ".update_incremental");
  using SmootherInterface = IncrementalInterface<gtsam::ISAM2>;
  SmootherInterface smoother_interface(smoother_.get());
  smoother_interface.setMaxExtraIterations(6);

  gtsam::ISAM2Result result;
  bool is_smoother_ok = smoother_interface.optimize(
      &result,
      [&](const gtsam::ISAM2&,
          SmootherInterface::UpdateArguments& update_arguments) {
        update_arguments.new_values = input->new_values;
        update_arguments.new_factors = input->new_factors;
      },
      error_hooks_);

  if (!is_smoother_ok) {
    LOG(FATAL) << "Failed...";
  }

  LOG(INFO) << "ISAM2 result. Error before " << result.getErrorBefore()
            << " error after " << result.getErrorAfter();
  gtsam::Values optimised_values = smoother_interface.calculateEstimate();
  formulation_->updateTheta(optimised_values);

  // for (const auto object_id : input->involved_objects) {
  //   StatusLandmarkVector local_landmarks =
  //       hybrid_accessor_->getLocalDynamicLandmarkEstimates(object_id);

  //   std::string path = dyno::getOutputFilePath(
  //       "refined_object_map_k" + std::to_string(input->frame_id) + "_j" +
  //       std::to_string(object_id) + ".pcd");
  //   VLOG(10) << "Writing object map of size " << local_landmarks.size() << "
  //   - "
  //            << path;
  //   saveAsPointCloud(local_landmarks, path);
  // }

  // alert frontend
  if (frontend_update_callback_) {
    frontend_update_callback_(input->frame_id, input->timestamp);
  }

  return makeOutput();
}

void PoseChangeVIBackendModule::registerFrontendUpdateCallback(
    const FrontendUpdateCallback& frontend_update_callback) {
  frontend_update_callback_ = frontend_update_callback;
}

}  // namespace dyno
