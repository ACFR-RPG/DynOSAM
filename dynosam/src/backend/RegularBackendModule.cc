/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam/backend/RegularBackendModule.hpp"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include "dynosam/backend/Accessor.hpp"
#include "dynosam/backend/BackendFactory.hpp"
#include "dynosam/backend/Formulation.hpp"
#include "dynosam_common/Flags.hpp"
#include "dynosam_common/logger/Logger.hpp"
#include "dynosam_common/utils/SafeCast.hpp"
#include "dynosam_common/utils/TimingStats.hpp"
#include "dynosam_opt/FactorGraphTools.hpp"
#include "dynosam_opt/ISAM2Params.hpp"
#include "dynosam_opt/ISAM2UpdateParams.hpp"
#include "dynosam_opt/IncrementalOptimization.hpp"
#include "dynosam_opt/SlidingWindowOptimization.hpp"

DEFINE_int32(opt_window_size, 10, "Sliding window size for optimisation");
DEFINE_int32(opt_window_overlap, 4, "Overlap for window size optimisation");

DEFINE_bool(
    use_identity_rot_L_for_init, false,
    "For experiments: set the initalisation point of L with identity rotation");
DEFINE_bool(corrupt_L_for_init, false,
            "For experiments: corrupt the initalisation point for L with "
            "gaussian noise");
DEFINE_double(corrupt_L_for_init_sigma, 0.2,
              "For experiments: sigma value to correupt initalisation point "
              "for L. When corrupt_L_for_init is true");

// declared in BackendModule.hpp so it can be used accross multiple backends
DEFINE_string(updater_suffix, "",
              "Suffix for updater to denote specific experiments");

DEFINE_int32(regular_backend_relinearize_skip, 10,
             "ISAM2 relinearize skip param for the regular backend");

DEFINE_bool(
    regular_backend_log_incremental_stats, false,
    "If ISAM2 stats should be logged to file when running incrementally."
    " This will slow down compute!!");

DEFINE_bool(
    regular_backend_static_only, false,
    "Run as a Static SLAM backend only (i.e ignore dynamic measurements!)");

namespace dyno {

RegularVIBackendModule::RegularVIBackendModule(
    const BackendParams& backend_params, Camera::Ptr camera,
    std::shared_ptr<RegularVIBackendModule::Factory> factory,
    const SharedGroundTruth& shared_ground_truth)
    : Base(backend_params, camera, shared_ground_truth) {
  setupOptimizers();
  setupFormulation(factory);
}

RegularVIBackendModule::RegularVIBackendModule(
    const BackendParams& backend_params, Camera::Ptr camera,
    const BackendType& backend_type,
    const SharedGroundTruth& shared_ground_truth)
    : RegularVIBackendModule(
          backend_params, camera,
          DefaultBackendFactory<MapVision>::Create(backend_type),
          shared_ground_truth) {}

RegularVIBackendModule::~RegularVIBackendModule() {
  if (backend_params_.use_logger_) {
    formulation_->postUpdate(PostUpdateData(this->latestFrameId()));

    FormulationLoggingParams logging_params;
    logging_params.logging_suffix = FLAGS_updater_suffix;
    formulation_->logBackendFromMap(logging_params);
  }
}

std::pair<gtsam::Values, gtsam::NonlinearFactorGraph>
RegularVIBackendModule::getActiveOptimisation() const {
  LOG(FATAL) << "TODO";
}

Accessor::Ptr RegularVIBackendModule::getAccessor() const {
  return formulation_->getAsVIOAccessor();
}

const VIOFormulation<MapVision>::Ptr RegularVIBackendModule::formulation()
    const {
  return formulation_;
}

BackendModuleDisplay::Ptr RegularVIBackendModule::formulationDisplay() const {
  return formulation_display_;
}

RegularVIBackendModule::SpinReturn RegularVIBackendModule::boostrapSpin(
    VisionImuPacket::ConstPtr input) {
  LOG(INFO) << "In RegularVIBackendModule boostrap " << input->frameId();
  const FrameId frame_k = input->frameId();
  const Timestamp timestamp = input->timestamp();

  gtsam::Values new_values;
  gtsam::NonlinearFactorGraph new_factors;

  addInitialStates(input, new_values, new_factors);

  PreUpdateData pre_update_data(frame_k);
  pre_update_data.input = input;
  formulation_->preUpdate(pre_update_data);

  UpdateObservationParams update_params;
  update_params.enable_debug_info = true;
  update_params.do_backtrack = false;

  PostUpdateData post_update_data(frame_k);
  addMeasurements(update_params, frame_k, new_values, new_factors,
                  post_update_data);

  updateAndOptimize(frame_k, new_values, new_factors, post_update_data);
  formulation_->postUpdate(post_update_data);

  debug_info_ = DebugInfo();

  return {State::Nominal, makeOutput()};
}

RegularVIBackendModule::SpinReturn RegularVIBackendModule::nominalSpin(
    VisionImuPacket::ConstPtr input) {
  LOG(INFO) << "In RegularVIBackendModule nominal " << input->frameId();

  const FrameId frame_k = input->frameId();
  const Timestamp timestamp = input->timestamp();

  gtsam::Values new_values;
  gtsam::NonlinearFactorGraph new_factors;

  addStates(input, new_values, new_factors);

  PreUpdateData pre_update_data(frame_k);
  pre_update_data.input = input;
  formulation_->preUpdate(pre_update_data);

  UpdateObservationParams update_params;
  update_params.enable_debug_info = true;
  update_params.do_backtrack = false;

  PostUpdateData post_update_data(frame_k);
  addMeasurements(update_params, frame_k, new_values, new_factors,
                  post_update_data);

  updateAndOptimize(frame_k, new_values, new_factors, post_update_data);
  formulation_->postUpdate(post_update_data);

  debug_info_ = DebugInfo();

  return {State::Nominal, makeOutput()};
}

void RegularVIBackendModule::setupOptimizers() {
  // 0: Full-batch, 1: sliding-window, 2: incremental
  const RegularOptimizationType& optimization_mode =
      backend_params_.optimization_mode;
  if (optimization_mode == RegularOptimizationType::SLIDING_WINDOW) {
    LOG(INFO) << "Setting up backend for Sliding Window Optimisation";
    SlidingWindowOptimization::Params sw_params;
    sw_params.window_size = FLAGS_opt_window_size;
    sw_params.overlap = FLAGS_opt_window_overlap;
    sliding_window_opt_ =
        std::make_unique<SlidingWindowOptimization>(sw_params);
  }

  if (optimization_mode == RegularOptimizationType::INCREMENTAL) {
    LOG(INFO) << "Setting up backend for Incremental Optimisation.";
    dyno::ISAM2Params isam2_params;
    isam2_params.relinearizeThreshold = 0.01;
    isam2_params.relinearizeSkip = FLAGS_regular_backend_relinearize_skip;
    isam2_params.keyFormatter = DynosamKeyFormatter;
    // isam2_params.enablePartialRelinearizationCheck = true;
    isam2_params.evaluateNonlinearError = true;
    smoother_ = std::make_unique<dyno::ISAM2>(isam2_params);
  }
}

void RegularVIBackendModule::setupFormulation(
    std::shared_ptr<Factory> factory) {
  CHECK_NOTNULL(factory);

  FormulationParams formulation_params = backend_params_;
  Sensors sensors;
  sensors.camera = camera_;

  FormulationHooks hooks;
  hooks.setGroundTruthPacketRequest(this->shared_ground_truth_);

  FormulationVizWrapper<MapVision> wrapper = factory->createFormulation(
      formulation_params, map(), noise_models_, sensors, hooks);

  if (!wrapper.formulation) {
    throw DynosamException("Loaded formulation is null!");
  }

  formulation_ =
      std::dynamic_pointer_cast<VIOFormulation<MapVision>>(wrapper.formulation);
  if (!formulation_) {
    throw DynosamException(
        "Formulation loaded but does not inherit from VIOFormulation!");
  }
  formulation_display_ = wrapper.display;
  error_hooks_ = formulation_->getCustomErrorHooks();
}

void RegularVIBackendModule::addInitialStates(
    const VisionImuPacket::ConstPtr& input, gtsam::Values& new_values,
    gtsam::NonlinearFactorGraph& new_factors) {
  const FrameId frame_k = input->frameId();
  const Timestamp timestamp_k = input->timestamp();
  const auto& X_W_k_initial = input->cameraPose();

  updateMapWithMeasurements(frame_k, input, X_W_k_initial);

  formulation_->addStatesInitalise(new_values, new_factors, frame_k,
                                   timestamp_k, X_W_k_initial,
                                   gtsam::Vector3(0, 0, 0));
}

void RegularVIBackendModule::addStates(
    const VisionImuPacket::ConstPtr& input, gtsam::Values& new_values,
    gtsam::NonlinearFactorGraph& new_factors) {
  const FrameId frame_k = input->frameId();
  const Timestamp timestamp_k = input->timestamp();

  const gtsam::NavState predicted_nav_state = formulation_->addStatesPropogate(
      new_values, new_factors, frame_k, timestamp_k,
      input->relativeCameraTransform(), input->pim());

  updateMapWithMeasurements(frame_k, input, predicted_nav_state.pose());
}

void RegularVIBackendModule::addMeasurements(
    const UpdateObservationParams& update_params, FrameId frame_k,
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors,
    PostUpdateData& post_update_data) {
  {
    LOG(INFO) << "Starting updateStaticObservations";
    utils::ChronoTimingStats timer("backend.update_static_obs");
    post_update_data.static_update_result =
        formulation_->updateStaticObservations(frame_k, new_values, new_factors,
                                               update_params);
  }

  {
    if (!FLAGS_regular_backend_static_only) {
      LOG(INFO) << "Starting updateDynamicObservations";
      utils::ChronoTimingStats timer("backend.update_dynamic_obs");
      post_update_data.dynamic_update_result =
          formulation_->updateDynamicObservations(frame_k, new_values,
                                                  new_factors, update_params);
    }
  }

  // if (post_formulation_update_cb_) {
  //   post_formulation_update_cb_(formulation_, frame_k, new_values,
  //   new_factors);
  // }
}

void RegularVIBackendModule::updateMapWithMeasurements(
    FrameId frame_id_k, const VisionImuPacket::ConstPtr& input,
    const gtsam::Pose3& X_W_k) {
  CHECK_EQ(frame_id_k, input->frameId());

  // update static and ego motion
  map_->updateObservations(input->staticMeasurements());
  map_->updateSensorPoseMeasurement(frame_id_k, input->timestamp(),
                                    Pose3Measurement(X_W_k));

  // update dynamic and motions
  MotionEstimateMap object_motions;
  for (const auto& [object_id, object_track] : input->objectTracks()) {
    map_->updateObservations(object_track.measurements);
    object_motions.insert2(object_id, object_track.H_W_k_1_k);
  }
  map_->updateObjectMotionMeasurements(frame_id_k, object_motions);
}

void RegularVIBackendModule::updateAndOptimize(
    FrameId frame_id_k, const gtsam::Values& new_values,
    const gtsam::NonlinearFactorGraph& new_factors,
    PostUpdateData& post_update_data) {
  // 0: Full-batch, 1: sliding-window, 2: incremental
  const RegularOptimizationType& optimization_mode =
      backend_params_.optimization_mode;
  if (optimization_mode == RegularOptimizationType::FULL_BATCH) {
    updateBatch(frame_id_k, new_values, new_factors, post_update_data);
  } else if (optimization_mode == RegularOptimizationType::SLIDING_WINDOW) {
    updateSlidingWindow(frame_id_k, new_values, new_factors, post_update_data);
  } else if (optimization_mode == RegularOptimizationType::INCREMENTAL) {
    updateIncremental(frame_id_k, new_values, new_factors, post_update_data);
  } else {
    LOG(FATAL) << "Unknown optimisation mode" << optimization_mode;
  }

  // if (frontend_update_callback_) frontend_update_callback_(frame_id_k, 0);
}
void RegularVIBackendModule::updateIncremental(
    FrameId frame_id_k, const gtsam::Values& new_values,
    const gtsam::NonlinearFactorGraph& new_factors,
    PostUpdateData& post_update_data) {
  CHECK(smoother_) << "updateIncremental run but smoother was not setup!";
  utils::ChronoTimingStats timer(formulation_->getFullyQualifiedName() +
                                 ".update_incremental");
  using SmootherInterface = IncrementalInterface<dyno::ISAM2>;
  SmootherInterface smoother_interface(smoother_.get());

  dyno::ISAM2Result result;
  bool is_smoother_ok = smoother_interface.optimize(
      &result,
      [&](const dyno::ISAM2&,
          SmootherInterface::UpdateArguments& update_arguments) {
        update_arguments.new_values = new_values;
        update_arguments.new_factors = new_factors;

        // TODO: for now only dynamic isam2 update params but eventually will
        // need to merge post_update_data should already be updated!!!!
        convert(post_update_data.dynamic_update_result.isam_update_params,
                update_arguments.update_params);

        // if(update_arguments.update_params.newAffectedKeys) {
        //    for(const auto& [idx, affected_keys] :
        //    *update_arguments.update_params.newAffectedKeys) {
        //     std::stringstream ss;
        //     for(const auto& key : affected_keys) ss <<
        //     DynosamKeyFormatter(key) << " "; LOG(INFO) << "Factor affected
        //     " << idx << " keys: " << ss.str();
        //   }
        // }
      },
      error_hooks_);

  if (!is_smoother_ok) {
    LOG(FATAL) << "Failed...";
  }

  LOG(INFO) << "ISAM2 result. Error before " << result.getErrorBefore()
            << " error after " << result.getErrorAfter();
  gtsam::Values optimised_values = smoother_interface.calculateEstimate();
  formulation_->updateTheta(optimised_values);

  // set and update post update incremental result
  PostUpdateData::IncrementalResult incremental_result;
  incremental_result.factors = smoother_interface.getFactors();

  convert(result, incremental_result.isam2);

  post_update_data.incremental_result = incremental_result;

  if (FLAGS_regular_backend_log_incremental_stats) {
    VLOG(10) << "Logging incremental stats at frame " << frame_id_k;
    logIncrementalStats(frame_id_k, smoother_interface);
  }
}

void RegularVIBackendModule::updateBatch(
    FrameId frame_id_k, const gtsam::Values& new_values,
    const gtsam::NonlinearFactorGraph& new_factors,
    PostUpdateData& post_update_data) {
  if (backend_params_.full_batch_frame - 1 == (int)frame_id_k) {
    LOG(INFO) << " Doing full batch at frame " << frame_id_k;

    gtsam::LevenbergMarquardtParams opt_params;
    opt_params.verbosity = gtsam::NonlinearOptimizerParams::Verbosity::ERROR;

    const auto theta = formulation_->getTheta();
    const auto graph = formulation_->getGraph();
    utils::StatsCollector(formulation_->getFullyQualifiedName() +
                          ".full_batch_opt_num_vars_all")
        .AddSample(theta.size());

    double error_before = graph.error(theta);
    utils::ChronoTimingStats timer(formulation_->getFullyQualifiedName() +
                                   ".full_batch_opt");

    gtsam::LevenbergMarquardtOptimizer problem(graph, theta, opt_params);
    gtsam::Values optimised_values = problem.optimize();
    double error_after = graph.error(optimised_values);

    utils::StatsCollector(formulation_->getFullyQualifiedName() +
                          ".inner_iterations")
        .AddSample(problem.getInnerIterations());
    utils::StatsCollector(formulation_->getFullyQualifiedName() + ".iterations")
        .AddSample(problem.iterations());

    formulation_->updateTheta(optimised_values);
    LOG(INFO) << " Error before: " << error_before
              << " error after: " << error_after;
  }
}
void RegularVIBackendModule::updateSlidingWindow(
    FrameId frame_id_k, const gtsam::Values& new_values,
    const gtsam::NonlinearFactorGraph& new_factors,
    PostUpdateData& post_update_data) {
  CHECK(sliding_window_opt_);
  const auto sw_result =
      sliding_window_opt_->update(new_factors, new_values, frame_id_k);
  LOG(INFO) << "Sliding window result - " << sw_result.optimized;

  if (sw_result.optimized) {
    formulation_->updateTheta(sw_result.result);
  }
}

void RegularVIBackendModule::logIncrementalStats(
    FrameId frame_id_k,
    const IncrementalInterface<dyno::ISAM2>& smoother_interface) const {
  auto file_name_maker = [&](const std::string& name,
                             const std::string& file_type =
                                 ".csv") -> std::string {
    std::string file_name = formulation_->getFullyQualifiedName() + name;
    file_name += file_type;
    return getOutputFilePath(file_name);
  };

  const auto& result = smoother_interface.result();
  const auto milliseconds = smoother_interface.timing();

  ISAM2Stats stats(smoother_interface);

  const std::string isam2_log_file = file_name_maker("_isam2_timing");

  static bool is_first = true;

  if (is_first) {
    // clear the file first
    std::ofstream clear_file(isam2_log_file, std::ios::out | std::ios::trunc);
    if (!clear_file.is_open()) {
      LOG(FATAL) << "Error clearing file: " << isam2_log_file;
    }
    clear_file.close();  // Close the stream to ensure truncation is complete
    is_first = false;

    std::ofstream header_file(isam2_log_file, std::ios::out | std::ios::trunc);
    if (!header_file.is_open()) {
      LOG(FATAL) << "Error writing file header file: " << isam2_log_file;
    }

    header_file
        << "timing [ms],frame id,num opt values,num factors,nnz (graph),"
           "nnz (isam),avg. clique size,max clique size,num variables "
           "re-elinm,num variables relinearized,num new,num involved,num "
           "(only) relin,num fluid,is batch \n",

        header_file
            .close();  // Close the stream to ensure truncation is complete
    is_first = false;
  }

  std::fstream file(isam2_log_file,
                    std::ios::in | std::ios::out | std::ios::app);
  file.precision(15);
  file << milliseconds << "," << frame_id_k << "," << stats.num_variables << ","
       << stats.num_factors << "," << stats.nnz_elements_R << ","
       << stats.nnz_elements_tree << "," << stats.average_clique_size << ","
       << stats.max_clique_size <<
      // number variables involved in the bayes tree (ie effected because they
      // are in cliques with marked variables)
      "," << result.getVariablesReeliminated() <<
      // number variables that are marked
      "," << result.getVariablesRelinearized() << "," << result.newVariables
       << "," << result.involvedVariables << ","
       << result.onlyRelinearizedVariables << "," << result.fluidVariables
       << "," << result.isBatch << "\n";
  file.close();
}

}  // namespace dyno
