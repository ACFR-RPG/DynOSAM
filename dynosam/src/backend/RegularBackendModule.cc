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
#include <gtsam_unstable/slam/PoseToPointFactor.h>

#include "dynosam/backend/Accessor.hpp"
#include "dynosam/backend/FactorGraphTools.hpp"
#include "dynosam/backend/Formulation.hpp"
#include "dynosam/backend/optimizers/ISAM2Params.hpp"
#include "dynosam/backend/optimizers/ISAM2UpdateParams.hpp"
#include "dynosam/backend/optimizers/IncrementalOptimization.hpp"
#include "dynosam/backend/optimizers/SlidingWindowOptimization.hpp"
#include "dynosam/backend/rgbd/HybridEstimator.hpp"
#include "dynosam/backend/rgbd/impl/test_HybridFormulations.hpp"
#include "dynosam/common/Flags.hpp"
#include "dynosam/factors/LandmarkMotionPoseFactor.hpp"
#include "dynosam/factors/LandmarkMotionTernaryFactor.hpp"
#include "dynosam/factors/LandmarkPoseSmoothingFactor.hpp"
#include "dynosam/factors/ObjectKinematicFactor.hpp"
#include "dynosam/logger/Logger.hpp"
#include "dynosam/utils/SafeCast.hpp"
#include "dynosam/utils/TimingStats.hpp"

DEFINE_int32(opt_window_size, 10, "Sliding window size for optimisation");
DEFINE_int32(opt_window_overlap, 4, "Overlap for window size optimisation");

DEFINE_bool(
    use_vo_factor, true,
    "If true, use visual odometry measurement as factor from the frontend");

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

namespace dyno {

RegularBackendModule::RegularBackendModule(const BackendParams& backend_params,
                                           Camera::Ptr camera,
                                           const UpdaterType& updater_type,
                                           ImageDisplayQueue* display_queue)
    : Base(backend_params, display_queue),
      camera_(CHECK_NOTNULL(camera)),
      updater_type_(updater_type) {
  CHECK_NOTNULL(map_);

  // // TODO: functioanlise and streamline with BackendModule
  // noise_models_.static_point_noise = gtsam::noiseModel::Isotropic::Sigma(
  //     3u, backend_params.static_point_noise_sigma_);
  // noise_models_.dynamic_point_noise = gtsam::noiseModel::Isotropic::Sigma(
  //     3u, backend_params.dynamic_point_noise_sigma_);
  // // set in base!
  // noise_models_.landmark_motion_noise = gtsam::noiseModel::Isotropic::Sigma(
  //     3u, backend_params.motion_ternary_factor_noise_sigma_);

  // if (backend_params.use_robust_kernals_) {
  //   noise_models_.static_point_noise = gtsam::noiseModel::Robust::Create(
  //       gtsam::noiseModel::mEstimator::Huber::Create(
  //           backend_params.k_huber_3d_points_),
  //       noise_models_.static_point_noise);

  //   noise_models_.dynamic_point_noise = gtsam::noiseModel::Robust::Create(
  //       gtsam::noiseModel::mEstimator::Huber::Create(
  //           backend_params.k_huber_3d_points_),
  //       noise_models_.dynamic_point_noise);

  //   // TODO: not k_huber_3d_points_ not just used for 3d points
  //   noise_models_.landmark_motion_noise = gtsam::noiseModel::Robust::Create(
  //       gtsam::noiseModel::mEstimator::Huber::Create(
  //           backend_params.k_huber_3d_points_),
  //       noise_models_.landmark_motion_noise);
  // }

  // CHECK_NOTNULL(noise_models_.static_point_noise);
  // CHECK_NOTNULL(noise_models_.dynamic_point_noise);
  // CHECK_NOTNULL(noise_models_.landmark_motion_noise);

  // noise_models_.dynamic_point_noise->print("Dynamic Point Noise");
  // noise_models_.landmark_motion_noise->print("Landmark motion noise");
  // // CHECK(false);

  noise_models_.print("RegularBackend noise models ");

  // setup smoother/optimizer variables
  setupUpdates();

  // set up formulation/some error handling
  formulation_ = std::move(makeFormulation());
}

RegularBackendModule::~RegularBackendModule() {
  LOG(INFO) << "Destructing RegularBackendModule";

  if (base_params_.use_logger_) {
    auto backend_info = createBackendMetadata();
    PostUpdateData post_update_data(spin_state_.frame_id);
    formulation_->postUpdate(post_update_data);
    formulation_->logBackendFromMap(backend_info);
  }
}

RegularBackendModule::SpinReturn RegularBackendModule::boostrapSpinImpl(
    RGBDInstanceOutputPacket::ConstPtr input) {
  const FrameId frame_k = input->getFrameId();
  CHECK_EQ(spin_state_.frame_id, frame_k);
  LOG(INFO) << "Running backend " << frame_k;
  gtsam::Values new_values;
  gtsam::NonlinearFactorGraph new_factors;

  addInitialStates(input, formulation_.get(), new_values, new_factors);

  PreUpdateData pre_update_data(frame_k);
  formulation_->preUpdate(pre_update_data);

  if (post_formulation_update_cb_) {
    post_formulation_update_cb_(formulation_, frame_k, new_values, new_factors);
  }

  PostUpdateData post_update_data(frame_k);
  updateAndOptimize(frame_k, new_values, new_factors, post_update_data);

  formulation_->postUpdate(post_update_data);

  // TODO: sanity checks that vision states are inline with the other frame idss
  // etc

  return {State::Nominal, nullptr};
}

RegularBackendModule::SpinReturn RegularBackendModule::nominalSpinImpl(
    RGBDInstanceOutputPacket::ConstPtr input) {
  const FrameId frame_k = input->getFrameId();
  const Timestamp timestamp = input->getTimestamp();
  LOG(INFO) << "Running backend " << frame_k;
  CHECK_EQ(spin_state_.frame_id, frame_k);

  // Pose estimate from the front-end
  gtsam::Values new_values;
  gtsam::NonlinearFactorGraph new_factors;

  addStates(input, formulation_.get(), new_values, new_factors);
  formulation_->preUpdate(PreUpdateData(frame_k));

  UpdateObservationParams update_params;
  update_params.enable_debug_info = true;
  update_params.do_backtrack =
      false;  // apparently this is v important for making the results == ICRA

  PostUpdateData post_update_data(frame_k);
  {
    LOG(INFO) << "Starting updateStaticObservations";
    utils::TimingStatsCollector timer("backend.update_static_obs");
    post_update_data.static_update_result =
        formulation_->updateStaticObservations(frame_k, new_values, new_factors,
                                               update_params);
  }

  {
    LOG(INFO) << "Starting updateDynamicObservations";
    utils::TimingStatsCollector timer("backend.update_dynamic_obs");
    post_update_data.dynamic_update_result =
        formulation_->updateDynamicObservations(frame_k, new_values,
                                                new_factors, update_params);
  }

  if (post_formulation_update_cb_) {
    post_formulation_update_cb_(formulation_, frame_k, new_values, new_factors);
  }

  LOG(INFO) << "Starting any updates";

  updateAndOptimize(frame_k, new_values, new_factors, post_update_data);
  LOG(INFO) << "Done any udpates";

  auto accessor = formulation_->accessorFromTheta();
  // update internal nav state based on the initial/optimised estimated in the
  // formulation this is also necessary to update the internal timestamp/frameid
  // variables within the VisionImuBackendModule
  updateNavStateFromFormulation(frame_k, formulation_.get());

  // TODO: sanity checks that vision states are inline with the other frame idss
  // etc

  utils::TimingStatsCollector timer(formulation_->getFullyQualifiedName() +
                                    ".post_update");
  formulation_->postUpdate(post_update_data);

  BackendOutputPacket::Ptr backend_output =
      constructOutputPacket(frame_k, timestamp);
  backend_output->involved_timestamp = input->involved_timestamps_;

  debug_info_ = DebugInfo();

  return {State::Nominal, backend_output};
}

void RegularBackendModule::setupUpdates() {
  // 0: Full-batch, 1: sliding-window, 2: incremental
  const int& optimization_mode = base_params_.optimization_mode;
  if (optimization_mode == 1) {
    LOG(INFO) << "Setting up backend for Sliding Window Optimisation";
    SlidingWindowOptimization::Params sw_params;
    sw_params.window_size = FLAGS_opt_window_size;
    sw_params.overlap = FLAGS_opt_window_overlap;
    sliding_window_opt_ =
        std::make_unique<SlidingWindowOptimization>(sw_params);
  }

  if (optimization_mode == 2) {
    LOG(INFO) << "Setting up backend for Incremental Optimisation.";
    dyno::ISAM2Params isam2_params;
    isam2_params.relinearizeThreshold = 0.01;
    isam2_params.relinearizeSkip = FLAGS_regular_backend_relinearize_skip;
    isam2_params.keyFormatter = DynoLikeKeyFormatter;
    // isam2_params.enablePartialRelinearizationCheck = true;
    isam2_params.evaluateNonlinearError = true;
    smoother_ = std::make_unique<dyno::ISAM2>(isam2_params);
  }
}

void RegularBackendModule::updateAndOptimize(
    FrameId frame_id_k, const gtsam::Values& new_values,
    const gtsam::NonlinearFactorGraph& new_factors,
    PostUpdateData& post_update_data) {
  // 0: Full-batch, 1: sliding-window, 2: incremental
  const int& optimization_mode = base_params_.optimization_mode;
  if (optimization_mode == 0) {
    updateBatch(frame_id_k, new_values, new_factors, post_update_data);
  } else if (optimization_mode == 1) {
    updateSlidingWindow(frame_id_k, new_values, new_factors, post_update_data);
  } else if (optimization_mode == 2) {
    updateIncremental(frame_id_k, new_values, new_factors, post_update_data);
  } else {
    LOG(FATAL) << "Unknown optimisation mode" << optimization_mode;
  }
}

void RegularBackendModule::updateIncremental(
    FrameId frame_id_k, const gtsam::Values& new_values,
    const gtsam::NonlinearFactorGraph& new_factors,
    PostUpdateData& post_update_data) {
  CHECK(smoother_) << "updateIncremental run but smoother was not setup!";
  utils::TimingStatsCollector timer(formulation_->getFullyQualifiedName() +
                                    ".update_incremental");

  using SmootherInterface = IncrementalInterface<dyno::ISAM2>;
  SmootherInterface smoother_interface(smoother_.get());

  // LOG(INFO) << "Adding incremental: new factors size " << new_factors.size();
  // for(size_t i = 0; i < new_factors.size(); i++) {
  //   std::cout << "idx " << i << " ";
  //   new_factors.at(i)->print("");
  //   std::cout << "\n";
  // }

  // error hooks should be updated for this formulation in the
  // makeFormulationFunction
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
        //     DynoLikeKeyFormatter(key) << " "; LOG(INFO) << "Factor affected
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

  // LOG(INFO) << "After increemental update " <<
  // incremental_result.factors.size(); for(size_t i = 0; i <
  // incremental_result.factors.size(); i++) {
  //   std::cout << "idx " << i << " ";
  //   incremental_result.factors.at(i)->print("");
  //   std::cout << "\n";
  // }

  convert(result, incremental_result.isam2);

  post_update_data.incremental_result = incremental_result;

  if (FLAGS_regular_backend_log_incremental_stats) {
    VLOG(10) << "Logging incremental stats at frame " << frame_id_k;
    logIncrementalStats(frame_id_k, smoother_interface);
  }
}

void RegularBackendModule::updateBatch(FrameId frame_id_k, const gtsam::Values&,
                                       const gtsam::NonlinearFactorGraph&,
                                       PostUpdateData& post_update_data) {
  if (base_params_.full_batch_frame - 1 == (int)frame_id_k) {
    LOG(INFO) << " Doing full batch at frame " << frame_id_k;

    gtsam::LevenbergMarquardtParams opt_params;
    opt_params.verbosity = gtsam::NonlinearOptimizerParams::Verbosity::ERROR;

    const auto theta = formulation_->getTheta();
    const auto graph = formulation_->getGraph();
    utils::StatsCollector(formulation_->getFullyQualifiedName() +
                          ".full_batch_opt_num_vars_all")
        .AddSample(theta.size());

    double error_before = graph.error(theta);
    utils::TimingStatsCollector timer(formulation_->getFullyQualifiedName() +
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

void RegularBackendModule::updateSlidingWindow(
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

void RegularBackendModule::logIncrementalStats(
    FrameId frame_id_k,
    const IncrementalInterface<dyno::ISAM2>& smoother_interface) const {
  auto file_name_maker = [&](const std::string& name,
                             const std::string& file_type =
                                 ".csv") -> std::string {
    std::string file_name = formulation_->getFullyQualifiedName() + name;
    // const std::string& suffix = base_params_.updater_suffix;
    // if (!suffix.empty()) {
    //   file_name += ("_" + suffix);
    // }
    file_name += file_type;
    return getOutputFilePath(file_name);
  };

  const auto& smoother = *smoother_interface.smoother();
  const auto& result = smoother_interface.result();
  const auto milliseconds = smoother_interface.timing();
  gtsam::Values theta =
      iOptimizationTraits<dyno::ISAM2>::getLinearizationPoint(smoother);
  gtsam::NonlinearFactorGraph graph =
      iOptimizationTraits<dyno::ISAM2>::getFactors(smoother);

  // this takes the longest time
  gtsam::GaussianFactorGraph::shared_ptr gfg = graph.linearize(theta);
  const auto sparsity_stats =
      factor_graph_tools::computeCholeskySparsityStats(gfg);

  size_t nnz_graph = sparsity_stats.nnz_elements;
  size_t nnz_bayes = smoother.roots().at(0)->calculate_nnz();

  const auto [max_clique_size, average_clique_size] =
      factor_graph_tools::getCliqueSize(smoother);

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
  file << milliseconds << "," << frame_id_k << "," << theta.size() << ","
       << graph.size() << "," << nnz_graph << "," << nnz_bayes << ","
       << average_clique_size << "," << max_clique_size <<
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

void RegularBackendModule::addInitialStates(
    const RGBDInstanceOutputPacket::ConstPtr& input,
    FormulationType* formulation, gtsam::Values& new_values,
    gtsam::NonlinearFactorGraph& new_factors) {
  CHECK(formulation);

  const FrameId frame_k = input->getFrameId();
  const Timestamp timestamp = input->getTimestamp();
  const auto& X_k_initial = input->T_world_camera_;

  // update map
  updateMapWithMeasurements(frame_k, input, X_k_initial);

  // update formulation with initial states
  if (input->pim_) {
    LOG(INFO) << "Initialising backend with IMU states!";
    this->addInitialVisualInertialState(
        frame_k, formulation, new_values, new_factors, noise_models_,
        gtsam::NavState(X_k_initial, gtsam::Vector3(0, 0, 0)),
        gtsam::imuBias::ConstantBias{});

  } else {
    LOG(INFO) << "Initialising backend with VO only states!";
    this->addInitialVisualState(frame_k, formulation, new_values, new_factors,
                                noise_models_, X_k_initial);
  }
}
void RegularBackendModule::addStates(
    const RGBDInstanceOutputPacket::ConstPtr& input,
    FormulationType* formulation, gtsam::Values& new_values,
    gtsam::NonlinearFactorGraph& new_factors) {
  CHECK(formulation);

  const FrameId frame_k = input->getFrameId();

  const gtsam::NavState predicted_nav_state = this->addVisualInertialStates(
      frame_k, formulation, new_values, new_factors, noise_models_,
      input->T_k_1_k_, input->pim_);

  updateMapWithMeasurements(frame_k, input, predicted_nav_state.pose());
}

void RegularBackendModule::updateMapWithMeasurements(
    FrameId frame_id_k, const RGBDInstanceOutputPacket::ConstPtr& input,
    const gtsam::Pose3& X_k_w) {
  CHECK_EQ(frame_id_k, input->getFrameId());

  map_->updateObservations(input->collectStaticLandmarkKeypointMeasurements());
  map_->updateObservations(input->collectDynamicLandmarkKeypointMeasurements());
  map_->updateSensorPoseMeasurement(frame_id_k, Pose3Measurement(X_k_w));

  // collected motion estimates for this current frame (ie. new motions!)
  // not handling the case where the update is incremental and other motions
  // have changed but right now the backend is not designed to handle this and
  // we currently dont run the backend with smoothing (tracking) in the
  // frontend.
  const auto estimated_motions =
      input->object_motions_.toEstimateMap(frame_id_k);
  map_->updateObjectMotionMeasurements(frame_id_k, estimated_motions);
}

Formulation<RegularBackendModule::RGBDMap>::UniquePtr
RegularBackendModule::makeFormulation() {
  FormulationParams formulation_params = base_params_;
  FormulationHooks hooks = createFormulationHooks();

  // setup error hooks
  ErrorHandlingHooks error_hooks;
  error_hooks.handle_ils_exception = [](const gtsam::Values& current_values,
                                        gtsam::Key problematic_key) {
    ErrorHandlingHooks::HandleILSResult ils_handle_result;
    // a little gross that I need to set this up in this function
    gtsam::NonlinearFactorGraph& prior_factors = ils_handle_result.pior_factors;
    auto& failed_on_object = ils_handle_result.failed_objects;

    ApplyFunctionalSymbol afs;
    afs.cameraPose([&prior_factors, &current_values](FrameId,
                                                     const gtsam::Symbol& sym) {
         const gtsam::Key& key = sym;
         gtsam::Pose3 pose = current_values.at<gtsam::Pose3>(key);
         gtsam::Vector6 sigmas;
         sigmas.head<3>().setConstant(0.001);  // rotation
         sigmas.tail<3>().setConstant(0.01);   // translation
         gtsam::SharedNoiseModel noise =
             gtsam::noiseModel::Diagonal::Sigmas(sigmas);
         prior_factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
             key, pose, noise);
       })
        .objectMotion(
            [&prior_factors, &current_values, &failed_on_object](
                FrameId k, ObjectId j, const gtsam::LabeledSymbol& sym) {
              const gtsam::Key& key = sym;
              gtsam::Pose3 pose = current_values.at<gtsam::Pose3>(key);
              gtsam::Vector6 sigmas;
              sigmas.head<3>().setConstant(0.001);  // rotation
              sigmas.tail<3>().setConstant(0.01);   // translation
              gtsam::SharedNoiseModel noise =
                  gtsam::noiseModel::Diagonal::Sigmas(sigmas);
              prior_factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
                  key, pose, noise);
              failed_on_object.push_back(std::make_pair(k, j));
            })
        .
        operator()(problematic_key);
    return ils_handle_result;
  };

  Formulation<RegularBackendModule::RGBDMap>::UniquePtr formulation;

  if (updater_type_ == RGBDFormulationType::WCME) {
    LOG(INFO) << "Using WCME";
    formulation = std::make_unique<WorldMotionFormulation>(
        formulation_params, getMap(), noise_models_, hooks);

  } else if (updater_type_ == RGBDFormulationType::WCPE) {
    LOG(INFO) << "Using WCPE";
    formulation = std::make_unique<WorldPoseFormulation>(
        formulation_params, getMap(), noise_models_, hooks);
  } else if (updater_type_ == RGBDFormulationType::HYBRID) {
    LOG(INFO) << "Using HYBRID";
    formulation = std::make_unique<RegularHybridFormulation>(
        formulation_params, getMap(), noise_models_, hooks);
  } else if (updater_type_ == RGBDFormulationType::TESTING_HYBRID_SD) {
    LOG(INFO) << "Using Hybrid Structureless Decoupled. Warning this is a "
                 "testing only formulation!";
    formulation =
        std::make_unique<test_hybrid::StructurelessDecoupledFormulation>(
            formulation_params, getMap(), noise_models_, hooks);
  } else if (updater_type_ == RGBDFormulationType::TESTING_HYBRID_D) {
    LOG(INFO) << "Using Hybrid Decoupled. Warning this is a testing only "
                 "formulation!";
    formulation = std::make_unique<test_hybrid::DecoupledFormulation>(
        formulation_params, getMap(), noise_models_, hooks);
  } else if (updater_type_ == RGBDFormulationType::TESTING_HYBRID_S) {
    LOG(INFO) << "Using Hybrid Structurless. Warning this is a testing only "
                 "formulation!";
    formulation = std::make_unique<test_hybrid::StructurlessFormulation>(
        formulation_params, getMap(), noise_models_, hooks);
  } else if (updater_type_ == RGBDFormulationType::TESTING_HYBRID_SMF) {
    LOG(INFO) << "Using Hybrid Smart Motion Factor. Warning this is a testing "
                 "only formulation!";
    formulation_params.min_dynamic_observations = 1u;
    formulation = std::make_unique<test_hybrid::SmartStructurlessFormulation>(
        formulation_params, getMap(), noise_models_, hooks);
  } else {
    CHECK(false) << "Not implemented";
  }

  CHECK_NOTNULL(formulation);
  // add additional error handling for incremental based on formulation
  auto* hybrid_formulation = static_cast<HybridFormulation*>(formulation.get());
  if (hybrid_formulation) {
    LOG(INFO) << "Adding additional error hooks for Hybrid formulation";
    error_hooks.handle_failed_object =
        [&hybrid_formulation](
            const std::pair<FrameId, ObjectId>& failed_on_object) {
          CHECK_NOTNULL(hybrid_formulation);
          const auto [frame_id, object_id] = failed_on_object;
          LOG(INFO) << "Is hybrid formulation with failed estimation at "
                    << info_string(frame_id, object_id);
          hybrid_formulation->forceNewKeyFrame(frame_id, object_id);
        };
  }
  error_hooks_ = error_hooks;

  return formulation;
}

BackendMetaData RegularBackendModule::createBackendMetadata() const {
  // TODO: cache?
  BackendMetaData backend_info;
  // backend_info.logging_suffix = base_params_.updater_suffix;
  backend_info.backend_params = &base_params_;
  return backend_info;
}

FormulationHooks RegularBackendModule::createFormulationHooks() const {
  // TODO: cache?
  FormulationHooks hooks;

  hooks.ground_truth_packets_request =
      [&]() -> std::optional<GroundTruthPacketMap> {
    return shared_module_info.getGroundTruthPackets();
  };

  return hooks;
}

BackendOutputPacket::Ptr RegularBackendModule::constructOutputPacket(
    FrameId frame_k, Timestamp timestamp) const {
  CHECK_NOTNULL(formulation_);
  return RegularBackendModule::constructOutputPacket(formulation_, frame_k,
                                                     timestamp);
}

BackendOutputPacket::Ptr RegularBackendModule::constructOutputPacket(
    const Formulation<RGBDMap>::UniquePtr& formulation, FrameId frame_k,
    Timestamp timestamp) {
  auto accessor = formulation->accessorFromTheta();

  auto backend_output = std::make_shared<BackendOutputPacket>();
  backend_output->timestamp = timestamp;
  backend_output->frame_id = frame_k;
  backend_output->T_world_camera = accessor->getSensorPose(frame_k).get();
  backend_output->static_landmarks = accessor->getFullStaticMap();
  // backend_output->optimized_object_motions =
  //     accessor->getObjectMotions(frame_k);
  backend_output->dynamic_landmarks =
      accessor->getDynamicLandmarkEstimates(frame_k);
  auto map = formulation->map();
  for (FrameId frame_id : map->getFrameIds()) {
    backend_output->optimized_camera_poses.push_back(
        accessor->getSensorPose(frame_id).get());
  }

  // fill temporal map information
  for (ObjectId object_id : map->getObjectIds()) {
    const auto& object_node = map->getObject(object_id);
    CHECK_NOTNULL(object_node);

    // TODO: based on measurements not on estimation so check that we have
    // landmarks for this object first?
    TemporalObjectMetaData temporal_object_info;
    temporal_object_info.object_id = object_id;
    temporal_object_info.first_seen = object_node->getFirstSeenFrame();
    temporal_object_info.last_seen = object_node->getLastSeenFrame();
    backend_output->temporal_object_data.push_back(temporal_object_info);
  }

  backend_output->optimized_object_motions = accessor->getObjectMotions();
  backend_output->optimized_object_poses = accessor->getObjectPoses();
  return backend_output;
}

}  // namespace dyno
