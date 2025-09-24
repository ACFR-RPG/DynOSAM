/*
 *   Copyright (c) 2025 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam/backend/ParallelHybridBackendModule.hpp"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/Marginals.h>
#include <tbb/tbb.h>

DEFINE_bool(use_relinearize_threshold_as_double, false,
            "If only a single value should be used as the realinerization "
            "threshold for all variables.");
DEFINE_double(relinearize_threshold, 0.01,
              "Relinearize threshold for ISAM2 params");

DEFINE_double(
    X_trans_relinearize_threshold, 0.01,
    "Relinearize threshold for X (cam pose) translation in ISAM2 params");
DEFINE_double(
    X_rot_relinearize_threshold, 0.01,
    "Relinearize threshold for X (cam pose) rotation in ISAM2 params");

DEFINE_double(
    H_trans_relinearize_threshold, 0.01,
    "Relinearize threshold for H (object motion) translation in ISAM2 params");
DEFINE_double(
    H_rot_relinearize_threshold, 0.01,
    "Relinearize threshold for H (object motion) rotation in ISAM2 params");

DEFINE_double(
    m_relinearize_threshold, 0.01,
    "Relinearize threshold for m (dynamic object point) in ISAM2 params");

DEFINE_int32(relinearize_skip, 1, "Relinearize skip for ISAM2 params");
DEFINE_int32(num_dynamic_optimize, 1,
             "Number of update steps to run for the object ISAM estimators");

DEFINE_bool(use_marginal_covariance, true,
            "If we should actually use the marginal covariance of X to "
            "condition the camera pose (as in the paper). "
            "Turning off is more computationall performant");

namespace dyno {

ParallelHybridBackendModule::ParallelHybridBackendModule(
    const BackendParams& backend_params, Camera::Ptr camera,
    ImageDisplayQueue* display_queue)
    : Base(backend_params, display_queue), camera_(CHECK_NOTNULL(camera)) {
  LOG(INFO) << "Creating ParallelHybridBackendModule";

  // TODO: set isam params
  dynamic_isam2_params_.keyFormatter = DynoLikeKeyFormatter;
  dynamic_isam2_params_.evaluateNonlinearError = true;
  dynamic_isam2_params_.enableDetailedResults = true;
  dynamic_isam2_params_.relinearizeSkip = FLAGS_relinearize_skip;

  if (FLAGS_use_relinearize_threshold_as_double) {
    LOG(INFO) << "Using FLAGS_relinearize_threshold for Parallel Object ISAM: "
              << FLAGS_relinearize_threshold;
    dynamic_isam2_params_.relinearizeThreshold = FLAGS_relinearize_threshold;
  } else {
    LOG(INFO) << "Using per-variable-type set of relinearisation thresholds "
                 "for Parallel Object ISAM";
    // Camera pose
    gtsam::Vector6 X_relinearize_threshold;
    X_relinearize_threshold.head<3>().setConstant(
        FLAGS_X_trans_relinearize_threshold);
    X_relinearize_threshold.head<3>().setConstant(
        FLAGS_X_rot_relinearize_threshold);
    // Object Motion
    gtsam::Vector6 H_relinearize_threshold;
    H_relinearize_threshold.head<3>().setConstant(
        FLAGS_H_trans_relinearize_threshold);
    H_relinearize_threshold.head<3>().setConstant(
        FLAGS_H_rot_relinearize_threshold);
    // Dynamic object point
    gtsam::Vector3 m_relinearize_threshold;
    m_relinearize_threshold.setConstant(FLAGS_m_relinearize_threshold);

    gtsam::FastMap<char, gtsam::Vector> thresholds;
    thresholds[kPoseSymbolChar] = X_relinearize_threshold;
    thresholds[kObjectMotionSymbolChar] = H_relinearize_threshold;
    thresholds[kDynamicLandmarkSymbolChar] = m_relinearize_threshold;

    dynamic_isam2_params_.relinearizeThreshold = thresholds;
  }

  static_isam2_params_.keyFormatter = DynoLikeKeyFormatter;
  static_isam2_params_.evaluateNonlinearError = true;
  // this value is very important for accuracy
  static_isam2_params_.relinearizeThreshold = 0.01;
  // this value is very important for accuracy
  static_isam2_params_.relinearizeSkip = 1;

  // sliding window of 20 frames...
  // this should be greater than the max track age to avoid adding static points
  // to poses that have been removed! (and becuase we dont keyframe...)
  static_estimator_ =
      gtsam::IncrementalFixedLagSmoother(25.0, static_isam2_params_);
  // static_estimator_ = gtsam::ISAM2(static_isam2_params_);

  Sensors sensors;
  sensors.camera = camera;

  FormulationHooks hooks;
  hooks.ground_truth_packets_request =
      [&]() -> std::optional<GroundTruthPacketMap> {
    return shared_module_info.getGroundTruthPackets();
  };

  FormulationParams formulation_params(base_params_);
  formulation_params.updater_suffix = "static";

  static_formulation_ = std::make_unique<HybridFormulation>(
      formulation_params, RGBDMap::create(), noise_models_, sensors, hooks);
}

ParallelHybridBackendModule::~ParallelHybridBackendModule() {
  LOG(INFO) << "Desctructing ParallelHybridBackendModule";

  if (base_params_.use_logger_) {
    logBackendFromEstimators();

    std::string file_name = "parallel_isam2_results";
    const std::string suffix = base_params_.updater_suffix;
    ;
    if (!suffix.empty()) {
      file_name += ("_" + suffix);
    }
    file_name += ".bson";

    const std::string file_path = getOutputFilePath(file_name);
    JsonConverter::WriteOutJson(result_map_, file_path,
                                JsonConverter::Format::BSON);
  }
}

// implementation taken from ObjectMotionSolverSAM
// TODO: update api
ParallelHybridBackendModule::SpinReturn
ParallelHybridBackendModule::boostrapSpinImpl(VisionImuPacket::ConstPtr input) {
  const FrameId frame_k = input->frameId();
  const Timestamp timestamp = input->timestamp();
  // TODO: sovle smoother
  //  non-sequentially?
  Pose3Measurement optimized_camera_pose =
      bootstrapUpdateStaticEstimator(input);

  parallelObjectSolve(input, optimized_camera_pose);

  // lazy update (not parallel)
  for (const ObjectId& object_id : input->getObjectIds()) {
    ParallelObjectISAM::Ptr estimator = getEstimator(object_id);
    CHECK(estimator) << "Somehow no estimator for j=" << object_id;
    const auto result = estimator->getResult();

    if (!result.was_smoother_ok) {
      LOG(WARNING) << "Could not record results for object smoother j="
                   << object_id << " as smoother was not ok";
      continue;
    }

    CHECK_EQ(result.frame_id, frame_k);
    result_map_.insert22(object_id, result.frame_id, result);
  }

  new_objects_estimators_.clear();
  return {State::Nominal, nullptr};
}

ParallelHybridBackendModule::SpinReturn
ParallelHybridBackendModule::nominalSpinImpl(VisionImuPacket::ConstPtr input) {
  const FrameId frame_k = input->frameId();
  const Timestamp timestamp = input->timestamp();

  const ObjectIds tracked_objects = input->getObjectIds();
  const bool has_objects = tracked_objects.size() > 0u;
  const bool requires_covariance_calc = has_objects;
  //  non-sequentially?
  Pose3Measurement optimized_camera_pose =
      nominalUpdateStaticEstimator(input, requires_covariance_calc);

  if (has_objects) {
    parallelObjectSolve(input, optimized_camera_pose);
  }
  // get estimator
  // should add previous measurements
  // updaet map
  // update estimator
  auto backend_output = constructOutputPacket(frame_k, timestamp);
  // TODO: this is gross - we need all the frame ids to timestamps for the
  // output to be valid assume that we get them from the shared module info
  // which is updated in the BackendModule registerInputCallback
  backend_output->involved_timestamp = shared_module_info.getTimestampMap();

  auto static_accessor = static_formulation_->accessorFromTheta();
  // draw trajectory on each object
  constexpr static int kWindow = 30;
  const int current_frame = static_cast<int>(frame_k);
  int start_frame = std::max(current_frame - kWindow, 1);

  if (input->debugImagery() && !input->debugImagery()->rgb_viz.empty()) {
    const cv::Mat& rgb = input->debugImagery()->rgb_viz;
    rgb.copyTo(backend_output->debug_image);
  }

  for (const ObjectId& object_id : tracked_objects) {
    ParallelObjectISAM::Ptr estimator = getEstimator(object_id);
    CHECK(estimator) << "Somehow no estimator for j=" << object_id;
    HybridAccessor::Ptr accessor = estimator->accessor();
    const auto result = estimator->getResult();

    if (!result.was_smoother_ok) {
      LOG(WARNING) << "Could not record results for object smoother j="
                   << object_id << " as smoother was not ok";
      continue;
    }

    CHECK_EQ(result.frame_id, frame_k);
    result_map_.insert22(object_id, result.frame_id, result);
    continue;

    // object poses in camera frame over some receeding time-horizon
    // for visualisation
    // very slow becuase we're query the accessor when we already have the
    // information in the backend output
    // TODO: fix!
    std::vector<gtsam::Point2> L_X_projected_vec;
    for (int i = start_frame; i <= current_frame; i++) {
      FrameId k = static_cast<FrameId>(i);
      StateQuery<gtsam::Pose3> X_W_k = static_accessor->getSensorPose(k);
      StateQuery<gtsam::Pose3> L_W_k = accessor->getObjectPose(k, object_id);
      if (X_W_k && L_W_k) {
        gtsam::Pose3 L_X_k = X_W_k->inverse() * L_W_k.get();
        // pose projected into the camera frame
        gtsam::Point2 L_X_k_projected;
        camera_->project(L_X_k.translation(), &L_X_k_projected);
        L_X_projected_vec.push_back(L_X_k_projected);
      }
    }

    if (!backend_output->debug_image.empty()) {
      const cv::Scalar colour = Color::uniqueId(object_id).bgra();

      for (size_t i = 0u; i < L_X_projected_vec.size(); i++) {
        const gtsam::Point2& projected_point = L_X_projected_vec.at(i);
        // https://github.com/mikel-brostrom/boxmot/blob/master/boxmot/trackers/basetracker.py
        int trajectory_thickness =
            static_cast<int>(std::sqrt(static_cast<float>(i + 1)) * 1.2f);
        // LOG(INFO) << trajectory_thickness;
        const auto pc_cur = utils::gtsamPointToCv(projected_point);
        // TODO: check point is in image?

        utils::drawCircleInPlace(backend_output->debug_image, pc_cur, colour, 1,
                                 trajectory_thickness);
      }
    }
  }

  // if (!backend_output->debug_image.empty() && display_queue_) {
  //   display_queue_->push(
  //       ImageToDisplay("Object Trajectories", backend_output->debug_image));
  // }

  new_objects_estimators_.clear();
  return {State::Nominal, backend_output};
}

Pose3Measurement ParallelHybridBackendModule::bootstrapUpdateStaticEstimator(
    VisionImuPacket::ConstPtr input) {
  utils::TimingStatsCollector timer("parallel_object_sam.static_estimator");

  const FrameId frame_k = input->frameId();
  auto map = static_formulation_->map();

  const auto& X_k_initial = input->cameraPose();

  map->updateObservations(input->staticMeasurements());
  map->updateSensorPoseMeasurement(frame_k, Pose3Measurement(X_k_initial));

  gtsam::Values new_values;
  gtsam::NonlinearFactorGraph new_factors;

  // update formulation with initial states
  // TODO: I think exactly the same as RegularBackendModule so could put into
  // base class?
  gtsam::NavState nav_state;
  if (input->pim()) {
    LOG(INFO) << "Initialising backend with IMU states!";
    nav_state = this->addInitialVisualInertialState(
        frame_k, static_formulation_.get(), new_values, new_factors,
        static_formulation_->noiseModels(),
        gtsam::NavState(X_k_initial, gtsam::Vector3(0, 0, 0)),
        gtsam::imuBias::ConstantBias{});

  } else {
    LOG(INFO) << "Initialising backend with VO only states!";
    nav_state = this->addInitialVisualState(
        frame_k, static_formulation_.get(), new_values, new_factors,
        static_formulation_->noiseModels(), X_k_initial);
  }

  // marginalise all values
  std::map<gtsam::Key, double> timestamps;
  double curr_id = static_cast<double>(this->spin_state_.iteration);
  for (const auto& key_value : new_values) {
    // LOG(INFO) << DynoLikeKeyFormatter(key_value.key);
    timestamps[key_value.key] = curr_id;
  }

  {
    utils::TimingStatsCollector timer(
        "parallel_object_sam.static_estimator.update");
    static_estimator_.update(new_factors, new_values, timestamps);
  }

  const auto& initial_pose_prior =
      static_formulation_->noiseModels().initial_pose_prior;
  gtsam::SharedGaussian gaussian_pose_prior =
      boost::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(
          initial_pose_prior);
  CHECK(gaussian_pose_prior)
      << "initial pose prior must be a Gaussian noise model!";

  return Pose3Measurement(nav_state.pose(), gaussian_pose_prior);
}

Pose3Measurement ParallelHybridBackendModule::nominalUpdateStaticEstimator(
    VisionImuPacket::ConstPtr input, bool should_calculate_covariance) {
  utils::TimingStatsCollector timer("parallel_object_sam.static_estimator");

  const FrameId frame_k = input->frameId();
  auto map = static_formulation_->map();
  map->updateObservations(input->staticMeasurements());

  gtsam::Values new_values;
  gtsam::NonlinearFactorGraph new_factors;

  const gtsam::NavState predicted_nav_state = this->addVisualInertialStates(
      frame_k, static_formulation_.get(), new_values, new_factors,
      noise_models_, input->relativeCameraTransform(), input->pim());
  // we dont have an uncertainty from the frontend
  map->updateSensorPoseMeasurement(
      frame_k, Pose3Measurement(predicted_nav_state.pose()));

  UpdateObservationParams update_params;
  update_params.enable_debug_info = true;
  // eventually should not need this if we start to use smart factors at least
  // for the static update
  update_params.do_backtrack = true;

  static_formulation_->updateStaticObservations(frame_k, new_values,
                                                new_factors, update_params);

  // marginalise all values
  std::map<gtsam::Key, double> timestamps;
  double curr_id = static_cast<double>(this->spin_state_.iteration);
  for (const auto& key_value : new_values) {
    // LOG(INFO) << DynoLikeKeyFormatter(key_value.key);
    timestamps[key_value.key] = curr_id;
  }

  utils::StatsCollector stats("parallel_object_sam.static_estimator.update");
  VLOG(10) << "Starting static estimator update...";
  auto tic = utils::Timer::tic();
  static_estimator_.update(new_factors, new_values, timestamps);
  auto toc = utils::Timer::toc<std::chrono::nanoseconds>(tic);
  int64_t milliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(toc).count();
  stats.AddSample(static_cast<double>(milliseconds));

  gtsam::ISAM2Result result = static_estimator_.getISAM2Result();

  VLOG(5) << "Finished LC Static update k" << frame_k
          << "  error before: " << result.errorBefore.value_or(NaN)
          << " error after: " << result.errorAfter.value_or(NaN)
          << " timing [ms]:" << milliseconds;

  // update results struct for timing data of static estimator
  ParallelObjectISAM::Result static_result;
  static_result.isam_result = result;
  static_result.was_smoother_ok = true;
  static_result.frame_id = frame_k;
  static_result.timing = milliseconds;
  result_map_.insert22(0, static_result.frame_id, static_result);

  gtsam::Values optimised_values = static_estimator_.calculateEstimate();
  static_formulation_->updateTheta(optimised_values);

  const gtsam::NavState& updated_nav_state =
      updateNavStateFromFormulation(frame_k, static_formulation_.get());

  auto accessor = static_formulation_->accessorFromTheta();
  StateQuery<gtsam::Pose3> X_w_k_opt_query = accessor->getSensorPose(frame_k);
  CHECK(X_w_k_opt_query);
  // TODO: should check that X_w_k_opt_query and updated_nav_state are close!!

  LOG(INFO) << "Nav state after estimation " << updated_nav_state;
  LOG(INFO) << "Bias after estimation " << imu_bias_;

  if (should_calculate_covariance) {
    if (FLAGS_use_marginal_covariance) {
      gtsam::Matrix66 X_w_k_cov;
      utils::TimingStatsCollector timer(
          "parallel_object_sam.camera_pose_cov_calc");
      gtsam::Marginals marginals(static_estimator_.getFactors(),
                                 optimised_values,
                                 gtsam::Marginals::Factorization::CHOLESKY);
      X_w_k_cov = marginals.marginalCovariance(X_w_k_opt_query.key());
      return Pose3Measurement(X_w_k_opt_query.get(), X_w_k_cov);
    } else {
      // arbitrary covariance to fix the camera pose in each DOFG
      const static double rotation_std = 0.01, translation_std = 0.1;
      gtsam::Matrix66 X_w_k_cov =
          (Eigen::Matrix<double, 6, 1>() << rotation_std * rotation_std,
           rotation_std * rotation_std, rotation_std * rotation_std,
           translation_std * translation_std, translation_std * translation_std,
           translation_std * translation_std)
              .finished()
              .asDiagonal();
      return Pose3Measurement(X_w_k_opt_query.get(), X_w_k_cov);
    }
  } else {
    return Pose3Measurement(X_w_k_opt_query.get());
  }
}

ParallelObjectISAM::Ptr ParallelHybridBackendModule::getEstimator(
    ObjectId object_id, bool* is_object_new) {
  std::lock_guard<std::mutex> lock(mutex_);

  bool is_new = false;
  // make new estimator if needed
  if (!sam_estimators_.exists(object_id)) {
    LOG(INFO) << "Making new ParallelObjectISAM for object " << object_id;

    FormulationHooks hooks;
    hooks.ground_truth_packets_request =
        [&]() -> std::optional<GroundTruthPacketMap> {
      return shared_module_info.getGroundTruthPackets();
    };

    Sensors sensors;
    sensors.camera = camera_;

    ParallelObjectISAM::Params params;
    params.num_optimzie = FLAGS_num_dynamic_optimize;
    params.isam = dynamic_isam2_params_;
    params.formulation = FormulationParams(base_params_);
    // // make this prior not SO small
    NoiseModels noise_models = NoiseModels::fromBackendParams(base_params_);
    sam_estimators_.insert2(
        object_id, std::make_shared<ParallelObjectISAM>(
                       params, object_id, noise_models, sensors, hooks));

    is_new = true;
  }
  if (is_object_new) *is_object_new = is_new;

  if (is_new) new_objects_estimators_.push_back(object_id);

  return sam_estimators_.at(object_id);
}

void ParallelHybridBackendModule::parallelObjectSolve(
    VisionImuPacket::ConstPtr input, const Pose3Measurement& X_W_k) {
  utils::TimingStatsCollector timer("parallel_object_sam.dynamic_estimator");
  const auto frame_id = input->frameId();
  const auto& object_tracks = input->objectTracks();
  tbb::parallel_for_each(
      object_tracks.begin(), object_tracks.end(),
      [&](const std::pair<ObjectId, VisionImuPacket::ObjectTracks>& update) {
        this->implSolvePerObject(frame_id, update.first, update.second, X_W_k);
      });
}

bool ParallelHybridBackendModule::implSolvePerObject(
    FrameId frame_id_k, ObjectId object_id,
    const VisionImuPacket::ObjectTracks& object_update,
    const Pose3Measurement& X_W_k) {
  bool is_object_new;
  ParallelObjectISAM::Ptr estimator = getEstimator(object_id, &is_object_new);

  // if object is new, dont update the smoother
  bool should_update_smoother = !is_object_new;

  CHECK_NOTNULL(estimator);
  auto map = estimator->map();

  const Motion3ReferenceFrame& H_W_k_1_k = object_update.H_W_k_1_k;

  // hack for now - if the object is new only update its map
  // this will create nodes in the Map but not update the estimator
  // only update the estimator otherwise!!

  // if new or last object update was more than 1 frame ago
  // this may be wrong if the smoother was not updated correctly...
  FrameId last_update_frame = estimator->getResult().frame_id;

  bool needs_new_key_frame = false;
  // Should this be last_update_frame == frame_id_k - 1u
  // if its more than that.... unsure
  if (!is_object_new && (frame_id_k > 0) &&
      (last_update_frame < (frame_id_k - 1u))) {
    VLOG(5) << "Only update k=" << frame_id_k << " j= " << object_id
            << " as object is not new but has reappeared. Previous update was "
            << last_update_frame;
    // only works if should_update_smoother makes sure that the formulation is
    // not updated but the map is
    should_update_smoother = false;
    needs_new_key_frame = true;
  }

  if (object_update.is_keyframe) {
    needs_new_key_frame = true;
  }

  estimator->update(frame_id_k, object_update.measurements, X_W_k, H_W_k_1_k,
                    should_update_smoother);

  if (needs_new_key_frame) {
    // needs the map to be updated for frame_id_k
    // this should happen in update
    // we dont want to update the formulation until the keyframe is inserted so
    // should_update_smoother must be false
    estimator->insertNewKeyFrame(frame_id_k);
  }
}

BackendOutputPacket::Ptr ParallelHybridBackendModule::constructOutputPacket(
    FrameId frame_k, Timestamp timestamp) const {
  auto backend_output = std::make_shared<BackendOutputPacket>();
  backend_output->timestamp = timestamp;
  backend_output->frame_id = frame_k;

  for (const auto& [object_id, estimator] : sam_estimators_) {
    // slow lookup for now!!
    // dont construct output if object is new
    // same logic as implSolvePerObject where we dont update the estimator on
    // the first pass, we just update the initial measurements etc...
    if (std::find(new_objects_estimators_.begin(),
                  new_objects_estimators_.end(),
                  object_id) != new_objects_estimators_.end()) {
      continue;
    }

    const ObjectPoseMap per_object_poses = estimator->getObjectPoses();
    const ObjectMotionMap per_object_motions =
        estimator->getFrame2FrameMotions();

    backend_output->optimized_object_motions += per_object_motions;
    backend_output->optimized_object_poses += per_object_poses;

    const auto& map = estimator->map();
    const auto& object_node = map->getObject(object_id);
    CHECK_NOTNULL(object_node);

    TemporalObjectMetaData temporal_object_info;
    temporal_object_info.object_id = object_id;
    temporal_object_info.first_seen = object_node->getFirstSeenFrame();
    temporal_object_info.last_seen = object_node->getLastSeenFrame();
    backend_output->temporal_object_data.push_back(temporal_object_info);

    // since we only show the current object map, get the landmarks only at the
    // current frame this should return an empty vector if the object does not
    // exist at the current frame
    backend_output->dynamic_landmarks +=
        estimator->getDynamicLandmarks(frame_k);
  }

  auto accessor = static_formulation_->accessorFromTheta();
  auto map = static_formulation_->map();

  backend_output->static_landmarks = accessor->getFullStaticMap();
  backend_output->T_world_camera = accessor->getSensorPose(frame_k).get();
  for (FrameId frame_id : map->getFrameIds()) {
    backend_output->optimized_camera_poses.push_back(
        accessor->getSensorPose(frame_id).get());
  }

  return backend_output;
}

void ParallelHybridBackendModule::logBackendFromEstimators() {
  // TODO: name + suffix
  std::string name = "parallel_hybrid";

  const std::string suffix = base_params_.updater_suffix;
  if (!suffix.empty()) {
    name += ("_" + suffix);
  }

  BackendLogger::UniquePtr logger = std::make_unique<BackendLogger>(name);

  Timestamp timestamp_k = this->spin_state_.timestamp;
  FrameId frame_id_k = this->spin_state_.frame_id;

  VLOG(20) << "Logging Parallel RGBD backend at frame " << frame_id_k;

  BackendOutputPacket::Ptr output =
      constructOutputPacket(frame_id_k, timestamp_k);

  const auto& gt_packets = shared_module_info.getGroundTruthPackets();

  logger->logObjectMotion(output->optimized_object_motions, gt_packets);
  logger->logObjectPose(output->optimized_object_poses, gt_packets);

  StatusLandmarkVector all_points = output->static_landmarks;
  // duplicated code from constructOutputPacket but we need the frame ids!!!
  auto accessor = static_formulation_->accessorFromTheta();
  auto map = static_formulation_->map();
  for (FrameId frame_id : map->getFrameIds()) {
    StateQuery<gtsam::Pose3> X_k_query = accessor->getSensorPose(frame_id);
    logger->logCameraPose(frame_id, X_k_query.get(), gt_packets);

    for (const auto& [object_id, estimator] : sam_estimators_) {
      all_points += estimator->getDynamicLandmarks(frame_id);
    }
  }
  logger->logMapPoints(all_points);

  logger.reset();
}

void ParallelHybridBackendModule::logGraphs() {
  FrameId frame_id_k = this->spin_state_.frame_id;
  for (const auto& [object_id, estimator] : sam_estimators_) {
    const auto& smoother = estimator->getSmoother();

    smoother.getFactorsUnsafe().saveGraph(
        dyno::getOutputFilePath("parallel_object_sam_k" +
                                std::to_string(frame_id_k) + "_j" +
                                std::to_string(object_id) + ".dot"),
        dyno::DynoLikeKeyFormatter);

    if (!smoother.empty()) {
      const auto isam_result = estimator->getResult().isam_result;
      gtsam::FastMap<gtsam::Key, std::string> colour_map;
      for (const auto& affected_keys : isam_result.markedKeys) {
        colour_map.insert2(affected_keys, "red");
      }

      dyno::factor_graph_tools::saveBayesTree(
          smoother,
          dyno::getOutputFilePath("parallel_object_sam_btree_k" +
                                  std::to_string(frame_id_k) + "_j" +
                                  std::to_string(object_id) + ".dot"),
          dyno::DynoLikeKeyFormatter, colour_map);
    }
  }

  // static
  static_estimator_.getFactors().saveGraph(
      dyno::getOutputFilePath("parallel_object_sam_k" +
                              std::to_string(frame_id_k) + "_static.dot"),
      dyno::DynoLikeKeyFormatter);

  const auto& smoother = static_estimator_.getISAM2();
  if (!smoother.empty()) {
    dyno::factor_graph_tools::saveBayesTree(
        smoother,
        dyno::getOutputFilePath("parallel_object_sam_btree_k" +
                                std::to_string(frame_id_k) + "_static.dot"),
        dyno::DynoLikeKeyFormatter);
  }
}

}  // namespace dyno
