#include "dynosam2/estimator/FullGraph.hpp"

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2Result.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/StereoFactor.h>

#include "dynosam_common/utils/TimingStats.hpp"
#include "dynosam_opt/FactorGraphTools.hpp"
#include "dynosam_opt/IncrementalOptimization.hpp"
#include "dynosam_sensors/RGBDCamera.hpp"

namespace dyno2 {

using GenericStereoFactor =
    gtsam::GenericStereoFactor<gtsam::Pose3, dyno::Landmark>;

FullGraph::FullGraph(dyno::Camera::Ptr camera)
    : K_stereo_(dyno::RGBDCamera::Calibration(camera)),
      is_running_full_opt_(false),
      has_update_(false),
      needs_full_graph_opt_(false) {
  gtsam::ISAM2Params params;
  params.evaluateNonlinearError = true;
  params.relinearizeThreshold = 0.01;
  params.relinearizeSkip = 1;
  isam_ = std::make_unique<gtsam::ISAM2>(params);
}

FullGraph::~FullGraph() {
  if (full_graph_opt_thread_.joinable()) {
    full_graph_opt_thread_.join();
  }
}

void FullGraph::addStaticMeasurements(
    const VioGraph& estimator,
    const std::vector<KeypointIdentifier>& vo_measurements) {
  const bool is_initalised = !states_.empty();

  std::unordered_set<dyno::FrameId> involved_frames;
  for (const auto& kpi : vo_measurements) {
    // only add for keyframes
    const dyno::FrameId fid = kpi.frameId();
    if (!estimator.isKeyframe(fid)) {
      continue;
    }

    involved_frames.insert(fid);

    StereoMeasurement stereo_measurement;
    CHECK(estimator.getMeasurement(kpi, stereo_measurement));

    const dyno::TrackletId& tid = kpi.trackletId();
    CHECK(estimator.landmarkExists(tid));

    auto [measurement, model] = stereo_measurement;
    model = dyno::factor_graph_tools::robustifyHuber(0.01, model);

    const gtsam::Symbol pose_sym('x', fid);
    const gtsam::Symbol lmk_sym('l', tid);

    if (!stateExists(fid)) {
      FullGraphState full_state;
      full_state.vio_state.emplace(estimator.getVioState(fid));
      states_[fid] = full_state;
      update_.new_states.vo.push_back(fid);
    }

    update_.new_factors.emplace_shared<GenericStereoFactor>(
        measurement, model, pose_sym, lmk_sym, K_stereo_);

    // not in the estimator and not yet added
    if (!this->landmarkExists(tid)) {
      CHECK(!isam_->valueExists(lmk_sym) &&
            !update_.new_values.exists(lmk_sym));
      const dyno::Landmark& lmk = estimator.getLandmark(tid);

      update_.new_values.insert(lmk_sym, lmk);
      this->setLandmark(tid, lmk);
    }

    if (!isam_->valueExists(pose_sym) && !update_.new_values.exists(pose_sym)) {
      // still take the best pose from the estimator
      const gtsam::Pose3& pose = estimator.getPose(fid);
      update_.new_values.insert(pose_sym, pose);
    }
  }

  LOG(INFO) << "New states add full graph: "
            << dyno::container_to_string(update_.new_states.vo);
  LOG(INFO) << "Affected sates full graph: "
            << dyno::container_to_string(involved_frames);

  // there are at least some frames involved
  if (involved_frames.size() > 0) {
    // wait until we have at least a few states before we begin to optimise
    if (states_.size() > 3) {
      needs_full_graph_opt_ = true;
    }

    if (!is_initalised) {
      // add prior on first pose
      const dyno::FrameId frame_id = firstFrameId();

      // TODO: eventually from state?
      update_.new_factors.addPrior<gtsam::Pose3>(
          gtsam::Symbol('x', frame_id), estimator.getPose(frame_id),
          gtsam::noiseModel::Isotropic::Sigma(6u, 0.00001));
    }
  }
}

bool FullGraph::synchronise(VioGraph* vio_graph, const DoGraphPtrMap& do_graphs,
                            UpdatedStates& involved_states) {
  // first get the involved frames
  if (vio_states_.numStates() == 0) {
    LOG(WARNING) << "No states to sync!";
    has_update_ = false;
    return false;
  }

  const dyno::FrameId vio_head = vio_graph->currentFrameId();
  const dyno::FrameId last_optimised_vio = vio_states_.currentFrameId();
  const dyno::FrameId first_optimied_vio = vio_states_.firstFrameId();
  // check this matches with the first frame in the graph
  CHECK_EQ(first_optimied_vio, vio_graph->firstFrameId());

  LOG(INFO) << "Building Camera PGO from k=" << first_optimied_vio << " -> "
            << last_optimised_vio << " with head=" << vio_head;

  CHECK_LE(last_optimised_vio, vio_head);

  // TODO: covariance from estimation!!!
  gtsam::SharedNoiseModel relative_noise_model =
      gtsam::noiseModel::Isotropic::Sigma(6u, 0.8);

  gtsam::SharedNoiseModel prior_noise_model =
      gtsam::noiseModel::Isotropic::Sigma(6u, 0.00001);

  BaProblemInterface problem;

  VioGraphState* vio_state;
  // build problem upto last optimized vio
  for (dyno::FrameId k0 = first_optimied_vio; k0 <= last_optimised_vio; k0++) {
    if (k0 > first_optimied_vio) {
      const ViFrame::Ptr vi_frame_k0 = vio_graph->getFramePtr(k0);
      const Motion3ReferenceFrame& T_ij = vi_frame_k0->getRelativePose();
      const dyno::FrameId frame_i = T_ij.from();

      // TODO: covariance from estimation!!
      problem.addFactor(boost::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
          gtsam::Symbol('x', frame_i), gtsam::Symbol('x', k0),
          static_cast<gtsam::Pose3>(T_ij), relative_noise_model));
    }

    const gtsam::Symbol sym('x', k0);
    // eqialent to checking if keyframe and in problem
    if (getVioState(k0, vio_state)) {
      const gtsam::Pose3& X_W_k0_refined = vio_state->pose_W;
      // problem.insert(sym, X_W_k0_refined);

      problem.addPrior<gtsam::Pose3>(sym, X_W_k0_refined, prior_noise_model);
    }
    const gtsam::Pose3& X_W_k0 = vio_graph->getPose(k0);
    problem.insert(sym, X_W_k0);
  }

  gtsam::Values optimised;
  dyno::NonlinearOptimizerSummary summary = problem.optimize(optimised, 5);

  VLOG(10) << "Camera PGO Initial error: " << summary.initial_error
           << " final error " << summary.final_error << " time[s] "
           << summary.cumulative_time_in_seconds
           << " #iterations= " << summary.numIterations();

  // recover all optimised poses
  for (dyno::FrameId k0 = first_optimied_vio; k0 <= last_optimised_vio; k0++) {
    gtsam::Pose3 X_W_k0 = optimised.at<gtsam::Pose3>(gtsam::Symbol('x', k0));
    vio_graph->setPose(k0, X_W_k0);
    // since we dont have IMU!
    vio_graph->setVelocityFromPose(k0);
    involved_states.vo.push_back(k0);
  }

  // for the unoptimised camera poses (ie. max optimised frame -> current frame)
  // propogate from last optimised frame using visual odometry
  for (dyno::FrameId k0 = last_optimised_vio + 1; k0 <= vio_head; k0++) {
    const ViFrame::Ptr vi_frame_j = vio_graph->getFramePtr(k0);
    const Motion3ReferenceFrame& T_ij = vi_frame_j->getRelativePose();
    dyno::FrameId frame_i = T_ij.from();
    const gtsam::Pose3& X_W_i = vio_graph->getPose(frame_i);

    const gtsam::Pose3 X_W_k0 = X_W_i * T_ij;
    // update propogated pose
    vio_graph->setPose(k0, X_W_k0);
    // since we dont have IMU!
    vio_graph->setVelocityFromPose(k0);
    involved_states.vo.push_back(k0);
  }

  // update landmarks
  const LandmarkMap& landmark_map = static_cast<const LandmarkMap&>(*this);
  for (const auto& [tid, idx] : landmark_map) {
    const auto& lmk = landmark_map.landmarkAt(idx);
    vio_graph->setLandmark(tid, lmk);
  }

  has_update_ = false;

  return true;
}

void FullGraph::optimize() {
  if (full_graph_opt_thread_.joinable()) {
    full_graph_opt_thread_.join();
  }
  LOG(INFO) << "Starting Full Graph opt!";
  // first make copy of update so other threads can freely add to the new one
  IncrementalUpdate update = update_;
  // mark which states are involved with the update
  // involved_states_.push_back(update.involved_states);
  // clear old update
  update_ = IncrementalUpdate();
  is_running_full_opt_ = true;
  // launch thread
  full_graph_opt_thread_ = std::thread(&FullGraph::optimizeImpl, this, update);
}

bool FullGraph::getVioState(const dyno::FrameId fid, VioGraphState*& state) {
  // do quick look up in vio states
  if (!vio_states_.stateExists(fid)) {
    return false;
  }
  state = vio_states_.state(fid);
  return true;
}
// bool FullGraph::getDoState(const dyno::TemporalObjectId& tid, DOGraphState*&
// state) {
//   auto it = states_.find(tid.frameId());
//   if(it == states_.end()) {
//     return false;
//   }

//   FullGraphState& fgs = it->second;
//   if(fgs.do_graph_states) {
//     auto it = fgs.do_graph_states->find(tid.objectId());
//     if(it == fgs.do_graph_states->end()) {
//       return false;
//     }

//     state = &it->second;
//     return true;
//   }
//   return false;
// }

void FullGraph::optimizeImpl(IncrementalUpdate update) {
  dyno::utils::ChronoTimingStats opt_t("full_graph.opt");
  is_running_full_opt_ = true;

  // do the work
  using SmootherInterface = dyno::IncrementalInterface<gtsam::ISAM2>;
  SmootherInterface smoother_interface(isam_.get());
  smoother_interface.setMaxExtraIterations(5);

  gtsam::ISAM2Result result;
  bool is_smoother_ok = smoother_interface.optimize(
      &result, [&](const gtsam::ISAM2&,
                   SmootherInterface::UpdateArguments& update_arguments) {
        update_arguments.new_values = update.new_values;
        update_arguments.new_factors = update.new_factors;
      });

  if (!is_smoother_ok) {
    LOG(FATAL) << "Failed...";
  }

  LOG(INFO) << "ISAM2 result. Error before " << result.getErrorBefore()
            << " error after " << result.getErrorAfter();

  LOG(INFO) << "New states add full graph: "
            << dyno::container_to_string(update.new_states.vo);

  // fill vio states for all new vio frames
  for (const FrameId k0 : update.new_states.vo) {
    CHECK(stateExists(k0));
    FullGraphState& fgs = this->state(k0);
    CHECK(fgs.vio_state);

    CHECK(!vio_states_.stateExists(k0));

    vio_states_.state(k0) = &fgs.vio_state.value();
    LOG(INFO) << "Adding new VO state in full graph k=" << k0;
  }

  // update all states
  const gtsam::Values values = isam_->calculateBestEstimate();
  for (auto [k0, vio_state] : vio_states_) {
    gtsam::Pose3 X_W_k0_refined =
        values.at<gtsam::Pose3>(gtsam::Symbol('x', k0));
    // cannot upate velocity unless with IMU?
    vio_state->pose_W = X_W_k0_refined;
  }

  is_running_full_opt_ = false;
  needs_full_graph_opt_ = false;
  has_update_ = true;
}

}  // namespace dyno2
