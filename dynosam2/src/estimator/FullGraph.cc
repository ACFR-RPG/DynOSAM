#include "dynosam2/estimator/FullGraph.hpp"

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2Result.h>
#include <gtsam/slam/StereoFactor.h>

#include "dynosam_common/utils/TimingStats.hpp"
#include "dynosam_opt/FactorGraphTools.hpp"
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

  dyno::StereoMeasurementStatusVector stereo_measurements;
  stereo_measurements.resize(vo_measurements.size());

  std::unordered_set<dyno::FrameId> involved_frames;
  for (const auto& kpi : vo_measurements) {
    // only add for keyframes
    const dyno::FrameId fid = kpi.frameId();
    if (!estimator.isKeyframe(fid)) {
      continue;
    }

    StereoMeasurement stereo_measurement;
    CHECK(estimator.getMeasurement(kpi, stereo_measurement));

    const dyno::TrackletId& tid = kpi.trackletId();
    CHECK(estimator.landmarkExists(tid));

    auto [measurement, model] = stereo_measurement;
    model = dyno::factor_graph_tools::robustifyHuber(0.01, model);

    const gtsam::Symbol pose_sym('x', fid);
    const gtsam::Symbol lmk_sym('l', tid);
    involved_frames.insert(fid);
    vio_keyframes_.insert(fid);

    if (!stateExists(fid)) {
      states_[fid] = FullGraphState();
    }

    // stereo_measurements.push_back(createStereoMeasurement(frame,
    // kpi.trackletId()));
    update_.new_factors.emplace_shared<GenericStereoFactor>(
        measurement, model, pose_sym, lmk_sym, K_stereo_);

    // not in the estimator and not yet added
    if (!isam_->valueExists(lmk_sym) && !update_.new_values.exists(lmk_sym)) {
      const dyno::Landmark& lmk = estimator.getLandmark(tid);
      update_.new_values.insert(lmk_sym, lmk);
    }

    if (!isam_->valueExists(pose_sym) && !update_.new_values.exists(pose_sym)) {
      const gtsam::Pose3& pose = estimator.getPose(fid);
      update_.new_values.insert(pose_sym, pose);
    }
  }

  update_.involved_states.vo.insert(update_.involved_states.vo.begin(),
                                    involved_frames.begin(),
                                    involved_frames.end());
  LOG(INFO) << "Added " << stereo_measurements.size()
            << " factors for full estimation";

  if (update_.involved_states.vo.size() > 0) {
    needs_full_graph_opt_ = true;

    if (!is_initalised) {
      // add prior on first pose
      auto it = states_.begin();
      const dyno::FrameId frame_id = it->first;

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
  if (involved_states_to_sync_.empty()) {
    LOG(WARNING) << "No states to sync!";
    has_update_ = false;
    return false;
  }

  UpdatedStates involved_states_internal;
  for (size_t idx : involved_states_to_sync_) {
    involved_states_internal.merge(involved_states_.at(idx));
  }

  // unfortunately updated really need to be applied in order and there is no
  // really good way to guarnatee this!
  std::vector<dyno::FrameId> involved_vo = involved_states_internal.vo;
  std::sort(involved_vo.begin(), involved_vo.end());

  involved_states_to_sync_.clear();
  has_update_ = false;
  // TODo: involved_states
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
  involved_states_to_sync_.push_back(involved_states_.size());
  involved_states_.push_back(update.involved_states);
  // clear old update
  update_ = IncrementalUpdate();
  needs_full_graph_opt_ = false;
  // launch thread
  full_graph_opt_thread_ = std::thread(&FullGraph::optimizeImpl, this, update);
}

void FullGraph::optimizeImpl(IncrementalUpdate update) {
  dyno::utils::ChronoTimingStats opt_t("full_graph.opt");
  is_running_full_opt_ = true;

  // do the work
  gtsam::ISAM2Result result =
      isam_->update(update.new_factors, update.new_values);
  LOG(INFO) << "ISAM2 result. Error before " << result.getErrorBefore()
            << " error after " << result.getErrorAfter();
  ///
  is_running_full_opt_ = false;
  has_update_ = true;
}

}  // namespace dyno2
