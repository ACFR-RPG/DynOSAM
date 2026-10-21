#include "dynosam2/estimator/VioGraph.hpp"

#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/slam/StereoFactor.h>

#include "dynosam2/frontend/Utils.hpp"
#include "dynosam_common/utils/TimingStats.hpp"
#include "dynosam_opt/FactorGraphTools.hpp"
#include "dynosam_opt/NonlinearOptimizer.hpp"
#include "dynosam_opt/factors/FixedStereoFactor.hpp"
#include "dynosam_sensors/RGBDCamera.hpp"

namespace dyno2 {

VioGraph::VioGraph(dyno::Camera::Ptr camera) {
  std::shared_ptr<dyno::RGBDCamera> rgbd_camera =
      CHECK_NOTNULL(camera->safeGetRGBDCamera());
  K_stereo_ = rgbd_camera->getFakeStereoCalib();
  CHECK_NOTNULL(K_stereo_);

  K_ = camera->getGtsamCalibration();
}

void VioGraph::initaliseStates(
    dyno::FrameId frame_id, dyno::Timestamp timestamp,
    const dyno::StereoMeasurementStatusVector& measurements,
    const gtsam::Pose3& pose) {
  VioGraphState& state =
      initaliseStatesWithPose(frame_id, timestamp, measurements, pose);
  setStateVelocityFromPose(state);
}

void VioGraph::initaliseStates(
    dyno::FrameId frame_id, dyno::Timestamp timestamp,
    const dyno::StereoMeasurementStatusVector& measurements,
    const gtsam::Pose3& pose, const Velocity3& velocity_W) {
  VioGraphState& state =
      initaliseStatesWithPose(frame_id, timestamp, measurements, pose);
  state.velocity_W = velocity_W;
}

VioGraphState& VioGraph::initaliseStatesWithPose(
    dyno::FrameId frame_id, dyno::Timestamp timestamp,
    const dyno::StereoMeasurementStatusVector& measurements,
    const gtsam::Pose3& pose) {
  this->addMeasurements(measurements);

  // last state?
  VioGraphState& state = this->state(frame_id);
  state.timestamp = timestamp;
  state.frame_id = frame_id;
  state.pose_W = pose;

  return state;
}

void VioGraph::setPose(dyno::FrameId frame_id, const gtsam::Pose3& pose) {
  this->state(frame_id).pose_W = pose;
}

void VioGraph::setVelocity(dyno::FrameId frame_id,
                           const Velocity3& velocity_W) {
  this->state(frame_id).velocity_W = velocity_W;
}

void VioGraph::setVelocityFromPose(dyno::FrameId frame_id) {
  VioGraphState& current_state = this->state(frame_id);
  setStateVelocityFromPose(current_state);
}

void VioGraph::setStateVelocityFromPose(VioGraphState& current_state) {
  VioGraphState* previous_state{nullptr};
  if (!stateByAge(current_state.frame_id, 1, previous_state)) {
    // no previous state, so hopefully velocity is zero?
    return;
  }

  CHECK_NOTNULL(previous_state);
  const dyno::Timestamp dt =
      current_state.timestamp - previous_state->timestamp;
  current_state.velocity_W =
      computeWorldVelocity(previous_state->pose_W, current_state.pose_W, dt);
}

const VioGraphState& VioGraph::getVioState(dyno::FrameId frame_id) const {
  return this->state(frame_id);
}

const gtsam::Pose3& VioGraph::getPose(dyno::FrameId frame_id) const {
  return this->state(frame_id).pose_W;
}

const Velocity3& VioGraph::getVelocity(dyno::FrameId frame_id) const {
  return this->state(frame_id).velocity_W;
}

const std::set<dyno::FrameId>& VioGraph::keyFrames() const {
  return keyframes_;
}

bool VioGraph::poseExists(dyno::FrameId frame_id) const {
  return stateExists(frame_id);
}

bool VioGraph::getLandmark(dyno::TrackletId id, MapPoint& map_point) const {
  auto it = LandmarkMap::find(id);
  if (it == LandmarkMap::end()) {
    return false;
  }

  const dyno::Landmark& lmk = LandmarkMap::landmarkAt(it->second);
  map_point = MapPoint{lmk, dyno::background_label, id};
  return true;
}

size_t VioGraph::getLandmarks(MapPointVector& map_points) const {
  if (LandmarkMap::isContiguous()) {
    const size_t n = LandmarkMap::size();
    map_points = MapPointVector(LandmarkMap::size());
    for (size_t i = 0; i < n; i++) {
      const auto [id, lmk] = LandmarkMap::at(i);
      map_points[i] = MapPoint{lmk, dyno::background_label, id};
    }
  } else {
    const size_t n = LandmarkMap::size();
    map_points = MapPointVector();
    map_points.reserve(n);

    // walk over valid indices
    for (auto it = LandmarkMap::begin(); it != LandmarkMap::end(); it++) {
      Index valid_index = it->second;
      const auto [id, lmk] = LandmarkMap::at(valid_index);
      map_points.push_back(MapPoint{lmk, dyno::background_label, id});
    }
  }
  return map_points.size();
}

size_t VioGraph::getLandmarks(MapPoints& map_points) const { return 0; }

bool VioGraph::pruneKeyframes(size_t num_keyframes, size_t num_motion_frames) {
  // we do this until we fix speed with covisiblity and growing checks as we
  // hae to remove observations to do that!
  dyno::utils::ChronoTimingStats timer_t("vio_graph.prune_kf");
  std::set<dyno::FrameId> keyframes;

  // Try the most recent keyframes first.
  size_t added_kfs = 0;
  for (auto it = keyframes_.rbegin();
       it != keyframes_.rend() && added_kfs < num_keyframes; ++it) {
    // add_unique(*it);
    keyframes.insert(*it);
    ++added_kfs;
  }

  keyframes_ = keyframes;
  return true;

  // missing tracking...

  // VLOG(10) << "Beginning keyframe pruing: Initial set="
  //          << container_to_string(keyframes_);

  // // we still have a good number of keyframes
  // if (keyframes_.size() <= num_keyframes) {
  //   return false;
  // }

  // {
  //   // only compute if necessary
  //   utils::ChronoTimingStats t("vio_graph.prune_kf.compute_covis");
  //   computeCovisibilities();
  // }

  // utils::ChronoTimingStats t("vio_graph.prune_kf.loop");
  // while (keyframes_.size() > num_keyframes) {
  //   const FrameId current_keyframe_id = currentKeyframeId();
  //   const FrameId current_frame_id = currentFrameId();

  //   auto min_it = keyframes_.end();
  //   int min_obs = std::numeric_limits<int>::max();

  //   for (auto it = keyframes_.begin(); it != keyframes_.end(); ++it) {
  //     const FrameId keyframe = *it;

  //     const int coObs = std::max(covisibilities(current_frame_id, keyframe),
  //                                covisibilities(current_keyframe_id,
  //                                keyframe));

  //     // Preserve the oldest keyframe if it has sufficient covisibility.
  //     if (keyframe == *keyframes_.begin() && coObs >= 2) {
  //       continue;
  //     }

  //     if (coObs < min_obs) {
  //       min_obs = coObs;
  //       min_it = it;
  //     }
  //   }

  //   // Fall back to the oldest keyframe if every candidate was skipped.
  //   if (min_it == keyframes_.end()) {
  //     min_it = keyframes_.begin();
  //   }

  //   keyframes_.erase(min_it);
  // }

  // VLOG(10) << "After keyframe pruing: " << container_to_string(keyframes_);
  // return true;
}

bool VioGraph::setKeyframe(dyno::FrameId id, bool is_keyframe) {
  CHECK(is_keyframe);
  keyframes_.insert(id);

  // state should exist!
  states_.at(id).is_keyframe = true;
  return true;
}

bool VioGraph::isKeyframe(dyno::FrameId id) const {
  auto it = states_.find(id);
  if (it == states_.end()) {
    return false;
  }
  return it->second.is_keyframe;
}

void VioGraphEstimator::optimize(dyno::FrameId frame_id,
                                 std::vector<dyno::FrameId>& frames_affected,
                                 bool only_states) {
  // covisible tracklets
  //  auto frame_node_ptr = observations_->getFrame(frame_id);
  dyno::utils::ChronoTimingStats timer_t("vio_graph.opt");
  size_t last_n_kfs = 3;
  const std::set<dyno::FrameId>& keyframes = keyframes_;

  std::vector<dyno::FrameId> frames_to_try;
  frames_to_try.reserve(last_n_kfs + 2);

  const auto add_unique = [&frames_to_try](dyno::FrameId id) {
    if (std::find(frames_to_try.begin(), frames_to_try.end(), id) ==
        frames_to_try.end()) {
      frames_to_try.push_back(id);
    }
  };

  // Try the most recent keyframes first.
  size_t added_kfs = 0;
  for (auto it = keyframes.rbegin();
       it != keyframes.rend() && added_kfs < last_n_kfs; ++it) {
    add_unique(*it);
    ++added_kfs;
  }

  // Also try the current and previous frames.
  add_unique(frame_id - 1);
  add_unique(frame_id);

  // this is silly - we can build frames_to_try in order...
  std::sort(frames_to_try.begin(), frames_to_try.end());

  // thius is taking longer and longer!
  // keep track of landmarks seen recently?
  dyno::utils::ChronoTimingStats covis_t("vio_graph.opt.covis");
  // dont want all landmarks, just the ones seen at frames_to_try!

  // using TrackletSet = std::unordered_set<TrackletId>;
  Covisibilities covisibilities;
  // TODO: use covisibilities from the base graph!

  for (const auto& [tracklet_id, lmk_node] : observations_->getLandmarks()) {
    if (!landmarkExists(tracklet_id)) {
      continue;
    }

    // only build covisabilies for the requested frames
    std::vector<dyno::FrameId> seen_at_frames;
    seen_at_frames.reserve(frames_to_try.size());
    for (dyno::FrameId k0 : frames_to_try) {
      if (lmk_node->seenAtFrame(k0)) {
        seen_at_frames.push_back(k0);
      }
    }
    // Build undirected covisibility edges.
    //
    // Each tracklet is inserted once into each frame-pair edge.
    // Only distinct frame pairs are stored; there are no self-edges.
    for (size_t i = 0; i < seen_at_frames.size(); ++i) {
      const dyno::FrameId frame_i = seen_at_frames[i];

      for (size_t j = 0; j < i; ++j) {
        const dyno::FrameId frame_j = seen_at_frames[j];

        covisibilities[frame_i][frame_j].push_back(tracklet_id);
      }
    }
  }
  covis_t.stop();

  dyno::utils::ChronoTimingStats build_t("vio_graph.opt.build");

  // ---------------------------------------------------------------------------
  // 1. Select tracklets using contiguous vectors.
  //
  // A tracklet may be shared by multiple qualifying edges incident to a frame.
  // Accumulate these tracklets first and deduplicate after processing all
  // edges.
  // ---------------------------------------------------------------------------

  dyno::FastUnorderedMap<dyno::FrameId, std::vector<dyno::TrackletId>>
      selected_tracklets_by_frame;

  constexpr static size_t kMinSharedTracklets = 2;

  for (const auto& [frame_i, neighbours] : covisibilities) {
    for (const auto& [frame_j, shared_tracklets] : neighbours) {
      if (shared_tracklets.size() < kMinSharedTracklets) {
        continue;
      }

      auto& tracklets_i = selected_tracklets_by_frame[frame_i];
      auto& tracklets_j = selected_tracklets_by_frame[frame_j];

      // No hashing or uniqueness checks in this inner loop.
      tracklets_i.insert(tracklets_i.end(), shared_tracklets.begin(),
                         shared_tracklets.end());

      tracklets_j.insert(tracklets_j.end(), shared_tracklets.begin(),
                         shared_tracklets.end());
    }
  }

  // Deduplicate each frame's selected tracklets.
  // TODO: should not actually need to do this as we expect tracklets to be
  // unique from the map!
  for (auto& [frame_id, tracklets] : selected_tracklets_by_frame) {
    std::sort(tracklets.begin(), tracklets.end());

    tracklets.erase(std::unique(tracklets.begin(), tracklets.end()),
                    tracklets.end());
  }

  // ---------------------------------------------------------------------------
  // 2. Build the factor graph.
  // ---------------------------------------------------------------------------

  gtsam::Values values;
  gtsam::NonlinearFactorGraph graph;

  dyno::TrackletIds tracklets_with_update;
  dyno::FrameIds poses_with_update;

  using GenericFixedStereoFactor =
      gtsam::GenericFixedStereoFactor<gtsam::Pose3>;

  using GenericStereoFactor =
      gtsam::GenericStereoFactor<gtsam::Pose3, dyno::Landmark>;

  std::stringstream ss;
  ss << "VIO factor graph: ";

  size_t frames_used{0};
  // very important frames_to_try  is in order!
  for (dyno::FrameId frame_id : frames_to_try) {
    auto selected_it = selected_tracklets_by_frame.find(frame_id);

    if (selected_it == selected_tracklets_by_frame.end() ||
        selected_it->second.empty()) {
      continue;
    }

    CHECK(poseExists(frame_id)) << frame_id;

    const gtsam::Symbol pose_sym('x', frame_id);

    // Each frame appears once in frames_to_try.
    values.insert(pose_sym, getPose(frame_id));
    poses_with_update.push_back(frame_id);
    ++frames_used;

    const auto& selected_tracklets = selected_it->second;
    ss << "k= " << frame_id << " with n=" << selected_tracklets.size();

    for (dyno::TrackletId tracklet_id : selected_tracklets) {
      const auto lmk_node = observations_->getLandmark(tracklet_id);
      CHECK(lmk_node);

      const dyno::Landmark lmk = LandmarkMap::getLandmark(tracklet_id);

      auto [measurement, model] = lmk_node->getMeasurement(frame_id);
      model = dyno::factor_graph_tools::robustifyHuber(0.01, model);

      if (only_states) {
        // Landmark is fixed; only the pose is optimised.
        graph.emplace_shared<GenericFixedStereoFactor>(
            measurement, model, pose_sym, lmk, K_stereo_);
      } else {
        // Insert each landmark variable only once.
        const gtsam::Symbol lmk_sym('l', tracklet_id);

        if (!values.exists(lmk_sym)) {
          values.insert(lmk_sym, lmk);
          tracklets_with_update.push_back(tracklet_id);
        }

        graph.emplace_shared<GenericStereoFactor>(measurement, model, pose_sym,
                                                  lmk_sym, K_stereo_);
      }
    }
  }

  VLOG(5) << ss.str();

  // this handles the weirdly specific case that we only have 2 frames
  // with covisibilities (which either occurs right at the start or when we have
  // massive occlusion and we loose all tracks)
  // and since we only track by optical flow (no cross frame matching sadly)
  // we have very minimal constraints
  if (frames_used < 3) {
    return;
  }

  // ---------------------------------------------------------------------------
  // 3. Anchor the first requested pose.
  // ---------------------------------------------------------------------------

  if (!frames_to_try.empty()) {
    const dyno::FrameId anchor_frame = frames_to_try.front();

    CHECK(poseExists(anchor_frame)) << anchor_frame;

    const gtsam::Symbol anchor_sym('x', anchor_frame);

    // The anchor may have no selected observations, so ensure its value exists.
    if (!values.exists(anchor_sym)) {
      values.insert(anchor_sym, getPose(anchor_frame));
    }

    graph.addPrior<gtsam::Pose3>(
        anchor_sym, getPose(anchor_frame),
        gtsam::noiseModel::Isotropic::Sigma(6u, 0.00001));
  }

  build_t.stop();

  // Dog leg is just way faster for will defined problems
  gtsam::DoglegParams opt_params;
  opt_params.setMaxIterations(2);

  dyno::NonlinearOptimizer<gtsam::DoglegOptimizer> solver(graph, values,
                                                          opt_params);

  dyno::NonlinearOptimizerSummary summary;
  dyno::utils::ChronoTimingStats solve_t("vio_graph.opt.solve");
  gtsam::Values optimised_values = values;
  { CHECK(solver.solve(optimised_values, options_, &summary)); }
  solve_t.stop();

  VLOG(10) << "Initial error: " << summary.initial_error << " final error "
           << summary.final_error << " time[s] "
           << summary.cumulative_time_in_seconds
           << " #iterations= " << summary.numIterations();

  dyno::utils::ChronoTimingStats recover_t("vio_graph.opt.recover");

  frames_affected.reserve(poses_with_update.size());
  // poses with update must be in order so the velocity update is correct
  for (dyno::FrameId frame_id : poses_with_update) {
    gtsam::Symbol pose_sym('x', frame_id);
    gtsam::Pose3 pose = optimised_values.at<gtsam::Pose3>(pose_sym);
    setPose(frame_id, pose);

    // since we dont have IMU!
    setVelocityFromPose(frame_id);

    frames_affected.push_back(frame_id);
  }

  LOG(INFO) << "Frames affected in VIO refinement: "
            << dyno::container_to_string(frames_affected);

  for (dyno::TrackletId id : tracklets_with_update) {
    gtsam::Symbol lmk_sym('l', id);
    gtsam::Point3 lmk = optimised_values.at<gtsam::Point3>(lmk_sym);
    setLandmark(id, lmk);
  }
}

}  // namespace dyno2
