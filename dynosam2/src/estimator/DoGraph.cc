#include "dynosam2/estimator/DoGraph.hpp"

#include <gtsam/nonlinear/DoglegOptimizer.h>

#include "dynosam2/frontend/Utils.hpp"
#include "dynosam_common/Transforms.hpp"
#include "dynosam_common/utils/TimingStats.hpp"
#include "dynosam_opt/FactorGraphTools.hpp"
#include "dynosam_opt/NonlinearOptimizer.hpp"
#include "dynosam_opt/factors/HybridFormulationFactors.hpp"
#include "dynosam_sensors/Camera.hpp"
#include "dynosam_sensors/RGBDCamera.hpp"

namespace dyno2 {

DoGraph::DoGraph(dyno::Camera::Ptr camera, dyno::ObjectId object_id)
    : object_id_(object_id) {
  std::shared_ptr<dyno::RGBDCamera> rgbd_camera =
      CHECK_NOTNULL(camera->safeGetRGBDCamera());
  K_stereo_ = rgbd_camera->getFakeStereoCalib();
  CHECK_NOTNULL(K_stereo_);

  K_ = camera->getGtsamCalibration();
}

void DoGraph::initalizeNewAnchorFrame(
    dyno::FrameId frame_id,
    const dyno::StereoMeasurementStatusVector& measurements,
    const gtsam::Pose3 L) {
  this->addStereoMeasurements(measurements);

  // update anchor point we are estimating relative to!
  DOGraphState& state = this->state(frame_id);
  state.state.motion = dyno::Motion3ReferenceFrame(
      gtsam::Pose3::Identity(), dyno::MotionRepresentationStyle::F2F,
      dyno::ReferenceFrame::GLOBAL, frame_id, frame_id);
  state.state.pose = L;
  state.keyframe_status = dyno::DoKeyframeType::AnchorKeyFrame;
  state.H_W_KF_k = dyno::Motion3ReferenceFrame(
      gtsam::Pose3::Identity(), dyno::MotionRepresentationStyle::KF,
      dyno::ReferenceFrame::GLOBAL, frame_id, frame_id);

  // reset active keyframes so there is no overlap in the estimator
  // and forces new keyframes at the start
  active_keyframes_.clear();
  active_keyframes_.insert(frame_id);

  // clear the map
  // what about when we synchronize from the backend?
  LandmarkMap::clear();
}

// J is current frame
void DoGraph::initalizeStatesPropogate(
    dyno::FrameId frame_j,
    const dyno::StereoMeasurementStatusVector& measurements,
    const dyno::Motion3ReferenceFrame& G_i_ij, const gtsam::Pose3 X_W_j,
    const gtsam::Pose3 X_W_i) {
  this->addStereoMeasurements(measurements);

  // state i
  const DOGraphState& last_state = lastState();
  const dyno::FrameId frame_i = last_state.state.motion.to();
  CHECK_EQ(frame_i, G_i_ij.from());
  CHECK_EQ(frame_i, last_state.H_W_KF_k.to());

  // TODO: add links between motions!
  const gtsam::Pose3 H_W_ij_pose = X_W_j * G_i_ij * X_W_i.inverse();

  const dyno::Motion3ReferenceFrame H_W_ij(
      H_W_ij_pose, dyno::MotionRepresentationStyle::F2F,
      dyno::ReferenceFrame::GLOBAL, frame_i, frame_j);

  gtsam::Pose3 L_W_j = H_W_ij->compose(last_state.state.pose);

  // in effect we dont use A_KF for anything except when we compute it for
  // optimisation store everything as pose!
  CHECK(!stateExists(frame_j));
  DOGraphState& state = this->state(frame_j);
  state.state.pose = L_W_j;
  state.state.motion = H_W_ij;
  // propogate using the frame-to-frame motion to construct a motion chain
  // from the anchor frame to the current frame
  gtsam::Pose3 H_W_KF_k = H_W_ij.estimate() * last_state.H_W_KF_k.estimate();

  state.H_W_KF_k = dyno::Motion3ReferenceFrame(
      H_W_KF_k, dyno::MotionRepresentationStyle::KF,
      dyno::ReferenceFrame::GLOBAL, last_state.H_W_KF_k.from(), frame_j);
}

void DoGraph::setRegularKeyframe(dyno::FrameId frame_id) {
  active_keyframes_.insert(frame_id);

  DOGraphState& state = this->state(frame_id);
  // check that state is not an anchor keyframe
  //  warn if so
  if (state.keyframe_status == dyno::DoKeyframeType::AnchorKeyFrame) {
    LOG(WARNING) << "Regular keyframe requested but object is already an "
                    "AnchorKeyframe!";
    return;
  }
  state.keyframe_status = dyno::DoKeyframeType::RegularKeyFrame;
}

bool DoGraph::getLandmark(dyno::TrackletId id, dyno::FrameId frame_id,
                          MapPoint& map_point) const {
  auto it = LandmarkMap::find(id);
  if (it == LandmarkMap::end()) {
    return false;
  }

  const dyno::Landmark& lmk_L = LandmarkMap::landmarkAt(it->second);
  const gtsam::Pose3 L_W_k = getPoseWithMotion(frame_id).pose;
  const dyno::Landmark lmk_W_k = L_W_k * lmk_L;

  map_point = MapPoint{lmk_W_k, object_id_, id};
  return true;
}

size_t DoGraph::getLandmarks(dyno::FrameId frame_id,
                             MapPointVector& map_points) const {
  const gtsam::Pose3 L_W_k = getPoseWithMotion(frame_id).pose;
  if (LandmarkMap::isContiguous()) {
    const size_t n = LandmarkMap::size();
    map_points = MapPointVector(LandmarkMap::size());
    for (size_t i = 0; i < n; i++) {
      const auto [id, lmk_L] = LandmarkMap::at(i);
      const dyno::Landmark lmk_W_k = L_W_k * lmk_L;
      map_points[i] = MapPoint{lmk_W_k, object_id_, id};
    }
  } else {
    const size_t n = LandmarkMap::size();
    map_points = MapPointVector();
    map_points.reserve(n);

    // walk over valid indices
    for (auto it = LandmarkMap::begin(); it != LandmarkMap::end(); it++) {
      Index valid_index = it->second;
      const auto [id, lmk_L] = LandmarkMap::at(valid_index);
      const dyno::Landmark lmk_W_k = L_W_k * lmk_L;
      map_points.push_back(MapPoint{lmk_W_k, object_id_, id});
    }
  }
  return map_points.size();
}

const dyno::PoseWithMotion& DoGraph::getPoseWithMotion(
    dyno::FrameId frame_id) const {
  return states_.at(frame_id).state;
}

const dyno::Motion3ReferenceFrame& DoGraph::keyframeMotion(
    dyno::FrameId frame_id) const {
  return states_.at(frame_id).H_W_KF_k;
}

dyno::DoKeyframeType DoGraph::keyframeStatus(dyno::FrameId frame_id) const {
  return states_.at(frame_id).keyframe_status;
}

const gtsam::Pose3& DoGraph::anchorPose(dyno::FrameId frame_id) const {
  const dyno::FrameId anchor = states_.at(frame_id).H_W_KF_k.from();
  return states_.at(anchor).state.pose;
}

const gtsam::Pose3& DoGraph::getPose(dyno::FrameId frame_id) const {
  return states_.at(frame_id).state.pose;
}

bool isValidRotation(const gtsam::Pose3& pose, double tolerance = 1e-6) {
  const auto& R = pose.rotation().matrix();

  return R.allFinite() &&
         (R.transpose() * R - Eigen::Matrix3d::Identity()).norm() < tolerance &&
         std::abs(R.determinant() - 1.0) < tolerance;
}

void DoGraphEstimator::optimize(dyno::FrameId frame_id,
                                const VioGraph& vi_graph,
                                std::vector<dyno::FrameId>& frames_affected,
                                size_t keyframes_to_include, bool only_states) {
  if (active_keyframes_.empty()) {
    VLOG(10) << "Cannot optimize for Dynamic object: no active keyframes!";
    return;
  }

  // keyframes to include is how far back to go when doing the optimization
  auto collectKeyframesToInclude = [&](size_t N) -> std::set<dyno::FrameId> {
    const size_t count = std::min(N, active_keyframes_.size());

    std::set<dyno::FrameId> result;

    auto it = active_keyframes_.rbegin();
    for (size_t i = 0; i < count; ++i, ++it) {
      result.insert(*it);
    }
    return result;
  };

  const std::set<dyno::FrameId> keyframes_to_use =
      collectKeyframesToInclude(keyframes_to_include);
  LOG(INFO) << "active keyframes "
            << dyno::container_to_string(keyframes_to_use);
  // try adding the most recent 3 frames as well (so we can add a motion
  // constraint)
  std::set<dyno::FrameId> frames_to_try = keyframes_to_use;
  const dyno::FrameId first_kf = *keyframes_to_use.begin();
  frames_to_try.insert(frame_id);

  if (frame_id - 1 >= first_kf) {
    frames_to_try.insert(frame_id - 1);
  }

  if (frame_id - 2 >= first_kf) {
    frames_to_try.insert(frame_id - 2);
  }
  LOG(INFO) << "frames to try: " << dyno::container_to_string(frames_to_try);

  dyno::utils::ChronoTimingStats covis_t("do_graph.opt.covis");

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

  dyno::utils::ChronoTimingStats build_t("do_graph.opt.build");

  dyno::FastUnorderedMap<dyno::FrameId, std::vector<dyno::TrackletId>>
      selected_tracklets_by_frame;

  constexpr static size_t kMinSharedTracklets = 3;

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

  // after checking covibilities we select the frames to actually use in the
  // optimization this may slightly change the base (ie. configuration) frame we
  // use this is the case where not frames have covisability and while we dont
  // need all frames to see each other, there needs to be some overlap with the
  // first frame!
  std::vector<dyno::FrameId> frames_to_use;
  for (dyno::FrameId frame_id : frames_to_try) {
    auto selected_it = selected_tracklets_by_frame.find(frame_id);

    if (selected_it == selected_tracklets_by_frame.end() ||
        selected_it->second.empty()) {
      continue;
    }
    frames_to_use.push_back(frame_id);
  }

  const dyno::FrameId head = frame_id;
  LOG(INFO) << "head frame =" << head << " keyframes to use "
            << dyno::container_to_string(frames_to_use);
  // this handles the weirdly specific case that we only have 2 frames
  // with covisibilities (which either occurs right at the start or when we have
  // massive occlusion and we loose all tracks)
  // and since we only track by optical flow (no cross frame matching sadly)
  // we have very minimal constraints
  if (frames_to_use.size() < 3) {
    return;
  }
  // the first frame to use: we will estimate a motion from this frame (ie
  // H_W_base_k) this may get updated if we
  const dyno::FrameId base = *frames_to_use.begin();

  // frame_id is the head, therefore we need at least 2, but ideally 3 frames
  // between the head and the most recent keyframe
  if (head - base < 2) {
    return;
  }

  std::unordered_map<dyno::FrameId, gtsam::Pose3> initial_motions;
  std::unordered_map<dyno::FrameId, gtsam::Pose3> X_W_k_poses;
  // Anchor -> Base motion in W frame
  const dyno::Motion3ReferenceFrame H_W_A_B = states_.at(base).H_W_KF_k;
  CHECK_EQ(H_W_A_B.to(), base);
  // the anchor frame and base pose will not move during the optimisation
  const dyno::FrameId anchor_frame = H_W_A_B.from();
  // anchor pose
  const gtsam::Pose3 L_W_A = getPose(anchor_frame);
  const gtsam::Pose3 L_W_B = getPose(base);

  // 'inverse': Base -> Anchor motion in W frame, needed for computing Base ->
  // Motion in W
  // const gtsam::Pose3 H_W_B_A = L_W_B * ( L_W_A.inverse() * H_W_A_B.estimate()
  // *  L_W_A).inverse() * L_W_B.inverse(); Transform from the base-frame motion
  // representation back to the original anchor-frame motion representation.
  const gtsam::Pose3 H_W_B_A = L_W_A * L_W_B.inverse();
  // for each frame to try build an initial motion w.r.t base
  for (dyno::FrameId k0 : frames_to_use) {
    // the full motion from the anchor frame to k0
    const dyno::Motion3ReferenceFrame H_W_A_k = states_.at(k0).H_W_KF_k;
    // sanity check we all have the same anchor frame
    CHECK_EQ(anchor_frame, H_W_A_k.from());
    CHECK_EQ(k0, H_W_A_k.to());

    if (!isValidRotation(H_W_A_k)) {
      LOG(ERROR) << "Invalid H_W_A_k at frame " << k0;
    }

    /**
     * After lots of debugging it was found that there is some numerical
     * issues in doing a round trip reference frame transform (ie. transforming
     * into some local frame, doing a delta update in the local frame, and then
     * returning to the original frame) when the intermediate update is used
     * recursively to update the coordinate frame. The result is that we get
     * some blow up of orthogonality error in the rotation matrix of some
     * motion, which is then compounded when we initalise the new pose L <- H *
     * L. We did some Gtests with the help of Chat-GPT to show that when
     * maintaining two evolving poses, repeatedly computing their relative
     * transform, reconstructing an updated pose, and feeding the result back
     * into the next iteration somewhat reconstructs this error. Doing the
     * update in the lie-algebra helps prevent this numerical drift but in the
     * case of motions (where everthing is applied on the left) this appears to
     * help less? Instead we just ensure that our composed (transformed) motion
     * is also a valid SO(3). We apply the correction here becuase extensive
     * debugging showed that this is the first place we start seeing errors in
     * the orthogonality of the pose.
     */
    const gtsam::Pose3 H_W_B_k =
        dyno::normalizeRotation(H_W_A_k.estimate() * H_W_B_A);
    if (!isValidRotation(H_W_B_k)) {
      LOG(ERROR) << "Invalid H_W_B_k at frame " << k0;
    }

    initial_motions[k0] = H_W_B_k;
    X_W_k_poses[k0] = vi_graph.getPose(k0);
  }

  gtsam::Values values;
  gtsam::NonlinearFactorGraph graph;

  dyno::TrackletIds tracklets_with_update;

  std::stringstream ss;
  ss << "Do factor graph: ";

  size_t frames_used{0};
  for (dyno::FrameId frame_id : frames_to_use) {
    CHECK(stateExists(frame_id)) << frame_id;
    const gtsam::Symbol motion_sym('H', frame_id);

    const gtsam::Pose3& X_W_k = X_W_k_poses.at(frame_id);
    // Each frame appears once in frames_to_try.
    values.insert(motion_sym, initial_motions[frame_id]);
    ++frames_used;

    const auto selected_it = selected_tracklets_by_frame.find(frame_id);
    const auto& selected_tracklets = selected_it->second;
    ss << "k= " << frame_id << " with n=" << selected_tracklets.size();

    for (dyno::TrackletId tracklet_id : selected_tracklets) {
      const auto lmk_node = observations_->getLandmark(tracklet_id);
      CHECK(lmk_node);

      const dyno::Landmark lmk_L = LandmarkMap::getLandmark(tracklet_id);

      auto [measurement, model] = lmk_node->getMeasurement(frame_id);
      model = dyno::factor_graph_tools::robustifyHuber(0.01, model);

      // Insert each landmark variable only once.
      const gtsam::Symbol lmk_sym('l', tracklet_id);
      if (!values.exists(lmk_sym)) {
        values.insert(lmk_sym, lmk_L);
        tracklets_with_update.push_back(tracklet_id);
      }

      graph.emplace_shared<dyno::StereoHybridMotionFactor2>(
          measurement, L_W_B, X_W_k, model, K_stereo_, motion_sym, lmk_sym,
          false);
    }
  }

  VLOG(5) << ss.str();

  LOG(INFO) << "Motions effected: " << dyno::container_to_string(frames_to_use);
  // add prior on base frame
  const gtsam::Symbol base_motion_sym('H', base);
  graph.addPrior<gtsam::Pose3>(
      base_motion_sym, gtsam::Pose3::Identity(),
      gtsam::noiseModel::Isotropic::Sigma(6u, 0.00001));

  const gtsam::Symbol head_sym('H', head);
  const gtsam::Symbol head_m1_sym('H', head - 1);
  const gtsam::Symbol head_m2_sym('H', head - 2);

  if (values.exists(head_sym) && values.exists(head_m1_sym) &&
      values.exists(head_m2_sym)) {
    LOG(INFO) << "Adding smoothing factor";
    // TODO: params
    gtsam::Vector6 sigmas;
    sigmas << 0.2, 0.2, 0.2, 0.1, 0.1, 0.1;
    gtsam::SharedNoiseModel model =
        gtsam::noiseModel::Isotropic::Sigmas(sigmas);
    graph.emplace_shared<dyno::HybridSmoothingFactor>(head_m2_sym, head_m1_sym,
                                                      head_sym, L_W_B, model);
  }

  // TODO: do the time based options: usually with DO
  //  we have many less points so we can afford to do more iterations...
  gtsam::DoglegParams opt_params;
  opt_params.setMaxIterations(10);

  graph.print("DO ");

  dyno::NonlinearOptimizer<gtsam::DoglegOptimizer> solver(graph, values,
                                                          opt_params);

  dyno::NonlinearOptimizerSummary summary;
  // TODO: move to common interface!
  dyno::NonlinearOptimizerOptions options;
  dyno::utils::ChronoTimingStats solve_t("do_graph.opt.solve");
  gtsam::Values optimised_values = values;
  { CHECK(solver.solve(optimised_values, options, &summary)); }
  solve_t.stop();

  VLOG(10) << "DO Initial error: " << summary.initial_error << " final error "
           << summary.final_error << " time[s] "
           << summary.cumulative_time_in_seconds
           << " #iterations= " << summary.numIterations();

  frames_affected = frames_to_use;
  for (dyno::FrameId k0 : frames_to_use) {
    // do not update anchor pose! Or i guess base frame?
    if (k0 == anchor_frame || k0 == base) {
      continue;
    }

    // TODO: update H_W_km1_k
    DOGraphState& state = states_.at(k0);
    const gtsam::Pose3 H_W_KF_k_old = state.H_W_KF_k;

    const gtsam::Pose3 H_W_B_k_refined =
        optimised_values.at<gtsam::Pose3>(gtsam::Symbol('H', k0));
    const gtsam::Pose3 L_W_k_refined = H_W_B_k_refined * L_W_B;
    const gtsam::Pose3 H_W_A_k_refined = H_W_B_k_refined * H_W_A_B.estimate();

    state.state.pose = L_W_k_refined;
    state.H_W_KF_k.estimate_ = H_W_A_k_refined;

    CHECK_EQ(state.H_W_KF_k.from(), anchor_frame);
    CHECK_EQ(state.H_W_KF_k.to(), k0);
    // break;
  }

  for (dyno::TrackletId id : tracklets_with_update) {
    gtsam::Point3 lmk =
        optimised_values.at<gtsam::Point3>(gtsam::Symbol('l', id));
    setLandmark(id, lmk);
  }
}

}  // namespace dyno2
