#pragma once

#include "dynosam/formulations/HybridEstimator.hpp"
#include "dynosam_opt/StateQuery.hpp"

namespace dyno {

template <typename MAP>
StateQuery<gtsam::Pose3> HybridAccessor<MAP>::getSensorPose(
    FrameId frame_id) const {
  const auto frame_node = this->map_->getFrame(frame_id);
  if (!frame_node) {
    return StateQuery<gtsam::Pose3>::InvalidMap();
  }
  // CHECK_NOTNULL(frame_node);
  return this->template query<gtsam::Pose3>(frame_node->makePoseKey());
}

template <typename MAP>
StateQuery<gtsam::Pose3> HybridAccessor<MAP>::getObjectMotion(
    FrameId frame_id, ObjectId object_id) const {
  StateQuery<Motion3ReferenceFrame> query =
      this->getObjectMotionReferenceFrame(frame_id, object_id);

  if (!query) {
    return StateQuery<gtsam::Pose3>(query.key(), query.status());
  } else {
    return StateQuery<gtsam::Pose3>(query.key(), query.get());
  }
}

template <typename MAP>
StateQuery<gtsam::Pose3> HybridAccessor<MAP>::getObjectPose(
    FrameId frame_id, ObjectId object_id) const {
  // we estimate a motion ^w_{s0}H_k, so we can compute a pose ^wL_k =
  // ^w_{s0}H_k * ^wL_{s0}
  const auto frame_node_k = this->map_->getFrame(frame_id);
  CHECK_NOTNULL(frame_node_k);
  if (!frame_node_k) {
    return StateQuery<gtsam::Pose3>::InvalidMap();
  }

  gtsam::Key motion_key = frame_node_k->makeObjectMotionKey(object_id);
  gtsam::Key pose_key = frame_node_k->makeObjectPoseKey(object_id);
  /// hmmm... if we do a query after we do an update but before an optimise then
  /// the motion will
  // be whatever we initalised it with
  // in the case of identity, the pose at k will just be L_s0 which we dont
  // want?
  StateQuery<gtsam::Pose3> e_H_k_world =
      this->template query<gtsam::Pose3>(motion_key);
  // CHECK(false);

  if (e_H_k_world) {
    auto key_frame_data =
        CHECK_NOTNULL(shared_hybrid_formulation_data_.key_frame_data);
    const auto range = CHECK_NOTNULL(key_frame_data->find(object_id, frame_id));
    const auto [s0, L0] = range->dataPair();

    const gtsam::Pose3 L_k = e_H_k_world.get() * L0;

    return StateQuery<gtsam::Pose3>(pose_key, L_k);
  } else {
    return StateQuery<gtsam::Pose3>::NotInMap(pose_key);
  }
}

template <typename MAP>
StateQuery<gtsam::Point3> HybridAccessor<MAP>::getDynamicLandmark(
    FrameId frame_id, TrackletId tracklet_id) const {
  StateQuery<gtsam::Point3> query_m_W;
  DynamicLandmarkQuery query;
  query.query_m_W = &query_m_W;

  getDynamicLandmarkImpl(frame_id, tracklet_id, query);

  return query_m_W;
}

template <typename MAP>
StatusLandmarkVector HybridAccessor<MAP>::getDynamicLandmarkEstimates(
    FrameId frame_id, ObjectId object_id) const {
  const auto frame_node = this->map_->getFrame(frame_id);

  // object may not exist at the frame query so allow invalid frame
  if (!frame_node) {
    return StatusLandmarkVector{};
  }

  const auto timestamp = frame_node->timestamp();

  const auto object_node = this->map_->getObject(object_id);
  CHECK(frame_node) << "Frame Null at k=" << frame_id << " j=" << object_id;
  CHECK(object_node) << "Object Null at k=" << frame_id << " j=" << object_id;

  if (!frame_node->objectObserved(object_id)) {
    return StatusLandmarkVector{};
  }

  StatusLandmarkVector estimates;
  // unlike in the base version, iterate over all points on the object (i.e all
  // tracklets) as we can propogate all of them!!!!
  const auto& dynamic_landmarks = object_node->landmarks();
  for (auto lmk_node : dynamic_landmarks) {
    const auto tracklet_id = lmk_node->trackletId();

    CHECK_EQ(object_id, lmk_node->objectId());

    // user defined function should put point in the world frame
    StateQuery<gtsam::Point3> lmk_query =
        this->getDynamicLandmark(frame_id, tracklet_id);
    if (lmk_query) {
      estimates.push_back(LandmarkStatus::DynamicInGlobal(
          Point3Measurement(lmk_query.get()), frame_id, timestamp, tracklet_id,
          object_id));
    }
  }
  return estimates;
}

template <typename MAP>
StatusLandmarkVector HybridAccessor<MAP>::getLocalDynamicLandmarkEstimates(
    ObjectId object_id) const {
  const auto object_node = this->map_->getObject(object_id);
  if (!object_node) {
    return StatusLandmarkVector{};
  }

  // what if we have multiple ranges?
  // pick the ones that have the most number of landmarks...?
  // this is a bad heuristic!!

  // iterate over tracklets and their keyframe to find the frame with the most
  // ids
  auto tracklet_id_to_keyframe =
      *CHECK_NOTNULL(shared_hybrid_formulation_data_.tracklet_id_to_keyframe);
  gtsam::FastMap<FrameId, int> keyframe_count;
  for (const auto& [_, e] : tracklet_id_to_keyframe) {
    if (!keyframe_count.exists(e)) keyframe_count[e] = 0;

    keyframe_count.at(e)++;
  }

  // get max
  int max_count = 0;
  FrameId kf_with_max_tracks;
  for (const auto [kf, count] : keyframe_count) {
    if (count > max_count) {
      max_count = count;
      kf_with_max_tracks = kf;
    }
  }

  VLOG(40) << "Collecting points for j=" << object_id
           << " kf with max tracks KF=" << kf_with_max_tracks
           << " count=" << max_count;

  StatusLandmarkVector estimates;
  if (max_count == 0) {
    return estimates;
  }

  const auto& dynamic_landmarks = object_node->landmarks();
  for (const auto& lmk_node : dynamic_landmarks) {
    const auto tracklet_id = lmk_node->trackletId();

    DynamicLandmarkQuery lmk_query;
    StateQuery<gtsam::Point3> query_m_L;
    lmk_query.query_m_L = &query_m_L;

    // TODO: lots of querties with same frame id
    //  make vector version of this function
    //  so motion/pose does not need to be querier each time
    if (getDynamicLandmarkImpl(kf_with_max_tracks, tracklet_id, lmk_query)) {
      CHECK(query_m_L);
      estimates.push_back(LandmarkStatus::DynamicInLocal(
          Point3Measurement(query_m_L.get()), LandmarkStatus::MeaninglessFrame,
          NaN, tracklet_id, object_id));
    }
  }

  return estimates;
}

template <typename MAP>
TrackletIds HybridAccessor<MAP>::collectPointsAtKeyFrame(
    ObjectId object_id, FrameId frame_id, FrameId* keyframe_id) const {
  if (!hasObjectKeyFrame(object_id, frame_id)) {
    return {};
  }

  TrackletIds tracklets;
  const auto& all_dynamic_landmarks =
      *shared_hybrid_formulation_data_.tracklet_id_to_keyframe;
  const auto [keyframe_k, _] = getObjectKeyFrame(object_id, frame_id);
  for (const auto& [tracklet_id, tracklet_keyframe] : all_dynamic_landmarks) {
    if (tracklet_keyframe == keyframe_k) {
      tracklets.push_back(tracklet_id);
    }
  }

  if (keyframe_id) {
    *keyframe_id = keyframe_k;
  }

  return tracklets;
}

template <typename MAP>
bool HybridAccessor<MAP>::getObjectKeyFrameHistory(
    ObjectId object_id, const KeyFrameRanges*& ranges) const {
  // CHECK_NOTNULL(ranges);
  const auto& key_frame_data = shared_hybrid_formulation_data_.key_frame_data;
  if (!key_frame_data->exists(object_id)) {
    return false;
  }

  ranges = &key_frame_data->at(object_id);
  return true;
}

template <typename MAP>
bool HybridAccessor<MAP>::hasObjectKeyFrame(ObjectId object_id,
                                            FrameId frame_id) const {
  const auto& key_frame_data = shared_hybrid_formulation_data_.key_frame_data;
  return static_cast<bool>(key_frame_data->find(object_id, frame_id));
}

template <typename MAP>
std::pair<FrameId, gtsam::Pose3> HybridAccessor<MAP>::getObjectKeyFrame(
    ObjectId object_id, FrameId frame_id) const {
  const auto& key_frame_data = shared_hybrid_formulation_data_.key_frame_data;
  const KeyFrameRange::ConstPtr range =
      key_frame_data->find(object_id, frame_id);
  CHECK_NOTNULL(range);
  return range->dataPair();
}

template <typename MAP>
StateQuery<Motion3ReferenceFrame> HybridAccessor<MAP>::getEstimatedMotion(
    ObjectId object_id, FrameId frame_id) const {
  // not in form of accessor but in form of estimation
  const auto frame_node_k = this->map_->getFrame(frame_id);
  CHECK_NOTNULL(frame_node_k);

  auto motion_key = frame_node_k->makeObjectMotionKey(object_id);
  StateQuery<gtsam::Pose3> e_H_k_world =
      this->template query<gtsam::Pose3>(motion_key);

  if (!e_H_k_world) {
    return StateQuery<Motion3ReferenceFrame>(e_H_k_world.key(),
                                             e_H_k_world.status());
  }

  CHECK(this->hasObjectKeyFrame(object_id, frame_id));
  // s0
  auto [reference_frame, _] = this->getObjectKeyFrame(object_id, frame_id);

  Motion3ReferenceFrame motion(e_H_k_world.get(), MotionRepresentationStyle::KF,
                               ReferenceFrame::GLOBAL, reference_frame,
                               frame_id);
  return StateQuery<Motion3ReferenceFrame>(e_H_k_world.key(), motion);
}

template <typename MAP>
std::optional<Motion3ReferenceFrame>
HybridAccessor<MAP>::getRelativeLocalMotion(FrameId frame_id,
                                            ObjectId object_id) const {
  const auto from = frame_id - 1u;
  const auto to = frame_id;
  auto L_W_k_1 = this->getObjectPose(from, object_id);
  auto L_W_k = this->getObjectPose(to, object_id);

  if (L_W_k_1 && L_W_k) {
    const gtsam::Pose3 L_k_1_k = L_W_k_1->inverse() * L_W_k.get();
    return Motion3ReferenceFrame(L_k_1_k, MotionRepresentationStyle::F2F,
                                 ReferenceFrame::OBJECT, from, to);
  } else {
    return {};
  }
}

template <typename MAP>
StateQueryStatus HybridAccessor<MAP>::getObjectMotionReferenceFrameHelper(
    FrameId frame_id, ObjectId object_id, gtsam::Key& motion_key,
    gtsam::Pose3& motion, FrameId& from, FrameId& to) const {
  const auto object_node = this->map_->getObject(object_id);
  const auto frame_node_k = this->map_->getFrame(frame_id);
  CHECK(object_node);

  if (!frame_node_k) {
    VLOG(30) << "Could not construct object motion frame id=" << frame_id
             << " object id=" << object_id << " as the frame does not exist!";
    return StateQueryStatus::INVALID_MAP;
  }

  motion_key = frame_node_k->makeObjectMotionKey(object_id);
  StateQuery<gtsam::Pose3> H_W_KF_k =
      this->template query<gtsam::Pose3>(motion_key);
  if (!H_W_KF_k) {
    VLOG(30) << "Could not construct object motion frame id=" << frame_id
             << " object id=" << object_id
             << ". Frame exists but motion is missing!!!";
    return StateQueryStatus::INVALID_MAP;
  }

  auto key_frame_data =
      CHECK_NOTNULL(shared_hybrid_formulation_data_.key_frame_data);

  FrameId last_seen;
  // check if and which frame the object was observed before frame id
  if (!object_node->previouslySeenFrame(frame_id, &last_seen)) {
    const auto range = CHECK_NOTNULL(key_frame_data->find(object_id, frame_id));
    const auto [kf_id, L0] = range->dataPair();
    // check that the first frame of the object motion is actually this frame
    // this motion should actually be identity
    CHECK_EQ(kf_id, frame_id);
    motion = H_W_KF_k.get();
    from = kf_id;
    to = frame_id;
    return StateQueryStatus::VALID;
  } else {
    CHECK_NOTNULL(frame_node_k);
    const auto frame_node_km1 = this->map_->getFrame(last_seen);
    CHECK_NOTNULL(frame_node_km1);

    StateQuery<gtsam::Pose3> H_W_KF_km1 = this->template query<gtsam::Pose3>(
        frame_node_km1->makeObjectMotionKey(object_id));

    if (H_W_KF_k && H_W_KF_km1) {
      // want a motion from k-1 to k, but we estimate s0 to k
      //^w_{k-1}H_k = ^w_{s0}H_k \: ^w_{s0}H_{k-1}^{-1}
      gtsam::Pose3 H_W_km1_k = H_W_KF_k.get() * H_W_KF_km1->inverse();
      motion = H_W_km1_k;
      from = frame_node_km1->frameId();
      to = frame_id;
      return StateQueryStatus::VALID;
    } else {
      return StateQueryStatus::NOT_IN_MAP;
    }
  }
  LOG(WARNING) << "Could not construct object motion frame id=" << frame_id
               << " object id=" << object_id;
  return StateQueryStatus::INVALID_MAP;
}

template <typename MAP>
StateQuery<Motion3ReferenceFrame>
HybridAccessor<MAP>::getObjectMotionReferenceFrame(FrameId frame_id,
                                                   ObjectId object_id) const {
  using Query = StateQuery<Motion3ReferenceFrame>;

  gtsam::Key motion_key;
  gtsam::Pose3 H_W_km1_k;
  FrameId from, to;
  const StateQueryStatus status = getObjectMotionReferenceFrameHelper(
      frame_id, object_id, motion_key, H_W_km1_k, from, to);
  if (status == StateQueryStatus::VALID) {
    // in base accessor expect motion to be a genuine F2F motion
    return Query(motion_key, Motion3ReferenceFrame(
                                 H_W_km1_k, Motion3ReferenceFrame::Style::F2F,
                                 ReferenceFrame::GLOBAL, from, to));
  } else {
    return Query(motion_key, status);
  }
}

template <typename MAP>
StateQuery<gtsam::Point3> HybridAccessor<MAP>::queryPoint(gtsam::Key point_key,
                                                          TrackletId) const {
  return this->template query<gtsam::Point3>(point_key);
}

template <typename MAP>
bool HybridAccessor<MAP>::getDynamicLandmarkImpl(
    FrameId frame_id, TrackletId tracklet_id,
    DynamicLandmarkQuery& query) const {
  auto tracklet_id_to_keyframe =
      CHECK_NOTNULL(shared_hybrid_formulation_data_.tracklet_id_to_keyframe);
  auto key_frame_data =
      CHECK_NOTNULL(shared_hybrid_formulation_data_.key_frame_data);

  if (!tracklet_id_to_keyframe->exists(tracklet_id)) {
    return false;
  }

  const auto lmk_node = this->map_->getLandmark(tracklet_id);
  const auto frame_node_k = this->map_->getFrame(frame_id);
  CHECK(frame_node_k);
  CHECK_NOTNULL(lmk_node);

  const auto object_id = lmk_node->objectId();

  // point in L_{e}
  gtsam::Key point_key = this->makeDynamicKey(tracklet_id);

  if (!tracklet_id_to_keyframe->exists(tracklet_id)) {
    return false;
  }

  // embedded frame (k) the point is represented in
  FrameId point_embedded_frame = tracklet_id_to_keyframe->at(tracklet_id);
  const auto range = key_frame_data->find(object_id, frame_id);

  // TODO: check the &= is right!
  //  we might mean result = result || (condition)
  bool result = true;

  // update intermediate queries
  if (query.frame_range_ptr) {
    *query.frame_range_ptr = range;
    result &= (bool)range;
  }

  // On a frame where the object has no motion (possibly between keyframes)
  // there will be no valid range!!
  if (!range) {
    return false;
  }
  // if the active keyframe is not the same as the reference frame the point is
  // represented in we (currentlly) have no way of propogating the point to the
  // query frame
  if (range->start != point_embedded_frame) {
    return false;
  }

  // point in local frame
  // StateQuery<gtsam::Point3> m_Le = this->query<gtsam::Point3>(point_key);
  StateQuery<gtsam::Point3> m_Le = this->queryPoint(point_key, tracklet_id);
  // get motion from S0 to k
  StateQuery<gtsam::Pose3> e_H_k_world = this->template query<gtsam::Pose3>(
      frame_node_k->makeObjectMotionKey(object_id));

  // update intermediate queries
  if (query.query_m_L) {
    *query.query_m_L = m_Le;
    result &= (bool)m_Le;
  }
  if (query.query_H_W_e_k) {
    *query.query_H_W_e_k = e_H_k_world;
    result &= (bool)e_H_k_world;
  }

  if (m_Le && e_H_k_world) {
    const auto [s0, L0] = range->dataPair();
    // since the motion has a range (and therefore may not be valid!!!)
    //  point in world at k
    const gtsam::Point3 m_W_k = e_H_k_world.get() * L0 * m_Le.get();
    StateQuery<gtsam::Point3> point_world(point_key, m_W_k);

    if (query.query_m_W) {
      *query.query_m_W = point_world;
      result &= (bool)point_world;
    }

    // TODO: result not actually used!!
    return true;

  } else {
    if (query.query_m_W) {
      *query.query_m_W = StateQuery<gtsam::Point3>::NotInMap(point_key);
    }
    return false;
  }
}

template <typename MAP>
bool HybridAccessor<MAP>::getDynamicLandmarkImpl(
    FrameId frame_id, TrackletId tracklet_id,
    StateQuery<gtsam::Point3>* query_m_W, StateQuery<gtsam::Point3>* query_m_L,
    StateQuery<gtsam::Pose3>* query_H_W_e_k,
    KeyFrameRange::ConstPtr* frame_range_ptr) const {
  DynamicLandmarkQuery query;
  query.query_m_W = query_m_W;
  query.query_m_L = query_m_L;
  query.query_H_W_e_k = query_H_W_e_k;
  query.frame_range_ptr = frame_range_ptr;

  return getDynamicLandmarkImpl(frame_id, tracklet_id, query);
}

}  // namespace dyno
