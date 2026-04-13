#include "dynosam/formulations/RegularHybridEstimator.hpp"

#include <pcl/common/centroid.h>

namespace dyno {

// this needs to happen (mostly) before factor graph construction to take
// effect!!
std::pair<FrameId, gtsam::Pose3> HybridFormulationV1::forceNewKeyFrame(
    FrameId frame_id, ObjectId object_id) {
  LOG(INFO) << "Starting new range of object k=" << frame_id
            << " j=" << object_id;
  gtsam::Pose3 center = calculateObjectCentroid(object_id, frame_id);

  auto result =
      key_frame_data_.startNewActiveRange(object_id, frame_id, center)
          ->dataPair();

  // clear meta-data to start new tracklets
  // TODO: somehow adding this back in causes a segfault when ISAM2::update step
  // happens... in combinating with clearing the internal graph
  // (this->clearGraph) and resetting the smoother in ParlallelObjectSAM. I
  // think we should do this!!
  // HACK - these vairblaes will still be in the values and therefore we will
  // get some kind of 'gtsam::ValuesKeyAlreadyExists' when updating the
  // formulation we therefore need to remove these from the theta - this will
  // remove old data
  // from the accessor (even though we keep track of the meta-data)
  // when the frontend is updated to include keyframe information this should
  // not be an issue as the frontend will ensure new measurements dont refer to
  // landmarks in old keypoints
  // for (const auto& [tracklet_id, _] : is_dynamic_tracklet_in_map_) {
  //     ObjectId lmk_object_id;
  //     CHECK(map()->getLandmarkObjectId(lmk_object_id, tracklet_id));
  //     //only delete for requested object
  //     if(lmk_object_id == object_id) {
  //       theta_.erase(this->makeDynamicKey(tracklet_id));
  //     }
  // }
  // is_dynamic_tracklet_in_map_.clear();

  // sanity check
  CHECK_EQ(result.first, frame_id);
  return result;
}

HybridFormulation::IntermediateMotionInfo
HybridFormulationV1::getIntermediateMotionInfo(ObjectId object_id,
                                               FrameId frame_id) {
  IntermediateMotionInfo info;
  bool keyframe_updated;
  info.H_W_e_k_initial =
      computeInitialH(object_id, frame_id, &keyframe_updated);
  (void)keyframe_updated;

  std::tie(info.kf_id, info.keyframe_pose) =
      getOrConstructL0(object_id, frame_id);
  return info;
}

std::pair<FrameId, gtsam::Pose3> HybridFormulationV1::getOrConstructL0(
    ObjectId object_id, FrameId frame_id) {
  const KeyFrameRange::ConstPtr range =
      key_frame_data_.find(object_id, frame_id);
  if (range) {
    return range->dataPair();
  }

  return forceNewKeyFrame(frame_id, object_id);
}

// TODO: can be massively more efficient
// should also check if the last object motion from the estimation can be used
// as the last motion
//  so only one composition is needed to get the latest motion
gtsam::Pose3 HybridFormulationV1::computeInitialH(ObjectId object_id,
                                                  FrameId frame_id,
                                                  bool* keyframe_updated) {
  // TODO: could this ever update the keyframe?
  auto [s0, L_e] = getOrConstructL0(object_id, frame_id);

  // LOG(INFO) << "computeInitialH " << info_string(frame_id, object_id);

  if (keyframe_updated) *keyframe_updated = false;

  FrameId current_frame_id = frame_id;
  CHECK_LE(s0, current_frame_id);
  if (current_frame_id == s0) {
    // same frame so motion between them should be identity!
    // except for rotation?
    return gtsam::Pose3::Identity();
  }

  bool has_initial = false;

  // check if we have an estimate from the previous frame
  const FrameId frame_id_km1 = frame_id - 1u;

  // only need an initial motion when k > s0
  Motion3ReferenceFrame initial_motion_frame;
  const bool has_frontend_motion = map()->getInitialObjectMotion(
      current_frame_id, object_id, initial_motion_frame);

  if (!has_frontend_motion) {
    // no motion estimation that takes us to this frame
    //  1. Check how far away the last motion we have is
    const auto object_node = CHECK_NOTNULL(map()->getObject(object_id));
    // assume continuous
    const auto seen_frame_ids_vec = object_node->getSeenFrameIds();
    std::set<FrameId> seen_frame_ids(seen_frame_ids_vec.begin(),
                                     seen_frame_ids_vec.end());
    // get smallest before current frame
    auto it = seen_frame_ids.lower_bound(current_frame_id);
    if (it != seen_frame_ids.begin() &&
        (it == seen_frame_ids.end() || *it >= current_frame_id)) {
      --it;

    } else {
      LOG(FATAL)
          << "Bookkeeping failure!! Cound not find a frame id for object "
          << object_id << " < " << current_frame_id
          << " but this frame is not s0!";
    }
    FrameId previous_frame = *it;
    // must actually be smaller than query frame
    CHECK_LT(previous_frame, current_frame_id);
    // should not be s0 becuase we have a condition for this!
    CHECK_GT(previous_frame, s0);
    // 2. If within threshold apply constant motion model to get us to current
    // frame and use that as initalisation (?)
    FrameId diff = current_frame_id - previous_frame;

    // TODO:hack!! This really depends on framerate etc...!!! just for now!!!!
    if (diff > 2) {
      LOG(WARNING) << "Motion intalisation failed for j= " << object_id
                   << ", motion missing at " << current_frame_id
                   << " and previous seen frame " << previous_frame
                   << " too far away!";
      std::tie(s0, L_e) = forceNewKeyFrame(frame_id, object_id);
      // start new key frame
      // gtsam::Pose3 center = calculateObjectCentroid(object_id, frame_id);
      // key_frame_data_.startNewActiveRange(object_id, frame_id, center);

      // // sanity check
      // std::tie(s0, L_e) = getOrConstructL0(object_id, frame_id);
      // LOG(INFO) << "Creating new KF for j=" << object_id << " k=" <<
      // frame_id; CHECK_EQ(s0, frame_id);
      // // TODO: need to tell other systems that the
      if (keyframe_updated) *keyframe_updated = true;

      return gtsam::Pose3::Identity();

    } else {
      // TODO: just use previous motion???
      CHECK(map()->getInitialObjectMotion(previous_frame, object_id,
                                          initial_motion_frame));
      // update current_frame_id to previous frame so that the composition loop
      // below stops at the right place!
      // TODO: will this mess up the frame_id - 1 check?
      // LOG(INFO) << "Updating current frame id to previous frame "
      //           << previous_frame << " to account for missing frame at "
      //           << current_frame_id;
      current_frame_id = previous_frame;
    }
  }

  // LOG(INFO) << "Gotten initial motion " << initial_motion_frame;

  // << "Missing initial motion at k= " << frame_id << " j= " << object_id;
  CHECK_EQ(initial_motion_frame.to(), current_frame_id);
  CHECK_EQ(initial_motion_frame.frame(), ReferenceFrame::GLOBAL);

  if (current_frame_id - 1 == s0) {
    // a motion that takes us from k-1 to k where k-1 == s0
    return initial_motion_frame;
  } else {
    // check representation
    if (initial_motion_frame.style() == MotionRepresentationStyle::KF) {
      // this motion should be from s0 to k and is already in the right
      // representation!!
      CHECK_EQ(initial_motion_frame.from(), s0);
      return initial_motion_frame;
    } else if (initial_motion_frame.style() == MotionRepresentationStyle::F2F) {
      HybridAccessor::Ptr accessor = this->derivedAccessor<HybridAccessor>();
      // we have a motion from the frontend that is k-1 to k
      // first check if we have a previous estimation motion that takes us from
      // s0 to k-1 in the map
      StateQuery<Motion3ReferenceFrame> H_W_s0_km1 =
          accessor->getEstimatedMotion(object_id, frame_id_km1);
      if (H_W_s0_km1) {
        CHECK_EQ(H_W_s0_km1->from(), s0);
        CHECK_EQ(H_W_s0_km1->to(), frame_id_km1);

        Motion3 H_W_km1_k = initial_motion_frame;
        Motion3 e_H_k_world = H_W_km1_k * H_W_s0_km1.get();
        return e_H_k_world;
      }
      // if we cant do this, try compouding all initial motions from s0 to k

      // compose frame-to-frame motion to construct the keyframe motion
      Motion3 composed_motion;
      Motion3 initial_motion = initial_motion_frame;

      // query from so+1 to k since we index backwards
      bool initalised_from_frontend = true;
      for (auto frame = s0 + 1; frame <= current_frame_id; frame++) {
        // LOG(INFO) << "frontend motion at frame " << frame << " object id "<<
        // object_id;
        Motion3ReferenceFrame motion_frame;  // if fail just use identity?
        if (!map()->getInitialObjectMotion(frame, object_id, motion_frame)) {
          // LOG(WARNING) << "No frontend motion at frame " << frame
          //              << " object id " << object_id;
          CHECK_EQ(motion_frame.style(), MotionRepresentationStyle::F2F)
              << "Motion representation is inconsistent!! ";
          initalised_from_frontend = false;
          break;
        }
        Motion3 motion = motion_frame;
        composed_motion = motion * composed_motion;
      }

      // if(initalised_from_frontend) {
      // after loop motion should be ^w_{s0}H_k
      return composed_motion;
      // }
      // else {
      //   // L0_.erase(object_id);

      // }
    } else {
      DYNO_THROW_MSG(DynosamException) << "Unknown MotionRepresentationStyle";
    }
  }
}

gtsam::Pose3 HybridFormulationV1::calculateObjectCentroid(
    ObjectId object_id, FrameId frame_id) const {
  if (FLAGS_init_object_pose_from_gt) {
    const auto gt_packets = hooks().ground_truth_packets_request();
    if (gt_packets && gt_packets->exists(frame_id)) {
      const auto& gt_packet = gt_packets->at(frame_id);

      ObjectPoseGT object_gt;
      if (gt_packet.getObject(object_id, object_gt)) {
        // return gtsam::Pose3(gtsam::Rot3::Identity(),
        // object_gt.L_world_.translation());
        return object_gt.L_world_;
        // L0_.insert2(object_id, std::make_pair(frame_id, object_gt.L_world_));
        // return L0_.at(object_id);
      } else {
        LOG(FATAL) << "COuld not get gt! object centroid";
      }
    } else {
      LOG(FATAL) << "COuld not get gt! object centroid";
    }
  }

  // else initalise from centroid?
  auto object_node = map()->getObject(object_id);
  CHECK(object_node);

  auto frame_node = map()->getFrame(frame_id);
  CHECK(frame_node);
  CHECK(frame_node->objectObserved(object_id));

  StatusLandmarkVector dynamic_landmarks;

  const auto timestamp = frame_node->timestamp();

  // TODO: could use computeObjectCentroid in accessor!!!?

  // measured/linearized camera pose at the first frame this object has been
  // seen
  const gtsam::Pose3 X_world = getInitialOrLinearizedSensorPose(frame_id);
  auto measurement_pairs = frame_node->dynamicMeasurements(object_id);

  for (const auto& [lmk_node, measurement] : measurement_pairs) {
    CHECK(lmk_node->seenAtFrame(frame_id));
    CHECK_EQ(lmk_node->objectId(), object_id);

    const gtsam::Point3 landmark_measurement_local =
        MeasurementTraits::point(measurement);
    // const gtsam::Point3 landmark_measurement_world = X_world *
    // landmark_measurement_local;

    dynamic_landmarks.push_back(LandmarkStatus::DynamicInGlobal(
        Point3Measurement(landmark_measurement_local), frame_id, timestamp,
        lmk_node->trackletId(), object_id));
  }

  CloudPerObject object_clouds = groupObjectCloud(dynamic_landmarks, X_world);
  CHECK_EQ(object_clouds.size(), 1u);

  CHECK(object_clouds.exists(object_id));

  const auto dynamic_point_cloud = object_clouds.at(object_id);
  pcl::PointXYZ centroid;
  pcl::computeCentroid(dynamic_point_cloud, centroid);
  // TODO: outlier reject?
  gtsam::Point3 translation = pclPointToGtsam(centroid);
  gtsam::Pose3 center(gtsam::Rot3::Identity(), X_world * translation);
  return center;
}

ErrorHandlingHooks HybridFormulationV1::getCustomErrorHooks() {
  // base error hooks set "handle ILS exceptions"
  auto handle_failed_object =
      [&](const std::pair<FrameId, ObjectId>& failed_on_object) {
        const auto [frame_id, object_id] = failed_on_object;
        LOG(INFO) << "Is hybrid formulation with failed estimation at "
                  << info_string(frame_id, object_id);
        this->forceNewKeyFrame(frame_id, object_id);
      };

  // make error handling hooks with default ILS and custom on failed object hook
  return getDefaultILSErrorHandlingHooks(handle_failed_object);
}

void RegularHybridFormulation::preUpdate(const PreUpdateData& data) {
  // get objects seen in this frame from the map
  const typename Map::Ptr map = this->map();
  CHECK(map) << "Now can map be null!?";
  const auto frame_id_k = data.frame_id;
  const auto frame_node = map->getFrame(frame_id_k);
  CHECK(frame_node) << "Frame node null at k=" << data.frame_id;

  for (const auto& obj_node : frame_node->objectsSeen()) {
    ObjectId obj_id = obj_node->objectId();
    // we have seen this object before
    if (objects_update_data_.exists(obj_id)) {
      const ObjectUpdateData& update_data = objects_update_data_.at(obj_id);

      // first appeared in this frame
      bool is_object_new = obj_node->getFirstSeenFrame() == frame_id_k;
      FrameId last_update_frame = update_data.frame_id;

      // duplicate logic from ParallelBackend!
      if (!is_object_new && (frame_id_k > 0) &&
          (last_update_frame < (frame_id_k - 1u))) {
        VLOG(5)
            << "Only update k=" << frame_id_k << " j= " << obj_id
            << " as object is not new but has reappeared. Previous update was "
            << last_update_frame << ". Making keyframe";

        this->forceNewKeyFrame(frame_id_k, obj_id);
      }
    }
  }
}

// TODO: we should actually get this information from the frontend (ie object
// keyframe!!!) the logic is then build into the formulation itself via the same
// calls (post/pre) but then the parallel has no need for additional logic and
// we dont have to have independant logic to decide if a new keyframe should be
// formed!!
void RegularHybridFormulation::postUpdate(const PostUpdateData& data) {
  const auto& affected_objects =
      data.dynamic_update_result.objects_affected_per_frame;

  // Jesse: I guess seen frames could be size > 1 but it SHOULD only matter if
  // its seen in the current frame
  for (const auto& [object_id, seen_frames] : affected_objects) {
    LOG(INFO) << "PostUpdate: adffected object " << object_id
              << "k = " << data.frame_id;
    CHECK(seen_frames.find(data.frame_id) != seen_frames.end());

    if (!objects_update_data_.exists(object_id)) {
      ObjectUpdateData oud;
      oud.frame_id = data.frame_id;
      oud.count = 1;
      objects_update_data_.insert2(object_id, oud);
    } else {
      auto& oud = objects_update_data_.at(object_id);
      oud.frame_id = data.frame_id;
      oud.count++;
    }
  }
}

}  // namespace dyno
