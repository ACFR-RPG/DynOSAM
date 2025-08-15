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

#include "dynosam/backend/rgbd/HybridEstimator.hpp"

#include <gtsam/slam/PoseRotationPrior.h>

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/factors/HybridFormulationFactors.hpp"

namespace dyno {

// class SmartHFactor
//     : public gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3,
//     gtsam::Pose3,
//                                       gtsam::Pose3> {
//  public:
//   typedef boost::shared_ptr<SmartHFactor> shared_ptr;
//   typedef SmartHFactor This;
//   typedef gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3,
//                                    gtsam::Pose3>
//       Base;

//   const gtsam::Point3 Z_previous_;
//   const gtsam::Point3 Z_current_;

//   SmartHFactor(gtsam::Key X_previous, gtsam::Key H_previous,
//                gtsam::Key X_current, gtsam::Key H_current,
//                const gtsam::Point3& Z_previous, const gtsam::Point3&
//                Z_current, gtsam::SharedNoiseModel model)
//       : Base(model, X_previous, H_previous, X_current, H_current),
//         Z_previous_(Z_previous),
//         Z_current_(Z_current) {}

//   gtsam::Vector evaluateError(
//       const gtsam::Pose3& X_previous, const gtsam::Pose3& H_previous,
//       const gtsam::Pose3& X_current, const gtsam::Pose3& H_current,
//       boost::optional<gtsam::Matrix&> J1 = boost::none,
//       boost::optional<gtsam::Matrix&> J2 = boost::none,
//       boost::optional<gtsam::Matrix&> J3 = boost::none,
//       boost::optional<gtsam::Matrix&> J4 = boost::none) const override {
//     if (J1) {
//       // error w.r.t to X_prev
//       Eigen::Matrix<double, 3, 6> df_dX_prev =
//           gtsam::numericalDerivative41<gtsam::Vector3, gtsam::Pose3,
//                                        gtsam::Pose3, gtsam::Pose3,
//                                        gtsam::Pose3>(
//               std::bind(&SmartHFactor::residual, std::placeholders::_1,
//                         std::placeholders::_2, std::placeholders::_3,
//                         std::placeholders::_4, Z_previous_, Z_current_),
//               X_previous, H_previous, X_current, H_current);
//       *J1 = df_dX_prev;
//     }

//     if (J2) {
//       // error w.r.t to P_prev
//       Eigen::Matrix<double, 3, 6> df_dP_prev =
//           gtsam::numericalDerivative42<gtsam::Vector3, gtsam::Pose3,
//                                        gtsam::Pose3, gtsam::Pose3,
//                                        gtsam::Pose3>(
//               std::bind(&SmartHFactor::residual, std::placeholders::_1,
//                         std::placeholders::_2, std::placeholders::_3,
//                         std::placeholders::_4, Z_previous_, Z_current_),
//               X_previous, H_previous, X_current, H_current);
//       *J2 = df_dP_prev;
//     }

//     if (J3) {
//       // error w.r.t to X_curr
//       Eigen::Matrix<double, 3, 6> df_dX_curr =
//           gtsam::numericalDerivative43<gtsam::Vector3, gtsam::Pose3,
//                                        gtsam::Pose3, gtsam::Pose3,
//                                        gtsam::Pose3>(
//               std::bind(&SmartHFactor::residual, std::placeholders::_1,
//                         std::placeholders::_2, std::placeholders::_3,
//                         std::placeholders::_4, Z_previous_, Z_current_),
//               X_previous, H_previous, X_current, H_current);
//       *J3 = df_dX_curr;
//     }

//     if (J4) {
//       // error w.r.t to P_curr
//       Eigen::Matrix<double, 3, 6> df_dP_curr =
//           gtsam::numericalDerivative44<gtsam::Vector3, gtsam::Pose3,
//                                        gtsam::Pose3, gtsam::Pose3,
//                                        gtsam::Pose3>(
//               std::bind(&SmartHFactor::residual, std::placeholders::_1,
//                         std::placeholders::_2, std::placeholders::_3,
//                         std::placeholders::_4, Z_previous_, Z_current_),
//               X_previous, H_previous, X_current, H_current);
//       *J4 = df_dP_curr;
//     }

//     return residual(X_previous, H_previous, X_current, H_current,
//     Z_previous_,
//                     Z_current_);
//   }

//   static gtsam::Vector residual(const gtsam::Pose3& X_previous,
//                                 const gtsam::Pose3& H_previous,
//                                 const gtsam::Pose3& X_current,
//                                 const gtsam::Pose3& H_current,
//                                 const gtsam::Point3& Z_previous,
//                                 const gtsam::Point3& Z_current) {
//     gtsam::Pose3 prev_H_current = H_current * H_previous.inverse();
//     gtsam::Point3 m_previous_world = X_previous * Z_previous;
//     gtsam::Point3 m_current_world = X_current * Z_current;
//     return m_current_world - prev_H_current * m_previous_world;
//   }
// };

StateQuery<gtsam::Pose3> HybridAccessor::getSensorPose(FrameId frame_id) const {
  const auto frame_node = map()->getFrame(frame_id);
  if (!frame_node) {
    return StateQuery<gtsam::Pose3>::InvalidMap();
  }
  // CHECK_NOTNULL(frame_node);
  return this->query<gtsam::Pose3>(frame_node->makePoseKey());
}

StateQuery<gtsam::Pose3> HybridAccessor::getObjectMotion(
    FrameId frame_id, ObjectId object_id) const {
  const auto frame_node_k = map()->getFrame(frame_id);
  const auto frame_node_k_1 = map()->getFrame(frame_id - 1u);

  if (!frame_node_k) {
    VLOG(30) << "Could not construct object motion frame id=" << frame_id
             << " object id=" << object_id << " as the frame does not exist!";
    return StateQuery<gtsam::Pose3>::InvalidMap();
  }

  auto motion_key = frame_node_k->makeObjectMotionKey(object_id);
  StateQuery<gtsam::Pose3> e_H_k_world = this->query<gtsam::Pose3>(motion_key);
  if (!e_H_k_world) {
    VLOG(30) << "Could not construct object motion frame id=" << frame_id
             << " object id=" << object_id
             << ". Frame exists but motion is missing!!!";
    return StateQuery<gtsam::Pose3>::InvalidMap();
  }

  // first object motion (ie s0 -> s1)
  auto key_frame_data =
      CHECK_NOTNULL(shared_hybrid_formulation_data_.key_frame_data);
  if (!frame_node_k_1) {
    CHECK_NOTNULL(frame_node_k);
    // FrameId s0 = L0_values_->at(object_id).first;
    const auto range = CHECK_NOTNULL(key_frame_data->find(object_id, frame_id));
    const auto [s0, L0] = range->dataPair();
    // check that the first frame of the object motion is actually this frame
    // this motion should actually be identity
    CHECK_EQ(s0, frame_id);
    return StateQuery<gtsam::Pose3>(motion_key, *e_H_k_world);
  } else {
    CHECK_NOTNULL(frame_node_k);
    CHECK_NOTNULL(frame_node_k_1);

    StateQuery<gtsam::Pose3> e_H_km1_world = this->query<gtsam::Pose3>(
        frame_node_k_1->makeObjectMotionKey(object_id));

    if (e_H_k_world && e_H_km1_world) {
      // want a motion from k-1 to k, but we estimate s0 to k
      //^w_{k-1}H_k = ^w_{s0}H_k \: ^w_{s0}H_{k-1}^{-1}
      gtsam::Pose3 motion = e_H_k_world.get() * e_H_km1_world->inverse();
      // LOG(INFO) << "Obj motion " << motion;
      return StateQuery<gtsam::Pose3>(motion_key, motion);
    } else {
      return StateQuery<gtsam::Pose3>::NotInMap(
          frame_node_k->makeObjectMotionKey(object_id));
    }
  }
  LOG(WARNING) << "Could not construct object motion frame id=" << frame_id
               << " object id=" << object_id;
  return StateQuery<gtsam::Pose3>::InvalidMap();
}

StateQuery<gtsam::Pose3> HybridAccessor::getObjectPose(
    FrameId frame_id, ObjectId object_id) const {
  // we estimate a motion ^w_{s0}H_k, so we can compute a pose ^wL_k =
  // ^w_{s0}H_k * ^wL_{s0}
  const auto frame_node_k = map()->getFrame(frame_id);
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
  StateQuery<gtsam::Pose3> e_H_k_world = this->query<gtsam::Pose3>(motion_key);
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
StateQuery<gtsam::Point3> HybridAccessor::getDynamicLandmark(
    FrameId frame_id, TrackletId tracklet_id) const {
  StateQuery<gtsam::Point3> query_m_W;
  DynamicLandmarkQuery query;
  query.query_m_W = &query_m_W;

  getDynamicLandmarkImpl(frame_id, tracklet_id, query);

  return query_m_W;
}

StatusLandmarkVector HybridAccessor::getDynamicLandmarkEstimates(
    FrameId frame_id, ObjectId object_id) const {
  const auto frame_node = map()->getFrame(frame_id);

  // object may not exist at the frame query so allow invalid frame
  if (!frame_node) {
    return StatusLandmarkVector{};
  }

  const auto object_node = map()->getObject(object_id);
  CHECK(frame_node) << "Frame Null at k=" << frame_id << " j=" << object_id;
  CHECK(object_node) << "Object Null at k=" << frame_id << " j=" << object_id;

  if (!frame_node->objectObserved(object_id)) {
    return StatusLandmarkVector{};
  }

  StatusLandmarkVector estimates;
  // unlike in the base version, iterate over all points on the object (i.e all
  // tracklets) as we can propogate all of them!!!!
  const auto& dynamic_landmarks = object_node->dynamic_landmarks;
  for (auto lmk_node : dynamic_landmarks) {
    const auto tracklet_id = lmk_node->tracklet_id;

    CHECK_EQ(object_id, lmk_node->object_id);

    // user defined function should put point in the world frame
    StateQuery<gtsam::Point3> lmk_query =
        this->getDynamicLandmark(frame_id, tracklet_id);
    if (lmk_query) {
      estimates.push_back(LandmarkStatus::DynamicInGLobal(
          Point3Measurement(lmk_query.get()),  // estimate
          frame_id, tracklet_id, object_id));
    }
  }
  return estimates;
}

StatusLandmarkVector HybridAccessor::getLocalDynamicLandmarkEstimates(
    ObjectId object_id) const {
  const auto object_node = map()->getObject(object_id);
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

  StatusLandmarkVector estimates;
  if (max_count == 0) {
    return estimates;
  }

  const auto& dynamic_landmarks = object_node->dynamic_landmarks;
  for (auto lmk_node : dynamic_landmarks) {
    const auto tracklet_id = lmk_node->tracklet_id;

    DynamicLandmarkQuery lmk_query;
    StateQuery<gtsam::Point3> query_m_L;
    lmk_query.query_m_L = &query_m_L;

    if (getDynamicLandmarkImpl(kf_with_max_tracks, tracklet_id, lmk_query)) {
      CHECK(query_m_L);
      estimates.push_back(LandmarkStatus::DynamicInLocal(
          Point3Measurement(query_m_L.get()), LandmarkStatus::MeaninglessFrame,
          tracklet_id, object_id));
    }
  }

  return estimates;
}

std::optional<Motion3ReferenceFrame> HybridAccessor::getRelativeLocalMotion(
    FrameId frame_id, ObjectId object_id) const {
  const auto from = frame_id - 1u;
  const auto to = frame_id;
  auto L_W_k_1 = this->getObjectPose(from, object_id);
  auto L_W_k = this->getObjectPose(to, object_id);

  if (L_W_k_1 && L_W_k) {
    const gtsam::Pose3 L_k_1_k = L_W_k_1->inverse() * L_W_k.value();
    return Motion3ReferenceFrame(L_k_1_k, MotionRepresentationStyle::F2F,
                                 ReferenceFrame::OBJECT, from, to);
  } else {
    return {};
  }
}

StateQuery<gtsam::Point3> HybridAccessor::queryPoint(gtsam::Key point_key,
                                                     TrackletId) const {
  return this->query<gtsam::Point3>(point_key);
}

bool HybridAccessor::getDynamicLandmarkImpl(FrameId frame_id,
                                            TrackletId tracklet_id,
                                            DynamicLandmarkQuery& query) const {
  auto tracklet_id_to_keyframe =
      CHECK_NOTNULL(shared_hybrid_formulation_data_.tracklet_id_to_keyframe);
  auto key_frame_data =
      CHECK_NOTNULL(shared_hybrid_formulation_data_.key_frame_data);

  if (!tracklet_id_to_keyframe->exists(tracklet_id)) {
    return false;
  }

  const auto lmk_node = map()->getLandmark(tracklet_id);
  const auto frame_node_k = map()->getFrame(frame_id);
  CHECK(frame_node_k);
  CHECK_NOTNULL(lmk_node);

  const auto object_id = lmk_node->object_id;

  // point in L_{e}
  gtsam::Key point_key = this->makeDynamicKey(tracklet_id);

  if (!tracklet_id_to_keyframe->exists(tracklet_id)) {
    return false;
  }

  // embedded frame (k) the point is represented in
  FrameId point_embedded_frame = tracklet_id_to_keyframe->at(tracklet_id);
  const auto range = key_frame_data->find(object_id, frame_id);

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
  StateQuery<gtsam::Pose3> e_H_k_world =
      this->query<gtsam::Pose3>(frame_node_k->makeObjectMotionKey(object_id));

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

bool HybridAccessor::getDynamicLandmarkImpl(
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

// this needs to happen (mostly) before factor graph construction to take
// effect!!
std::pair<FrameId, gtsam::Pose3> HybridFormulation::forceNewKeyFrame(
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

bool HybridFormulation::getObjectKeyFrameHistory(
    ObjectId object_id, const KeyFrameRanges* ranges) const {
  CHECK_NOTNULL(ranges);
  if (!key_frame_data_.exists(object_id)) {
    return false;
  }

  ranges = &key_frame_data_.at(object_id);
  return true;
}

bool HybridFormulation::hasObjectKeyFrame(ObjectId object_id,
                                          FrameId frame_id) const {
  return static_cast<bool>(key_frame_data_.find(object_id, frame_id));
}
std::pair<FrameId, gtsam::Pose3> HybridFormulation::getObjectKeyFrame(
    ObjectId object_id, FrameId frame_id) const {
  const KeyFrameRange::ConstPtr range =
      key_frame_data_.find(object_id, frame_id);
  CHECK_NOTNULL(range);
  return range->dataPair();
}

StateQuery<Motion3ReferenceFrame> HybridFormulation::getEstimatedMotion(
    ObjectId object_id, FrameId frame_id) const {
  // not in form of accessor but in form of estimation
  const auto frame_node_k = map()->getFrame(frame_id);
  CHECK_NOTNULL(frame_node_k);

  auto motion_key = frame_node_k->makeObjectMotionKey(object_id);
  // raw access the theta in the accessor!!
  auto theta_accessor = this->accessorFromTheta();
  StateQuery<gtsam::Pose3> e_H_k_world =
      theta_accessor->query<gtsam::Pose3>(motion_key);

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

TrackletIds HybridFormulation::collectPointsAtKeyFrame(ObjectId object_id,
                                                       FrameId frame_id) const {
  if (!hasObjectKeyFrame(object_id, frame_id)) {
    return {};
  }

  TrackletIds tracklets;
  const auto [keyframe_k, _] = getObjectKeyFrame(object_id, frame_id);
  for (const auto& [tracklet_id, tracklet_keyframe] : all_dynamic_landmarks_) {
    if (tracklet_keyframe == keyframe_k) {
      tracklets.push_back(tracklet_id);
    }
  }
  return tracklets;
}

void HybridFormulation::dynamicPointUpdateCallback(
    const PointUpdateContextType& context, UpdateObservationResult& result,
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors) {
  const auto lmk_node = context.lmk_node;
  const auto frame_node_k_1 = context.frame_node_k_1;
  const auto frame_node_k = context.frame_node_k;

  auto theta_accessor = this->accessorFromTheta();

  gtsam::Key point_key = this->makeDynamicKey(context.getTrackletId());

  const gtsam::Key object_motion_key_k =
      frame_node_k->makeObjectMotionKey(context.getObjectId());
  const gtsam::Key object_motion_key_k_1 =
      frame_node_k_1->makeObjectMotionKey(context.getObjectId());

  gtsam::Pose3 L_e;
  FrameId s0;
  std::tie(s0, L_e) =
      getOrConstructL0(context.getObjectId(), frame_node_k_1->getId());
  auto landmark_motion_noise = noise_models_.landmark_motion_noise;
  // check that the first frame id is at least the initial frame for s0

  // TODO:this will not be the case with sliding/window as we reconstruct the
  // graph from a different starting point!!
  //  CHECK_GE(frame_node_k_1->getId(), s0);

  if (!isDynamicTrackletInMap(lmk_node)) {
    // TODO: this will not hold in the batch case as the first dynamic point we
    // get will not be the first point on the object (we will get the first
    // point seen within the window) so, where should be initalise the object
    // pose!?
    //  //this is a totally new tracklet so should be the first time we've seen
    //  it! CHECK_EQ(lmk_node->getFirstSeenFrame(), frame_node_k_1->getId());

    // use first point as initalisation?
    // in this case k is k-1 as we use frame_node_k_1
    bool keyframe_updated;
    gtsam::Pose3 e_H_k_world = computeInitialH(
        context.getObjectId(), frame_node_k_1->getId(), &keyframe_updated);

    // TODO: we should never actually let this happen during an update
    //  it should only happen before measurements are added
    // want to avoid somehow a situation where some (landmark)variables are at
    // an old keyframe I dont think this will happen with the current
    // implementation...
    if (keyframe_updated) {
      // TODO: gross I have to re-get them again!!
      std::tie(s0, L_e) =
          getOrConstructL0(context.getObjectId(), frame_node_k_1->getId());
    }

    // mark as now in map and include associated frame!!s
    is_dynamic_tracklet_in_map_.insert2(context.getTrackletId(), s0);
    all_dynamic_landmarks_.insert2(context.getTrackletId(), s0);
    CHECK(isDynamicTrackletInMap(lmk_node));

    // gtsam::Pose3 L_k = e_H_k_world * L_e;
    // // H from k to s0 in frame k (^wL_k)
    // //  gtsam::Pose3 k_H_s0_k = L_e * e_H_k_world.inverse() * L_e.inverse();
    // gtsam::Pose3 k_H_s0_k = (L_e.inverse() * e_H_k_world * L_e).inverse();
    // gtsam::Pose3 k_H_s0_W = L_k * k_H_s0_k * L_k.inverse();
    // const gtsam::Point3 m_camera =
    //     lmk_node->getMeasurement(frame_node_k_1).landmark;
    // Landmark lmk_L0_init =
    //     L_e.inverse() * k_H_s0_W * context.X_k_1_measured * m_camera;
    Landmark lmk_L0_init = HybridObjectMotion::projectToObject3(
        context.X_k_1_measured, e_H_k_world, L_e,
        MeasurementTraits::point(lmk_node->getMeasurement(frame_node_k_1)));

    // TODO: this should not every be true as this is a new value!!!
    Landmark lmk_L0;
    getSafeQuery(lmk_L0, theta_accessor->query<Landmark>(point_key),
                 lmk_L0_init);
    // TODO: cache what s0 the landmark is made at so we can propogate them
    // later using the right motions within the correct Keyframe range!!!!
    new_values.insert(point_key, lmk_L0);
    result.updateAffectedObject(frame_node_k_1->frame_id,
                                context.getObjectId());
    if (result.debug_info)
      result.debug_info->getObjectInfo(context.getObjectId())
          .num_new_dynamic_points++;
  }

  if (context.is_starting_motion_frame) {
    // add factor at k-1
    Landmark measured_point_local;
    gtsam::SharedNoiseModel measurement_covariance;
    std::tie(measured_point_local, measurement_covariance) =
        MeasurementTraits::pointWithCovariance(
            lmk_node->getMeasurement(frame_node_k_1));

    if (params_.makeDynamicMeasurementsRobust()) {
      measurement_covariance = factor_graph_tools::robustifyHuber(
          params_.k_huber_3d_points_, measurement_covariance);
    }

    new_factors.emplace_shared<HybridMotionFactor>(
        frame_node_k_1->makePoseKey(),  // pose key at previous frames,
        object_motion_key_k_1, point_key, measured_point_local, L_e,
        measurement_covariance);
    if (result.debug_info)
      result.debug_info->getObjectInfo(context.getObjectId())
          .num_dynamic_factors++;
  }

  // add factor at k

  Landmark measured_point_local;
  gtsam::SharedNoiseModel measurement_covariance;
  std::tie(measured_point_local, measurement_covariance) =
      MeasurementTraits::pointWithCovariance(
          lmk_node->getMeasurement(frame_node_k));

  if (params_.makeDynamicMeasurementsRobust()) {
    measurement_covariance = factor_graph_tools::robustifyHuber(
        params_.k_huber_3d_points_, measurement_covariance);
  }

  new_factors.emplace_shared<HybridMotionFactor>(
      frame_node_k->makePoseKey(),  // pose key at previous frames,
      object_motion_key_k, point_key, measured_point_local, L_e,
      measurement_covariance);

  result.updateAffectedObject(frame_node_k->frame_id, context.getObjectId());
  if (result.debug_info)
    result.debug_info->getObjectInfo(context.getObjectId())
        .num_dynamic_factors++;
}

void HybridFormulation::objectUpdateContext(
    const ObjectUpdateContextType& context, UpdateObservationResult& result,
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors) {
  auto frame_node_k = context.frame_node_k;
  auto object_node = context.object_node;
  const gtsam::Key object_motion_key_k =
      frame_node_k->makeObjectMotionKey(context.getObjectId());

  auto theta_accessor = this->accessorFromTheta();
  const auto frame_id = context.getFrameId();
  const auto object_id = context.getObjectId();

  if (!is_other_values_in_map.exists(object_motion_key_k)) {
    // gtsam::Pose3 motion;
    const gtsam::Pose3 X_world = getInitialOrLinearizedSensorPose(frame_id);
    gtsam::Pose3 motion = computeInitialH(object_id, frame_id);
    VLOG(5) << "Added motion at  " << DynoLikeKeyFormatter(object_motion_key_k);
    // gtsam::Pose3 motion;
    new_values.insert(object_motion_key_k, motion);
    is_other_values_in_map.insert2(object_motion_key_k, true);

    // for now lets treat num_motion_factors as motion (values) added!!
    if (result.debug_info)
      result.debug_info->getObjectInfo(context.getObjectId())
          .num_motion_factors++;

    const auto [s0, L0] = getOrConstructL0(object_id, frame_id);
    if (s0 == frame_id) {
      // add prior
      new_factors.addPrior<gtsam::Pose3>(object_motion_key_k,
                                         gtsam::Pose3::Identity(),
                                         noise_models_.initial_pose_prior);
    }

    // test stuff
    FrameId first_seen_object_frame = object_node->getFirstSeenFrame();
    if (first_seen_object_frame == frame_id) {
      CHECK_EQ(s0, frame_id);
    }
  }

  if (frame_id < 2) return;

  auto frame_node_k_1 = map()->getFrame(frame_id - 1u);
  auto frame_node_k_2 = map()->getFrame(frame_id - 2u);
  if (!frame_node_k_1 || !frame_node_k_2) {
    return;
  }

  if (params_.use_smoothing_factor &&
      frame_node_k_1->objectObserved(object_id) &&
      frame_node_k_2->objectObserved(object_id)) {
    // motion key at previous frame
    const gtsam::Symbol object_motion_key_k_1 =
        frame_node_k_1->makeObjectMotionKey(object_id);

    const gtsam::Symbol object_motion_key_k_2 =
        frame_node_k_2->makeObjectMotionKey(object_id);

    auto object_smoothing_noise = noise_models_.object_smoothing_noise;
    CHECK(object_smoothing_noise);
    CHECK_EQ(object_smoothing_noise->dim(), 6u);

    {
      ObjectId object_label_k_1, object_label_k;
      FrameId frame_id_k_1, frame_id_k;
      CHECK(reconstructMotionInfo(object_motion_key_k_1, object_label_k_1,
                                  frame_id_k_1));
      CHECK(reconstructMotionInfo(object_motion_key_k, object_label_k,
                                  frame_id_k));
      CHECK_EQ(object_label_k_1, object_label_k);
      CHECK_EQ(frame_id_k_1 + 1, frame_id_k);  // assumes
      // consequative frames
    }

    // if the motion key at k (motion from k-1 to k), and key at k-1 (motion
    //  from k-2 to k-1)
    // exists in the map or is about to exist via new values, add the
    //  smoothing factor
    bool smoothing_factor_added =
        smoothing_factors_added_.exists(object_motion_key_k);
    if (!smoothing_factor_added &&
        is_other_values_in_map.exists(object_motion_key_k_2) &&
        is_other_values_in_map.exists(object_motion_key_k_1) &&
        is_other_values_in_map.exists(object_motion_key_k)) {
      new_factors.emplace_shared<HybridSmoothingFactor>(
          object_motion_key_k_2, object_motion_key_k_1, object_motion_key_k,
          getOrConstructL0(object_id, frame_id).second, object_smoothing_noise);
      if (result.debug_info)
        result.debug_info->getObjectInfo(context.getObjectId())
            .smoothing_factor_added = true;

      // update internal containers
      smoothing_factors_added_.insert(object_motion_key_k);
    }
  }
}

std::pair<FrameId, gtsam::Pose3> HybridFormulation::getOrConstructL0(
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
gtsam::Pose3 HybridFormulation::computeInitialH(ObjectId object_id,
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
  const bool has_frontend_motion = map()->hasInitialObjectMotion(
      current_frame_id, object_id, &initial_motion_frame);

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
      CHECK(map()->hasInitialObjectMotion(previous_frame, object_id,
                                          &initial_motion_frame));
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
      // we have a motion from the frontend that is k-1 to k
      // first check if we have a previous estimation motion that takes us from
      // s0 to k-1 in the map
      StateQuery<Motion3ReferenceFrame> H_W_s0_km1 =
          getEstimatedMotion(object_id, frame_id_km1);
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
        if (!map()->hasInitialObjectMotion(frame, object_id, &motion_frame)) {
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

gtsam::Pose3 HybridFormulation::calculateObjectCentroid(
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

  // TODO: could use computeObjectCentroid in accessor!!!?

  // measured/linearized camera pose at the first frame this object has been
  // seen
  const gtsam::Pose3 X_world = getInitialOrLinearizedSensorPose(frame_id);
  auto measurement_pairs = frame_node->getDynamicMeasurements(object_id);

  for (const auto& [lmk_node, measurement] : measurement_pairs) {
    CHECK(lmk_node->seenAtFrame(frame_id));
    CHECK_EQ(lmk_node->object_id, object_id);

    const gtsam::Point3 landmark_measurement_local =
        MeasurementTraits::point(measurement);
    // const gtsam::Point3 landmark_measurement_world = X_world *
    // landmark_measurement_local;

    dynamic_landmarks.push_back(LandmarkStatus::DynamicInGLobal(
        Point3Measurement(landmark_measurement_local), frame_id,
        lmk_node->tracklet_id, object_id));
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

void RegularHybridFormulation::preUpdate(const PreUpdateData& data) {
  // get objects seen in this frame from the map
  const typename Map::Ptr map = this->map();
  CHECK(map) << "Now can map be null!?";
  const auto frame_id_k = data.frame_id;
  const auto frame_node = map->getFrame(frame_id_k);
  CHECK(frame_node) << "Frame node null at k=" << data.frame_id;

  for (const auto& obj_node : frame_node->objects_seen) {
    ObjectId obj_id = obj_node->getId();
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
