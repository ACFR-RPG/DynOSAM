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

#include "dynosam/formulations/HybridEstimator.hpp"

#include <gtsam/slam/PoseRotationPrior.h>
#include <pcl/common/centroid.h>

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
  StateQuery<Motion3ReferenceFrame> query =
      this->getObjectMotionReferenceFrame(frame_id, object_id);

  if (!query) {
    return StateQuery<gtsam::Pose3>(query.key(), query.status());
  } else {
    return StateQuery<gtsam::Pose3>(query.key(), query.get());
  }
}

StateQuery<gtsam::Pose3> HybridAccessor::getObjectPose(
    FrameId frame_id, ObjectId object_id) const {
  // we estimate a motion ^w_{s0}H_k, so we can compute a pose ^wL_k =
  // ^w_{s0}H_k * ^wL_{s0}
  const auto frame_node_k = map()->getFrame(frame_id);
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

  const auto timestamp = frame_node->timestamp();

  const auto object_node = map()->getObject(object_id);
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

TrackletIds HybridAccessor::collectPointsAtKeyFrame(
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

bool HybridAccessor::getObjectKeyFrameHistory(
    ObjectId object_id, const KeyFrameRanges*& ranges) const {
  // CHECK_NOTNULL(ranges);
  const auto& key_frame_data = shared_hybrid_formulation_data_.key_frame_data;
  if (!key_frame_data->exists(object_id)) {
    return false;
  }

  ranges = &key_frame_data->at(object_id);
  return true;
}

bool HybridAccessor::hasObjectKeyFrame(ObjectId object_id,
                                       FrameId frame_id) const {
  const auto& key_frame_data = shared_hybrid_formulation_data_.key_frame_data;
  return static_cast<bool>(key_frame_data->find(object_id, frame_id));
}

std::pair<FrameId, gtsam::Pose3> HybridAccessor::getObjectKeyFrame(
    ObjectId object_id, FrameId frame_id) const {
  const auto& key_frame_data = shared_hybrid_formulation_data_.key_frame_data;
  const KeyFrameRange::ConstPtr range =
      key_frame_data->find(object_id, frame_id);
  CHECK_NOTNULL(range);
  return range->dataPair();
}

StateQuery<Motion3ReferenceFrame> HybridAccessor::getEstimatedMotion(
    ObjectId object_id, FrameId frame_id) const {
  // not in form of accessor but in form of estimation
  const auto frame_node_k = map()->getFrame(frame_id);
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

std::optional<Motion3ReferenceFrame> HybridAccessor::getRelativeLocalMotion(
    FrameId frame_id, ObjectId object_id) const {
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

StateQueryStatus HybridAccessor::getObjectMotionReferenceFrameHelper(
    FrameId frame_id, ObjectId object_id, gtsam::Key& motion_key,
    gtsam::Pose3& motion, FrameId& from, FrameId& to) const {
  const auto object_node = map()->getObject(object_id);
  const auto frame_node_k = map()->getFrame(frame_id);
  CHECK(object_node);

  if (!frame_node_k) {
    VLOG(30) << "Could not construct object motion frame id=" << frame_id
             << " object id=" << object_id << " as the frame does not exist!";
    return StateQueryStatus::INVALID_MAP;
  }

  motion_key = frame_node_k->makeObjectMotionKey(object_id);
  StateQuery<gtsam::Pose3> H_W_KF_k = this->query<gtsam::Pose3>(motion_key);
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
    const auto frame_node_km1 = map()->getFrame(last_seen);
    CHECK_NOTNULL(frame_node_km1);

    StateQuery<gtsam::Pose3> H_W_KF_km1 = this->query<gtsam::Pose3>(
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

StateQuery<Motion3ReferenceFrame> HybridAccessor::getObjectMotionReferenceFrame(
    FrameId frame_id, ObjectId object_id) const {
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

HybridFormulation::HybridFormulation(const FormulationParams& params,
                                     typename Map::Ptr map,
                                     const NoiseModels& noise_models,
                                     const Sensors& sensors,
                                     const FormulationHooks& hooks)
    : Base(params, map, noise_models, sensors, hooks) {
  auto camera = sensors_.camera;
  CHECK_NOTNULL(camera);
  rgbd_camera_ = camera->safeGetRGBDCamera();
  CHECK_NOTNULL(rgbd_camera_);
}

void HybridFormulation::dynamicPointUpdateCallback(
    const PointUpdateContextType& context, UpdateObservationResult& result,
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors) {
  const auto lmk_node = context.lmk_node;
  const auto frame_node_k_1 = context.frame_node_k_1;
  const auto frame_node_k = context.frame_node_k;
  const auto object_id = context.getObjectId();
  const auto frame_id_k_1 = frame_node_k_1->getId();

  auto theta_accessor = this->accessorFromTheta();

  gtsam::Key point_key = this->makeDynamicKey(context.getTrackletId());

  const gtsam::Key object_motion_key_k =
      frame_node_k->makeObjectMotionKey(object_id);
  const gtsam::Key object_motion_key_k_1 =
      frame_node_k_1->makeObjectMotionKey(object_id);

  // gtsam::Pose3 L_e;
  const IntermediateMotionInfo keyframe_info =
      getIntermediateMotionInfo(object_id, frame_id_k_1);

  const FrameId& s0 = keyframe_info.kf_id;
  const gtsam::Pose3& L_e = keyframe_info.keyframe_pose;
  const gtsam::Pose3& H_W_e_k_initial = keyframe_info.H_W_e_k_initial;
  // FrameId s0;
  // std::tie(s0, L_e) =
  //     getOrConstructL0(context.getObjectId(), frame_node_k_1->getId());
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
    // bool keyframe_updated;
    // gtsam::Pose3 e_H_k_world = computeInitialH(
    //     context.getObjectId(), frame_node_k_1->getId(), &keyframe_updated);

    // TODO: we should never actually let this happen during an update
    //  it should only happen before measurements are added
    // want to avoid somehow a situation where some (landmark)variables are at
    // an old keyframe I dont think this will happen with the current
    // implementation...
    // if (keyframe_updated) {
    //   // TODO: gross I have to re-get them again!!
    //   std::tie(s0, L_e) =
    //       getOrConstructL0(context.getObjectId(), frame_node_k_1->getId());
    // }

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
        context.X_k_1_measured, H_W_e_k_initial, L_e,
        MeasurementTraits::point(lmk_node->getMeasurement(frame_node_k_1)));

    // TODO: this should not every be true as this is a new value!!!
    Landmark lmk_L0;
    getSafeQuery(lmk_L0, theta_accessor->query<Landmark>(point_key),
                 lmk_L0_init);
    // TODO: cache what s0 the landmark is made at so we can propogate them
    // later using the right motions within the correct Keyframe range!!!!
    new_values.insert(point_key, lmk_L0);
    result.updateAffectedObject(frame_node_k_1->frameId(),
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

  result.updateAffectedObject(frame_node_k->frameId(), context.getObjectId());
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

  const IntermediateMotionInfo keyframe_info =
      getIntermediateMotionInfo(object_id, frame_id);

  if (!is_other_values_in_map.exists(object_motion_key_k)) {
    // gtsam::Pose3 motion;
    const gtsam::Pose3 X_world = getInitialOrLinearizedSensorPose(frame_id);
    // gtsam::Pose3 motion = computeInitialH(object_id, frame_id);
    VLOG(5) << "Added motion at  " << DynosamKeyFormatter(object_motion_key_k);
    // gtsam::Pose3 motion;
    new_values.insert(object_motion_key_k, keyframe_info.H_W_e_k_initial);
    is_other_values_in_map.insert2(object_motion_key_k, true);

    // for now lets treat num_motion_factors as motion (values) added!!
    if (result.debug_info)
      result.debug_info->getObjectInfo(context.getObjectId())
          .num_motion_factors++;

    // we are at object keyframe
    // NOTE: this should never happen for hybrid KF!!
    if (keyframe_info.kf_id == frame_id) {
      // add prior
      new_factors.addPrior<gtsam::Pose3>(object_motion_key_k,
                                         gtsam::Pose3::Identity(),
                                         noise_models_.initial_pose_prior);
    }

    // test stuff
    FrameId first_seen_object_frame = object_node->getFirstSeenFrame();
    if (first_seen_object_frame == frame_id) {
      CHECK_EQ(keyframe_info.kf_id, frame_id);
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
          keyframe_info.keyframe_pose, object_smoothing_noise);
      if (result.debug_info)
        result.debug_info->getObjectInfo(context.getObjectId())
            .smoothing_factor_added = true;

      // update internal containers
      smoothing_factors_added_.insert(object_motion_key_k);
    }
  }
}

}  // namespace dyno
