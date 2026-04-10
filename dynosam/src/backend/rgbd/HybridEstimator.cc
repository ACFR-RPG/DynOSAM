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
#include <pcl/common/centroid.h>

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/factors/HybridFormulationFactors.hpp"
#include "dynosam_opt/NonlinearOptimizer.hpp"

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
  const auto object_node = map()->getObject(object_id);
  const auto frame_node_k = map()->getFrame(frame_id);
  CHECK(object_node);
  // const auto frame_node_k_1 = map()->getFrame(frame_id - 1u);

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

  FrameId last_seen;
  if (!object_node->previouslySeenFrame(&last_seen)) {
    const auto range = CHECK_NOTNULL(key_frame_data->find(object_id, frame_id));
    const auto [s0, L0] = range->dataPair();
    // check that the first frame of the object motion is actually this frame
    // this motion should actually be identity
    CHECK_EQ(s0, frame_id);
    return StateQuery<gtsam::Pose3>(motion_key, *e_H_k_world);
  }
  // if (!frame_node_k_1) {
  //   CHECK_NOTNULL(frame_node_k);
  //   // const auto range = CHECK_NOTNULL(key_frame_data->find(object_id,
  //   frame_id));
  //   // const auto [s0, L0] = range->dataPair();
  //   // // check that the first frame of the object motion is actually this
  //   frame
  //   // // this motion should actually be identity
  //   // CHECK_EQ(s0, frame_id);
  //   // return StateQuery<gtsam::Pose3>(motion_key, *e_H_k_world);
  // }
  else {
    CHECK_NOTNULL(frame_node_k);
    const auto frame_node_k_1 = map()->getFrame(last_seen);
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

TrackedPointsPerObject HybridFormulationKeyFrame::getObjectPoints(
    const ObjectIds& objects) const {
  auto hybrid_accessor = this->derivedAccessor<HybridAccessor>();

  TrackedPointsPerObject points_per_object;
  for (ObjectId object_id : objects) {
    StatusLandmarkVector estimates =
        hybrid_accessor->getLocalDynamicLandmarkEstimates(object_id);

    std::vector<std::pair<TrackletId, gtsam::Point3>> tracklet_pairs;
    tracklet_pairs.reserve(estimates.size());

    for (const auto& lmk_status : estimates) {
      tracklet_pairs.push_back({lmk_status.trackletId(), lmk_status.value()});
    }
    points_per_object.insert2(object_id, std::move(tracklet_pairs));
  }

  return points_per_object;
}

TrackedPointsPerObject HybridFormulationKeyFrame::getObjectPoints() const {
  auto hybrid_accessor = this->derivedAccessor<HybridAccessor>();
  return getObjectPoints(hybrid_accessor->getObjectIds());
}

TrackedPointsPerObject HybridFormulationKeyFrame::getObjectPoints(
    FrameId frame_id) const {
  auto frame_node = map()->getFrame(frame_id);
  CHECK_NOTNULL(frame_node);
  return getObjectPoints(frame_node->objectSeenIds());
}

HybridKeyFrameUpdate HybridFormulationKeyFrame::generateUpdateInfo() const {
  auto hybrid_accessor = this->derivedAccessor<HybridAccessor>();

  // what if things are optimising when we do this...?
  HybridKeyFrameUpdate info;
  info.frame_id = hybrid_accessor->getLatestFrameId();
  info.timestamp = hybrid_accessor->getLatestTimestamp();
  info.camera_trajectory = hybrid_accessor->getCameraTrajectory();

  auto object_trajectories = hybrid_accessor->getMultiObjectTrajectories();
  auto object_points = getObjectPoints();

  CHECK_EQ(object_trajectories.size(), object_points.size());

  info.object_infos.reserve(object_trajectories.size());
  for (const auto& [object_id, trajectory] : object_trajectories) {
    CHECK(object_points.exists(object_id));

    HybridKeyFrameUpdate::Object object_info;
    object_info.object_id = object_id;
    object_info.trajectory = trajectory;
    object_info.object_points = object_points.at(object_id);

    info.object_infos.push_back(std::move(object_info));
  }

  return info;
}

MultiObjectTrajectories HybridFormulationKeyFrame::refinePerFrameMotionsPGO(
    const MultiObjectTrajectories& full_trajectories) const {
  MultiObjectTrajectories full_trajectories_refined = full_trajectories;

  auto map = this->map();
  HybridAccessor::Ptr accessor = this->derivedAccessor<HybridAccessor>();
  for (const auto& [object_id, trajectory_j] : full_trajectories) {
    gtsam::Values values;
    gtsam::NonlinearFactorGraph graph;

    std::vector<std::pair<FrameId, FrameId>> pose_frame_pairs;

    LOG(INFO) << "Building PGO for j=" << object_id << " " << trajectory_j;

    // do sanity check on first frame to ensure it matches the Anchor Keyframe
    // stored in the backend
    const auto& first_entry = trajectory_j.first().data;
    const Motion3ReferenceFrame& first_motion = first_entry.motion;
    const gtsam::Pose3& first_pose = first_entry.pose;
    const FrameId to_frame = first_motion.to();
    const FrameId from_frame = first_motion.from();
    CHECK_EQ(first_motion.style(), MotionRepresentationStyle::F2F);

    // // set of shared frame nodes
    // const auto frames_with_refined_motion = object_node->getSeenFrames();
    for (const auto& entry_k : trajectory_j) {
      const Motion3ReferenceFrame& f2f_motion = entry_k.data.motion;
      const gtsam::Pose3 pose_k = entry_k.data.pose;
      const FrameId to_frame = f2f_motion.to();
      const FrameId from_frame = f2f_motion.from();
      CHECK_EQ(f2f_motion.style(), MotionRepresentationStyle::F2F);

      LOG(INFO) << "j=" << object_id << "entry: " << f2f_motion;

      pose_frame_pairs.push_back(std::make_pair(from_frame, to_frame));

      // use frontend poses as initial values
      // these will then be validated against the backend keyframe pose data
      // and priors added when necessary for anchor keyframes
      gtsam::Key object_pose_key = ObjectPoseSymbol(object_id, to_frame);
      gtsam::Key object_pose_key_from = ObjectPoseSymbol(object_id, from_frame);
      values.insert(object_pose_key, pose_k);

      // convert H_W_km1_k to relative motion constraint
      gtsam::Pose3 L_W_from = trajectory_j.at(from_frame).pose;
      gtsam::Pose3 H_L_from_to =
          L_W_from.inverse() * f2f_motion.estimate() * L_W_from;

      // TODO: for now!!
      gtsam::SharedNoiseModel relative_noise_model =
          gtsam::noiseModel::Isotropic::Sigma(6u, 4.0);

      auto relative_object_motion =
          boost::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
              object_pose_key_from, object_pose_key, H_L_from_to,
              relative_noise_model);
      graph += relative_object_motion;

      LOG(INFO) << "Adding relative motion constraint " << from_frame << " -> "
                << to_frame;

      if (isObjectKeyFrame(object_id, to_frame)) {
        LOG(INFO) << to_frame << " is KF";
        const KeyFrameMetaData& kf_data =
            key_frames_per_object_.at(object_id, to_frame);

        // sanity check frames are good
        CHECK_EQ(kf_data.H_W_lRKF_KF.to(), to_frame);

        auto object_node = map->getObject(object_id);
        CHECK_NOTNULL(object_node);

        const KeyFrameRange::ConstPtr akf_range =
            key_frame_data_.find(object_id, to_frame);

        if (!akf_range) {
          DYNO_THROW_MSG(DynosamException) << "Missing anchor frame for motion "
                                           << info_string(to_frame, object_id);
        }

        // we will validate the backend pose is the same as the frontend pose
        auto [akf_id, akf_pose_backend] = akf_range->dataPair();
        LOG(INFO) << akf_id;

        // we expect to have a refined motion from some previous anchor frame
        // to the to_frame
        // importantly this motion will inform us what the anchor frame is
        // via the "from" frame
        Motion3ReferenceFrame H_W_akf_k_refined = DYNO_GET_QUERY_DEBUG(
            accessor->getEstimatedMotion(object_id, to_frame));
        CHECK_EQ(H_W_akf_k_refined.to(), to_frame);

        // convert H_W_KF_k to relative motion constraint
        CHECK(trajectory_j.exists(akf_id));
        gtsam::Pose3 L_W_akf = trajectory_j.at(akf_id).pose;
        gtsam::Pose3 H_L_akf_to =
            L_W_akf.inverse() * H_W_akf_k_refined.estimate() * L_W_akf;

        gtsam::Key object_pose_key_akf = ObjectPoseSymbol(object_id, akf_id);
        // add keyframe motion as constraint to hold the graph together!
        // TODO: odometry should be different noise!!
        auto refined_keyframe_motion =
            boost::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
                object_pose_key_akf, object_pose_key, H_L_akf_to,
                this->noiseModels().odometry_noise);
        graph += refined_keyframe_motion;

        LOG(INFO) << "Adding refined KF relative motion constraint " << akf_id
                  << " -> " << to_frame;

        // TODO: currently intermediate frame nodes will not exist!!

        if (kf_data.keyframe_status == ObjectKeyFrameStatus::AnchorKeyFrame) {
          CHECK_EQ(H_W_akf_k_refined.from(), akf_id);

          // check that the per-frame motions start from the same pose as the
          // backemd although the value does not NEED to be the same, it ensures
          // that there is good synchronisation between the frontend and the
          // estimator get the pose associated with the anchor frame
          gtsam::Pose3 akf_pose_frontend = trajectory_j.at(akf_id).pose;
          CHECK(gtsam::traits<gtsam::Pose3>::Equals(akf_pose_backend,
                                                    akf_pose_frontend, 1e-4))
              << "KF pose " << akf_pose_backend << " pose " << akf_pose_frontend
              << " akf id " << akf_id;

          gtsam::Key anchor_object_pose_key =
              ObjectPoseSymbol(object_id, akf_id);
          LOG(INFO) << "Adding pose prior at KF pose " << akf_id;
          // add a strong prior to anchor pose as this will not change!
          graph.addPrior<gtsam::Pose3>(anchor_object_pose_key,
                                       akf_pose_frontend,
                                       this->noiseModels().initial_pose_prior);

        } else if (kf_data.keyframe_status ==
                   ObjectKeyFrameStatus::RegularKeyFrame) {
        } else {
          throw DynosamException("keyframe status cannot be NonKeyFrame!");
        }

        // // more sanity checks
        // right now this is a fail because only KF measurements are added to
        // the map!! CHECK_EQ(from_frame, object_node->getFirstSeenFrame());
      }
    }

    dyno::NonlinearOptimizer<gtsam::LevenbergMarquardtOptimizer> solver(graph,
                                                                        values);

    NonlinearOptimizerSummary summary;
    NonlinearOptimizerOptions options;

    LOG(INFO) << "Beginning PGO j=" << object_id;
    gtsam::Values optimised_values;
    CHECK(solver.solve(optimised_values, options, &summary));

    LOG(INFO) << "Initial error: " << summary.initial_error << " final error "
              << summary.final_error << " time[s] "
              << summary.cumulative_time_in_seconds
              << " #iterations= " << summary.numIterations();

    // recover pose values and correct motions
    PoseWithMotionTrajectory& refined_trajectory_j =
        full_trajectories_refined.at(object_id);
    for (const auto& [from_frame, to_frame] : pose_frame_pairs) {
      gtsam::Key object_pose_key_from = ObjectPoseSymbol(object_id, from_frame);
      gtsam::Key object_pose_key_to = ObjectPoseSymbol(object_id, to_frame);

      CHECK(object_pose_key_from == object_pose_key_to - 1 ||
            object_pose_key_from == object_pose_key_to);

      gtsam::Pose3 L_W_from =
          optimised_values.at<gtsam::Pose3>(object_pose_key_from);
      gtsam::Pose3 L_W_to =
          optimised_values.at<gtsam::Pose3>(object_pose_key_to);
      gtsam::Pose3 H_W_from_to = L_W_to * L_W_from.inverse();

      Motion3ReferenceFrame f2f_motion_refined(
          H_W_from_to, Motion3ReferenceFrame::Style::F2F,
          ReferenceFrame::GLOBAL, from_frame, to_frame);

      PoseWithMotion refined_entry;
      refined_entry.pose = L_W_to;
      refined_entry.motion = f2f_motion_refined;

      // always update the "to" frame
      // we will have identity motions where from==to in which case we still
      // update every frame since L_W_from==L_W_to
      CHECK(refined_trajectory_j.update(to_frame, refined_entry));
    }
  }
  return full_trajectories_refined;
}

UpdateObservationResult HybridFormulationKeyFrame::updateDynamicObservations(
    FrameId frame_id_k, gtsam::Values& new_values,
    gtsam::NonlinearFactorGraph& new_factors,
    const UpdateObservationParams& update_params) {
  typename Map::Ptr map = this->map();
  auto accessor = this->accessorFromTheta();

  // keep track of the new factors added in this function
  // these are then appended to the internal factors_ and new_factors
  gtsam::NonlinearFactorGraph internal_new_factors;
  // keep track of the new values added in this function
  // these are then appended to the internal values_ and new_values
  gtsam::Values internal_new_values;

  UpdateObservationResult result(update_params);

  // starting slot number is size of new factors
  // as long as the new factor slot is calculated before adding a new factor
  const Slot starting_factor_slot = new_factors.size();

  const auto frame_node_k = map->getFrame(frame_id_k);

  for (const auto& object_node : frame_node_k->objectsSeen()) {
    const ObjectId object_id = object_node->getId();

    // depending on how the frontend is implemented we may have measurements in
    // the map at non-keyframes but only want to update objects at KFs
    if (isObjectKeyFrame(object_id, frame_id_k)) {
      Context context;
      context.object_node = object_node;
      context.frame_node = frame_node_k;
      context.X_k_measured = getInitialOrLinearizedSensorPose(frame_id_k);
      context.starting_factor_slot = starting_factor_slot;

      LOG(INFO) << "Updating object " << info_string(frame_id_k, object_id);
      updateObject(context, result, internal_new_values, internal_new_factors);
    }
  }

  if (result.debug_info && VLOG_IS_ON(20)) {
    for (const auto& [object_id, object_info] :
         result.debug_info->getObjectInfos()) {
      std::stringstream ss;
      ss << "Object id debug info: " << object_id << "\n";
      ss << object_info;
      LOG(INFO) << ss.str();
    }
  }

  factors_ += internal_new_factors;
  new_factors += internal_new_factors;
  // update internal theta and factors
  // theta_.insert(internal_new_values);
  shared_data_->threadSafeInsertTheta(internal_new_values);
  // add to the external new_values
  new_values.insert(internal_new_values);
  return result;
}

bool HybridFormulationKeyFrame::isObjectKeyFrame(ObjectId object_id,
                                                 FrameId frame_id) const {
  return key_frames_per_object_.exists(object_id, frame_id);
}

void HybridFormulationKeyFrame::updateObject(
    const Context& context, UpdateObservationResult& result,
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors) {
  auto object_node = context.object_node;
  auto frame_node_kf = context.frame_node;
  const auto frame_id_kf = context.getFrameId();
  const auto object_id = context.getObjectId();

  const gtsam::Key object_motion_key_kf =
      frame_node_kf->makeObjectMotionKey(object_id);
  const gtsam::Key pose_key_kf = frame_node_kf->makePoseKey();

  auto seen_lmks_k = object_node->landmarksSeenAtFrame(frame_id_kf);

  CHECK(!is_other_values_in_map.exists(object_motion_key_kf));
  CHECK(initial_H_W_AKF_k_.exists(object_id, frame_id_kf));

  const KeyFrameRange::ConstPtr kf_range =
      key_frame_data_.find(object_id, frame_id_kf);
  CHECK(kf_range);

  // TODO: should check if the AKF_id is different for the from and the to?
  const auto [AKF_id, AKF_pose] = kf_range->dataPair();

  Motion3ReferenceFrame H_W_AKF_k =
      initial_H_W_AKF_k_.at(object_id, frame_id_kf);
  CHECK_EQ(H_W_AKF_k.from(), AKF_id);
  CHECK_EQ(H_W_AKF_k.to(), frame_id_kf);
  // Must add measurements at both AKF and KF (ie. multi view)
  // since the motion is constructed between these two frames
  const FrameId frame_id_akf = AKF_id;

  CHECK(key_frames_per_object_.exists(object_id, frame_id_kf));
  const KeyFrameMetaData& kf_meta_data =
      key_frames_per_object_.at(object_id, frame_id_kf);
  // measured motion from the frontend.
  const Motion3ReferenceFrame& H_W_lRKF_KF = kf_meta_data.H_W_lRKF_KF;
  CHECK_EQ(H_W_lRKF_KF.to(), frame_id_kf);
  // measured from frame
  const auto& lRKF_id = H_W_lRKF_KF.from();

  typename Map::Ptr map = this->map();
  auto frame_node_akf = map->getFrame(frame_id_akf);
  CHECK(frame_node_akf);

  // frame node for last regular KF
  auto frame_node_lrkf = map->getFrame(lRKF_id);
  CHECK(frame_node_lrkf);
  // object motion key from the 'from' frame
  const gtsam::Key object_motion_key_lkf =
      frame_node_lrkf->makeObjectMotionKey(object_id);

  new_values.insert(object_motion_key_kf, H_W_AKF_k.estimate());
  is_other_values_in_map.insert2(object_motion_key_kf, true);

  result.updateAffectedObject(frame_id_kf, object_id);

  // add zero motion at AKF
  // TODO: assumes that at least SOME points were seen at AKF!
  // TODO: should only add this if weve seen some points AKF -> should enforce
  // this actually happens!!!
  const gtsam::Key object_motion_key_akf =
      frame_node_akf->makeObjectMotionKey(object_id);
  if (!is_other_values_in_map.exists(object_motion_key_akf)) {
    new_values.insert(object_motion_key_akf, gtsam::Pose3::Identity());
    is_other_values_in_map.insert2(object_motion_key_akf, true);

    // add strong prior on initaion motion (which is just identity)
    new_factors.addPrior<gtsam::Pose3>(object_motion_key_akf,
                                       gtsam::Pose3::Identity(),
                                       noise_models_.initial_pose_prior);
  }

  size_t num_points_seen_akf = 0;
  for (const auto& obj_lmk_node : seen_lmks_k) {
    CHECK_EQ(obj_lmk_node->objectId(), object_id);
    const TrackletId tracklet_id = obj_lmk_node->trackletId();
    // LOG(INFO) << "Iterating through dynamic lmk " << tracklet_id;
    const gtsam::Key point_key = this->makeDynamicKey(tracklet_id);

    // becuase we dont anchor the motion (like in the original Hybrid with an
    // identity motion) we dont always have a motion at an anchor keyframe so
    // the point doesnt need to be seen there
    // TODO: depending in implementation of regular vs anchor KF, we may expect
    // that at anchor frames a point is not necessarily
    // seen at both the AKF and the RKF but should be seen at RKF-1 and RKF-k
    // (ie. if the previous KF was only a RKF, points should be seen at both?
    // MAYBE) CHECK(obj_lmk_node->seenAtFrame(frame_id_akf)) << "Lmk i=" <<
    // tracklet_id << " Object " << object_id << " not seen at " << frame_id_akf
    // << " but this is the from motion";
    CHECK(obj_lmk_node->seenAtFrame(frame_id_kf))
        << "Lmk i=" << tracklet_id << "Object " << object_id << " not seen at "
        << frame_id_kf << " but this is the to motion";

    if (!isDynamicTrackletInMap(obj_lmk_node)) {
      // we may have more seen landmarks than points in the filter
      // This "shouldn't" happen but is maybe some slightly bug in bookkeeping
      // somewhere CHECK(m_L_initial_.exists(object_id, tracklet_id)) <<
      // "Missing initalisation for j=" << object_id << " i=" << tracklet_id;
      if (!m_L_initial_.exists(object_id, tracklet_id)) {
        continue;
      }

      // uuuh need to update these becuase something in the accessor
      //  needs them!
      // TODO: double check implementation and write comment!
      is_dynamic_tracklet_in_map_.insert2(tracklet_id, AKF_id);
      all_dynamic_landmarks_.insert2(tracklet_id, AKF_id);

      factors_added_.insert2(tracklet_id, std::set<FrameId>{});

      CHECK(isDynamicTrackletInMap(obj_lmk_node));

      gtsam::Point3 m_L_initial = m_L_initial_.at(object_id, tracklet_id);
      new_values.insert(point_key, m_L_initial);

      if (result.debug_info) {
        result.debug_info->getObjectInfo(object_id).num_new_dynamic_points++;
      }

      // at measurements of point if also seen at AKF
      // most measurements wont actually be seen at the AFK
      // since this is the first frame!
      // what we really want to do is make sure we add measurements for the
      // "from" -> "to"
      // frame of the original motion (ie. the one from the frontend)
      // if(obj_lmk_node->seenAtFrame(frame_id_akf)) {
      //   addHybridMotionFactor(new_factors, pose_key_akf,
      //   object_motion_key_akf, point_key,
      //                     AKF_pose, obj_lmk_node, frame_node_akf);
      //   if (result.debug_info) {
      //     result.debug_info->getObjectInfo(context.getObjectId())
      //         .num_dynamic_factors++;
      //   }
      //   num_points_seen_akf++;
      // }

      // add measurements at both Keyframes a point is seen at
      // only needed if a point is new (I think!)
      // if (obj_lmk_node->seenAtFrame(lRKF_id)) {
      //   const gtsam::Key object_motion_key_lrkf =
      //       frame_node_lrkf->makeObjectMotionKey(object_id);
      //   const gtsam::Key pose_key_lrkf = frame_node_lrkf->makePoseKey();

      //   addHybridMotionFactor(new_factors, pose_key_lrkf,
      //                         object_motion_key_lrkf, point_key, AKF_pose,
      //                         obj_lmk_node, frame_node_lrkf);
      //   if (result.debug_info) {
      //     result.debug_info->getObjectInfo(context.getObjectId())
      //         .num_dynamic_factors++;
      //   }
      //   num_points_seen_akf++;
      // }

      // add at from frame if point is new
      //  assume that once we have seen it we only need to add measurements
      //  at the newest KF, since we will have added measurements
      //  for the previous KF last iteration (if all works well!)
      // addHybridMotionFactor(new_factors, pose_key, object_motion_key,
      // point_key,
      //                       AKF_pose, obj_lmk_node, frame_node_akf);
    }

    // check if we've added a factor at the regular from frame
    // actually could just sanity check we've added factors at any/every frame
    // that we have a motion for

    // CHECK(obj_lmk_node->seenAtFrame(lRKF_id))
    //     << info_string(lRKF_id, obj_lmk_node->object_id);
    std::set<FrameId>& frames_with_factors_added =
        factors_added_.at(tracklet_id);
    const bool factor_not_added_for_lRKF =
        frames_with_factors_added.find(lRKF_id) ==
        frames_with_factors_added.end();
    if (factor_not_added_for_lRKF && obj_lmk_node->seenAtFrame(lRKF_id)) {
      const gtsam::Key object_motion_key_lrkf =
          frame_node_lrkf->makeObjectMotionKey(object_id);
      const gtsam::Key pose_key_lrkf = frame_node_lrkf->makePoseKey();

      addHybridMotionFactor(new_factors, pose_key_lrkf, object_motion_key_lrkf,
                            point_key, AKF_pose, obj_lmk_node, frame_node_lrkf);
      if (result.debug_info) {
        result.debug_info->getObjectInfo(context.getObjectId())
            .num_dynamic_factors++;
      }
      num_points_seen_akf++;
      frames_with_factors_added.insert(frame_node_lrkf->frameId());
    }

    addHybridMotionFactor(new_factors, pose_key_kf, object_motion_key_kf,
                          point_key, AKF_pose, obj_lmk_node, frame_node_kf);
    frames_with_factors_added.insert(frame_id_kf);

    // sanity check/ backwards adding of points to ensure measurements
    // are added for all possible frames
    const FrameIds frames_with_measurements = obj_lmk_node->getSeenFrameIds();
    for (const FrameId frame_with_z : frames_with_measurements) {
      // check if this frame is in the same backend range
      // if it is not, ignore it!
      const KeyFrameRange::ConstPtr range =
          key_frame_data_.find(object_id, frame_with_z);
      CHECK(range);
      const auto [AKF_id_for_z, _] = range->dataPair();
      // if we have not added a measurement
      if (AKF_id_for_z != AKF_id) {
        continue;
      }

      // we have not added a factor at frame id with z
      // TODO: code duplication as above!!
      const bool factor_not_added_for_frame_with_z =
          frames_with_factors_added.find(frame_with_z) ==
          frames_with_factors_added.end();
      if (factor_not_added_for_frame_with_z &&
          obj_lmk_node->seenAtFrame(frame_with_z)) {
        auto frame_node_with_z = map->getFrame(frame_with_z);
        CHECK_NOTNULL(frame_node_with_z);

        const gtsam::Key object_motion_key =
            frame_node_with_z->makeObjectMotionKey(object_id);
        const gtsam::Key pose_key = frame_node_with_z->makePoseKey();

        addHybridMotionFactor(new_factors, pose_key, object_motion_key,
                              point_key, AKF_pose, obj_lmk_node,
                              frame_node_with_z);
        if (result.debug_info) {
          result.debug_info->getObjectInfo(context.getObjectId())
              .num_dynamic_factors++;
        }
        frames_with_factors_added.insert(frame_with_z);
      }
    }

    {
      // do a sanity check that we've added all meaasurement factors for this
      // point
      // const FrameIds frames_with_measurements =
      // obj_lmk_node->getSeenFrameIds(); FrameIds
      // frames_with_factors_added_vec(frames_with_factors_added.begin(),
      //                                        frames_with_factors_added.end());
      // CHECK(equals_with_abs_tol(frames_with_measurements,
      //                           frames_with_factors_added_vec))
      //   << "Frames with measurements:" <<
      //   container_to_string(frames_with_measurements)
      //   << "\n"
      //   << "Frames with factors:" <<
      //   container_to_string(frames_with_factors_added_vec);
    }

    // if(!smoothing_factors_added_.exists(object_motion_key_kf)) {
    //   // check we have the previous keyframed motion
    //   is_other_values_in_map.exists(object_motion_key_lkf);

    //   auto object_smoothing_noise = noise_models_.object_smoothing_noise;
    //   CHECK(object_smoothing_noise);
    //   CHECK_EQ(object_smoothing_noise->dim(), 6u);

    //   new_factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
    //     object_motion_key_lkf, object_motion_key_kf, H_W_lRKF_KF.estimate(),
    //     object_smoothing_noise
    //   );

    //   if (result.debug_info)
    //     result.debug_info->getObjectInfo(context.getObjectId())
    //         .smoothing_factor_added = true;

    //     // update internal containers
    //   smoothing_factors_added_.insert(object_motion_key_kf);
    // }

    if (result.debug_info) {
      result.debug_info->getObjectInfo(context.getObjectId())
          .num_dynamic_factors++;
    }
  }

  LOG(INFO) << "Num factors added for measuemenets at AKF "
            << num_points_seen_akf;
}

void HybridFormulationKeyFrame::addHybridMotionFactor(
    gtsam::NonlinearFactorGraph& new_factors, gtsam::Key pose_key,
    gtsam::Key object_motion_key, gtsam::Key point_key,
    const gtsam::Pose3& KF_pose, SharedLandmarkNode lmk_node,
    SharedFrameNode frame_node) {
  // Landmark measured_point_local;
  // gtsam::SharedNoiseModel measurement_covariance;
  // std::tie(measured_point_local, measurement_covariance) =
  //     MeasurementTraits::pointWithCovariance(
  //         lmk_node->getMeasurement(frame_node));

  auto stereo_measurement =
      MeasurementTraits::stereo(lmk_node->getMeasurement(frame_node));

  // FOR NOW
  CHECK(stereo_measurement);
  auto [measurement, noise_model] = *stereo_measurement;

  if (params_.makeDynamicMeasurementsRobust()) {
    noise_model = factor_graph_tools::robustifyHuber(params_.k_huber_3d_points_,
                                                     noise_model);
  }

  new_factors.emplace_shared<StereoHybridMotionFactor>(
      measurement, KF_pose, noise_model, rgbd_camera_->getFakeStereoCalib(),
      pose_key, object_motion_key, point_key, true /* throw cheirality*/
  );

  // new_factors.emplace_shared<HybridMotionFactor>(
  //     pose_key, object_motion_key, point_key, measured_point_local, KF_pose,
  //     noise_model);
}

void HybridFormulationKeyFrame::addObjects(
    FrameId frame_id, const ObjectPoseChangeInfoMap& object_motion_info) {
  HybridAccessor::Ptr accessor = this->derivedAccessor<HybridAccessor>();
  CHECK_NOTNULL(accessor);

  for (const auto& [object_id, object_info] : object_motion_info) {
    CHECK(object_info.isKeyFrame());
    // estimated keyframe motioa from the frontend
    // in this case k is the current but will now also be the latest KF
    const Motion3ReferenceFrame& H_W_RKF_k = object_info.H_W_KF_k;
    const ObjectKeyFrameStatus keyframe_status = object_info.keyframe_status;

    KeyFrameMetaData kf_data;
    kf_data.keyframe_status = keyframe_status;
    kf_data.H_W_lRKF_KF = H_W_RKF_k;

    CHECK_EQ(frame_id, H_W_RKF_k.to());

    key_frames_per_object_.insert22(object_id, frame_id, kf_data);

    LOG(INFO) << "Processing object track: " << info_string(frame_id, object_id)
              << " keyframe status: " << keyframe_status;

    // TODO: we initalie the new KF with the pose provided from the frontend
    // for consistency. This is fine when the object observations are
    // continuous
    // but at some point the KF pose in the frontend and back-end will change
    // (i.e after opt!)

    // ad new initialisation points to backend
    // misleading print as we dont add this many points! Only new ones
    VLOG(10) << "Adding initial object points of size "
             << object_info.initial_object_points.size();
    for (const auto& landmark_status : object_info.initial_object_points) {
      const TrackletId& tracklet_id = landmark_status.trackletId();
      const gtsam::Point3& m_L = landmark_status.value();
      // only add new ones?
      if (!m_L_initial_.exists(object_id, tracklet_id)) {
        // LOG(INFO) << "Making initial object points j=" << object_id << "
        // i="
        // << tracklet_id;
        m_L_initial_.insert22(object_id, tracklet_id, m_L);
      }
    }

    if (keyframe_status == ObjectKeyFrameStatus::AnchorKeyFrame) {
      key_frame_data_.startNewActiveRange(object_id, H_W_RKF_k.from(),
                                          object_info.L_W_KF);
      LOG(INFO) << "Making Anchor KF for NEW object "
                << info_string(H_W_RKF_k.from(), object_id) << " with motion "
                << H_W_RKF_k.from() << " -> " << H_W_RKF_k.to();

      // the frontend range is always "to" because it indicates the start
      // of the next range and a single motion represents one
      front_end_keyframes_.startNewActiveRange(object_id, H_W_RKF_k.from(),
                                               object_info.L_W_KF);
      LOG(INFO) << "Making Regular KF for NEW object "
                << info_string(H_W_RKF_k.from(), object_id) << " with motion "
                << H_W_RKF_k.from() << " -> " << H_W_RKF_k.to();
      initial_H_W_AKF_k_.insert22(object_id, H_W_RKF_k.to(), H_W_RKF_k);
    } else {
      CHECK_EQ(keyframe_status, ObjectKeyFrameStatus::RegularKeyFrame);

      const KeyFrameRange::ConstPtr last_frontend_range =
          front_end_keyframes_.find(object_id, frame_id);
      CHECK(last_frontend_range)
          << "Failed for tracked object " << info_string(frame_id, object_id);
      auto [lRKF_id, L_lRKF] = last_frontend_range->dataPair();

      // hopefully the last regular KF is the from frame
      lRKF_id = H_W_RKF_k.from();
      LOG(INFO) << "Last regular KF " << lRKF_id;

      // TODO: if this is regular KF then the position of this KF will change
      // according to the motion that is refined
      //  as L_W_k = L_W_KF = H_W_AKF_KF * L_AKF
      front_end_keyframes_.startNewActiveRange(object_id, H_W_RKF_k.to(),
                                               object_info.L_W_k);
      LOG(INFO) << "Making Regular KF for tracked object "
                << info_string(H_W_RKF_k.to(), object_id) << " with motion "
                << H_W_RKF_k.from() << " -> " << H_W_RKF_k.to();

      const KeyFrameRange::ConstPtr frontend_range =
          front_end_keyframes_.find(object_id, frame_id);
      CHECK(frontend_range);
      // the most recent motion added to the estimator should take us from
      // backend_kf_id to last_kf_id
      const auto [current_kf_id, current_kf_pose] = frontend_range->dataPair();
      LOG(INFO) << "Current regular KF " << current_kf_id;

      // get backend anchor point and confert motion if necessary
      const KeyFrameRange::ConstPtr backend_range =
          CHECK_NOTNULL(key_frame_data_.find(object_id, frame_id));
      const auto [backend_kf_id, backend_kf_pose] = backend_range->dataPair();
      LOG(INFO) << "Anchor KF id: " << backend_kf_id;

      LOG(INFO) << "Provided object odometry " << H_W_RKF_k.from() << " -> "
                << H_W_RKF_k.to();

      // motion from anchor point to current k
      // this value will be added to the estimator
      Motion3ReferenceFrame H_W_AKF_KF_initial;
      if (H_W_RKF_k.from() == backend_kf_id) {
        // TODO: also check pose is close?
        CHECK_EQ(H_W_RKF_k.from(), lRKF_id);
        H_W_AKF_KF_initial = H_W_RKF_k;
      } else {
        // motion does not match, so start a new starting point and transform
        // the motio update frontend range
        // NOTE: this uses the "to" motion (not the from)

        // need to transform into correct frame using (ideally the most up to
        // date, i.e estimated motion)
        LOG(INFO) << "Looking up estimated motion from " << backend_kf_id
                  << " -> " << lRKF_id;

        // TODO: eventually should come from optimizer
        CHECK(initial_H_W_AKF_k_.exists(object_id, lRKF_id));
        // from current anchor keyframe to last regular kf
        const auto H_W_AKF_lKF_initial =
            initial_H_W_AKF_k_.at(object_id, lRKF_id);

        const StateQuery<Motion3ReferenceFrame> H_W_AKF_lKF_refined =
            accessor->getEstimatedMotion(object_id, lRKF_id);

        Motion3ReferenceFrame H_W_AKF_lKF;
        // use refined motion estimate if available, otherwise fall back to
        // initial NOTE: I think the refined query should ALways be available
        // getSafeQuery(H_W_AKF_lKF, H_W_AKF_lKF_refined, H_W_AKF_lKF_initial);
        H_W_AKF_lKF = H_W_AKF_lKF_initial;

        CHECK_EQ(H_W_AKF_lKF.from(), backend_kf_id);
        CHECK_EQ(H_W_AKF_lKF.to(), lRKF_id);

        // this motion does us from the last keyframe to the current frame k
        CHECK_EQ(H_W_RKF_k.from(), lRKF_id);
        CHECK_EQ(H_W_RKF_k.to(), current_kf_id);

        H_W_AKF_KF_initial = Motion3ReferenceFrame(
            H_W_RKF_k.estimate() * H_W_AKF_lKF.estimate(),
            Motion3ReferenceFrame::Style::KF, ReferenceFrame::GLOBAL,
            backend_kf_id, current_kf_id);
      }
      initial_H_W_AKF_k_.insert22(object_id, H_W_AKF_KF_initial.to(),
                                  H_W_AKF_KF_initial);
    }
  }
}

ObjectPoseMap HybridFormulationKeyFrame::getInitialObjectPoses() const {
  ObjectPoseMap kf_poses;

  for (const auto& [object_id, H_W_AKF_k_per_frame] : initial_H_W_AKF_k_) {
    CHECK(key_frame_data_.exists(object_id));

    const auto& kf_ranges = key_frame_data_.at(object_id);

    for (const auto& [kf_frame_id, H_W_AKF_KF] : H_W_AKF_k_per_frame) {
      // should be the anchor frame
      const auto from_frame = H_W_AKF_KF.from();
      // ranges are stored by their "to" range
      const auto to_frame = H_W_AKF_KF.to();

      const auto anchor_range = kf_ranges.find(to_frame);
      CHECK(anchor_range);
      const auto [anchor_frame, anchor_pose] = anchor_range->dataPair();
      CHECK_EQ(from_frame, anchor_frame);

      gtsam::Pose3 L_w = H_W_AKF_KF.estimate() * anchor_pose;
      kf_poses.insert22(object_id, to_frame, L_w);
    }
  }

  return kf_poses;
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
