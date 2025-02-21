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

#include "dynosam/backend/rgbd/ObjectCentricEstimator.hpp"

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/factors/ObjectCentricFactors.hpp"

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

StateQuery<gtsam::Pose3> ObjectCentricAccessor::getSensorPose(
    FrameId frame_id) const {
  const auto frame_node = map()->getFrame(frame_id);
  CHECK_NOTNULL(frame_node);
  return this->query<gtsam::Pose3>(frame_node->makePoseKey());
}

StateQuery<gtsam::Pose3> ObjectCentricAccessor::getObjectMotion(
    FrameId frame_id, ObjectId object_id) const {
  const auto frame_node_k = map()->getFrame(frame_id);
  const auto frame_node_k_1 = map()->getFrame(frame_id - 1u);

  if (!frame_node_k) {
    LOG(WARNING) << "Could not construct object motion frame id=" << frame_id
                 << " object id=" << object_id
                 << " as the frame does not exist!";
    return StateQuery<gtsam::Pose3>::InvalidMap();
  }

  auto motion_key = frame_node_k->makeObjectMotionKey(object_id);
  StateQuery<gtsam::Pose3> motion_s0_k = this->query<gtsam::Pose3>(motion_key);
  if (!motion_s0_k) {
    LOG(WARNING) << "Could not construct object motion frame id=" << frame_id
                 << " object id=" << object_id
                 << ". Frame exists by motion is missing!!!";
    return StateQuery<gtsam::Pose3>::InvalidMap();
  }

  // first object motion (ie s0 -> s1)
  if (!frame_node_k_1) {
    CHECK_NOTNULL(frame_node_k);
    FrameId s0 = L0_values_->at(object_id).first;
    // check that the first frame of the object motion is actually this frame
    // this motion should actually be identity
    CHECK_EQ(s0, frame_id);
    return StateQuery<gtsam::Pose3>(motion_key, *motion_s0_k);
  } else {
    CHECK_NOTNULL(frame_node_k);
    CHECK_NOTNULL(frame_node_k_1);

    StateQuery<gtsam::Pose3> motion_s0_k_1 = this->query<gtsam::Pose3>(
        frame_node_k_1->makeObjectMotionKey(object_id));

    if (motion_s0_k && motion_s0_k_1) {
      // want a motion from k-1 to k, but we estimate s0 to k
      //^w_{k-1}H_k = ^w_{s0}H_k \: ^w_{s0}H_{k-1}^{-1}
      gtsam::Pose3 motion = motion_s0_k.get() * motion_s0_k_1->inverse();
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

StateQuery<gtsam::Pose3> ObjectCentricAccessor::getObjectPose(
    FrameId frame_id, ObjectId object_id) const {
  // we estimate a motion ^w_{s0}H_k, so we can compute a pose ^wL_k =
  // ^w_{s0}H_k * ^wL_{s0}
  const auto frame_node_k = map()->getFrame(frame_id);
  gtsam::Key motion_key = frame_node_k->makeObjectMotionKey(object_id);
  gtsam::Key pose_key = frame_node_k->makeObjectPoseKey(object_id);
  CHECK(frame_node_k);
  /// hmmm... if we do a query after we do an update but before an optimise then
  /// the motion will
  // be whatever we initalised it with
  // in the case of identity, the pose at k will just be L_s0 which we dont
  // want?
  StateQuery<gtsam::Pose3> motion_s0_k = this->query<gtsam::Pose3>(motion_key);
  // CHECK(false);

  if (motion_s0_k) {
    CHECK(L0_values_->exists(object_id));
    const gtsam::Pose3& L0 = L0_values_->at(object_id).second;
    // LOG(INFO) << "Frame " << frame_id << " obj id " << object_id;
    // LOG(INFO) << "Object pose s0 " << L0;
    // LOG(INFO) << "Object motion " << motion_s0_k.get();
    const gtsam::Pose3 L_k = motion_s0_k.get() * L0;
    // LOG(INFO) << "Object pose " << L_k;

    return StateQuery<gtsam::Pose3>(pose_key, L_k);
  } else {
    return StateQuery<gtsam::Pose3>::NotInMap(pose_key);
  }
}
StateQuery<gtsam::Point3> ObjectCentricAccessor::getDynamicLandmark(
    FrameId frame_id, TrackletId tracklet_id) const {
  // we estimate a motion ^w_{s0}H_k, so we can compute a point ^wm_k =
  // ^w_{s0}H_k * ^wL_{s0} * ^{L_{s0}}m
  const auto frame_node_k = map()->getFrame(frame_id);
  const auto lmk_node = map()->getLandmark(tracklet_id);
  CHECK(frame_node_k);
  CHECK_NOTNULL(lmk_node);
  const auto object_id = lmk_node->object_id;
  // point in L_{s0}
  // NOTE: we use STATIC point key here
  gtsam::Key point_key = this->makeDynamicKey(tracklet_id);
  StateQuery<gtsam::Point3> point_local = this->query<gtsam::Point3>(point_key);

  // get motion from S0 to k
  gtsam::Key motion_key = frame_node_k->makeObjectMotionKey(object_id);
  StateQuery<gtsam::Pose3> motion_s0_k = this->query<gtsam::Pose3>(motion_key);

  // TODO: I guess can happen if we miss a motion becuae an object is not seen
  // for one frame?!?
  //  if (point_local)
  //    CHECK(motion_s0_k) << "We have a point " <<
  //    DynoLikeKeyFormatter(point_key)
  //                       << " but no motion at frame " << frame_id << " with
  //                       key: " << DynoLikeKeyFormatter(motion_key);
  if (point_local && motion_s0_k) {
    CHECK(L0_values_->exists(object_id));
    const gtsam::Pose3& L0 = L0_values_->at(object_id).second;
    // point in world at k
    const gtsam::Point3 m_k = motion_s0_k.get() * L0 * point_local.get();
    return StateQuery<gtsam::Point3>(point_key, m_k);
  } else {
    return StateQuery<gtsam::Point3>::NotInMap(point_key);
  }
}

StatusLandmarkVector ObjectCentricAccessor::getDynamicLandmarkEstimates(
    FrameId frame_id, ObjectId object_id) const {
  const auto frame_node = map()->getFrame(frame_id);
  const auto object_node = map()->getObject(object_id);
  CHECK_NOTNULL(frame_node);
  CHECK_NOTNULL(object_node);

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
      estimates.push_back(
          LandmarkStatus::DynamicInGLobal(lmk_query.get(),  // estimate
                                          frame_id, tracklet_id, object_id));
    }
  }
  return estimates;
}

// TODO: no keyframing
bool ObjectCentricFormulation::hasObjectKeyFrame(ObjectId object_id,
                                                 FrameId) const {
  return L0_.exists(object_id);
}
std::pair<FrameId, gtsam::Pose3> ObjectCentricFormulation::getObjectKeyFrame(
    ObjectId object_id, FrameId frame_id) const {
  CHECK(hasObjectKeyFrame(object_id, frame_id));
  return L0_.at(object_id);
}

Motion3ReferenceFrame ObjectCentricFormulation::getEstimatedMotion(
    ObjectId object_id, FrameId frame_id) const {
  // not in form of accessor but in form of estimation
  const auto frame_node_k = map()->getFrame(frame_id);
  CHECK_NOTNULL(frame_node_k);

  auto motion_key = frame_node_k->makeObjectMotionKey(object_id);
  // raw access the theta in the accessor!!
  auto theta_accessor = this->accessorFromTheta();
  StateQuery<gtsam::Pose3> H_W_s0_k =
      theta_accessor->query<gtsam::Pose3>(motion_key);
  CHECK(H_W_s0_k);

  CHECK(this->hasObjectKeyFrame(object_id, frame_id));
  // s0
  auto [reference_frame, _] = this->getObjectKeyFrame(object_id, frame_id);

  return Motion3ReferenceFrame(H_W_s0_k.get(), MotionRepresentationStyle::KF,
                               ReferenceFrame::GLOBAL, reference_frame,
                               frame_id);
}

void ObjectCentricFormulation::dynamicPointUpdateCallback(
    const PointUpdateContextType& context, UpdateObservationResult& result,
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors) {
  // //acrew PointUpdateContextType for each object and trigger the update
  // const auto frame_node_k_1 = context.frame_node_k_1;
  // const auto frame_node_k = context.frame_node_k;
  // const auto object_id = context.getObjectId();

  // //TODO: for now lets just use k (which means we are dropping a
  // measurement!)
  // //just for initial testing!!
  // result.updateAffectedObject(frame_node_k_1->frame_id, object_id);
  // result.updateAffectedObject(frame_node_k->frame_id, object_id);

  // if(!point_contexts_.exists(object_id)) {
  //     point_contexts_.insert2(object_id,
  //     std::vector<PointUpdateContextType>());
  // }
  // point_contexts_.at(object_id).push_back(context);
  const auto lmk_node = context.lmk_node;
  const auto frame_node_k_1 = context.frame_node_k_1;
  const auto frame_node_k = context.frame_node_k;

  // if(frame_node_k_1->getId() % 2 == 0) {
  //   return;
  // }

  auto theta_accessor = this->accessorFromTheta();

  gtsam::Key point_key = this->makeDynamicKey(context.getTrackletId());

  const gtsam::Key object_motion_key_k =
      frame_node_k->makeObjectMotionKey(context.getObjectId());
  const gtsam::Key object_motion_key_k_1 =
      frame_node_k_1->makeObjectMotionKey(context.getObjectId());

  // LOG(INFO) << "Dynamic point update context tracklet " <<
  // context.getTrackletId() << " object id " << context.getObjectId() << " "
  //   << DynoLikeKeyFormatter(object_motion_key_k_1) << " " <<
  //   DynoLikeKeyFormatter(object_motion_key_k);

  gtsam::Pose3 L_0;
  FrameId s0;
  std::tie(s0, L_0) =
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

    // mark as now in map
    is_dynamic_tracklet_in_map_.insert2(context.getTrackletId(), true);
    CHECK(isDynamicTrackletInMap(lmk_node));

    // use first point as initalisation?
    // in this case k is k-1 as we use frame_node_k_1
    gtsam::Pose3 s0_H_k_world =
        computeInitialH(context.getObjectId(), frame_node_k_1->getId());
    gtsam::Pose3 L_k = s0_H_k_world * L_0;
    // H from k to s0 in frame k (^wL_k)
    //  gtsam::Pose3 k_H_s0_k = L_0 * s0_H_k_world.inverse() *  L_0.inverse();
    gtsam::Pose3 k_H_s0_k = (L_0.inverse() * s0_H_k_world * L_0).inverse();
    gtsam::Pose3 k_H_s0_W = L_k * k_H_s0_k * L_k.inverse();
    // LOG(INFO) << "s0_H_k " << s0_H_k;
    // measured point in camera frame
    const gtsam::Point3 m_camera =
        lmk_node->getMeasurement(frame_node_k_1).landmark;
    Landmark lmk_L0_init =
        L_0.inverse() * k_H_s0_W * context.X_k_1_measured * m_camera;

    // initalise value //cannot initalise again the same -> it depends where L_0
    // is created, no?
    Landmark lmk_L0;
    getSafeQuery(lmk_L0, theta_accessor->query<Landmark>(point_key),
                 lmk_L0_init);
    new_values.insert(point_key, lmk_L0);
    result.updateAffectedObject(frame_node_k_1->frame_id,
                                context.getObjectId());

    if (result.debug_info)
      result.debug_info->getObjectInfo(context.getObjectId())
          .num_new_dynamic_points++;
  }

  auto dynamic_point_noise = noise_models_.dynamic_point_noise;
  if (context.is_starting_motion_frame) {
    // add factor at k-1
    // ------ good motion factor/////
    new_factors.emplace_shared<ObjectCentricMotionFactor>(
        frame_node_k_1->makePoseKey(),  // pose key at previous frames,
        object_motion_key_k_1, point_key,
        lmk_node->getMeasurement(frame_node_k_1).landmark, L_0,
        dynamic_point_noise);
    if (result.debug_info)
      result.debug_info->getObjectInfo(context.getObjectId())
          .num_motion_factors++;
  }

  // add factor at k
  // ------ good motion factor/////
  new_factors.emplace_shared<ObjectCentricMotionFactor>(
      frame_node_k->makePoseKey(),  // pose key at previous frames,
      object_motion_key_k, point_key,
      lmk_node->getMeasurement(frame_node_k).landmark, L_0,
      dynamic_point_noise);

  if (result.debug_info)
    result.debug_info->getObjectInfo(context.getObjectId())
        .num_motion_factors++;

  result.updateAffectedObject(frame_node_k->frame_id, context.getObjectId());
}

void ObjectCentricFormulation::objectUpdateContext(
    const ObjectUpdateContextType& context, UpdateObservationResult& result,
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors) {
  auto frame_node_k = context.frame_node_k;
  const gtsam::Key object_motion_key_k =
      frame_node_k->makeObjectMotionKey(context.getObjectId());

  auto theta_accessor = this->accessorFromTheta();
  const auto frame_id = context.getFrameId();
  const auto object_id = context.getObjectId();

  if (!is_other_values_in_map.exists(object_motion_key_k)) {
    // gtsam::Pose3 motion;
    const gtsam::Pose3 X_world = getInitialOrLinearizedSensorPose(frame_id);
    gtsam::Pose3 motion = computeInitialH(object_id, frame_id);
    // LOG(INFO) << "Added motion at  "
    //           << DynoLikeKeyFormatter(object_motion_key_k);
    // gtsam::Pose3 motion;
    new_values.insert(object_motion_key_k, motion);
    is_other_values_in_map.insert2(object_motion_key_k, true);
    result.updateAffectedObject(frame_id, object_id);

    FrameId s0 = getOrConstructL0(object_id, frame_id).first;
    if (s0 == frame_id) {
      // add prior
      new_factors.addPrior<gtsam::Pose3>(object_motion_key_k,
                                         gtsam::Pose3::Identity(),
                                         noise_models_.initial_pose_prior);
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
    if (is_other_values_in_map.exists(object_motion_key_k_2) &&
        is_other_values_in_map.exists(object_motion_key_k_1) &&
        is_other_values_in_map.exists(object_motion_key_k)) {
      new_factors.emplace_shared<ObjectCentricSmoothing>(
          object_motion_key_k_2, object_motion_key_k_1, object_motion_key_k,
          getOrConstructL0(object_id, frame_id).second, object_smoothing_noise);
      if (result.debug_info)
        result.debug_info->getObjectInfo(context.getObjectId())
            .smoothing_factor_added = true;
    }
  }
}

std::pair<FrameId, gtsam::Pose3> ObjectCentricFormulation::getOrConstructL0(
    ObjectId object_id, FrameId frame_id) {
  if (L0_.exists(object_id)) {
    // LOG(INFO) << "Getting L0 from cache " << object_id << " SE(3) "
    //           << L0_.at(object_id).second;
    return L0_.at(object_id);
  }

  if (FLAGS_init_object_pose_from_gt) {
    const auto gt_packets = hooks().ground_truth_packets_request();
    if (gt_packets && gt_packets->exists(frame_id)) {
      const auto& gt_packet = gt_packets->at(frame_id);

      ObjectPoseGT object_gt;
      if (gt_packet.getObject(object_id, object_gt)) {
        L0_.insert2(object_id, std::make_pair(frame_id, object_gt.L_world_));
        return L0_.at(object_id);
      }
    }
  }

  // else initalise from centroid?
  auto object_node = map()->getObject(object_id);
  CHECK(object_node);

  auto frame_node = map()->getFrame(frame_id);
  CHECK(frame_node);
  CHECK(frame_node->objectObserved(object_id));

  StatusLandmarkVector dynamic_landmarks;

  // measured/linearized camera pose at the first frame this object has been
  // seen
  const gtsam::Pose3 X_world = getInitialOrLinearizedSensorPose(frame_id);
  auto measurement_pairs = frame_node->getDynamicMeasurements(object_id);

  for (const auto& [lmk_node, measurement] : measurement_pairs) {
    CHECK(lmk_node->seenAtFrame(frame_id));
    CHECK_EQ(lmk_node->object_id, object_id);

    const gtsam::Point3 landmark_measurement_local = measurement.landmark;
    // const gtsam::Point3 landmark_measurement_world = X_world *
    // landmark_measurement_local;

    dynamic_landmarks.push_back(
        LandmarkStatus::DynamicInGLobal(landmark_measurement_local, frame_id,
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

  // frame id should coincide with the first seen frame of this object...?
  L0_.insert2(object_id, std::make_pair(frame_id, center));
  return L0_.at(object_id);
}

// TODO: can be massively more efficient
// should also check if the last object motion from the estimation can be used
// as the last motion
//  so only one composition is needed to get the latest motion
gtsam::Pose3 ObjectCentricFormulation::computeInitialH(ObjectId object_id,
                                                       FrameId frame_id) {
  auto [s0, L_0] = getOrConstructL0(object_id, frame_id);

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
  const auto frame_node_km1 = map()->getFrame(frame_id_km1);
  if (frame_node_km1) {
    auto motion_key = frame_node_km1->makeObjectMotionKey(object_id);
    // if(this->exists)
    // TODO: initalise s -> k-1 from last update and then compound?
  }

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
    if (diff > 5) {
      LOG(FATAL) << "Motion intalisation failed for j= " << object_id
                 << ", motion missing at " << current_frame_id
                 << " and previous seen frame " << previous_frame
                 << " too far away!";
    } else {
      // TODO: just use previous motion???
      CHECK(map()->hasInitialObjectMotion(previous_frame, object_id,
                                          &initial_motion_frame));
      // update current_frame_id to previous frame so that the composition loop
      // below stops at the right place!
      // TODO: will this mess up the frame_id - 1 check?
      //  LOG(INFO) << "Updating current frame id to previous frame " <<
      //  previous_frame << " to account for missing frame at " <<
      //  current_frame_id;
      current_frame_id = previous_frame;
    }
  }

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
          LOG(WARNING) << "No frontend motion at frame " << frame
                       << " object id " << object_id;
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
    }
  }
}

}  // namespace dyno
