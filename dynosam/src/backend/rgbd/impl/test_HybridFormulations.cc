/*
 *   Copyright (c) 2025 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam/backend/rgbd/impl/test_HybridFormulations.hpp"

#include "dynosam/factors/HybridFormulationFactors.hpp"

namespace dyno {
namespace test_hybrid {

gtsam::Vector DecoupledObjectCentricMotionFactor::evaluateError(
    const gtsam::Pose3& e_H_k_world, const gtsam::Point3& m_L,
    boost::optional<gtsam::Matrix&> J1,
    boost::optional<gtsam::Matrix&> J2) const {
  auto reordered_resiudal = [&](const gtsam::Pose3& e_H_k_world,
                                const gtsam::Point3& m_L) {
    return residual(X_k_, e_H_k_world, m_L, Z_k_, L_e_);
  };

  if (J1) {
    Eigen::Matrix<double, 3, 6> J =
        gtsam::numericalDerivative21<gtsam::Vector3, gtsam::Pose3,
                                     gtsam::Point3>(reordered_resiudal,
                                                    e_H_k_world, m_L);
    *J1 = J;
  }

  if (J2) {
    Eigen::Matrix<double, 3, 3> J =
        gtsam::numericalDerivative22<gtsam::Vector3, gtsam::Pose3,
                                     gtsam::Point3>(reordered_resiudal,
                                                    e_H_k_world, m_L);
    *J2 = J;
  }

  return reordered_resiudal(e_H_k_world, m_L);
}

gtsam::Vector StructurelessObjectCentricMotion2::residual(
    const gtsam::Pose3& X_k_1, const gtsam::Pose3& H_k_1,
    const gtsam::Pose3& X_k, const gtsam::Pose3& H_k,
    const gtsam::Point3& Z_k_1, const gtsam::Point3& Z_k,
    const gtsam::Pose3& L_e) {
  return HybridObjectMotion::projectToObject3(X_k_1, H_k_1, L_e, Z_k_1) -
         HybridObjectMotion::projectToObject3(X_k, H_k, L_e, Z_k);
}

gtsam::Vector StructurelessDecoupledObjectCentricMotion::evaluateError(
    const gtsam::Pose3& H_k_1, const gtsam::Pose3& H_k,
    boost::optional<gtsam::Matrix&> J1,
    boost::optional<gtsam::Matrix&> J2) const {
  // use lambda to create residual with arguments and variables
  auto reordered_resiudal = [&](const gtsam::Pose3& H_k_1,
                                const gtsam::Pose3& H_k) -> gtsam::Vector3 {
    return residual(X_k_1_, H_k_1, X_k_, H_k, Z_k_1_, Z_k_, L_e_);
  };

  if (J1) {
    Eigen::Matrix<double, 3, 6> J =
        gtsam::numericalDerivative21<gtsam::Vector3, gtsam::Pose3,
                                     gtsam::Pose3>(reordered_resiudal, H_k_1,
                                                   H_k);
    *J1 = J;
  }

  if (J2) {
    Eigen::Matrix<double, 3, 6> J =
        gtsam::numericalDerivative22<gtsam::Vector3, gtsam::Pose3,
                                     gtsam::Pose3>(reordered_resiudal, H_k_1,
                                                   H_k);
    *J2 = J;
  }

  return reordered_resiudal(H_k_1, H_k);
}

gtsam::Vector StructurelessObjectCentricMotionFactor2::evaluateError(
    const gtsam::Pose3& X_k_1, const gtsam::Pose3& H_k_1,
    const gtsam::Pose3& X_k, const gtsam::Pose3& H_k,
    boost::optional<gtsam::Matrix&> J1, boost::optional<gtsam::Matrix&> J2,
    boost::optional<gtsam::Matrix&> J3,
    boost::optional<gtsam::Matrix&> J4) const {
  if (J1) {
    Eigen::Matrix<double, 3, 6> J =
        gtsam::numericalDerivative41<gtsam::Vector3, gtsam::Pose3, gtsam::Pose3,
                                     gtsam::Pose3, gtsam::Pose3>(
            std::bind(&StructurelessObjectCentricMotionFactor2::residual,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, std::placeholders::_4, Z_k_1_,
                      Z_k_, L_e_),
            X_k_1, H_k_1, X_k, H_k);
    *J1 = J;
  }

  if (J1) {
    Eigen::Matrix<double, 3, 6> J =
        gtsam::numericalDerivative41<gtsam::Vector3, gtsam::Pose3, gtsam::Pose3,
                                     gtsam::Pose3, gtsam::Pose3>(
            std::bind(&StructurelessObjectCentricMotionFactor2::residual,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, std::placeholders::_4, Z_k_1_,
                      Z_k_, L_e_),
            X_k_1, H_k_1, X_k, H_k);
    *J1 = J;
  }

  if (J2) {
    Eigen::Matrix<double, 3, 6> J =
        gtsam::numericalDerivative42<gtsam::Vector3, gtsam::Pose3, gtsam::Pose3,
                                     gtsam::Pose3, gtsam::Pose3>(
            std::bind(&StructurelessObjectCentricMotionFactor2::residual,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, std::placeholders::_4, Z_k_1_,
                      Z_k_, L_e_),
            X_k_1, H_k_1, X_k, H_k);
    *J2 = J;
  }

  if (J3) {
    Eigen::Matrix<double, 3, 6> J =
        gtsam::numericalDerivative43<gtsam::Vector3, gtsam::Pose3, gtsam::Pose3,
                                     gtsam::Pose3, gtsam::Pose3>(
            std::bind(&StructurelessObjectCentricMotionFactor2::residual,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, std::placeholders::_4, Z_k_1_,
                      Z_k_, L_e_),
            X_k_1, H_k_1, X_k, H_k);
    *J3 = J;
  }

  if (J4) {
    Eigen::Matrix<double, 3, 6> J =
        gtsam::numericalDerivative44<gtsam::Vector3, gtsam::Pose3, gtsam::Pose3,
                                     gtsam::Pose3, gtsam::Pose3>(
            std::bind(&StructurelessObjectCentricMotionFactor2::residual,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, std::placeholders::_4, Z_k_1_,
                      Z_k_, L_e_),
            X_k_1, H_k_1, X_k, H_k);
    *J4 = J;
  }

  return residual(X_k_1, H_k_1, X_k, H_k, Z_k_1_, Z_k_, L_e_);
}

void StructurelessDecoupledFormulation::dynamicPointUpdateCallback(
    const PointUpdateContextType& context, UpdateObservationResult& result,
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors) {
  const auto lmk_node = context.lmk_node;
  const auto frame_node_k_1 = context.frame_node_k_1;
  const auto frame_node_k = context.frame_node_k;

  gtsam::Key point_key = this->makeDynamicKey(context.getTrackletId());

  const gtsam::Key object_motion_key_k =
      frame_node_k->makeObjectMotionKey(context.getObjectId());
  const gtsam::Key object_motion_key_k_1 =
      frame_node_k_1->makeObjectMotionKey(context.getObjectId());
  auto landmark_motion_noise = noise_models_.landmark_motion_noise;

  gtsam::Pose3 X_K_1 =
      this->getInitialOrLinearizedSensorPose(frame_node_k_1->frame_id);
  gtsam::Pose3 X_K =
      this->getInitialOrLinearizedSensorPose(frame_node_k->frame_id);

  gtsam::Pose3 L_e;
  FrameId s0;
  std::tie(s0, L_e) =
      getOrConstructL0(context.getObjectId(), frame_node_k_1->getId());
  auto dynamic_point_noise = noise_models_.dynamic_point_noise;

  new_factors.emplace_shared<StructurelessDecoupledObjectCentricMotion>(
      object_motion_key_k_1, object_motion_key_k, X_K_1, X_K,
      lmk_node->getMeasurement(frame_node_k_1).landmark,
      lmk_node->getMeasurement(frame_node_k).landmark, L_e,
      dynamic_point_noise);

  result.updateAffectedObject(frame_node_k_1->frame_id, context.getObjectId());
  result.updateAffectedObject(frame_node_k->frame_id, context.getObjectId());
}

void DecoupledFormulation::dynamicPointUpdateCallback(
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

  // TODO:this will not be the case with sliding/window as we reconstruct the
  // graph from a different starting point!!
  //  CHECK_GE(frame_node_k_1->getId(), s0);

  if (!isDynamicTrackletInMap(lmk_node)) {
    // mark as now in map
    is_dynamic_tracklet_in_map_.insert2(context.getTrackletId(), true);
    CHECK(isDynamicTrackletInMap(lmk_node));

    // use first point as initalisation?
    // in this case k is k-1 as we use frame_node_k_1
    gtsam::Pose3 e_H_k_world =
        computeInitialH(context.getObjectId(), frame_node_k_1->getId());
    gtsam::Pose3 L_k = e_H_k_world * L_e;
    // H from k to s0 in frame k (^wL_k)
    //  gtsam::Pose3 k_H_s0_k = L_e * e_H_k_world.inverse() *  L_e.inverse();
    gtsam::Pose3 k_H_s0_k = (L_e.inverse() * e_H_k_world * L_e).inverse();
    gtsam::Pose3 k_H_s0_W = L_k * k_H_s0_k * L_k.inverse();
    // LOG(INFO) << "e_H_k_world " << e_H_k_world;
    // measured point in camera frame
    const gtsam::Point3 m_camera =
        lmk_node->getMeasurement(frame_node_k_1).landmark;
    Landmark lmk_L0_init =
        L_e.inverse() * k_H_s0_W * context.X_k_1_measured * m_camera;

    // initalise value //cannot initalise again the same -> it depends where L_e
    // is created, no?
    Landmark lmk_L0;
    getSafeQuery(lmk_L0, theta_accessor->query<Landmark>(point_key),
                 lmk_L0_init);
    new_values.insert(point_key, lmk_L0);
    result.updateAffectedObject(frame_node_k_1->frame_id,
                                context.getObjectId());
  }

  auto dynamic_point_noise = noise_models_.dynamic_point_noise;
  if (context.is_starting_motion_frame) {
    gtsam::Pose3 X_k_1 =
        this->getInitialOrLinearizedSensorPose(frame_node_k_1->frame_id);
    new_factors.emplace_shared<DecoupledObjectCentricMotionFactor>(
        object_motion_key_k_1, point_key,
        lmk_node->getMeasurement(frame_node_k_1).landmark, L_e, X_k_1,
        dynamic_point_noise);
    result.updateAffectedObject(frame_node_k_1->frame_id,
                                context.getObjectId());
  }

  gtsam::Pose3 X_k =
      this->getInitialOrLinearizedSensorPose(frame_node_k->frame_id);

  new_factors.emplace_shared<DecoupledObjectCentricMotionFactor>(
      object_motion_key_k, point_key,
      lmk_node->getMeasurement(frame_node_k).landmark, L_e, X_k,
      dynamic_point_noise);
  result.updateAffectedObject(frame_node_k->frame_id, context.getObjectId());
}

void StructurlessFormulation::dynamicPointUpdateCallback(
    const PointUpdateContextType& context, UpdateObservationResult& result,
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors) {
  const auto lmk_node = context.lmk_node;
  const auto frame_node_k_1 = context.frame_node_k_1;
  const auto frame_node_k = context.frame_node_k;

  gtsam::Key point_key = this->makeDynamicKey(context.getTrackletId());

  const gtsam::Key object_motion_key_k =
      frame_node_k->makeObjectMotionKey(context.getObjectId());
  const gtsam::Key object_motion_key_k_1 =
      frame_node_k_1->makeObjectMotionKey(context.getObjectId());
  auto landmark_motion_noise = noise_models_.landmark_motion_noise;

  gtsam::Pose3 L_e;
  FrameId s0;
  std::tie(s0, L_e) =
      getOrConstructL0(context.getObjectId(), frame_node_k_1->getId());
  auto dynamic_point_noise = noise_models_.dynamic_point_noise;

  new_factors.emplace_shared<StructurelessObjectCentricMotionFactor2>(
      frame_node_k_1->makePoseKey(), object_motion_key_k_1,
      frame_node_k->makePoseKey(), object_motion_key_k,
      lmk_node->getMeasurement(frame_node_k_1).landmark,
      lmk_node->getMeasurement(frame_node_k).landmark, L_e,
      dynamic_point_noise);

  result.updateAffectedObject(frame_node_k_1->frame_id, context.getObjectId());
  result.updateAffectedObject(frame_node_k->frame_id, context.getObjectId());
}

StateQuery<gtsam::Point3> SmartStructurlessAccessor::queryPoint(
    gtsam::Key point_key, TrackletId tracklet_id) const {
  if (tracklet_id_to_smart_factor_->exists(tracklet_id)) {
    HybridSmartFactor::shared_ptr smart_factor =
        tracklet_id_to_smart_factor_->at(tracklet_id);

    gtsam::TriangulationResult point_result = smart_factor->point();

    if (point_result) {
      return StateQuery<gtsam::Point3>(point_key, *point_result);
    } else {
      // TODO: not actually the resason!!
      return StateQuery<gtsam::Point3>::NotInMap(point_key);
    }

  } else {
    return StateQuery<gtsam::Point3>::NotInMap(point_key);
  }
}

void SmartStructurlessFormulation::dynamicPointUpdateCallback(
    const PointUpdateContextType& context, UpdateObservationResult& result,
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors) {
  const auto lmk_node = context.lmk_node;
  const auto frame_node_k_1 = context.frame_node_k_1;
  const auto frame_node_k = context.frame_node_k;

  auto theta_accessor = this->accessorFromTheta();
  auto dynamic_point_noise = noise_models_.dynamic_point_noise;

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

  if (!isDynamicTrackletInMap(lmk_node)) {
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
        lmk_node->getMeasurement(frame_node_k_1).landmark);

    // TODO: this should not every be true as this is a new value!!!
    Landmark lmk_L0;
    getSafeQuery(lmk_L0, theta_accessor->query<Landmark>(point_key),
                 lmk_L0_init);

    HybridSmartFactor::shared_ptr smart_factor =
        boost::make_shared<HybridSmartFactor>(L_e, dynamic_point_noise,
                                              lmk_L0_init);
    // HybridSmartFactor::shared_ptr smart_factor =
    //     boost::make_shared<HybridSmartFactor>(L_e, dynamic_point_noise);
    // TODO: actually not needed!!
    Slot new_factor_slot = context.starting_factor_slot + new_factors.size();

    new_factors.push_back(smart_factor);
    tracklet_id_to_smart_factor_.insert2(context.getTrackletId(), smart_factor);
    tracklet_id_to_smart_factor_index_.insert2(context.getTrackletId(),
                                               new_factor_slot);
    tracklet_ids_of_new_smart_factors_.push_back(context.getTrackletId());

    result.updateAffectedObject(frame_node_k_1->frame_id,
                                context.getObjectId());
    if (result.debug_info)
      result.debug_info->getObjectInfo(context.getObjectId())
          .num_new_dynamic_points++;
  }

  HybridSmartFactor::shared_ptr smart_factor =
      tracklet_id_to_smart_factor_.at(context.getTrackletId());
  CHECK_NOTNULL(smart_factor);

  // expecting slot to have been updated
  // TODO: do not only want to do this if the smart factor is old...?
  Slot slot = tracklet_id_to_smart_factor_index_.at(context.getTrackletId());

  if (!result.isam_update_params.newAffectedKeys) {
    result.isam_update_params.newAffectedKeys =
        gtsam::FastMap<gtsam::FactorIndex, gtsam::KeySet>{};
  }

  gtsam::KeySet affected_keys;

  if (context.is_starting_motion_frame) {
    // add factor at k-1
    // ------ good motion factor/////
    smart_factor->add(lmk_node->getMeasurement(frame_node_k_1).landmark,
                      object_motion_key_k_1, frame_node_k_1->makePoseKey());
    if (result.debug_info)
      result.debug_info->getObjectInfo(context.getObjectId())
          .num_dynamic_factors++;

    affected_keys.insert(object_motion_key_k_1);
    affected_keys.insert(frame_node_k_1->makePoseKey());
  }

  smart_factor->add(lmk_node->getMeasurement(frame_node_k).landmark,
                    object_motion_key_k, frame_node_k->makePoseKey());

  affected_keys.insert(object_motion_key_k);
  affected_keys.insert(frame_node_k->makePoseKey());

  // altert the incremental solver of the updates to this factor!!!
  result.isam_update_params.newAffectedKeys->insert2(slot, affected_keys);

  result.updateAffectedObject(frame_node_k->frame_id, context.getObjectId());
  if (result.debug_info)
    result.debug_info->getObjectInfo(context.getObjectId())
        .num_dynamic_factors++;
}

void SmartStructurlessFormulation::postUpdate(const PostUpdateData& data) {
  // call base class first
  Base::postUpdate(data);

  // if incremental, update factor slots
  if (data.incremental_result) {
    const auto& incremental_result = data.incremental_result.value();

    VLOG(15) << "Received incremental result. Updating slots for "
             << tracklet_ids_of_new_smart_factors_.size() << " smart factors";
    for (size_t i = 0u; i < tracklet_ids_of_new_smart_factors_.size(); ++i) {
      DCHECK(i < incremental_result.newFactorsIndices.size())
          << "There are more new smart factors than new factors added to the "
             "graph.";
      // Get new slot in the graph for the newly added smart factor.
      const size_t& slot = incremental_result.newFactorsIndices.at(i);

      const auto& it = tracklet_id_to_smart_factor_index_.find(
          tracklet_ids_of_new_smart_factors_.at(i));

      DCHECK(it != tracklet_id_to_smart_factor_index_.end())
          << "Trying to access unavailable factor.";

      // update slot number
      it->second = slot;
    }

    // only clear if incremental update...?
    tracklet_ids_of_new_smart_factors_.clear();
  }
}

}  // namespace test_hybrid
}  // namespace dyno
