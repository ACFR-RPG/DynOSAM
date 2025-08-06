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

// gtsam::Vector DecoupledObjectCentricMotionFactor::evaluateError(
//     const gtsam::Pose3& e_H_k_world, const gtsam::Point3& m_L,
//     boost::optional<gtsam::Matrix&> J1,
//     boost::optional<gtsam::Matrix&> J2) const {
//   auto reordered_resiudal = [&](const gtsam::Pose3& e_H_k_world,
//                                 const gtsam::Point3& m_L) {
//     return residual(X_k_, e_H_k_world, m_L, Z_k_, L_e_);
//   };

//   if (J1) {
//     Eigen::Matrix<double, 3, 6> J =
//         gtsam::numericalDerivative21<gtsam::Vector3, gtsam::Pose3,
//                                      gtsam::Point3>(reordered_resiudal,
//                                                     e_H_k_world, m_L);
//     *J1 = J;
//   }

//   if (J2) {
//     Eigen::Matrix<double, 3, 3> J =
//         gtsam::numericalDerivative22<gtsam::Vector3, gtsam::Pose3,
//                                      gtsam::Point3>(reordered_resiudal,
//                                                     e_H_k_world, m_L);
//     *J2 = J;
//   }

//   return reordered_resiudal(e_H_k_world, m_L);
// }

// gtsam::Vector StructurelessObjectCentricMotion2::residual(
//     const gtsam::Pose3& X_k_1, const gtsam::Pose3& H_k_1,
//     const gtsam::Pose3& X_k, const gtsam::Pose3& H_k,
//     const gtsam::Point3& Z_k_1, const gtsam::Point3& Z_k,
//     const gtsam::Pose3& L_e) {
//   return HybridObjectMotion::projectToObject3(X_k_1, H_k_1, L_e, Z_k_1) -
//          HybridObjectMotion::projectToObject3(X_k, H_k, L_e, Z_k);
// }

// gtsam::Vector StructurelessDecoupledObjectCentricMotion::evaluateError(
//     const gtsam::Pose3& H_k_1, const gtsam::Pose3& H_k,
//     boost::optional<gtsam::Matrix&> J1,
//     boost::optional<gtsam::Matrix&> J2) const {
//   // use lambda to create residual with arguments and variables
//   auto reordered_resiudal = [&](const gtsam::Pose3& H_k_1,
//                                 const gtsam::Pose3& H_k) -> gtsam::Vector3 {
//     return residual(X_k_1_, H_k_1, X_k_, H_k, Z_k_1_, Z_k_, L_e_);
//   };

//   if (J1) {
//     Eigen::Matrix<double, 3, 6> J =
//         gtsam::numericalDerivative21<gtsam::Vector3, gtsam::Pose3,
//                                      gtsam::Pose3>(reordered_resiudal, H_k_1,
//                                                    H_k);
//     *J1 = J;
//   }

//   if (J2) {
//     Eigen::Matrix<double, 3, 6> J =
//         gtsam::numericalDerivative22<gtsam::Vector3, gtsam::Pose3,
//                                      gtsam::Pose3>(reordered_resiudal, H_k_1,
//                                                    H_k);
//     *J2 = J;
//   }

//   return reordered_resiudal(H_k_1, H_k);
// }

// gtsam::Vector StructurelessObjectCentricMotionFactor2::evaluateError(
//     const gtsam::Pose3& X_k_1, const gtsam::Pose3& H_k_1,
//     const gtsam::Pose3& X_k, const gtsam::Pose3& H_k,
//     boost::optional<gtsam::Matrix&> J1, boost::optional<gtsam::Matrix&> J2,
//     boost::optional<gtsam::Matrix&> J3,
//     boost::optional<gtsam::Matrix&> J4) const {
//   if (J1) {
//     Eigen::Matrix<double, 3, 6> J =
//         gtsam::numericalDerivative41<gtsam::Vector3, gtsam::Pose3,
//         gtsam::Pose3,
//                                      gtsam::Pose3, gtsam::Pose3>(
//             std::bind(&StructurelessObjectCentricMotionFactor2::residual,
//                       std::placeholders::_1, std::placeholders::_2,
//                       std::placeholders::_3, std::placeholders::_4, Z_k_1_,
//                       Z_k_, L_e_),
//             X_k_1, H_k_1, X_k, H_k);
//     *J1 = J;
//   }

//   if (J1) {
//     Eigen::Matrix<double, 3, 6> J =
//         gtsam::numericalDerivative41<gtsam::Vector3, gtsam::Pose3,
//         gtsam::Pose3,
//                                      gtsam::Pose3, gtsam::Pose3>(
//             std::bind(&StructurelessObjectCentricMotionFactor2::residual,
//                       std::placeholders::_1, std::placeholders::_2,
//                       std::placeholders::_3, std::placeholders::_4, Z_k_1_,
//                       Z_k_, L_e_),
//             X_k_1, H_k_1, X_k, H_k);
//     *J1 = J;
//   }

//   if (J2) {
//     Eigen::Matrix<double, 3, 6> J =
//         gtsam::numericalDerivative42<gtsam::Vector3, gtsam::Pose3,
//         gtsam::Pose3,
//                                      gtsam::Pose3, gtsam::Pose3>(
//             std::bind(&StructurelessObjectCentricMotionFactor2::residual,
//                       std::placeholders::_1, std::placeholders::_2,
//                       std::placeholders::_3, std::placeholders::_4, Z_k_1_,
//                       Z_k_, L_e_),
//             X_k_1, H_k_1, X_k, H_k);
//     *J2 = J;
//   }

//   if (J3) {
//     Eigen::Matrix<double, 3, 6> J =
//         gtsam::numericalDerivative43<gtsam::Vector3, gtsam::Pose3,
//         gtsam::Pose3,
//                                      gtsam::Pose3, gtsam::Pose3>(
//             std::bind(&StructurelessObjectCentricMotionFactor2::residual,
//                       std::placeholders::_1, std::placeholders::_2,
//                       std::placeholders::_3, std::placeholders::_4, Z_k_1_,
//                       Z_k_, L_e_),
//             X_k_1, H_k_1, X_k, H_k);
//     *J3 = J;
//   }

//   if (J4) {
//     Eigen::Matrix<double, 3, 6> J =
//         gtsam::numericalDerivative44<gtsam::Vector3, gtsam::Pose3,
//         gtsam::Pose3,
//                                      gtsam::Pose3, gtsam::Pose3>(
//             std::bind(&StructurelessObjectCentricMotionFactor2::residual,
//                       std::placeholders::_1, std::placeholders::_2,
//                       std::placeholders::_3, std::placeholders::_4, Z_k_1_,
//                       Z_k_, L_e_),
//             X_k_1, H_k_1, X_k, H_k);
//     *J4 = J;
//   }

//   return residual(X_k_1, H_k_1, X_k, H_k, Z_k_1_, Z_k_, L_e_);
// }

// void StructurelessDecoupledFormulation::dynamicPointUpdateCallback(
//     const PointUpdateContextType& context, UpdateObservationResult& result,
//     gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors) {
//   const auto lmk_node = context.lmk_node;
//   const auto frame_node_k_1 = context.frame_node_k_1;
//   const auto frame_node_k = context.frame_node_k;

//   gtsam::Key point_key = this->makeDynamicKey(context.getTrackletId());

//   const gtsam::Key object_motion_key_k =
//       frame_node_k->makeObjectMotionKey(context.getObjectId());
//   const gtsam::Key object_motion_key_k_1 =
//       frame_node_k_1->makeObjectMotionKey(context.getObjectId());
//   auto landmark_motion_noise = noise_models_.landmark_motion_noise;

//   gtsam::Pose3 X_K_1 =
//       this->getInitialOrLinearizedSensorPose(frame_node_k_1->frame_id);
//   gtsam::Pose3 X_K =
//       this->getInitialOrLinearizedSensorPose(frame_node_k->frame_id);

//   gtsam::Pose3 L_e;
//   FrameId s0;
//   std::tie(s0, L_e) =
//       getOrConstructL0(context.getObjectId(), frame_node_k_1->getId());
//   auto dynamic_point_noise = noise_models_.dynamic_point_noise;

//   new_factors.emplace_shared<StructurelessDecoupledObjectCentricMotion>(
//       object_motion_key_k_1, object_motion_key_k, X_K_1, X_K,
//       lmk_node->getMeasurement(frame_node_k_1).landmark,
//       lmk_node->getMeasurement(frame_node_k).landmark, L_e,
//       dynamic_point_noise);

//   result.updateAffectedObject(frame_node_k_1->frame_id,
//   context.getObjectId()); result.updateAffectedObject(frame_node_k->frame_id,
//   context.getObjectId());
// }

// void DecoupledFormulation::dynamicPointUpdateCallback(
//     const PointUpdateContextType& context, UpdateObservationResult& result,
//     gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors) {
//   const auto lmk_node = context.lmk_node;
//   const auto frame_node_k_1 = context.frame_node_k_1;
//   const auto frame_node_k = context.frame_node_k;

//   auto theta_accessor = this->accessorFromTheta();

//   gtsam::Key point_key = this->makeDynamicKey(context.getTrackletId());

//   const gtsam::Key object_motion_key_k =
//       frame_node_k->makeObjectMotionKey(context.getObjectId());
//   const gtsam::Key object_motion_key_k_1 =
//       frame_node_k_1->makeObjectMotionKey(context.getObjectId());

//   gtsam::Pose3 L_e;
//   FrameId s0;
//   std::tie(s0, L_e) =
//       getOrConstructL0(context.getObjectId(), frame_node_k_1->getId());
//   auto landmark_motion_noise = noise_models_.landmark_motion_noise;

//   // TODO:this will not be the case with sliding/window as we reconstruct the
//   // graph from a different starting point!!
//   //  CHECK_GE(frame_node_k_1->getId(), s0);

//   if (!isDynamicTrackletInMap(lmk_node)) {
//     // mark as now in map
//     is_dynamic_tracklet_in_map_.insert2(context.getTrackletId(), true);
//     CHECK(isDynamicTrackletInMap(lmk_node));

//     // use first point as initalisation?
//     // in this case k is k-1 as we use frame_node_k_1
//     gtsam::Pose3 e_H_k_world =
//         computeInitialH(context.getObjectId(), frame_node_k_1->getId());
//     gtsam::Pose3 L_k = e_H_k_world * L_e;
//     // H from k to s0 in frame k (^wL_k)
//     //  gtsam::Pose3 k_H_s0_k = L_e * e_H_k_world.inverse() *  L_e.inverse();
//     gtsam::Pose3 k_H_s0_k = (L_e.inverse() * e_H_k_world * L_e).inverse();
//     gtsam::Pose3 k_H_s0_W = L_k * k_H_s0_k * L_k.inverse();
//     // LOG(INFO) << "e_H_k_world " << e_H_k_world;
//     // measured point in camera frame
//     const gtsam::Point3 m_camera =
//         lmk_node->getMeasurement(frame_node_k_1).landmark;
//     Landmark lmk_L0_init =
//         L_e.inverse() * k_H_s0_W * context.X_k_1_measured * m_camera;

//     // initalise value //cannot initalise again the same -> it depends where
//     L_e
//     // is created, no?
//     Landmark lmk_L0;
//     getSafeQuery(lmk_L0, theta_accessor->query<Landmark>(point_key),
//                  lmk_L0_init);
//     new_values.insert(point_key, lmk_L0);
//     result.updateAffectedObject(frame_node_k_1->frame_id,
//                                 context.getObjectId());
//   }

//   auto dynamic_point_noise = noise_models_.dynamic_point_noise;
//   if (context.is_starting_motion_frame) {
//     gtsam::Pose3 X_k_1 =
//         this->getInitialOrLinearizedSensorPose(frame_node_k_1->frame_id);
//     new_factors.emplace_shared<DecoupledObjectCentricMotionFactor>(
//         object_motion_key_k_1, point_key,
//         lmk_node->getMeasurement(frame_node_k_1).landmark, L_e, X_k_1,
//         dynamic_point_noise);
//     result.updateAffectedObject(frame_node_k_1->frame_id,
//                                 context.getObjectId());
//   }

//   gtsam::Pose3 X_k =
//       this->getInitialOrLinearizedSensorPose(frame_node_k->frame_id);

//   new_factors.emplace_shared<DecoupledObjectCentricMotionFactor>(
//       object_motion_key_k, point_key,
//       lmk_node->getMeasurement(frame_node_k).landmark, L_e, X_k,
//       dynamic_point_noise);
//   result.updateAffectedObject(frame_node_k->frame_id, context.getObjectId());
// }

// void StructurlessFormulation::dynamicPointUpdateCallback(
//     const PointUpdateContextType& context, UpdateObservationResult& result,
//     gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors) {
//   const auto lmk_node = context.lmk_node;
//   const auto frame_node_k_1 = context.frame_node_k_1;
//   const auto frame_node_k = context.frame_node_k;

//   gtsam::Key point_key = this->makeDynamicKey(context.getTrackletId());

//   const gtsam::Key object_motion_key_k =
//       frame_node_k->makeObjectMotionKey(context.getObjectId());
//   const gtsam::Key object_motion_key_k_1 =
//       frame_node_k_1->makeObjectMotionKey(context.getObjectId());
//   auto landmark_motion_noise = noise_models_.landmark_motion_noise;

//   gtsam::Pose3 L_e;
//   FrameId s0;
//   std::tie(s0, L_e) =
//       getOrConstructL0(context.getObjectId(), frame_node_k_1->getId());
//   auto dynamic_point_noise = noise_models_.dynamic_point_noise;

//   new_factors.emplace_shared<StructurelessObjectCentricMotionFactor2>(
//       frame_node_k_1->makePoseKey(), object_motion_key_k_1,
//       frame_node_k->makePoseKey(), object_motion_key_k,
//       lmk_node->getMeasurement(frame_node_k_1).landmark,
//       lmk_node->getMeasurement(frame_node_k).landmark, L_e,
//       dynamic_point_noise);

//   result.updateAffectedObject(frame_node_k_1->frame_id,
//   context.getObjectId()); result.updateAffectedObject(frame_node_k->frame_id,
//   context.getObjectId());
// }

StateQuery<gtsam::Point3> SmartStructurlessAccessor::queryPoint(
    gtsam::Key point_key, TrackletId tracklet_id) const {
  if (smart_factor_map_->exists(tracklet_id)) {
    HybridSmartFactor::shared_ptr smart_factor =
        smart_factor_map_->at(tracklet_id).first;

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

  bool is_smart_factor_new = false;

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
        MeasurementTraits::point(lmk_node->getMeasurement(frame_node_k_1)));

    // TODO: this should not every be true as this is a new value!!!
    Landmark lmk_L0;
    getSafeQuery(lmk_L0, theta_accessor->query<Landmark>(point_key),
                 lmk_L0_init);

    SmartMotionFactorParams smart_factor_params;
    smart_factor_params.dyanmic_outlier_rejection_threshold = 1.0;
    HybridSmartFactor::shared_ptr smart_factor =
        boost::make_shared<HybridSmartFactor>(L_e, dynamic_point_noise,
                                              smart_factor_params, lmk_L0_init);
    // HybridSmartFactor::shared_ptr smart_factor =
    //     boost::make_shared<HybridSmartFactor>(L_e, dynamic_point_noise);

    // slot for a new factor MUST be its relative position in the set of new
    // factors so that when we update its new factor index (after incremental
    // update) we can calculate the corresponding position in 1-1
    // newFactorsIndices
    Slot starting_factor_slot =
        context.starting_factor_slot + new_factors.size();
    // LOG(INFO) << "Expecting SMF at slot " <<  starting_factor_slot;

    new_factors.push_back(smart_factor);
    smart_factor_map_.insert2(
        context.getTrackletId(),
        std::make_pair(smart_factor, starting_factor_slot));
    tracklet_ids_of_new_smart_factors_.push_back(context.getTrackletId());

    is_smart_factor_new = true;

    result.updateAffectedObject(frame_node_k_1->frame_id,
                                context.getObjectId());
    if (result.debug_info)
      result.debug_info->getObjectInfo(context.getObjectId())
          .num_new_dynamic_points++;
  }

  // expecting slot to have been updated
  auto [smart_factor, slot] = smart_factor_map_.at(context.getTrackletId());
  CHECK_NOTNULL(smart_factor);

  if (!is_smart_factor_new && !result.isam_update_params.newAffectedKeys) {
    result.isam_update_params.newAffectedKeys =
        gtsam::FastMap<gtsam::FactorIndex, gtsam::KeySet>{};
  }

  gtsam::KeySet affected_keys;

  if (context.is_starting_motion_frame) {
    smart_factor->add(
        MeasurementTraits::point(lmk_node->getMeasurement(frame_node_k_1)),
        object_motion_key_k_1, frame_node_k_1->makePoseKey());
    if (result.debug_info)
      result.debug_info->getObjectInfo(context.getObjectId())
          .num_dynamic_factors++;

    affected_keys.insert(object_motion_key_k_1);
    affected_keys.insert(frame_node_k_1->makePoseKey());
  }

  smart_factor->add(
      MeasurementTraits::point(lmk_node->getMeasurement(frame_node_k)),
      object_motion_key_k, frame_node_k->makePoseKey());

  affected_keys.insert(object_motion_key_k);
  affected_keys.insert(frame_node_k->makePoseKey());

  // if this factor is old alert the smoother to new updates
  // its slot in the smoother should be updated in the postUpdate hook
  if (!is_smart_factor_new) {
    result.isam_update_params.newAffectedKeys->insert2(slot, affected_keys);
  }

  result.updateAffectedObject(frame_node_k->frame_id, context.getObjectId());
  if (result.debug_info)
    result.debug_info->getObjectInfo(context.getObjectId())
        .num_dynamic_factors++;
}

void SmartStructurlessFormulation::postUpdate(const PostUpdateData& data) {
  // call base class first
  Base::postUpdate(data);

  // now take the opportunity to update all points
  const gtsam::Values& estimate = this->getTheta();

  std::string file_name =
      getOutputFilePath("smf_stats_" + this->getFullyQualifiedName() + ".csv");

  static bool is_first = true;

  if (is_first) {
    // clear the file first
    std::ofstream clear_file(file_name, std::ios::out | std::ios::trunc);
    if (!clear_file.is_open()) {
      LOG(FATAL) << "Error clearing file: " << file_name;
    }
    clear_file.close();  // Close the stream to ensure truncation is complete
    is_first = false;

    std::ofstream header_file(file_name, std::ios::out | std::ios::trunc);
    if (!header_file.is_open()) {
      LOG(FATAL) << "Error writing file header file: " << file_name;
    }

    header_file << "tracklet_id,frame_id,reprojection_error,is_good\n";

    header_file.close();  // Close the stream to ensure truncation is complete
    is_first = false;
  }

  std::fstream file(file_name, std::ios::in | std::ios::out | std::ios::app);
  file.precision(15);

  for (auto& [tracklet_id, shf_pair] : smart_factor_map_) {
    HybridSmartFactor::shared_ptr smart_factor = shf_pair.first;
    // TODO: would be nice to take the reprojection error before and after but
    // current implementation will update the internal result_ variable with any
    // call to totalReprojectionError
    auto result = smart_factor->point(estimate);
    double error = smart_factor->reprojectionError(estimate).norm();

    file << tracklet_id << "," << data.frame_id << "," << error << ","
         << result.valid() << "\n";
  }

  file.close();

  // if incremental, update factor slots
  if (data.incremental_result) {
    const auto& incremental_result = data.incremental_result.value();
    const auto isam2_result = incremental_result.isam2;
    const auto isam2_factors = incremental_result.factors;

    VLOG(15) << "Received incremental result. Updating slots for "
             << tracklet_ids_of_new_smart_factors_.size() << " smart factors";
    for (size_t i = 0u; i < tracklet_ids_of_new_smart_factors_.size(); ++i) {
      DCHECK(i < isam2_result.newFactorsIndices.size())
          << "There are more new smart factors than new factors added to the "
             "graph.";

      TrackletId tracklet_id_of_smart_factor =
          tracklet_ids_of_new_smart_factors_.at(i);
      const auto& it = smart_factor_map_.find(tracklet_id_of_smart_factor);
      CHECK(it != smart_factor_map_.end())
          << "Trying to access unavailable factor.";

      CHECK_EQ(it->first, tracklet_id_of_smart_factor);

      // calculate relative position of this factor in the set of new factors
      // that was added to the smoother
      const auto& starting_slot = it->second.second;
      // Get new slot in the graph for the newly added smart factor.
      const size_t& slot = isam2_result.newFactorsIndices.at(starting_slot);

      const auto shptr =
          dynamic_cast<const HybridSmartFactor*>(isam2_factors.at(slot).get());

      // isam2_factors.at(slot)->print("Not shf: ", DynoLikeKeyFormatter);
      // it->second.first->print("shf: ", DynoLikeKeyFormatter);

      CHECK(shptr);
      // check the factors are the same!!!
      if (shptr != it->second.first.get()) {
        isam2_factors.at(slot)->print("Not shf: ", DynoLikeKeyFormatter);
        it->second.first->print("shf: ", DynoLikeKeyFormatter);
        CHECK_EQ(shptr, it->second.first.get());
      }

      // update slot number
      it->second.second = slot;
    }

    // only clear if incremental update...?
    tracklet_ids_of_new_smart_factors_.clear();
  }
}

}  // namespace test_hybrid
}  // namespace dyno
