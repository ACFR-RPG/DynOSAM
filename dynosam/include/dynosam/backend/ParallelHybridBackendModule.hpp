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

#pragma once

#include <gtsam/navigation/NavState.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/BackendInputPacket.hpp"
#include "dynosam/backend/BackendModule.hpp"
#include "dynosam/backend/Formulation.hpp"
#include "dynosam/backend/ParallelObjectISAM.hpp"
#include "dynosam/backend/RegularBackendDefinitions.hpp"
#include "dynosam/backend/VisionImuBackendModule.hpp"
#include "dynosam_common/Flags.hpp"
#include "dynosam_opt/Map.hpp"

namespace dyno {

class ParallelHybridBackendModule;

/**
 * @brief Special accessor class to access all values via a single interface
 * since the optimisation of the static scene and dynamic objects are decoupled
 * and therefore handled by their own formulation
 *
 */
class ParallelHybridAccessor : public Accessor {
 public:
  DYNO_POINTER_TYPEDEFS(ParallelHybridAccessor)

  ParallelHybridAccessor(ParallelHybridBackendModule* parallel_hybrid_module);

  StateQuery<gtsam::Pose3> getSensorPose(FrameId frame_id) const override;

  StateQuery<gtsam::Pose3> getObjectMotion(FrameId frame_id,
                                           ObjectId object_id) const override;

  StateQuery<gtsam::Pose3> getObjectPose(FrameId frame_id,
                                         ObjectId object_id) const override;

  StateQuery<gtsam::Point3> getDynamicLandmark(
      FrameId frame_id, TrackletId tracklet_id) const override;

  StateQuery<gtsam::Point3> getStaticLandmark(
      TrackletId tracklet_id) const override;

  EstimateMap<ObjectId, gtsam::Pose3> getObjectPoses(
      FrameId frame_id) const override;

  MotionEstimateMap getObjectMotions(FrameId frame_id) const override;

  ObjectPoseMap getObjectPoses() const override;

  ObjectMotionMap getObjectMotions() const override;

  StatusLandmarkVector getDynamicLandmarkEstimates(
      FrameId frame_id) const override;

  StatusLandmarkVector getDynamicLandmarkEstimates(
      FrameId frame_id, ObjectId object_id) const override;

  StatusLandmarkVector getStaticLandmarkEstimates(
      FrameId frame_id) const override;

  StatusLandmarkVector getFullStaticMap() const override;
  bool hasObjectMotionEstimate(FrameId frame_id, ObjectId object_id,
                               Motion3* motion) const override;

  bool hasObjectPoseEstimate(FrameId frame_id, ObjectId object_id,
                             gtsam::Pose3* pose) const override;

  gtsam::FastMap<ObjectId, gtsam::Point3> computeObjectCentroids(
      FrameId frame_id) const override;

 protected:
  // Helper function to call a function on an ObjectEstimator if it exists
  //  in the map of sam estimators
  //  otherwise the result of the FallbackFunc will be used (as default)
  template <typename Func, typename FallbackFunc>
  auto withOr(ObjectId object_id, Func&& func, FallbackFunc&& fallback) const;

  ParallelHybridBackendModule* parallel_hybrid_module_;
  HybridAccessor::Ptr static_accessor_;
};

class ParallelHybridBackendModule
    : public VisionImuBackendModule<RegularBackendModuleTraits> {
 public:
  DYNO_POINTER_TYPEDEFS(ParallelHybridBackendModule)

  using Base = VisionImuBackendModule<RegularBackendModuleTraits>;
  using RGBDMap = Base::MapType;

  ParallelHybridBackendModule(const BackendParams& backend_params,
                              Camera::Ptr camera,
                              ImageDisplayQueue* display_queue = nullptr);
  ~ParallelHybridBackendModule();

  void logGraphs();

  const gtsam::FastMap<ObjectId, ParallelObjectISAM::Ptr>& objectEstimators()
      const;
  HybridFormulation::Ptr staticEstimator() const;

  std::pair<gtsam::Values, gtsam::NonlinearFactorGraph> getActiveOptimisation()
      const override;

  Accessor::Ptr getAccessor() override;

 private:
  using SpinReturn = Base::SpinReturn;

  SpinReturn boostrapSpinImpl(VisionImuPacket::ConstPtr input) override;
  SpinReturn nominalSpinImpl(VisionImuPacket::ConstPtr input) override;

  Pose3Measurement bootstrapUpdateStaticEstimator(
      VisionImuPacket::ConstPtr input);
  Pose3Measurement nominalUpdateStaticEstimator(
      VisionImuPacket::ConstPtr input, bool should_calculate_covariance = true);

  ParallelObjectISAM::Ptr getEstimator(ObjectId object_id,
                                       bool* is_object_new = nullptr);

  // SHOULD Returns objects with successful sovle
  void parallelObjectSolve(VisionImuPacket::ConstPtr input,
                           const Pose3Measurement& X_W_k);
  void implSolvePerObject(FrameId frame_id, ObjectId object_id,
                          const VisionImuPacket::ObjectTracks& object_update,
                          const Pose3Measurement& X_W_k);

  BackendOutputPacket::Ptr constructOutputPacket(FrameId frame_k,
                                                 Timestamp timestamp) const;

  void logBackendFromEstimators();
  void updateTrackletMapping(const VisionImuPacket::ConstPtr input);

 private:
  Camera::Ptr camera_;

  mutable std::mutex mutex_;

  gtsam::ISAM2Params static_isam2_params_;
  HybridFormulation::Ptr static_formulation_;
  gtsam::IncrementalFixedLagSmoother static_estimator_;

  gtsam::ISAM2Params dynamic_isam2_params_;
  gtsam::FastMap<ObjectId, ParallelObjectISAM::Ptr> sam_estimators_;

  //! Vector of object ids that are new for this frame. Cleared after each spin
  ObjectIds new_objects_estimators_;

  friend class ParallelHybridAccessor;
  ParallelHybridAccessor::Ptr combined_accessor_;

  //! Fast look up of object ids for each point
  //! Needed to look up the right object estimator for each point
  FastUnorderedMap<TrackletId, ObjectId> tracklet_id_to_object_;

  //! used to cache the result of each update which will we log to file
  GenericObjectCentricMap<ParallelObjectISAM::Result> result_map_;
};

// implement function after ParallelHybridBackendModule has been fully defined
template <typename Func, typename FallbackFunc>
auto ParallelHybridAccessor::withOr(ObjectId object_id, Func&& func,
                                    FallbackFunc&& fallback) const {
  const auto& object_estimators = parallel_hybrid_module_->sam_estimators_;
  auto it = object_estimators.find(object_id);
  if (it != object_estimators.end()) return func(it->second);
  return fallback();
}

}  // namespace dyno
