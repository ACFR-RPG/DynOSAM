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
#include "dynosam/common/Flags.hpp"
#include "dynosam/common/Map.hpp"

namespace dyno {

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

 private:
  Camera::Ptr camera_;

  mutable std::mutex mutex_;

  gtsam::ISAM2Params static_isam2_params_;
  HybridFormulation::UniquePtr static_formulation_;
  gtsam::IncrementalFixedLagSmoother static_estimator_;

  gtsam::ISAM2Params dynamic_isam2_params_;
  gtsam::FastMap<ObjectId, ParallelObjectISAM::Ptr> sam_estimators_;

  //! Vector of object ids that are new for this frame. Cleared after each spin
  ObjectIds new_objects_estimators_;

  //! used to cache the result of each update which will we log to file
  GenericObjectCentricMap<ParallelObjectISAM::Result> result_map_;
};

}  // namespace dyno
