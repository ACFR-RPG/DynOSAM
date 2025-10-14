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

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/ISAM2Params.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <nlohmann/json.hpp>

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/rgbd/HybridEstimator.hpp"
#include "dynosam_common/Exceptions.hpp"
#include "dynosam_common/Map.hpp"
#include "dynosam_common/Types.hpp"  //only needed for factors
#include "dynosam_opt/IncrementalOptimization.hpp"

namespace dyno {

class ParallelObjectISAM {
 public:
  DYNO_POINTER_TYPEDEFS(ParallelObjectISAM)

  struct Params {
    //! Number additional iSAM updates to run
    int num_optimzie = 2;
    FormulationParams formulation{};
    gtsam::ISAM2Params isam{};
  };

  struct Result {
    //! If smoother is not okay, then all results other than frame id will be
    //! invalid!
    bool was_smoother_ok{false};
    FrameId frame_id{0};
    gtsam::ISAM2Result isam_result;
    FrameIds motions_with_large_change;
    double large_motion_change_delta;
    //! Timing (in ms) for optimisation
    int64_t timing;

    double average_clique_size;
    double max_clique_size;
    size_t num_factors;
    size_t num_variables;

    // Information about marked variables
    size_t num_landmarks_marked = 0;
    size_t num_motions_marked = 0;

    using PointCloud = Eigen::Matrix<double, -1, 3>;
    std::optional<PointCloud> dynamic_map{};

    using VariableStatus = gtsam::ISAM2Result::DetailedResults::VariableStatus;
    gtsam::FastMap<FrameId, VariableStatus> motion_variable_status{};
  };

  using Map = HybridFormulation::Map;

  template <typename DERIVEDSTATUS>
  using MeasurementStatusVector = Map::MeasurementStatusVector<DERIVEDSTATUS>;

  ParallelObjectISAM(const Params& params, ObjectId object_id,
                     const NoiseModels& noise_models, const Sensors& sensors,
                     const FormulationHooks& formulation_hooks);

  // what motion representation should this be in? GLOBAL? Do ne need a new
  // repsentation for KF object centric?
  template <typename DERIVEDSTATUS>
  void update(FrameId frame_k,
              const MeasurementStatusVector<DERIVEDSTATUS>& measurements,
              const Pose3Measurement& X_world_k,
              const Motion3ReferenceFrame& motion_frame,
              bool update_smoother = true) {
    VLOG(5) << "ParallelObjectISAM::update running for k= " << frame_k
            << ", j= " << object_id_;
    // frame id must get updated each time regardless as things outside this
    // class depend on it maybe need finer-grained variables in result struct
    // (e.g last map update / last smoother update)
    result_.frame_id = frame_k;
    result_.was_smoother_ok = false;

    VLOG(50) << "ParallelObjectISAM::updateMap, j=" << object_id_;
    this->updateMap(frame_k, measurements, X_world_k, motion_frame);

    if (!update_smoother) {
      return;
    }

    // updating the smoothing will update the formulation and run
    // update on the optimizer. the internal results_ object is updated
    VLOG(50) << "ParallelObjectISAM::updateSmoother, j=" << object_id_;
    const bool is_smoother_ok = this->updateSmoother(frame_k, X_world_k);

    if (is_smoother_ok) {
      VLOG(50) << "ParallelObjectISAM::updateStates, j=" << object_id_;
      updateStates();
    }
  }

  const gtsam::Values& getEstimate() const {
    return decoupled_formulation_->getTheta();
  }
  const Result& getResult() const { return result_; }

  inline Map::Ptr map() const { return map_; }

  StateQuery<Motion3ReferenceFrame> getFrame2FrameMotion(
      FrameId frame_id) const;
  // this assumes the motion exists for this frame... which it may not, I
  // guess....
  Motion3ReferenceFrame getKeyFramedMotion(FrameId frame_id) const;

  // all frames
  ObjectPoseMap getObjectPoses() const { return accessor_->getObjectPoses(); }
  // all frames
  ObjectMotionMap getFrame2FrameMotions() const;
  ObjectMotionMap getKeyFramedMotions() const;

  // due to the nature of this formulation, this will be the accumulated cloud!!
  StatusLandmarkVector getDynamicLandmarks(FrameId frame_id) const;

  std::pair<FrameId, gtsam::Pose3> insertNewKeyFrame(FrameId frame_id);

  inline const gtsam::ISAM2& getSmoother() const { return *smoother_; }
  inline HybridAccessor::Ptr accessor() const { return accessor_; }

 private:
  template <typename DERIVEDSTATUS>
  void updateMap(FrameId frame_k,
                 const MeasurementStatusVector<DERIVEDSTATUS>& measurements,
                 const Pose3Measurement& X_world_k,
                 const Motion3ReferenceFrame& motion_frame) {
    map_->updateObservations(measurements);
    map_->updateSensorPoseMeasurement(frame_k, X_world_k);
    const FrameId to = motion_frame.to();
    if (to != frame_k) {
      throw DynosamException(
          "ParallelObjectISAM::updateMap failed as the 'to' frame of the "
          "initial motion was not the same as expected frame id");
    }

    // check style of motion is self consistent
    if (!expected_style_) {
      expected_style_ = motion_frame.style();
    } else {
      CHECK_EQ(expected_style_.value(), motion_frame.style());
    }

    // do we want global?
    MotionEstimateMap motion_estimate;
    motion_estimate.insert({object_id_, motion_frame});
    map_->updateObjectMotionMeasurements(frame_k, motion_estimate);
  }

  bool updateSmoother(FrameId frame_k, const Pose3Measurement& X_world_k);

  void updateFormulation(FrameId frame_k, const Pose3Measurement& X_world_k,
                         gtsam::NonlinearFactorGraph& new_factors,
                         gtsam::Values& new_values);

  void updateStates();

  void setupErrorHandlingHooks();

 private:
  const Params params_;
  const ObjectId object_id_;
  Map::Ptr map_;
  HybridFormulation::Ptr decoupled_formulation_;
  HybridAccessor::Ptr accessor_;
  std::shared_ptr<gtsam::ISAM2> smoother_;
  Result result_;
  //! style of motion expected to be used as input. Set on the first run and all
  //! motions are expected to then follow the same style
  std::optional<MotionRepresentationStyle> expected_style_;
  ErrorHandlingHooks error_hooks_;
};

using json = nlohmann::json;

void to_json(json& j, const ParallelObjectISAM::Result& result);

}  // namespace dyno
