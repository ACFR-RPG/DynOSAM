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

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/rgbd/impl/ObjectCentricFormulations.hpp"
#include "dynosam/common/Exceptions.hpp"
#include "dynosam/common/Map.hpp"
#include "dynosam/common/Types.hpp"  //only needed for factors

namespace dyno {

using namespace keyframe_object_centric;

class DecoupledObjectSAM {
 public:
  DYNO_POINTER_TYPEDEFS(DecoupledObjectSAM)

  using Map = DecoupledFormulation::Map;

  template <typename DERIVEDSTATUS>
  using MeasurementStatusVector = Map::MeasurementStatusVector<DERIVEDSTATUS>;

  DecoupledObjectSAM(ObjectId object_id, const gtsam::ISAM2Params& isam_params);

  // what motion representation should this be in? GLOBAL? Do ne need a new
  // repsentation for KF object centric?
  template <typename DERIVEDSTATUS>
  void update(FrameId frame_k,
              const MeasurementStatusVector<DERIVEDSTATUS>& measurements,
              const gtsam::Pose3& X_world_k,
              const Motion3ReferenceFrame& motion_frame) {
    VLOG(5) << "DecoupledObjectSAM::update running for k= " << frame_k
            << ", j= " << object_id_;

    this->updateMap(frame_k, measurements, X_world_k, motion_frame);
    this->updateSmoother(frame_k);
  }

  const gtsam::Values& getEstimate() const { return estimate_; }

  inline Map::Ptr map() { return map_; }

  Motion3ReferenceFrame getFrame2FrameMotion(FrameId frame_id) const;
  Motion3ReferenceFrame getKeyFramedMotion(FrameId frame_id) const;

 private:
  template <typename DERIVEDSTATUS>
  void updateMap(FrameId frame_k,
                 const MeasurementStatusVector<DERIVEDSTATUS>& measurements,
                 const gtsam::Pose3& X_world_k,
                 const Motion3ReferenceFrame& motion_frame) {
    map_->updateObservations(measurements);
    map_->updateSensorPoseMeasurement(frame_k, X_world_k);

    const FrameId to = motion_frame.to();
    if (to != frame_k) {
      throw DynosamException(
          "DecoupledObjectSAM::updateMap failed as the 'to' frame of the "
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

  void updateSmoother(FrameId frame_k);

 private:
  const ObjectId object_id_;
  Map::Ptr map_;
  DecoupledFormulation::Ptr decoupled_formulation_;
  Accessor<Map>::Ptr accessor_;
  std::shared_ptr<gtsam::ISAM2> smoother_;
  gtsam::ISAM2Result result_;
  gtsam::Values estimate_;
  //! style of motion expected to be used as input. Set on the first run and all
  //! motions are expected to then follow the same style
  std::optional<MotionRepresentationStyle> expected_style_;
};

}  // namespace dyno
