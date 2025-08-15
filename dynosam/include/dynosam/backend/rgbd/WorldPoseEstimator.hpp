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

#pragma once

#include "dynosam/backend/Accessor.hpp"
#include "dynosam/backend/Formulation.hpp"
#include "dynosam/common/Map.hpp"

namespace dyno {

class WorldPoseAccessor : public Accessor<MapVision> {
 public:
  WorldPoseAccessor(const SharedFormulationData& shared_data,
                    MapVision::Ptr map)
      : Accessor<MapVision>(shared_data, map) {}
  virtual ~WorldPoseAccessor() {}

  StateQuery<gtsam::Pose3> getSensorPose(FrameId frame_id) const override;
  virtual StateQuery<gtsam::Pose3> getObjectMotion(
      FrameId frame_id, ObjectId object_id) const override;
  virtual StateQuery<gtsam::Pose3> getObjectPose(
      FrameId frame_id, ObjectId object_id) const override;
  StateQuery<gtsam::Point3> getDynamicLandmark(
      FrameId frame_id, TrackletId tracklet_id) const override;
};

class WorldPoseFormulation : public Formulation<MapVision> {
 public:
  using Base = Formulation<MapVision>;
  using Base::AccessorTypePointer;
  using Base::MapTraitsType;
  using Base::ObjectUpdateContextType;
  using Base::PointUpdateContextType;

  DYNO_POINTER_TYPEDEFS(WorldPoseFormulation)

  WorldPoseFormulation(const FormulationParams& params, typename Map::Ptr map,
                       const NoiseModels& noise_models, const Sensors& sensors,
                       const FormulationHooks& hooks)
      : Base(params, map, noise_models, sensors, hooks) {}
  virtual ~WorldPoseFormulation() {}

  void dynamicPointUpdateCallback(
      const PointUpdateContextType& context, UpdateObservationResult& result,
      gtsam::Values& new_values,
      gtsam::NonlinearFactorGraph& new_factors) override;
  void objectUpdateContext(const ObjectUpdateContextType& context,
                           UpdateObservationResult& result,
                           gtsam::Values& new_values,
                           gtsam::NonlinearFactorGraph& new_factors) override;

  inline bool isDynamicTrackletInMap(
      const typename MapTraitsType::LandmarkNodePtr& lmk_node) const override {
    const TrackletId tracklet_id = lmk_node->tracklet_id;
    return is_dynamic_tracklet_in_map_.exists(tracklet_id);
  }

 protected:
  AccessorTypePointer createAccessor(
      const SharedFormulationData& shared_data) const override {
    return std::make_shared<WorldPoseAccessor>(shared_data, this->map());
  }

  std::string loggerPrefix() const override { return "wcpe"; }

 protected:
  // we need a separate way of tracking if a dynamic tracklet is in the map,
  // since each point is modelled uniquely simply used as an O(1) lookup, the
  // value is not actually used. If the key exists, we assume that the tracklet
  // is in the map
  gtsam::FastMap<TrackletId, bool>
      is_dynamic_tracklet_in_map_;  //! thr set of dynamic points that have been
                                    //! added by this updater. We use a separate
                                    //! map containing the tracklets as the keys
                                    //! are non-unique
};

}  // namespace dyno
