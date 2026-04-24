/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/BackendParams.hpp"
#include "dynosam/backend/Formulation.hpp"
#include "dynosam_common/DynoState.hpp"
#include "dynosam_common/Exceptions.hpp"
#include "dynosam_common/ModuleBase.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/SafeCast.hpp"
#include "dynosam_opt/Map.hpp"

namespace dyno {

class Backend {
 public:
  DYNO_POINTER_TYPEDEFS(Backend)

  Backend() {}
  virtual ~Backend() {}
};

template <typename INPUT>
class BackendModule : public ModuleBase<INPUT, DynoState>, public Backend {
 public:
  using Base = ModuleBase<INPUT, DynoState>;
  using Base::SpinReturn;

  using This = BackendModule<INPUT>;
  DYNO_POINTER_TYPEDEFS(This)

  BackendModule(const BackendParams& params, Camera::Ptr camera,
                const SharedGroundTruth& shared_ground_truth)
      : Base("backend"),
        backend_params_(params),
        camera_(CHECK_NOTNULL(camera)),
        noise_models_(NoiseModels::fromBackendParams(params)),
        shared_ground_truth_(shared_ground_truth) {
    if (VLOG_IS_ON(10)) {
      if (shared_ground_truth_.valid()) {
        LOG(INFO) << "Backend initalised with valid ground truth";
      } else {
        LOG(INFO) << "Backend initalised without ground truth";
      }
    }
  }

  virtual ~BackendModule() = default;

  // something about callback here too?
  // and get noise models!
  const BackendParams& getParams() const { return backend_params_; }
  const NoiseModels& getNoiseModels() const { return noise_models_; }

  // Maybe need better name? This should be the lates frame with updates?
  /**
   * @brief Uses the accessor to retruieve the greatest (ie last) frame id
   * available.
   *
   * The accessor will use the map object to retrieve the value
   * and this function assumes the Accessor#getFrameIds() returns frame
   * values as an ordered vector
   *
   * @return FrameId
   */
  FrameId latestFrameId() const {
    return this->getAccessor()->getLatestFrameId();
  }

  /**
   * @brief Get the accessor the the underlying formulation, allowing the
   * optimised values to be directly accessed
   *
   * @return Accessor::Ptr
   */
  virtual Accessor::Ptr getAccessor() const = 0;
  virtual std::pair<gtsam::Values, gtsam::NonlinearFactorGraph>
  getActiveOptimisation() const = 0;

  virtual DynoState::Ptr makeOutput() const {
    Accessor::Ptr accessor = this->getAccessor();

    DynoState::Ptr state = std::make_shared<DynoState>();

    // shoudl really only go up to last propogated frame...
    const auto camera_trajectory = accessor->getCameraTrajectory();

    // expect frame and timestamp to be from the last entry
    const auto last_camera_entry = camera_trajectory.last();
    state->frame_id = last_camera_entry.frame_id;
    state->timestamp = last_camera_entry.timestamp;

    state->camera_trajectory = camera_trajectory;
    state->object_trajectories = accessor->getMultiObjectTrajectories();

    // TODO: should be global!?
    state->local_static_map = accessor->getFullStaticMap();

    // TODO: this wrong for hybrid KF as there may be not motions at this CKF
    // TODO: hack for dynamic objects in KF but may also be okay general
    // solution
    // TODO: actaully still not right...
    for (const auto& [object_id, trajectory_j] : state->object_trajectories) {
      // get the structural position of the object at its last optimized frame
      // not the latest camera frame (ie. state->frame_id)
      state->dynamic_map += accessor->getDynamicLandmarkEstimates(
          trajectory_j.maxFrame(), object_id);
    }
    // state->dynamic_map =
    // accessor->getDynamicLandmarkEstimates(state->frame_id);

    return state;
  }

 private:
  // called in ModuleBase immediately before the spin function is called
  virtual inline void validateInput(
      const typename Base::InputConstPtr& /*input*/) const override {}

 protected:
  // TODO: maybe put in Backend
  const BackendParams backend_params_;
  Camera::Ptr camera_;
  const NoiseModels noise_models_;
  const SharedGroundTruth shared_ground_truth_;
};

// For modules that have a single internal map
// ie. not PH
template <typename MAP, typename INPUT>
class BackendModuleT : public BackendModule<INPUT> {
 public:
  using MapT = MAP;
  using Base = BackendModule<INPUT>;
  using Formulation = FormulationT<MapT>;

  using PostFormulationUpdateCallback = std::function<void(
      const typename Formulation::Ptr&, FrameId, const gtsam::Values&,
      const gtsam::NonlinearFactorGraph&)>;

  BackendModuleT(const BackendParams& params, Camera::Ptr camera,
                 const SharedGroundTruth& shared_ground_truth)
      : Base(params, camera, shared_ground_truth), map_(MapT::create()) {}
  virtual ~BackendModuleT() = default;

  const typename MapT::Ptr map() { return map_; }

  void registerPostFormulationUpdateCallback(
      const PostFormulationUpdateCallback& cb) {
    post_formulation_update_cb_ = cb;
  }

 protected:
  typename MapT::Ptr map_;

  // //! Formulation that will share the map_
  // VIOFormulation<MapT>::Ptr formulation_;
  // BackendModuleDisplay::Ptr formulation_display_;
  // // Cached error hooks
  // ErrorHandlingHooks error_hooks_;
  //! External callback containing formulation data and new values and factors
  PostFormulationUpdateCallback post_formulation_update_cb_;
};

}  // namespace dyno
