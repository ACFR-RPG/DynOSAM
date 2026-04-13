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

#include <variant>

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/BackendFormulationFactory.hpp"
#include "dynosam/backend/BackendModule.hpp"
#include "dynosam/backend/Formulation.hpp"
#include "dynosam/formulations/VIOFormulation.hpp"
#include "dynosam_common/Flags.hpp"
#include "dynosam_opt/ISAM2.hpp"
#include "dynosam_opt/IncrementalOptimization.hpp"
#include "dynosam_opt/Map.hpp"
#include "dynosam_opt/SlidingWindowOptimization.hpp"

namespace dyno {

class RegularVIBackendModule
    : public BackendModuleT<MapVision, VisionImuPacket> {
 public:
  //! Alias needed in BackendFactory
  using Factory = BackendFormulationFactory<MapVision>;

  using Base = BackendModuleT<MapVision, VisionImuPacket>;
  DYNO_POINTER_TYPEDEFS(RegularVIBackendModule)

  RegularVIBackendModule(const BackendParams& backend_params,
                         Camera::Ptr camera, std::shared_ptr<Factory> factory,
                         const SharedGroundTruth& shared_ground_truth = {});

  /**
   * @brief Construct a new Regular V I Backend Module object A secondary
   * constructor the RegularBackend that does not take an explicit factory but
   * instead just the type of formulation to be used. In this case the
   * DefaultBackendFactory will be used which has no special behaviour.
   *
   * This constructor is mostly used for unit-tests
   *
   * @param backend_params const BackendParams&
   * @param camera Camera::Ptr
   * @param backend_type const BackendType&
   */
  RegularVIBackendModule(const BackendParams& backend_params,
                         Camera::Ptr camera, const BackendType& backend_type,
                         const SharedGroundTruth& shared_ground_truth = {});

  ~RegularVIBackendModule();

  std::pair<gtsam::Values, gtsam::NonlinearFactorGraph> getActiveOptimisation()
      const override;

  Accessor::Ptr getAccessor() const override;
  const VIOFormulation<MapVision>::Ptr formulation() const;
  BackendModuleDisplay::Ptr formulationDisplay() const;

 protected:
  using SpinReturn = Base::SpinReturn;
  SpinReturn boostrapSpin(VisionImuPacket::ConstPtr input) override;
  SpinReturn nominalSpin(VisionImuPacket::ConstPtr input) override;

 private:
  void setupOptimizers();
  void setupFormulation(std::shared_ptr<Factory> factor);

  void addInitialStates(const VisionImuPacket::ConstPtr& input,
                        gtsam::Values& new_values,
                        gtsam::NonlinearFactorGraph& new_factors);
  void addStates(const VisionImuPacket::ConstPtr& input,
                 gtsam::Values& new_values,
                 gtsam::NonlinearFactorGraph& new_factors);

  /**
   * @brief Construct factors and new values for static and dynamic features.
   * Does the bulk of the graph construction by calling
   * Formulation::updateStaticObservations and
   * Formulation::updateDynamicObservations.
   *
   * @param update_params const UpdateObservationParams&
   * @param frame_k FrameId
   * @param new_values gtsam::Values&
   * @param new_factors gtsam::NonlinearFactorGraph&
   * @param post_update_data PostUpdateData&
   */
  virtual void addMeasurements(const UpdateObservationParams& update_params,
                               FrameId frame_k, gtsam::Values& new_values,
                               gtsam::NonlinearFactorGraph& new_factors,
                               PostUpdateData& post_update_data);

  // initial pose can come from many sources
  void updateMapWithMeasurements(FrameId frame_id_k,
                                 const VisionImuPacket::ConstPtr& input,
                                 const gtsam::Pose3& X_W_k);

  void updateAndOptimize(FrameId frame_id_k, const gtsam::Values& new_values,
                         const gtsam::NonlinearFactorGraph& new_factors,
                         PostUpdateData& post_update_data);
  void updateIncremental(FrameId frame_id_k, const gtsam::Values& new_values,
                         const gtsam::NonlinearFactorGraph& new_factors,
                         PostUpdateData& post_update_data);
  void updateBatch(FrameId frame_id_k, const gtsam::Values& new_values,
                   const gtsam::NonlinearFactorGraph& new_factors,
                   PostUpdateData& post_update_data);
  void updateSlidingWindow(FrameId frame_id_k, const gtsam::Values& new_values,
                           const gtsam::NonlinearFactorGraph& new_factors,
                           PostUpdateData& post_update_data);

  void logIncrementalStats(
      FrameId frame_id_k,
      const IncrementalInterface<dyno::ISAM2>& smoother_interface) const;

 private:
  VIOFormulation<MapVision>::Ptr formulation_;
  BackendModuleDisplay::Ptr formulation_display_;

  // Cached debug info
  DebugInfo debug_info_;
  // Cached error hooks
  ErrorHandlingHooks error_hooks_;

  // optimizers are set in setupUpdates() depending on
  SlidingWindowOptimization::UniquePtr sliding_window_opt_;
  std::unique_ptr<dyno::ISAM2> smoother_;
};

}  // namespace dyno
