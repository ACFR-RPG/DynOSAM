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
#include "dynosam/backend/BackendInputPacket.hpp"
#include "dynosam/backend/BackendModule.hpp"
#include "dynosam/backend/Formulation.hpp"
#include "dynosam/backend/RegularBackendDefinitions.hpp"
#include "dynosam/backend/VisionImuBackendModule.hpp"
#include "dynosam/visualizer/Visualizer-Definitions.hpp"  //for ImageDisplayQueueOptional,
#include "dynosam_common/Flags.hpp"
#include "dynosam_opt/ISAM2.hpp"
#include "dynosam_opt/IncrementalOptimization.hpp"
#include "dynosam_opt/Map.hpp"
#include "dynosam_opt/SlidingWindowOptimization.hpp"

namespace dyno {

class RegularBackendModule
    : public VisionImuBackendModule<RegularBackendModuleTraits> {
 public:
  DYNO_POINTER_TYPEDEFS(RegularBackendModule)

  using Base = VisionImuBackendModule<RegularBackendModuleTraits>;
  using RGBDMap = Base::MapType;
  using FormulationType = Base::FormulationType;

  RegularBackendModule(const BackendParams& backend_params, Camera::Ptr camera,
                       std::shared_ptr<RegularFormulationFactory> factory,
                       ImageDisplayQueue* display_queue = nullptr);

  /**
   * @brief A secondary constructor the RegularBackend that does not take an
   * explicit factory but instead just the type of formulation to be used. In
   * this case the DefaultBackendFactory will be used which has no special
   * behaviour.
   *
   * This constructor is mostly used for unit-tests
   * @param backend_params
   * @param camera
   * @param display_queue
   */
  RegularBackendModule(const BackendParams& backend_params, Camera::Ptr camera,
                       const BackendType& backend_type,
                       ImageDisplayQueue* display_queue = nullptr);

  ~RegularBackendModule();

  using SpinReturn = Base::SpinReturn;

  // const FormulationType* formulation() const { return formulation_.get(); }

  // also provide non-const access (this should only be used with caution and is
  // really only there to enable specific unit-tests!)
  const typename FormulationType::Ptr formulation() const {
    return formulation_;
  }
  BackendModuleDisplay::Ptr formulationDisplay() const {
    return formulation_display_;
  }

  using PostFormulationUpdateCallback = std::function<void(
      const Formulation<RGBDMap>::Ptr&, FrameId, const gtsam::Values&,
      const gtsam::NonlinearFactorGraph&)>;
  void registerPostFormulationUpdateCallback(
      const PostFormulationUpdateCallback& cb) {
    post_formulation_update_cb_ = cb;
  }

  std::pair<gtsam::Values, gtsam::NonlinearFactorGraph> getActiveOptimisation()
      const override;

  Accessor::Ptr getAccessor() override;

 protected:
  void setupUpdates();

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

 protected:
  SpinReturn boostrapSpinImpl(VisionImuPacket::ConstPtr input) override;
  SpinReturn nominalSpinImpl(VisionImuPacket::ConstPtr input) override;

  void addInitialStates(const VisionImuPacket::ConstPtr& input,
                        FormulationType* formulation, gtsam::Values& new_values,
                        gtsam::NonlinearFactorGraph& new_factors);
  void addStates(const VisionImuPacket::ConstPtr& input,
                 FormulationType* formulation, gtsam::Values& new_values,
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
                                 const gtsam::Pose3& X_k_w);

 private:
  // Also sets up error hooks based on the formulation
  void setFormulation(std::shared_ptr<RegularFormulationFactory> factory);

  BackendMetaData createBackendMetadata() const;
  FormulationHooks createFormulationHooks() const;
  BackendOutputPacket::Ptr constructOutputPacket(FrameId frame_k,
                                                 Timestamp timestamp) const;

  Camera::Ptr camera_;
  Formulation<RGBDMap>::Ptr formulation_;
  BackendModuleDisplay::Ptr formulation_display_;

  // new calibration every time
  inline auto getGtsamCalibration() const {
    const CameraParams& camera_params = camera_->getParams();
    return boost::make_shared<Camera::CalibrationType>(
        camera_params.constructGtsamCalibration<Camera::CalibrationType>());
  }

  // logger here!!
  BackendLogger::UniquePtr logger_{nullptr};
  DebugInfo debug_info_;
  ErrorHandlingHooks error_hooks_;

  // optimizers are set in setupUpdates() depending on
  SlidingWindowOptimization::UniquePtr sliding_window_opt_;
  std::unique_ptr<dyno::ISAM2> smoother_;

  //! External callback containing formulation data and new values and factors
  PostFormulationUpdateCallback post_formulation_update_cb_;
};

}  // namespace dyno
