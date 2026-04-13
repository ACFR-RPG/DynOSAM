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

#include <memory>

#include "dynosam/backend/BackendFormulationFactory.hpp"
#include "dynosam/backend/BackendModule.hpp"
#include "dynosam/backend/BackendModuleFactory.hpp"
#include "dynosam/backend/BackendParams.hpp"
#include "dynosam/backend/Formulation.hpp"
#include "dynosam/backend/ParallelHybridBackendModule.hpp"
#include "dynosam/backend/PoseChangeBackendModule.hpp"
#include "dynosam/backend/RegularBackendModule.hpp"
#include "dynosam/formulations/HybridEstimator.hpp"
#include "dynosam/formulations/WorldMotionEstimator.hpp"
#include "dynosam/formulations/WorldPoseEstimator.hpp"
#include "dynosam/visualizer/VisualizerPipelines.hpp"  //for BackendModuleDisplay
#include "dynosam_opt/IncrementalOptimization.hpp"     // for ErrorHandlingHooks

namespace dyno {

class IncorrectBackendModuleConstruction : public DynosamException {
 public:
  IncorrectBackendModuleConstruction(const std::string& what)
      : DynosamException(what) {}
};

/**
 * @brief Factory to create the backend module and associated formulations.
 * This class is quite complex due to the interdepencies between backend and
 * formulations.
 *
 * In general a module refers to some derived BackendModule which represents the
 * whole backend (e.g. RegularBackendModule, but also
 * ParallelHybridBackendModule which is a special backend) while a formulation
 * is derived from Formulation<MAP>.
 *
 * We also template the BackendFactory on a Policy class which must implement
 * createDisplay<T>(std::shared_ptr<T> module) ->
 * std::shared_ptr<BackendModuleDisplayRos> where T is a Formulation (except in
 * the Parallel-Hybrid case where it is the BackendModule itself) but can be
 * anything loaded by the BackendFactory. This allows module/formulation
 * specific displays to be written independantly from the class and injected
 * into the loader. If non null, this display will be called once per iteration
 * after the backend has spun.
 *
 * @tparam Policy
 * @tparam MAP
 */
template <typename Policy>
class BackendFactory
    : public Policy,
      public BackendModuleFactory,
      public std::enable_shared_from_this<BackendFactory<Policy>> {
  struct PrivateBackendType {
    const BackendType backend_type;
    PrivateBackendType(const BackendType& type) : backend_type(type) {}
  };

 private:
  //! Selection of which module/formulation to load
  const BackendType backend_type_;

 public:
  using This = BackendFactory<Policy>;
  DYNO_POINTER_TYPEDEFS(This)

  BackendFactory(const PrivateBackendType& p_type, const Policy& policy)
      : backend_type_(p_type.backend_type), Policy(policy) {}

  template <typename... Args>
  BackendFactory(const PrivateBackendType& p_type, Args&&... args)
      : backend_type_(p_type.backend_type),
        Policy(std::forward<Args>(args)...) {}

  std::shared_ptr<This> getPtr() { return this->shared_from_this(); }

  static std::shared_ptr<This> Create(const BackendType& backend_type,
                                      const Policy& policy) {
    return std::make_shared<This>(PrivateBackendType{backend_type}, policy);
  }

  template <typename... Args>
  static std::shared_ptr<This> Create(const BackendType& backend_type,
                                      Args&&... args) {
    return std::make_shared<This>(PrivateBackendType{backend_type},
                                  std::forward<Args>(args)...);
  }

  virtual ~BackendFactory() = default;

  /**
   * @brief Get the formulation factory (ie. creation of formulations
   * independant of the modules) used for regular formulation construction.
   *
   * The formulation factory contains the desired policy behaviour for displays.
   *
   * This only works if the formulation that is loaded by the factory uses the
   * desired MAP so the using module should know the map type for each
   * formulation.
   *
   *
   * @tparam MAP
   * @return std::shared_ptr<BackendFormulationFactory<MAP>>
   */
  template <typename MAP>
  std::shared_ptr<BackendFormulationFactory<MAP>> asFormulationFactory() {
    std::shared_ptr<Policy> shared_policy =
        std::dynamic_pointer_cast<Policy>(this->getPtr());
    CHECK_NOTNULL(shared_policy);

    using ImplFactory = FormulationFactoryImpl<MAP>;
    return std::make_shared<ImplFactory>(this->backend_type_, shared_policy);
  }

  BackendWrapper createModule(const ModuleParams& params) override {
    BackendWrapper wrapper;

    if (this->backend_type_ == BackendType::PARALLEL_HYBRID) {
      std::shared_ptr<ParallelHybridBackendModule> backend =
          std::make_shared<ParallelHybridBackendModule>(
              params.backend_params, params.sensors.camera,
              params.shared_ground_truth);

      wrapper.backend = backend;
      // Parallel Hybrid is a special case where we have a vizualiser over
      // the whole module
      wrapper.backend_viz = this->createDisplay(backend);
      VLOG(20) << "Creating ParallelHybridBackendModule "
               << (wrapper.backend_viz ? " with additional display"
                                       : " without additional display");

    } else if (this->backend_type_ == BackendType::KF_HYBRID) {
      FormulationParams formulation_params = params.backend_params;

      FormulationHooks hooks;
      hooks.setGroundTruthPacketRequest(params.shared_ground_truth);

      NoiseModels noise_models =
          NoiseModels::fromBackendParams(params.backend_params);

      std::shared_ptr<HybridFormulationKeyFrame> formulation =
          std::make_shared<HybridFormulationKeyFrame>(
              formulation_params, HybridFormulationKeyFrame::Map::create(),
              noise_models, params.sensors, hooks);

      std::shared_ptr<PoseChangeVIBackendModule> pose_change_backend =
          std::make_shared<PoseChangeVIBackendModule>(
              params.backend_params, params.sensors.camera, formulation,
              params.shared_ground_truth);

      wrapper.backend = pose_change_backend;
      wrapper.backend_viz = this->createDisplay(pose_change_backend);
      VLOG(20) << "Creating PoseChangeVIBackendModule "
               << (wrapper.backend_viz ? " with additional display"
                                       : " without additional display");
    } else {
      // The factory type (including the map) expected by the regular backend
      // Expect all formulations used by the regular backend to have the same
      // map type!
      using RegularBackendFactory = RegularVIBackendModule::Factory;
      using DesiredMap = RegularBackendFactory::Map;

      std::shared_ptr<RegularBackendFactory> formulation_factory =
          this->asFormulationFactory<DesiredMap>();
      CHECK_NOTNULL(formulation_factory);

      std::shared_ptr<RegularVIBackendModule> backend =
          std::make_shared<RegularVIBackendModule>(
              params.backend_params, params.sensors.camera, formulation_factory,
              params.shared_ground_truth);

      wrapper.backend = backend;

      // the formulation is tighly wrapper in the regular backend module and can
      // be any formulation (while the Parallel Hybrid has to be Hybrid) so we
      // now retrieve the visualiser
      wrapper.backend_viz = backend->formulationDisplay();
      VLOG(20) << "Creating RegularVIBackendModule"
               << (wrapper.backend_viz ? " with additional display"
                                       : " without additional display");
    }
    return wrapper;
  }

 private:
  template <typename MAP>
  class FormulationFactoryImpl : public BackendFormulationFactory<MAP> {
   public:
    FormulationFactoryImpl(BackendType backend_type,
                           std::shared_ptr<Policy> policy)
        : BackendFormulationFactory<MAP>(backend_type), policy_(policy) {}

    // implements the BackendFormulationFactory<MAP> class
    FormulationVizWrapper createFormulation(
        const FormulationParams& formulation_params, std::shared_ptr<MAP> map,
        const NoiseModels& noise_models, const Sensors& sensors,
        const FormulationHooks& formulation_hooks) override {
      FormulationVizWrapper wrapper;

      // TODO: or KF_HYBRDI!!
      if (this->backend_type_ == BackendType::PARALLEL_HYBRID) {
        DYNO_THROW_MSG(IncorrectBackendModuleConstruction)
            << "Cannot construct PARALLEL_HYBRID backend with a call to "
               "BackendFactory::createFormulation. "
               "Use BackendFactory::createModule instead!";
        throw;
      } else if (this->backend_type_ == BackendType::KF_HYBRID) {
        DYNO_THROW_MSG(IncorrectBackendModuleConstruction)
            << "Cannot construct KF_HYBRID backend with a call to "
               "BackendFactory::createFormulation. "
               "Use BackendFactory::createModule instead!";
        throw;
      } else if (this->backend_type_ == BackendType::WCME) {
        LOG(INFO) << "Using WCME";
        std::shared_ptr<WorldMotionFormulation> formulation =
            std::make_shared<WorldMotionFormulation>(formulation_params, map,
                                                     noise_models, sensors,
                                                     formulation_hooks);

        // call polciy function
        wrapper.display = policy_->createDisplay(formulation);
        wrapper.formulation = formulation;

      } else if (this->backend_type_ == BackendType::WCPE) {
        LOG(INFO) << "Using WCPE";
        std::shared_ptr<WorldPoseFormulation> formulation =
            std::make_shared<WorldPoseFormulation>(formulation_params, map,
                                                   noise_models, sensors,
                                                   formulation_hooks);
        // call polciy function
        wrapper.display = policy_->createDisplay(formulation);
        wrapper.formulation = formulation;

      } else if (this->backend_type_ == BackendType::HYBRID) {
        LOG(INFO) << "Using Regular HYBRID";
        std::shared_ptr<RegularHybridFormulation> formulation =
            std::make_shared<RegularHybridFormulation>(formulation_params, map,
                                                       noise_models, sensors,
                                                       formulation_hooks);

        // call polciy function
        wrapper.display = policy_->createDisplay(formulation);
        wrapper.formulation = formulation;

      } else if (this->backend_type_ == BackendType::TESTING_HYBRID_SD) {
        LOG(FATAL) << "Using Hybrid Structureless Decoupled. Warning this is a "
                      "testing only formulation!";
      } else if (this->backend_type_ == BackendType::TESTING_HYBRID_D) {
        LOG(FATAL) << "Using Hybrid Decoupled. Warning this is a testing only "
                      "formulation!";
      } else if (this->backend_type_ == BackendType::TESTING_HYBRID_S) {
        LOG(FATAL)
            << "Using Hybrid Structurless. Warning this is a testing only "
               "formulation!";
      } else if (this->backend_type_ == BackendType::TESTING_HYBRID_SMF) {
        LOG(FATAL)
            << "Using Hybrid Smart Motion Factor. Warning this is a testing "
               "only formulation!";

      } else {
        CHECK(false) << "Not implemented";
        return wrapper;
      }

      return wrapper;
    }

   private:
    std::shared_ptr<Policy> policy_;
  };
};

struct NoVizPolicy {
  template <typename Formulation>
  BackendModuleDisplay::Ptr createDisplay(std::shared_ptr<Formulation>) {
    VLOG(20) << "No display will be created for formulation "
             << type_name<Formulation>();
    return nullptr;
  }
};

/// @brief a BackendFactory with a Policy that creates no additional displays
using DefaultBackendFactory = BackendFactory<NoVizPolicy>;

}  // namespace dyno
