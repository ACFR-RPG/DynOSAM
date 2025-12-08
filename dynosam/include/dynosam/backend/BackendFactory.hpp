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
#include "dynosam/backend/RegularBackendModule.hpp"
#include "dynosam/backend/rgbd/HybridEstimator.hpp"
#include "dynosam/backend/rgbd/WorldMotionEstimator.hpp"
#include "dynosam/backend/rgbd/WorldPoseEstimator.hpp"
#include "dynosam/backend/rgbd/impl/test_HybridFormulations.hpp"
#include "dynosam/visualizer/VisualizerPipelines.hpp"  //for BackendModuleDisplay
#include "dynosam_opt/IncrementalOptimization.hpp"     // for ErrorHandlingHooks

namespace dyno {

class IncorrectParallelHybridConstruction : public DynosamException {
 public:
  IncorrectParallelHybridConstruction(const std::string& what)
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
template <typename Policy, typename MAP>
class BackendFactory
    : public BackendFormulationFactory<MAP>,
      public Policy,
      public BackendModuleFactory,
      public std::enable_shared_from_this<BackendFactory<Policy, MAP>> {
  struct PrivateBackendType {
    const BackendType backend_type;
    PrivateBackendType(const BackendType& type) : backend_type(type) {}
  };

 public:
  using This = BackendFactory<Policy, MAP>;
  DYNO_POINTER_TYPEDEFS(This)

  BackendFactory(const PrivateBackendType& p_type, const Policy& policy)
      : BackendFormulationFactory<MAP>(p_type.backend_type), Policy(policy) {}

  template <typename... Args>
  BackendFactory(const PrivateBackendType& p_type, Args&&... args)
      : BackendFormulationFactory<MAP>(p_type.backend_type),
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

  BackendWrapper createModule(const ModuleParams& params) override {
    BackendWrapper wrapper;

    if (this->backend_type_ == BackendType::PARALLEL_HYBRID) {
      std::shared_ptr<ParallelHybridBackendModule> backend =
          std::make_shared<ParallelHybridBackendModule>(params.backend_params,
                                                        params.sensors.camera,
                                                        params.display_queue);

      wrapper.backend = backend;
      // Parallel Hybrid is a special case where we have a vizualiser over the
      // whole module
      wrapper.backend_viz = this->createDisplay(backend);
      VLOG(20) << "Creating ParallelHybridBackendModule"
               << (wrapper.backend_viz ? " with additional display"
                                       : " without additional display");

    } else {
      std::shared_ptr<BackendFormulationFactory<MAP>> formulation_factory =
          std::dynamic_pointer_cast<BackendFormulationFactory<MAP>>(
              this->getPtr());

      CHECK_NOTNULL(formulation_factory);

      std::shared_ptr<RegularBackendModule> backend =
          std::make_shared<RegularBackendModule>(
              params.backend_params, params.sensors.camera, formulation_factory,
              params.display_queue);

      wrapper.backend = backend;

      // the formulation is tighly wrapper in the regular backend module and can
      // be any formulation (while the Parallel Hybrid has to be Hybrid) so we
      // now retrieve the visualiser
      wrapper.backend_viz = backend->formulationDisplay();
      VLOG(20) << "Creating RegularBackendModule"
               << (wrapper.backend_viz ? " with additional display"
                                       : " without additional display");
    }
    return wrapper;
  }

 private:
  // implements the BackendFormulationFactory<MAP> class
  FormulationVizWrapper<MAP> createFormulation(
      const FormulationParams& formulation_params, std::shared_ptr<MAP> map,
      const NoiseModels& noise_models, const Sensors& sensors,
      const FormulationHooks& formulation_hooks) override {
    FormulationVizWrapper<MAP> wrapper;

    if (this->backend_type_ == BackendType::PARALLEL_HYBRID) {
      DYNO_THROW_MSG(IncorrectParallelHybridConstruction)
          << "Cannot construct PARALLEL_HYBRID backend with a call to "
             "BackendFactory::createFormulation"
          << " Use BackendFactory::createModule instead!";
      return wrapper;
    } else if (this->backend_type_ == BackendType::WCME) {
      LOG(INFO) << "Using WCME";
      std::shared_ptr<WorldMotionFormulation> formulation =
          std::make_shared<WorldMotionFormulation>(formulation_params, map,
                                                   noise_models, sensors,
                                                   formulation_hooks);

      // call polciy function
      wrapper.display = this->createDisplay(formulation);
      wrapper.formulation = formulation;

    } else if (this->backend_type_ == BackendType::WCPE) {
      LOG(INFO) << "Using WCPE";
      std::shared_ptr<WorldPoseFormulation> formulation =
          std::make_shared<WorldPoseFormulation>(formulation_params, map,
                                                 noise_models, sensors,
                                                 formulation_hooks);
      // call polciy function
      wrapper.display = this->createDisplay(formulation);
      wrapper.formulation = formulation;

    } else if (this->backend_type_ == BackendType::HYBRID) {
      LOG(INFO) << "Using HYBRID";
      std::shared_ptr<RegularHybridFormulation> formulation =
          std::make_shared<RegularHybridFormulation>(formulation_params, map,
                                                     noise_models, sensors,
                                                     formulation_hooks);

      // call polciy function
      wrapper.display = this->createDisplay(formulation);
      wrapper.formulation = formulation;

    } else if (this->backend_type_ == BackendType::TESTING_HYBRID_SD) {
      LOG(FATAL) << "Using Hybrid Structureless Decoupled. Warning this is a "
                    "testing only formulation!";
    } else if (this->backend_type_ == BackendType::TESTING_HYBRID_D) {
      LOG(FATAL) << "Using Hybrid Decoupled. Warning this is a testing only "
                    "formulation!";
    } else if (this->backend_type_ == BackendType::TESTING_HYBRID_S) {
      LOG(FATAL) << "Using Hybrid Structurless. Warning this is a testing only "
                    "formulation!";
    } else if (this->backend_type_ == BackendType::TESTING_HYBRID_SMF) {
      LOG(INFO)
          << "Using Hybrid Smart Motion Factor. Warning this is a testing "
             "only formulation!";
      FormulationParams fp = formulation_params;
      fp.min_dynamic_observations = 1u;
      std::shared_ptr<test_hybrid::SmartStructurlessFormulation> formulation =
          std::make_shared<test_hybrid::SmartStructurlessFormulation>(
              fp, map, noise_models, sensors, formulation_hooks);
      wrapper.display = this->createDisplay(formulation);
      wrapper.formulation = formulation;

    } else {
      CHECK(false) << "Not implemented";
      return wrapper;
    }

    return wrapper;
  }

 private:
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
template <typename MAP>
using DefaultBackendFactory = BackendFactory<NoVizPolicy, MAP>;

/// @brief BackendModuleFactory templated on the correct map type and with the
/// default (NoVizPolicy) policy
using DefaultRegularBackendModuleFactory =
    DefaultBackendFactory<RegularBackendModuleTraits::MapType>;

/// @brief BackendModuleFactory templated on the regular map type but with a
/// templated Policy
template <typename Policy>
using RegularBackendModuleFactory =
    BackendFactory<Policy, RegularBackendModuleTraits::MapType>;

}  // namespace dyno
