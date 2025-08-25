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

#include "dynosam/backend/BackendModule.hpp"
#include "dynosam/backend/BackendParams.hpp"
#include "dynosam/backend/Formulation.hpp"
#include "dynosam/backend/ParallelHybridBackendModule.hpp"
#include "dynosam/backend/RegularBackendModule.hpp"
#include "dynosam/backend/rgbd/HybridEstimator.hpp"
#include "dynosam/backend/rgbd/MPCEstimator.hpp"
#include "dynosam/backend/rgbd/WorldMotionEstimator.hpp"
#include "dynosam/backend/rgbd/WorldPoseEstimator.hpp"
#include "dynosam/backend/rgbd/impl/test_HybridFormulations.hpp"

namespace dyno {

class IncorrectParallelHybridConstruction : public DynosamException {
 public:
  IncorrectParallelHybridConstruction(const std::string& what)
      : DynosamException(what) {}
};

class BackendFactory {
 public:
  DYNO_DELETE_COPY_CONSTRUCTORS(BackendFactory)
  BackendFactory() = delete;
  virtual ~BackendFactory() = default;

  static BackendModule::Ptr createModule(
      const BackendType& backend_type, const BackendParams& backend_params,
      Camera::Ptr camera, ImageDisplayQueue* display_queue = nullptr) {
    if (backend_type == BackendType::PARALLEL_HYBRID) {
      return std::make_shared<ParallelHybridBackendModule>(
          backend_params, camera, display_queue);
    } else {
      return std::make_shared<RegularBackendModule>(
          backend_params, camera, backend_type, display_queue);
    }
  }

  template <typename MAP>
  static typename Formulation<MAP>::UniquePtr createFormulation(
      const BackendType& backend_type,
      const FormulationParams& formulation_params, std::shared_ptr<MAP> map,
      const NoiseModels& noise_models, const Sensors& sensors,
      const FormulationHooks& formulation_hooks) {
    if (backend_type == BackendType::PARALLEL_HYBRID) {
      DYNO_THROW_MSG(IncorrectParallelHybridConstruction)
          << "Cannot construct PARALLEL_HYBRID backend with a call to "
             "BackendFactory::createFormulation"
          << " Use BackendFactory::createModule instead!";
      return nullptr;
    } else if (backend_type == BackendType::WCME) {
      LOG(INFO) << "Using WCME";
      return std::make_unique<WorldMotionFormulation>(
          formulation_params, map, noise_models, sensors, formulation_hooks);

    } else if (backend_type == BackendType::WCPE) {
      LOG(INFO) << "Using WCPE";
      return std::make_unique<WorldPoseFormulation>(
          formulation_params, map, noise_models, sensors, formulation_hooks);
    } else if (backend_type == BackendType::HYBRID) {
      LOG(INFO) << "Using HYBRID";
      return std::make_unique<RegularHybridFormulation>(
          formulation_params, map, noise_models, sensors, formulation_hooks);
    } else if (backend_type == BackendType::TESTING_HYBRID_SD) {
      LOG(FATAL) << "Using Hybrid Structureless Decoupled. Warning this is a "
                    "testing only formulation!";
    } else if (backend_type == BackendType::TESTING_HYBRID_D) {
      LOG(FATAL) << "Using Hybrid Decoupled. Warning this is a testing only "
                    "formulation!";
    } else if (backend_type == BackendType::TESTING_HYBRID_S) {
      LOG(FATAL) << "Using Hybrid Structurless. Warning this is a testing only "
                    "formulation!";
    } else if (backend_type == BackendType::TESTING_HYBRID_SMF) {
      LOG(INFO)
          << "Using Hybrid Smart Motion Factor. Warning this is a testing "
             "only formulation!";
      FormulationParams fp = formulation_params;
      fp.min_dynamic_observations = 1u;
      return std::make_unique<test_hybrid::SmartStructurlessFormulation>(
          fp, map, noise_models, sensors, formulation_hooks);
    } else if (backend_type == BackendType::MPC_ESTIMATOR) {
      LOG(INFO) << "Using MPC-Estimation";
      return std::make_unique<MPCFormulation>(
          formulation_params, map, noise_models, sensors, formulation_hooks);
    }

    else {
      CHECK(false) << "Not implemented";
      return nullptr;
    }
  }
};

}  // namespace dyno
