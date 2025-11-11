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
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include "dynosam/backend/ParallelHybridBackendModule.hpp"
#include "dynosam/backend/RegularBackendModule.hpp"

namespace dyno_testing {

using namespace dyno;

struct TesterBase {
  virtual ~TesterBase() = default;

  virtual BackendModule::Ptr getBackend() = 0;
  virtual void onFinish() = 0;
  virtual void preSpin() {}
  virtual void postSpin() {}
};

struct RegularBackendTester : public TesterBase {
  RegularBackendTester(dyno::RegularBackendModule::Ptr backend_)
      : backend(backend_) {}

  BackendModule::Ptr getBackend() override { return backend; }

  dyno::RegularBackendModule::Ptr backend;
};

struct ParallelHybridBackendTester : public TesterBase {
  ParallelHybridBackendTester(dyno::ParallelHybridBackendModule::Ptr backend_)
      : backend(backend_) {}

  BackendModule::Ptr getBackend() override { return backend; }

  virtual void postSpin() { CHECK_NOTNULL(backend)->logGraphs(); }

  void onFinish() override {}

  dyno::ParallelHybridBackendModule::Ptr backend;
};

struct IncrementalTester : public RegularBackendTester {
  struct Data {
    dyno::RegularBackendModule::Ptr backend;
    std::shared_ptr<gtsam::ISAM2> isam2;
    gtsam::Values opt_values;
  };

  IncrementalTester(dyno::RegularBackendModule::Ptr backend_)
      : RegularBackendTester(backend_) {
    data = std::make_shared<Data>();
    gtsam::ISAM2Params isam2_params;
    isam2_params.evaluateNonlinearError = true;
    isam2_params.factorization = gtsam::ISAM2Params::Factorization::CHOLESKY;
    data->isam2 = std::make_shared<gtsam::ISAM2>(isam2_params);

    backend->registerPostFormulationUpdateCallback(
        [&](const RegularBackendModule::FormulationType::Ptr& formulation,
            dyno::FrameId frame_id, const gtsam::Values& new_values,
            const gtsam::NonlinearFactorGraph& new_factors) -> void {
          LOG(INFO) << "Running isam2 update " << frame_id
                    << " for formulation "
                    << formulation->getFullyQualifiedName();
          CHECK_NOTNULL(data);
          CHECK_NOTNULL(data->isam2);
          auto isam = data->isam2;
          gtsam::ISAM2Result result;
          {
            dyno::utils::TimingStatsCollector timer(
                "isam2_oc_test_update." + formulation->getFullyQualifiedName());
            result = isam->update(new_factors, new_values);
          }

          LOG(INFO) << "ISAM2 result. Error before " << result.getErrorBefore()
                    << " error after " << result.getErrorAfter();
          data->opt_values = isam->calculateEstimate();

          isam->getFactorsUnsafe().saveGraph(
              dyno::getOutputFilePath(
                  "isam_graph_" + std::to_string(frame_id) + "_" +
                  formulation->getFullyQualifiedName() + ".dot"),
              dyno::DynosamKeyFormatter);

          if (!isam->empty()) {
            dyno::factor_graph_tools::saveBayesTree(
                *isam,
                dyno::getOutputFilePath(
                    "bayes_tree_" + std::to_string(frame_id) + "_" +
                    formulation->getFullyQualifiedName() + ".dot"),
                dyno::DynosamKeyFormatter);
          }
        });

    data->backend = backend;
  }

  void onFinish() override {
    auto backend = data->backend;
    dyno::BackendMetaData backend_info;
    backend_info.backend_params = &backend->getParams();

    dyno::PostUpdateData post_update(backend->getSpinState().frame_id);
    backend->formulation()->postUpdate(post_update);
    backend->formulation()->logBackendFromMap(backend_info);

    backend_info.logging_suffix = "isam_opt";
    backend->formulation()->updateTheta(data->opt_values);
    backend->formulation()->postUpdate(post_update);
    backend->formulation()->logBackendFromMap(backend_info);
  }

  std::shared_ptr<Data> data;
};

struct BatchTester : public RegularBackendTester {
  struct Data {
    dyno::RegularBackendModule::Ptr backend;

    gtsam::Values values;
    gtsam::NonlinearFactorGraph factors;
  };

  BatchTester(dyno::RegularBackendModule::Ptr backend_)
      : RegularBackendTester(backend_) {
    data = std::make_shared<Data>();
    CHECK(backend);
    data->backend = backend;

    backend->registerPostFormulationUpdateCallback(
        [&](const RegularBackendModule::FormulationType::Ptr& formulation,
            dyno::FrameId frame_id, const gtsam::Values& new_values,
            const gtsam::NonlinearFactorGraph& new_factors) -> void {
          data->values = formulation->getTheta();
          data->factors = formulation->getGraph();

          data->factors.saveGraph(
              dyno::getOutputFilePath(
                  "batch_graph_" + std::to_string(frame_id) + "_" +
                  formulation->getFullyQualifiedName() + ".dot"),
              dyno::DynosamKeyFormatter);
        });
  }

  void onFinish() override {
    auto backend = data->backend;

    dyno::BackendMetaData backend_info;
    backend_info.backend_params = &backend->getParams();

    dyno::PostUpdateData post_update(backend->getSpinState().frame_id);
    backend->formulation()->postUpdate(post_update);
    backend->formulation()->logBackendFromMap(backend_info);

    LOG(INFO) << "Starting batch opt";
    try {
      gtsam::LevenbergMarquardtParams opt_params;
      opt_params.verbosityLM =
          gtsam::LevenbergMarquardtParams::VerbosityLM::SUMMARY;
      gtsam::Values opt_values = gtsam::LevenbergMarquardtOptimizer(
                                     data->factors, data->values, opt_params)
                                     .optimize();
      backend_info.logging_suffix = "batch_opt";
      backend->formulation()->updateTheta(opt_values);
      backend->formulation()->postUpdate(post_update);
      backend->formulation()->logBackendFromMap(backend_info);
    } catch (const std::exception& e) {
      LOG(FATAL) << "Batch opt failed with exception: " << e.what();
    }
  }
  std::shared_ptr<Data> data;
};

struct RGBDBackendTester {
  void addTester(std::shared_ptr<TesterBase> t) { testers.push_back(t); }

  void processAll(dyno::VisionImuPacket::Ptr output_packet) {
    for (auto t : testers) {
      t->preSpin();
      auto backend = t->getBackend();
      CHECK(backend);
      backend->spinOnce(output_packet);
      t->postSpin();
    }
    // for (auto b : backends) {
    //   b->
    //   b->spinOnce(output_packet);
    // }
  }

  void finishAll() {
    for (auto t : testers) {
      t->onFinish();
    }
  }

  std::vector<std::shared_ptr<TesterBase>> testers;
};

}  // namespace dyno_testing
