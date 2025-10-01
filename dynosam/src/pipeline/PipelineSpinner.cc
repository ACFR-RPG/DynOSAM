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

#include "dynosam/pipeline/PipelineSpinner.hpp"

#include <glog/logging.h>

#include <functional>

#include "dynosam_common/Exceptions.hpp"

namespace dyno {

PipelineSpinner::~PipelineSpinner() { shutdownAll(); }

void PipelineSpinner::registerPipeline(PipelineBase::Ptr pipeline,
                                       bool parallel_run) {
  const std::string name = pipeline->getModuleName();
  checkAndThrow(!pipelines_.exists(name),
                "Cannot register pipeline with name " + name +
                    " as it has already been registered");

  if (parallel_run) {
    pipelines_.insert(
        {name, std::move(std::make_unique<AsynchronousSpinnerPair>(pipeline))});
  } else {
    pipelines_.insert(
        {name, std::move(std::make_unique<SynchronousSpinnerPair>(pipeline))});
  }

  ordering_.push_back(name);
}

void PipelineSpinner::launchAll() {
  auto lambda = [](SpinnerPair::UniquePtr& spinner_pair) {
    spinner_pair->launch();
  };
  appyAll(lambda);

  is_launched_ = true;
}
void PipelineSpinner::spinOnceAll() {
  if (!is_launched_) {
    LOG(WARNING)
        << "spinOnceAll called but the pipelines have not been launched yet!";
  }

  auto lambda = [](SpinnerPair::UniquePtr& spinner_pair) {
    spinner_pair->spinOnce();
  };
  appyAll(lambda);
}

void PipelineSpinner::shutdownAll() {
  auto lambda = [](SpinnerPair::UniquePtr& spinner_pair) {
    spinner_pair->shutdown();
  };
  appyAll(lambda);
}

void PipelineSpinner::appyAll(
    std::function<void(SpinnerPair::UniquePtr&)> apply) {
  for (const std::string& name : ordering_) {
    SpinnerPair::UniquePtr& spinner_pair = pipelines_.at(name);
    apply(spinner_pair);
  }
}

PipelineSpinner::SynchronousSpinnerPair::SynchronousSpinnerPair(
    PipelineBase::Ptr p)
    : SpinnerPair(p) {}

bool PipelineSpinner::SynchronousSpinnerPair::spinOnce() {
  return p_->spinOnce();
}

void PipelineSpinner::SynchronousSpinnerPair::shutdown() {
  LOG(INFO) << "Shutting down pipeline: " + p_->getModuleName();
  // shutdown pipeline
  p_->shutdown();
}

PipelineSpinner::AsynchronousSpinnerPair::AsynchronousSpinnerPair(
    PipelineBase::Ptr p)
    : SpinnerPair(p) {}

void PipelineSpinner::AsynchronousSpinnerPair::launch() {
  CHECK(spinner_ == nullptr);

  const std::string spinner_name = p_->getModuleName() + "-spinner";
  spinner_ = std::make_unique<Spinner>(std::bind(&PipelineBase::spin, p_.get()),
                                       spinner_name);
  LOG(INFO) << "Launching spinner: " << spinner_name;
}

bool PipelineSpinner::AsynchronousSpinnerPair::spinOnce() {
  return spinner_->isRunning();
}

void PipelineSpinner::AsynchronousSpinnerPair::shutdown() {
  LOG(INFO) << "Shutting down pipeline and associated spinner: " +
                   p_->getModuleName();
  // shutdown pipeline
  p_->shutdown();
  // then shutdown spinner
  spinner_->shutdown();
}

}  // namespace dyno
