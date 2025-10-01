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

#include <gtsam/base/FastMap.h>

#include "dynosam/pipeline/PipelineBase.hpp"
#include "dynosam_common/utils/Macros.hpp"
#include "dynosam_common/utils/Spinner.hpp"

namespace dyno {

// TODO: I think not used?
class PipelineSpinner {
 public:
  PipelineSpinner() = default;
  virtual ~PipelineSpinner();

  void registerPipeline(PipelineBase::Ptr pipeline, bool parallel_run);

  void launchAll();
  void spinOnceAll();

 protected:
  void shutdownAll();

 private:
  struct SpinnerPair {
    DYNO_POINTER_TYPEDEFS(SpinnerPair)

    PipelineBase::Ptr p_{nullptr};  // pipeline;

    SpinnerPair(PipelineBase::Ptr p) : p_(p) {}
    virtual ~SpinnerPair() = default;
    virtual void launch() {}
    virtual bool spinOnce() = 0;
    virtual void shutdown() = 0;
  };

  struct SynchronousSpinnerPair : SpinnerPair {
    SynchronousSpinnerPair(PipelineBase::Ptr p);
    bool spinOnce() override;
    void shutdown() override;
  };

  struct AsynchronousSpinnerPair : SpinnerPair {
    Spinner::UniquePtr spinner_;

    AsynchronousSpinnerPair(PipelineBase::Ptr p);
    void launch() override;
    bool spinOnce() override;
    void shutdown() override;
  };

  void appyAll(std::function<void(SpinnerPair::UniquePtr&)> apply);

 private:
  bool is_launched_{false};
  std::vector<std::string> ordering_;  //! Enumeration order determines which
                                       //! order the pipelines will be spun in
  gtsam::FastMap<std::string, SpinnerPair::UniquePtr>
      pipelines_;  //! Map of module name to spinning pairs
};

}  // namespace dyno
