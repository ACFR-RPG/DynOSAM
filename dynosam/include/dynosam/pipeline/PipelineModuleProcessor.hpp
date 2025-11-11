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

#include <glog/logging.h>

#include "dynosam/pipeline/PipelineBase.hpp"
#include "dynosam_common/ModuleBase.hpp"

namespace dyno {

/**
 * @brief A SIMO (Single-in-multi-out) pipeline module that processes a data
 * module.
 *
 * This defines the a basic pipeline (which connects to 1 or more pipelines via
 * threadsafe queues), processes data via a module (also tempalated on INPUT and
 * OUTPUT) and sends the data to the output queues.
 *
 * Output queues may be attached by This::registerOutputQueue
 * and data is taken from the InputQueue, recquired by construction.
 *
 * See SIMOPipelineModule and MIMOPipelineModule in PipelineBase.hpp for more
 * derived functions.
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT>
class PipelineModuleProcessor : public SIMOPipelineModule<INPUT, OUTPUT> {
 public:
  using Input = INPUT;
  using Output = OUTPUT;
  using Base = SIMOPipelineModule<Input, Output>;
  using BaseMIMO = typename Base::Base;

  using This = PipelineModuleProcessor<Input, Output>;
  using typename Base::InputConstSharedPtr;
  using typename Base::InputQueue;
  using typename Base::OutputConstSharedPtr;
  using typename Base::OutputQueue;
  using typename Base::OutputRegistra;

  DYNO_POINTER_TYPEDEFS(This)

  using Module = ModuleBase<Input, Output>;
  using ModulePtr = std::shared_ptr<Module>;

  explicit PipelineModuleProcessor(const std::string& module_name,
                                   InputQueue* input_queue, ModulePtr module)
      : Base(module_name, CHECK_NOTNULL(input_queue)),
        module_(CHECK_NOTNULL(module)) {}

  virtual ~PipelineModuleProcessor() = default;

  virtual inline OutputConstSharedPtr process(
      const InputConstSharedPtr& input) override {
    return module_->spinOnce(CHECK_NOTNULL(input));
  }

 protected:
  ModulePtr module_;
};

}  // namespace dyno
