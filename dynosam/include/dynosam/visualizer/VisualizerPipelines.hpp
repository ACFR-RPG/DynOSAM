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

#include "dynosam_common/Types.hpp"
#include "dynosam_core/DynoState.hpp"
#include "dynosam_core/GroundTruthPacket.hpp"
#include "dynosam_core/RealtimeOutput.hpp"
#include "dynosam_processing/ImageViewer.hpp"
#include "dynosam_processing/PipelineBase.hpp"

namespace dyno {

template <typename TYPE>
class DisplayBase {
 public:
  using Type = TYPE;
  using This = DisplayBase<TYPE>;
  DYNO_POINTER_TYPEDEFS(This)

  using TypeConstPtr = typename TYPE::ConstPtr;

  DisplayBase() {}
  virtual ~DisplayBase() {}

  virtual void spinOnce(const TypeConstPtr& input) = 0;
};

/**
 * @brief Generic pipeline for display
 *
 * @tparam INPUT
 */
template <typename INPUT>
class DisplayPipeline : public SIMOPipelineModule<INPUT, EmptyPayload> {
 public:
  using Input = INPUT;
  using This = DisplayPipeline<Input>;
  using Base = SIMOPipelineModule<Input, EmptyPayload>;
  using Display = DisplayBase<Input>;
  DYNO_POINTER_TYPEDEFS(This)

  using InputQueue = typename Base::InputQueue;
  using InputConstPtr = typename Display::TypeConstPtr;

  DisplayPipeline(const std::string& name, InputQueue* input_queue,
                  typename Display::Ptr display)
      : Base(name, input_queue), display_(CHECK_NOTNULL(display)) {
    empty_payload_ = std::make_shared<EmptyPayload>();
  }

  EmptyPayload::ConstPtr process(const InputConstPtr& input) override {
    display_->spinOnce(CHECK_NOTNULL(input));
    return empty_payload_;
  }

 private:
  EmptyPayload::Ptr empty_payload_;
  typename Display::Ptr display_;
};

class FrontendDisplay : public DisplayBase<RealtimeOutput> {
 public:
  FrontendDisplay() = default;
  virtual ~FrontendDisplay() = default;
};
using BackendDisplay = DisplayBase<DynoState>;

using FrontendVizPipeline = DisplayPipeline<FrontendDisplay::Type>;
using BackendVizPipeline = DisplayPipeline<BackendDisplay::Type>;

/**
 * @brief Vizualisation class that can be associated with a
 * BackendModule/formulation to allow custom vizualistions for the specifics of
 * the BackendModule/formulation in addition to the standard frontend/backend
 * output
 *
 */
class BackendModuleDisplay : public DisplayBase<DynoState> {
 public:
  DYNO_POINTER_TYPEDEFS(BackendModuleDisplay)

  BackendModuleDisplay() = default;
  virtual ~BackendModuleDisplay() = default;
};

}  // namespace dyno
