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

#include <glog/logging.h>

#include "dynosam/backend/BackendOutputPacket.hpp"
#include "dynosam/frontend/VisionImuOutputPacket.hpp"
#include "dynosam/pipeline/PipelineBase.hpp"
#include "dynosam/pipeline/PipelinePayload.hpp"
#include "dynosam/visualizer/Display.hpp"
#include "dynosam_common/Types.hpp"

namespace dyno {

/**
 * @brief Generic pipeline for display
 *
 * @tparam INPUT
 */
template <typename INPUT>
class DisplayPipeline : public SIMOPipelineModule<INPUT, NullPipelinePayload> {
 public:
  using Input = INPUT;
  using This = DisplayPipeline<Input>;
  using Base = SIMOPipelineModule<Input, NullPipelinePayload>;
  using Display = DisplayBase<Input>;
  DYNO_POINTER_TYPEDEFS(This)

  using InputQueue = typename Base::InputQueue;
  using InputConstPtr = typename Display::InputConstPtr;

  DisplayPipeline(const std::string& name, InputQueue* input_queue,
                  typename Display::Ptr display)
      : Base(name, input_queue), display_(CHECK_NOTNULL(display)) {}

  NullPipelinePayload::ConstPtr process(const InputConstPtr& input) override {
    display_->spinOnce(CHECK_NOTNULL(input));
    static auto null_payload = std::make_shared<NullPipelinePayload>();
    return null_payload;
  }

 private:
  typename Display::Ptr display_;
};

using FrontendVizPipeline = DisplayPipeline<VisionImuPacket>;
using BackendVizPipeline = DisplayPipeline<BackendOutputPacket>;

/**
 * @brief Vizualisation class that can be associated with a
 * BackendModule/formulation to allow custom vizualistions for the specifics of
 * the BackendModule/formulation in addition to the standard frontend/backend
 * output
 *
 */
class BackendModuleDisplay {
 public:
  DYNO_POINTER_TYPEDEFS(BackendModuleDisplay)

  BackendModuleDisplay() = default;
  virtual ~BackendModuleDisplay() = default;

  virtual void spin(Timestamp timestamp, FrameId frame_id) = 0;
};

}  // namespace dyno
