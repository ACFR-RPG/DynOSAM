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

#include "dynosam/backend/BackendModule.hpp"

#include <glog/logging.h>

#include "dynosam/backend/BackendDefinitions.hpp"

namespace dyno {

BackendModule::BackendModule(const BackendParams& params,
                             ImageDisplayQueue* display_queue)
    : Base("backend"),
      base_params_(params),
      display_queue_(display_queue)

{
  setFactorParams(params);

  // create callback to update gt_packet_map_ values so the derived classes dont
  // need to manage this
  // TODO: this logic is exactly the same as in FrontendModule - functionalise!!
  registerInputCallback([=](BackendInputPacket::ConstPtr input) {
    if (input->groundTruthPacket())
      shared_module_info.updateGroundTruthPacket(input->frameId(),
                                                 *input->groundTruthPacket());
    shared_module_info.updateTimestampMapping(input->frameId(),
                                              input->timestamp());

    const BackendSpinState previous_spin_state = spin_state_;

    // update spin state
    spin_state_ = BackendSpinState(input->frameId(), input->timestamp(),
                                   previous_spin_state.iteration + 1);
  });
}

void BackendModule::setFactorParams(const BackendParams& backend_params) {
  noise_models_ = NoiseModels::fromBackendParams(backend_params);
}

void BackendModule::validateInput(const BackendInputPacket::ConstPtr&) const {}

}  // namespace dyno
