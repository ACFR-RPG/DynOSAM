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

#include "dynosam/frontend/FrontendModule.hpp"

#include <glog/logging.h>

#include "dynosam_common/logger/Logger.hpp"

namespace dyno {

FrontendModule::FrontendModule(const DynoParams& params,
                               ImageDisplayQueue* display_queue)
    : Base("frontend"), params_(params), display_queue_(display_queue) {
  // create callback to update gt_packet_map_ values so the derived classes dont
  // need to manage this
  registerInputCallback([=](FrontendInputPacketBase::ConstPtr input) {
    // if(input->optional_gt_) gt_packet_map_.insert2(input->getFrameId(),
    // *input->optional_gt_);
    if (input->optional_gt_)
      shared_module_info.updateGroundTruthPacket(input->getFrameId(),
                                                 *input->optional_gt_);
    shared_module_info.updateTimestampMapping(input->getFrameId(),
                                              input->getTimestamp());
  });
}

FrontendModule::~FrontendModule() { VLOG(5) << "Destructing frontend module"; }

void FrontendModule::validateInput(
    const FrontendInputPacketBase::ConstPtr& input) const {
  CHECK(input->image_container_);

  const ImageValidationResult result =
      validateImageContainer(input->image_container_);

  // throw exception if result is false
  checkAndThrow<InvalidImageContainerException>(
      result.valid_, *input->image_container_, result.requirement_);
}

}  // namespace dyno
