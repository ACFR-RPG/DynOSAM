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

namespace dyno {

struct PipelinePayload {
  DYNO_POINTER_TYPEDEFS(PipelinePayload)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  explicit PipelinePayload(const Timestamp& timestamp)
      : timestamp_(timestamp) {}
  PipelinePayload() {}
  virtual ~PipelinePayload() = default;

  // Untouchable timestamp of the payload.
  Timestamp timestamp_;
};

/**
 * @brief The NullPipelinePayload is an empty payload, used for those modules
 * that do not return a payload, such as the display module, which only
 * displays images and returns nothing.
 */
struct NullPipelinePayload : public PipelinePayload {
  DYNO_POINTER_TYPEDEFS(NullPipelinePayload)
  DYNO_DELETE_COPY_CONSTRUCTORS(NullPipelinePayload)
  explicit NullPipelinePayload() : PipelinePayload(Timestamp()) {}
  virtual ~NullPipelinePayload() = default;
};

}  // namespace dyno
