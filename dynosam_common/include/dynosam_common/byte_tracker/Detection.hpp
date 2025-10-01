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

#include <memory>

#include "dynosam_common/byte_tracker/Rect.hpp"
#include "dynosam_common/utils/Macros.hpp"

namespace dyno {
namespace byte_track {

class DetectionBase {
 public:
  DYNO_POINTER_TYPEDEFS(DetectionBase)

  virtual ~DetectionBase() = default;

  virtual const RectBase &rect() const = 0;
  virtual float score() const = 0;

  virtual void set_rect(const RectBase &rect) = 0;
  virtual void set_score(float score) = 0;
};

class Detection : public DetectionBase {
  TlwhRect rect_;
  float score_ = 0;

 public:
  DYNO_POINTER_TYPEDEFS(Detection)

  Detection(const TlwhRect &rect, float score);

  const TlwhRect &rect() const override;
  float score() const override;

  void set_rect(const RectBase &rect) override;
  void set_score(float score) override;
};

}  // namespace byte_track
}  // namespace dyno
