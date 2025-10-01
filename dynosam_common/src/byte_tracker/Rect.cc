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

#include "dynosam_common/byte_tracker/Rect.hpp"

#include <algorithm>

namespace dyno {
namespace byte_track {

float calc_iou(const RectBase& A, const RectBase& B) {
  const float iw = std::min(A.left() + A.width(), B.left() + B.width()) -
                   std::max(A.left(), B.left());
  float iou = 0;
  if (iw > 0) {
    const float ih = std::min(A.top() + A.height(), B.top() + B.height()) -
                     std::max(A.top(), B.top());
    if (ih > 0) {
      const float ua =
          A.width() * A.height() + B.width() * B.height() - iw * ih;
      iou = iw * ih / ua;
    }
  }
  return iou;
}

TlwhRect::TlwhRect(float top, float left, float width, float height)
    : top_(top), left_(left), width_(width), height_(height) {}

TlwhRect::TlwhRect(const cv::Rect& cv_other)
    : top_(static_cast<float>(cv_other.y)),
      left_(static_cast<float>(cv_other.x)),
      width_(static_cast<float>(cv_other.width)),
      height_(static_cast<float>(cv_other.height)) {}

TlwhRect::TlwhRect(const RectBase& other)
    : top_(other.top()),
      left_(other.left()),
      width_(other.width()),
      height_(other.height()) {}

float TlwhRect::top() const { return top_; }
float TlwhRect::left() const { return left_; }
float TlwhRect::width() const { return width_; }
float TlwhRect::height() const { return height_; }

}  // namespace byte_track
}  // namespace dyno
