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

#include <opencv4/opencv2/core/types.hpp>  // for cv::Rect

namespace dyno {
namespace byte_track {

class RectBase {
 public:
  virtual ~RectBase() = default;

  virtual float top() const = 0;
  virtual float left() const = 0;
  virtual float width() const = 0;
  virtual float height() const = 0;

  inline operator cv::Rect() const {
    return cv::Rect(this->left(), this->top(), this->width(), this->height());
  }
};

float calc_iou(const RectBase& A, const RectBase& B);

class TlwhRect : public RectBase {
  float top_;
  float left_;
  float width_;
  float height_;

 public:
  TlwhRect(float top = 0, float left = 0, float width = 0, float height = 0);

  TlwhRect(const cv::Rect& cv_other);
  TlwhRect(const RectBase& other);

  virtual float top() const override;
  virtual float left() const override;
  virtual float width() const override;
  virtual float height() const override;
};

}  // namespace byte_track
}  // namespace dyno
