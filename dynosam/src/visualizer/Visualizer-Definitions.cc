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

#include "dynosam/visualizer/Visualizer-Definitions.hpp"

#include <glog/logging.h>

#include <opencv4/opencv2/opencv.hpp>

namespace dyno {

OpenCVImageDisplayQueue::OpenCVImageDisplayQueue(
    ImageDisplayQueue* display_queue, bool parallel_run)
    : display_queue_(CHECK_NOTNULL(display_queue)),
      parallel_run_(parallel_run) {}

void OpenCVImageDisplayQueue::process() {
  bool queue_state = false;
  if (parallel_run_) {
    ImageToDisplay image_to_display;
    queue_state = display_queue_->popBlockingWithTimeout(image_to_display, 2);

    if (queue_state) {
      // cv::namedWindow(image_to_display.name_);
      cv::imshow(image_to_display.name_, image_to_display.image_);
      cv::waitKey(1);  // Not needed because we are using startWindowThread()
    }
  } else {
    std::vector<ImageToDisplay> images_to_display;
    queue_state = display_queue_->popAll(images_to_display, 2);

    if (queue_state) {
      for (const auto& image_to_display : images_to_display) {
        cv::imshow(image_to_display.name_, image_to_display.image_);
      }
      cv::waitKey(1);
    }
  }
}

}  // namespace dyno
