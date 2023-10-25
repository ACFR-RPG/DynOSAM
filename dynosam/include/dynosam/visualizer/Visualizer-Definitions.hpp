/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 */

#pragma once

#include "dynosam/pipeline/PipelineBase-Definitions.hpp"
#include "dynosam/frontend/FrontendOutputPacket.hpp"
#include "dynosam/pipeline/ThreadSafeQueue.hpp"

#include <glog/logging.h>
#include <opencv4/opencv2/opencv.hpp>

namespace dyno {


struct ImageToDisplay {
  ImageToDisplay() = default;
  ImageToDisplay(const std::string& name, const cv::Mat& image)
    //clone necessary?
      : name_(name), image_(image.clone()) {}

  std::string name_;
  cv::Mat image_;
};

using ImageDisplayQueue = ThreadsafeQueue<ImageToDisplay>;

class OpenCVImageDisplayQueue {

public:
    OpenCVImageDisplayQueue(ImageDisplayQueue* display_queue, bool parallel_run);

    void process();

private:
    ImageDisplayQueue* display_queue_;
    bool parallel_run_;
};

} //dyno
