#pragma once

#include <opencv4/opencv2/core.hpp>

#include "dynosam_common/utils/Macros.hpp"
#include "dynosam_processing/ThreadSafeQueue.hpp"

namespace dyno {

struct ImageToDisplay {
  ImageToDisplay(const std::string& name, const cv::Mat& image)
      : name_(name), image_(image) {}

  std::string name_;
  cv::Mat image_;
};

using ImageDisplayQueue = ThreadsafeQueue<ImageToDisplay>;

class ImageDisplayer {
 public:
  DYNO_POINTER_TYPEDEFS(ImageDisplayer);
  ImageDisplayer(ImageDisplayQueue* display_queue);

  void displayFromQueue(size_t wait_time_ms = 2);

 private:
  ImageDisplayQueue* display_queue_;
};

}  // namespace dyno
