#include "dynosam_processing/ImageViewer.hpp"

#include <glog/logging.h>

#include <opencv4/opencv2/highgui.hpp>

namespace dyno {

ImageDisplayer::ImageDisplayer(ImageDisplayQueue* display_queue)
    : display_queue_(CHECK_NOTNULL(display_queue)) {}

void ImageDisplayer::displayFromQueue(size_t wait_time_ms) {
  std::vector<ImageToDisplay> images_to_display;
  if (display_queue_->popAll(images_to_display, wait_time_ms)) {
    for (const auto& image_to_display : images_to_display) {
      cv::imshow(image_to_display.name_, image_to_display.image_);
    }
    cv::waitKey(1);
  }
}

}  // namespace dyno
