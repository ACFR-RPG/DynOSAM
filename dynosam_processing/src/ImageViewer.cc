#include "dynosam_processing/ImageViewer.hpp"

#include <glog/logging.h>

#include <cstdlib>
#include <iostream>
#include <opencv4/opencv2/highgui.hpp>

namespace dyno {

ImageBatchDisplayer::ImageBatchDisplayer(ImageDisplayQueue* display_queue)
    : display_queue_(CHECK_NOTNULL(display_queue)) {}

void ImageBatchDisplayer::displayFromQueue(size_t wait_time_ms) {
  std::vector<ImageToDisplay> images_to_display;
  if (getFromQueue(images_to_display, wait_time_ms)) {
    for (auto d : displayers_) {
      d->display(images_to_display);
    }
  }
}

void ImageBatchDisplayer::addDisplay(ImageDisplay::Ptr display) {
  CHECK_NOTNULL(display);
  displayers_.push_back(display);
}

bool ImageBatchDisplayer::getFromQueue(std::vector<ImageToDisplay>& images,
                                       size_t wait_time_ms) {
  return display_queue_->popAll(images, wait_time_ms);
}

bool SmartOpenCvDisplayer::isGuiAvailable() {
  // 1. Runtime Environment Check (Essential for Linux/Jetson Orin)
  // If running over headless SSH or a background systemd service without
  // X11/Wayland, the DISPLAY variable will be null or empty.
  const char* displayEnv = std::getenv("DISPLAY");
  if (!displayEnv || std::string(displayEnv).empty()) {
    return false;
  }

  // 2. HighGUI Implementation Check
  // Even if DISPLAY is set, OpenCV might be built headless (without GTK/Qt
  // support).
  try {
    // Attempt to create a temporary, invisible window placeholder.
    // If HighGUI support is missing, this immediately throws an un-implemented
    // exception.
    cv::namedWindow("test_gui_presence", cv::WINDOW_AUTOSIZE);
    cv::destroyWindow("test_gui_presence");
    return true;
  } catch (const cv::Exception& e) {
    // Catch error code -2 (Stands for "The function is not implemented")
    if (e.code == cv::Error::StsNotImplemented) {
      LOG(WARNING) << "[Warning] OpenCV GUI window capabilities are not "
                      "compiled into this build."
                   << std::endl;
    } else {
      LOG(WARNING) << "[Error] OpenCV encountered a problem testing the GUI: "
                   << e.what() << std::endl;
    }
    return false;
  }
}

SmartOpenCvDisplayer::SmartOpenCvDisplayer()
    : is_gui_available_(isGuiAvailable()) {
  if (is_gui_available_) {
    LOG(INFO) << "OpenCV gui available";
  } else {
    LOG(INFO) << "OpenCV gui not available: CV image display will be skipped!";
  }
}

void SmartOpenCvDisplayer::display(const std::vector<ImageToDisplay>& images) {
  if (!isDisplayAvailable()) {
    return;
  }
  for (const auto& image_to_display : images) {
    cv::imshow(image_to_display.name_, image_to_display.image_);
  }
  cv::waitKey(1);
}

}  // namespace dyno
