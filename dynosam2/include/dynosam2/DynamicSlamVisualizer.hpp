#pragma once

#include <glog/logging.h>

#include "dynosam2/common/DynamicSlamInterface.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_processing/ImageViewer.hpp"

namespace dyno2 {

class DynamicSlamVisualizer {
 public:
  DynamicSlamVisualizer() {}
  virtual ~DynamicSlamVisualizer() {}

  // viz data AND estimator update!
  // this function is called asynchronously!
  virtual void publishEstimatorUpdate(const PublicationData& publication_data) {
    LOG(INFO) << "Here";
  }

  /// @brief Set external image display queue
  /// @param display_queue
  void setDisplayQueue(dyno::ImageDisplayQueue* display_queue);

  /// @brief Draw images from the image displayer (if set) using OpenCV
  void displayImages();

 private:
  dyno::ImageDisplayer::UniquePtr image_displayer_{nullptr};
};

}  // namespace dyno2
