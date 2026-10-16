#include "dynosam2/DynamicSlamVisualizer.hpp"

namespace dyno2 {

void DynamicSlamVisualizer::setDisplayQueue(
    dyno::ImageDisplayQueue* display_queue) {
  image_displayer_ = std::make_unique<dyno::ImageDisplayer>(display_queue);
}

void DynamicSlamVisualizer::displayImages() {
  if (image_displayer_) {
    image_displayer_->displayFromQueue(2);
  }
}

}  // namespace dyno2
