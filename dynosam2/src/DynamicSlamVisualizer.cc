#include "dynosam2/DynamicSlamVisualizer.hpp"

namespace dyno2 {

DynamicSlamVisualizer::DynamicSlamVisualizer(
    dyno::CanonicalSensorRig::Ptr sensor_rig, const SlamBackend* estimator)
    : sensor_rig_(sensor_rig),
      estimator_(CHECK_NOTNULL(estimator)),
      T_RC_(sensor_rig->getCanonicalExtrinsics()),
      reference_frames_(sensor_rig->getReferenceFrames()) {}

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
