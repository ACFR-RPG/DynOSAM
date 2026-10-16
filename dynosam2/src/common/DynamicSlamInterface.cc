#include "dynosam2/common/DynamicSlamInterface.hpp"

namespace dyno2 {

// should be in common but whatever
struct InvalidImageContainerException : public dyno::DynosamException {
  InvalidImageContainerException(const dyno::ImageContainer& container,
                                 const std::string& what)
      : dyno::DynosamException(
            "Image container with config: " + container.toString() +
            "\n was invalid - " + what) {}
};

DynamicSlamInterface::DynamicSlamInterface() : Base("dynosam2") {}

void DynamicSlamInterface::setPublicationQueue(
    ThreadsafeQueue<PublicationData::ConstPtr>* queue) {
  CHECK_NOTNULL(queue);
  publication_queue_ = queue;
}

void DynamicSlamInterface::setImageDisplayQueue(
    dyno::ImageDisplayQueue* queue) {
  CHECK_NOTNULL(queue);
  image_display_queue_ = queue;
}

bool DynamicSlamInterface::pushImageToDisplayQueue(const std::string& wname,
                                                   const cv::Mat& image) {
  if (!image_display_queue_) {
    return false;
  }

  image_display_queue_->push(dyno::ImageToDisplay(wname, image));
  return true;
}

bool DynamicSlamInterface::pushPublicationDataToQueue(
    PublicationData::ConstPtr publication_data) {
  if (!publication_queue_) {
    return false;
  }

  publication_queue_->push(publication_data);
  return true;
}

void DynamicSlamInterface::validateInput(
    const dyno::VIFrontendInput::ConstPtr& input) const {
  const auto image_container = input->image_container_;

  const bool has_rgb = image_container->hasRgb();
  const bool has_depth_image = image_container->hasDepth();
  const bool has_stereo = image_container->hasRightRgb();

  if (!has_rgb) {
    throw InvalidImageContainerException(*image_container, "Missing RGB");
  }

  if (!has_depth_image && !has_stereo) {
    throw InvalidImageContainerException(*image_container,
                                         "Missing Depth or Stereo");
  }
}

}  // namespace dyno2
