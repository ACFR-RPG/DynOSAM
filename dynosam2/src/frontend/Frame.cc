#include "dynosam2/frontend/Frame.hpp"

namespace dyno2 {

void Frame::setImages(const dyno::ImageContainer& other) {
  releaseImages();

  static constexpr bool kCopyImageData = false;
  image_container = dyno::ImageContainer::CreateSubset(
      other, {dyno::ImageContainer::kRGB, dyno::ImageContainer::kObjectMask},
      kCopyImageData);
}

void Frame::releaseImages() {
  if (image_container) {
    image_container.reset();
  }
}

void ViFrame::setRelativePose(RelativePoseErrorBase::Ptr relative_pose) {
  CHECK_EQ(relative_pose->to(), this->getFrameId());
  relative_pose_ = relative_pose;
}

bool DoFrame::getObjectMask(cv::Mat& mask) const {
  if (image_container && image_container->hasObjectMask()) {
    const cv::Mat& object_mask = image_container->objectMotionMask();
    mask = object_mask == getObjectId();
    return true;
  }
  return false;
}

}  // namespace dyno2
