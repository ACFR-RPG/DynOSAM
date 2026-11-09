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

bool ViFrame::describe(cv::Ptr<cv::Feature2D> describer) {
  if (image_container && numKeypoints() > 0) {
    const cv::Mat mono =
        dyno::ImageType::RGBMono::toMono(image_container->rgb());

    std::vector<cv::KeyPoint> keypoints;
    keypoints.reserve(this->numKeypoints());

    const int num_keypoints = numKeypoints();
    const int descriptor_size = describer->descriptorSize();

    // TODO: for now dont fill out anything else usefil in the Keypoint
    cv::Point2f pt;
    for (int i = 0; i < num_keypoints; i++) {
      this->getCvKeypointByIndex(i, pt);

      cv::KeyPoint kp(pt, 3);
      kp.class_id = i;
      keypoints.push_back(kp);
    }

    // BRIEF descriptors will drop some keypoints near image boundaries so we
    // need to preserve all our existing keypoints
    std::vector<cv::KeyPoint> modified_keypoints = keypoints;
    cv::Mat computed_descriptors;
    describer->compute(mono, modified_keypoints, computed_descriptors);

    // Initialise descriptors to zero and validity to false.
    descriptors_ = cv::Mat::zeros(num_keypoints, descriptor_size,
                                  describer->descriptorType());
    valid_descriptors_.assign(num_keypoints, 0);

    // Populate descriptors using the original keypoint index stored in
    // class_id.
    for (int i = 0; i < computed_descriptors.rows; ++i) {
      const int original_idx = modified_keypoints[i].class_id;

      CHECK_GE(original_idx, 0);
      CHECK_LT(original_idx, num_keypoints);

      computed_descriptors.row(i).copyTo(descriptors_.row(original_idx));
      valid_descriptors_[original_idx] = 1;
    }

    return true;
  }
  return false;
}

std::pair<bool, const uint8_t*> ViFrame::getDescriptorByIndex(Index idx) const {
  const uint8_t* descriptor = descriptors_.ptr<uint8_t>(idx);
  const u_char valid = valid_descriptors_.at(idx);
  return {static_cast<bool>(valid), descriptor};
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
