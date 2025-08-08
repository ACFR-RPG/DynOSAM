/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a
 copy
 *   of this software and associated documentation files (the "Software"), to
 deal
 *   in the Software without restriction, including without limitation the
 rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in
 all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE
 *   SOFTWARE.
 */

#include "dynosam/common/ImageContainer.hpp"

#include <exception>
#include <opencv4/opencv2/opencv.hpp>

#include "dynosam/frontend/vision/UndistortRectifier.hpp"  //annoything this has to be here - better make (undisort iamges) into a free fucntion later!!
#include "dynosam/utils/OpenCVUtils.hpp"

namespace dyno {

ImageContainer ImageContainer::clone() const {
  ImageContainer container(this->frameId(), this->timestamp());

  // explicit deep copies for all key image pairs
  for (const auto& [k, v] : images_) {
    container.images_.emplace(k, v.clone());
  }

  return container;
}

const ImageWrapper<ImageType::RGBMono>& ImageContainer::rgb() const {
  return this->at<ImageType::RGBMono>(kRGB);
}
const ImageWrapper<ImageType::Depth>& ImageContainer::depth() const {
  return this->at<ImageType::Depth>(kDepth);
}
const ImageWrapper<ImageType::OpticalFlow>& ImageContainer::opticalFlow()
    const {
  return this->at<ImageType::OpticalFlow>(kOPticalFlow);
}
const ImageWrapper<ImageType::MotionMask>& ImageContainer::objectMotionMask()
    const {
  return this->at<ImageType::MotionMask>(kObjectMask);
}
const ImageWrapper<ImageType::RGBMono>& ImageContainer::rightRgb() const {
  return this->at<ImageType::RGBMono>(kRightRgb);
}

ImageWrapper<ImageType::RGBMono>& ImageContainer::rgb() {
  return this->at<ImageType::RGBMono>(kRGB);
}
ImageWrapper<ImageType::Depth>& ImageContainer::depth() {
  return this->at<ImageType::Depth>(kDepth);
}
ImageWrapper<ImageType::OpticalFlow>& ImageContainer::opticalFlow() {
  return this->at<ImageType::OpticalFlow>(kOPticalFlow);
}
ImageWrapper<ImageType::MotionMask>& ImageContainer::objectMotionMask() {
  return this->at<ImageType::MotionMask>(kObjectMask);
}
ImageWrapper<ImageType::RGBMono>& ImageContainer::rightRgb() {
  return this->at<ImageType::RGBMono>(kRightRgb);
}

ImageContainer& ImageContainer::rgb(const cv::Mat& image) {
  return this->add<ImageType::RGBMono>(kRGB, image);
}

ImageContainer& ImageContainer::depth(const cv::Mat& image) {
  return this->add<ImageType::Depth>(kDepth, image);
}

ImageContainer& ImageContainer::opticalFlow(const cv::Mat& image) {
  return this->add<ImageType::OpticalFlow>(kOPticalFlow, image);
}

ImageContainer& ImageContainer::objectMotionMask(const cv::Mat& image) {
  return this->add<ImageType::MotionMask>(kObjectMask, image);
}

ImageContainer& ImageContainer::rightRgb(const cv::Mat& image) {
  return this->add<ImageType::RGBMono>(kRightRgb, image);
}

std::string ImageContainer::toString() const {
  std::stringstream ss;
  ss << "[ ";
  for (const auto& [key, key_image_pair] : images_) {
    CHECK_NOTNULL(key_image_pair.ptr);
    ss << "{" << key << ": " << key_image_pair.ptr->toString() << "} ";
  }
  ss << "]";
  return ss.str();
}

std::ostream& operator<<(std::ostream& os,
                         const ImageContainer& image_container) {
  os << image_container.toString();
  return os;
}

}  // namespace dyno
