/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam_vision_common/ImageTypes.hpp"

#include <opencv4/opencv2/opencv.hpp>

#include "dynosam_common/utils/OpenCVUtils.hpp"

namespace dyno {

ImageBase::ImageBase(const cv::Mat& img) : image(img) {}

bool ImageBase::exists() const { return !image.empty(); }

void validateMask(const cv::Mat& input, const std::string& name) {
  // we guanrantee with a static assert that the SemanticMask and MotionMask
  // types are the same
  const static std::string expected_type =
      utils::cvTypeToString(ImageType::MotionMask::OpenCVType);
  if (input.type() != ImageType::MotionMask::OpenCVType) {
    throw InvalidImageTypeException(name + " image was not " + expected_type +
                                    ". Input image type was " +
                                    utils::cvTypeToString(input.type()));
  }
}

void validateRGBMono(const cv::Mat& input) {
  if (input.type() != CV_8UC1 && input.type() != CV_8UC3 &&
      input.type() != CV_8UC4) {
    throw InvalidImageTypeException(
        "RGBMono image was not CV_8UC1, CV_8UC3 or CV_8UC4. Input image type "
        "was " +
        utils::cvTypeToString(input.type()));
  }
}

void validateSingleImage(const cv::Mat& input, int expected_type,
                         const std::string& name) {
  if (input.type() != expected_type) {
    throw InvalidImageTypeException(
        name + " image was not " + utils::cvTypeToString(expected_type) +
        " - Input image type was " + utils::cvTypeToString(input.type()));
  }
}

cv::Mat RGBMonoToRGB(const cv::Mat& mat) {
  const auto channels = mat.channels();
  if (channels == 3) {
    return mat;
  } else if (channels == 4) {
    cv::Mat rgb;
    cv::cvtColor(mat, rgb, cv::COLOR_RGBA2RGB);
    return rgb;
  } else if (channels == 1) {
    // grey scale but we want rgb so that we can draw colours on it or whatever
    cv::Mat rgb;
    cv::cvtColor(mat, rgb, cv::COLOR_GRAY2RGB);
    return rgb;
  } else {
    return mat;
  }
}

cv::Mat RGBMonoToMono(const cv::Mat& mat) {
  const auto channels = mat.channels();
  if (channels == 1) {
    return mat;
  }
  if (channels == 3) {
    cv::Mat mono;
    cv::cvtColor(mat, mono, cv::COLOR_RGB2GRAY);
    return mono;
  } else if (channels == 4) {
    cv::Mat mono;
    cv::cvtColor(mat, mono, cv::COLOR_RGBA2GRAY);
    return mono;
  } else {
    return mat;
  }
}

// RGBMono
void ImageType::RGBMono::validate(const cv::Mat& input) {
  validateRGBMono(input);
}

cv::Mat ImageType::RGBMono::toRGB(const ImageWrapper<RGBMono>& image) {
  return RGBMonoToRGB(image);
}

cv::Mat ImageType::RGBMono::toMono(const ImageWrapper<RGBMono>& image) {
  return RGBMonoToMono(image);
}

// Depth
void ImageType::Depth::validate(const cv::Mat& input) {
  validateSingleImage(input, OpenCVType, name());
}

// TODO: should this use getDisparityVis?
// depth is metric so we should actually just scale it!
cv::Mat ImageType::Depth::toRGB(const ImageWrapper<Depth>& image) {
  cv::Mat metric_depth_map = image.clone();
  // convert to floats to allow patch nan's to work
  metric_depth_map.convertTo(metric_depth_map, CV_32F);

  static constexpr float max_value = 25.0;

  cv::patchNaNs(metric_depth_map, max_value);
  for (int i = 0; i < metric_depth_map.rows; ++i) {
    for (int j = 0; j < metric_depth_map.cols; ++j) {
      float& pixel = metric_depth_map.at<float>(i, j);
      if (std::isinf(pixel)) {
        pixel = 0.0;  // Replace Inf with small value so image is not saturated
                      // to a large number
      }
    }
  }

  // // truncate to max value (e.g. there is some outlier in the depth value
  // that is a super larger number) cv::threshold(metric_depth_map,
  // metric_depth_map, max_value, max_value, cv::THRESH_TRUNC);

  // //fix a pixel as the maximum depth to avoid flickering!!
  // metric_depth_map.at<float>(1, 1) = max_value;

  // Normalize the depth map to the range [0, 1]
  cv::Mat normalised_depth_map;
  cv::normalize(metric_depth_map, normalised_depth_map, 0.0, 1.0,
                cv::NORM_MINMAX);
  // // Convert the normalized depth map to an 8-bit image [0, 255]
  cv::Mat depth_map_8u;
  normalised_depth_map.convertTo(depth_map_8u, CV_8U, 255.0, 20.0);

  return depth_map_8u;
}

void ImageType::OpticalFlow::validate(const cv::Mat& input) {
  validateSingleImage(input, OpenCVType, name());
}

cv::Mat ImageType::OpticalFlow::toRGB(const ImageWrapper<OpticalFlow>& image) {
  const cv::Mat& optical_flow = image;
  cv::Mat flow_viz;
  utils::flowToRgb(optical_flow, flow_viz);
  return flow_viz;
}

void ImageType::SemanticMask::validate(const cv::Mat& input) {
  validateMask(input, name());
}

cv::Mat ImageType::SemanticMask::toRGB(
    const ImageWrapper<SemanticMask>& image) {
  const cv::Mat& semantic_mask = image;
  return utils::labelMaskToRGB(semantic_mask, background_label);
}

void ImageType::MotionMask::validate(const cv::Mat& input) {
  validateMask(input, name());
}

cv::Mat ImageType::MotionMask::toRGB(const ImageWrapper<MotionMask>& image) {
  const cv::Mat motion_mask = image.clone();
  return utils::labelMaskToRGB(motion_mask, background_label);
}

void ImageType::ClassSegmentation::validate(const cv::Mat& input) {
  validateMask(input, name());
}

cv::Mat ImageType::ClassSegmentation::toRGB(
    const ImageWrapper<ClassSegmentation>& image) {
  const cv::Mat& class_seg_mask = image;
  return utils::labelMaskToRGB(class_seg_mask,
                               (int)ClassSegmentation::Labels::Undefined);
}

}  // namespace dyno
