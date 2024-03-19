/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 */

#include "dynosam/common/ImageTypes.hpp"
#include "dynosam/utils/OpenCVUtils.hpp"
#include <opencv4/opencv2/opencv.hpp>

namespace dyno {

void validateMask(const cv::Mat& input, const std::string& name) {
    //we guanrantee with a static assert that the SemanticMask and MotionMask types are the same
    const static std::string expected_type = utils::cvTypeToString(ImageType::MotionMask::OpenCVType);
    if(input.type() != ImageType::MotionMask::OpenCVType) {
        throw InvalidImageTypeException(
            name + " image was not " + expected_type + ". Input image type was " + utils::cvTypeToString(input.type())
        );
    }
}

void validateSingleImage(const cv::Mat& input, int expected_type, const std::string& name) {
    if(input.type() != expected_type) {
        throw InvalidImageTypeException(
            name + " image was not " +  utils::cvTypeToString(expected_type) + " - Input image type was " + utils::cvTypeToString(input.type())
        );
    }
}


void ImageType::RGBMono::validate(const cv::Mat& input) {
    if(input.type() != CV_8UC1 && input.type() != CV_8UC3) {
        throw InvalidImageTypeException(
            "RGBMono image was not CV_8UC1 or CV_8UC3. Input image type was " + utils::cvTypeToString(input.type())
        );
    }
}

cv::Mat ImageType::RGBMono::toRGB(const ImageWrapper<RGBMono>& image) {
    const cv::Mat& mat = image;
    const auto channels = mat.channels();
    if(channels == 3) {
        return image;
    }
    else if(channels == 4) {
        cv::Mat rgb;
        cv::cvtColor(mat, rgb, cv::COLOR_RGBA2RGB);
        return rgb;
    }
    else if (channels == 1) {
        //grey scale but we want rgb so that we can draw colours on it or whatever
        cv::Mat rgb;
        cv::cvtColor(mat, rgb, cv::COLOR_GRAY2RGB);
        return rgb;
    }
    else {
        return image;
    }
}

cv::Mat ImageType::RGBMono::toMono(const ImageWrapper<RGBMono>& image) {
    const cv::Mat& mat = image;
    const auto channels = mat.channels();
    if(channels == 1) {
        return image;
    }
    if (channels == 3)
    {
        cv::Mat mono;
        cv::cvtColor(mat, mono, cv::COLOR_RGB2GRAY);
        return mono;
    }
    else if (channels == 4)
    {
        cv::Mat mono;
        cv::cvtColor(mat, mono, cv::COLOR_RGBA2GRAY);
        return mono;
    }
    else {
        return image;
    }
}


void ImageType::Depth::validate(const cv::Mat& input){
    validateSingleImage(input, OpenCVType, name());
}

cv::Mat ImageType::Depth::toRGB(const ImageWrapper<Depth>& image) {
   return image;
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

cv::Mat ImageType::SemanticMask::toRGB(const ImageWrapper<SemanticMask>& image) {
    const cv::Mat& semantic_mask = image;
    return utils::labelMaskToRGB(semantic_mask, background_label);
}


void ImageType::MotionMask::validate(const cv::Mat& input) {
    validateMask(input, name());
}

cv::Mat ImageType::MotionMask::toRGB(const ImageWrapper<MotionMask>& image) {
    const cv::Mat& motion_mask = image;
    return utils::labelMaskToRGB(motion_mask, background_label);
}

void ImageType::ClassSegmentation::validate(const cv::Mat& input) {
    validateMask(input, name());
}


cv::Mat ImageType::ClassSegmentation::toRGB(const ImageWrapper<ClassSegmentation>& image) {
    const cv::Mat& class_seg_mask = image;
    return utils::labelMaskToRGB(class_seg_mask, (int)ClassSegmentation::Labels::Undefined);
}


}
