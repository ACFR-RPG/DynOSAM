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
#pragma once

#include <opencv4/opencv2/core/types.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "dynosam/common/Types.hpp"

#define CHECK_MAT_TYPES(mat1, mat2)                                            \
  using namespace dyno::utils;                                                 \
  CHECK_EQ(mat1.type(), mat2.type())                                           \
      << "Matricies should be of the same type ( "                             \
      << cvTypeToString(mat1.type()) << " vs. " << cvTypeToString(mat2.type()) \
      << ")."

namespace dyno {

class CameraParams;

template <typename T>
class ParallelOpenCVMat : public cv::ParallelLoopBody {
 public:
  using This = ParallelOpenCVMat<T>;

  ParallelOpenCVMat(cv::Mat& img_ref) : img_(img_ref) {}

  int dataSize() const { return img_.rows * img_.cols; }

  void operator()(const cv::Range& range) const CV_OVERRIDE {
    const T* derived = static_cast<const T*>(this);
    CHECK_NOTNULL(derived);
    for (int r = range.start; r < range.end; r++) {
      int i = r / img_.cols;
      int j = r % img_.cols;
      derived->call(img_, i, j);
    }
  }

  This& run() {
    cv::parallel_for_(cv::Range(0, dataSize()), *this);
    return *this;
  }

  ParallelOpenCVMat& operator=(const ParallelOpenCVMat&) { return *this; };

 protected:
  virtual void call(cv::Mat& img_ref, int i, int j) const = 0;

 private:
  cv::Mat& img_;
};

class FunctionalParallelOpenCVMat
    : public ParallelOpenCVMat<FunctionalParallelOpenCVMat> {
 public:
  using Call = std::function<void(cv::Mat&, int, int)>;
  using Base = ParallelOpenCVMat<FunctionalParallelOpenCVMat>;

  FunctionalParallelOpenCVMat(cv::Mat& img_ref, const Call& call)
      : Base(img_ref), call_(call) {}

 private:
  // allow the base class to access the derived call
  friend Base;

  void call(cv::Mat& img_ref, int i, int j) const override {
    call_(img_ref, i, j);
  }

 private:
  mutable Call call_;
};

namespace utils {

// TODO: this function is really if opencv AVAILABLE, not just enabled (ie. it
// is enabled AND can find a GPU!)
bool opencvCudaAvailable(std::string* message = nullptr);

double calculateIoU(const cv::Rect& a, const cv::Rect& b);

bool cvSizeEqual(const cv::Size& a, const cv::Size& b);
bool cvSizeEqual(const cv::Mat& a, const cv::Mat& b);

/**
 * @brief Returns true of the point is inside bounds of the cv::Mat.
 *
 * @tparam T Point template to work with any cv::Point type
 * @param mat const cv::Mat&
 * @param point const cv::Point_<T>&
 * @return true
 * @return false
 */
template <typename T>
inline bool matContains(const cv::Mat& mat, const cv::Point_<T>& point) {
  const auto x = static_cast<int>(point.x);
  const auto y = static_cast<int>(point.y);
  return x >= 0 && y >= 0 && x < mat.cols && y < mat.rows;
}

// checks if point is within image
void drawCircleInPlace(cv::Mat& img, const cv::Point2d& point,
                       const cv::Scalar& colour, const int radius = 1,
                       const int thickness = 2);

std::string cvTypeToString(int type);
std::string cvTypeToString(const cv::Mat& mat);

// Compares two opencv matrices up to a certain tolerance error.
bool compareCvMatsUpToTol(const cv::Mat& mat1, const cv::Mat& mat2,
                          const double& tol = 1e-7);

cv::Mat concatenateImagesHorizontally(const cv::Mat& left_img,
                                      const cv::Mat& right_img);

cv::Mat concatenateImagesVertically(const cv::Mat& top_img,
                                    const cv::Mat& bottom_img);

void flowToRgb(const cv::Mat& flow, cv::Mat& rgb);

/**
 * @brief Given an image that operates like a mask (ie. is a single channel
 * image where each pixel value correspondes to a label of some type) draws the
 * mask over an input image.
 *
 * The value at each pixel (labels) are treated like object labels which are
 * used to generate a unique colour for the mask. The background label is used
 * to indicate which pixel value should be ignored - this could be the
 * background or simply unknown pixel values.
 *
 *
 * Alpha is used to blend with mask ontop of the rgb image.
 *
 * @param mask const cv::Mat&
 * @param rgb_input const cv::Mat&
 * @param output cv::Mat&
 * @param alpha float default value is 0.7
 * @param background_label int default value is 0
 */
void labelMaskToRGB(const cv::Mat& mask, const cv::Mat& rgb_input,
                    cv::Mat& output, float alpha = 0.7,
                    int background_label = 0);

/**
 * @brief Given an image that operates like a mask (ie. is a single channel
 * image where each pixel value correspondes to a label of some type) draws the
 * mask over an input image.
 *
 * The value at each pixel (labels) are treated like object labels which are
 * used to generate a unique colour for the mask. The background label is used
 * to indicate which pixel value should be ignored - this could be the
 * background or simply unknown pixel values
 *
 * @param mask
 * @param background_label
 * @param rgb
 * @return cv::Mat
 */
cv::Mat labelMaskToRGB(const cv::Mat& mask, int background_label,
                       const cv::Mat& rgb);

/**
 * @brief Same as labelMaskToRGB(mask, background label) but instead draws the
 * mask over a black image so only the mask values are shown
 *
 * @param mask
 * @param background_label
 * @return cv::Mat
 */
cv::Mat labelMaskToRGB(const cv::Mat& mask, int background_label);

/**
 * @brief Allows visualization of disparity image (which is
 * usually outputed as CV_16S or CV_32F from stereo depth reconstruction).
 *
 * @param src cv::InputArray
 * @param dst cv::OutputArray
 * @param unknown_disparity
 */
void getDisparityVis(cv::InputArray src, cv::OutputArray dst,
                     int unknown_disparity = 16320);

// TODO: uh, this was const cv::Mat& and things were still being modified...
// how...?
/**
 * @brief Draws a coloured bounding box and associated label onto a RGB image.
 *
 * Used to draw object bounding boxes and their labels.
 *
 * @param image cv::Mat& rgb image to modify
 * @param label const std::string& label to draw in the top-right corner of the
 * bounding box
 * @param colour const cv::Scalar& colour to draw the bounding box with
 * @param bounding_box const cv::Rect& bounding box to draw.
 * @param bb_thickness const int& Line thickness for the boundong box
 */
void drawLabeledBoundingBox(cv::Mat& image, const std::string& label,
                            const cv::Scalar& colour,
                            const cv::Rect& bounding_box,
                            const int& bb_thickness = 2);

/**
 * @brief Projects SE(3) poses onto the image as an RGB axes.
 * Input poses are expected to be in the camera (local) frame
 *
 * @param image cv::Mat&
 * @param K const cv::Mat& 3x3 calibration matrix
 * @param D const cv::Mat& distortion coeeffs
 * @param poses_c const std::vector<gtsam::Pose3>& Poses in the camera (local)
 * frame to be projected
 * @param scale float The size of the axis in 2D after projection (default: 0.2)
 */
void drawObjectPoseAxes(cv::Mat& image, const cv::Mat& K, const cv::Mat& D,
                        const std::vector<gtsam::Pose3>& poses_c,
                        float scale = 0.2);

/**
 * I have absolutely no idea why but OpenCV seemds to have removed support for
 * the read/write optical flow functions in 4.x I have taken this implementation
 * from OpenCV 3.4
 * (https://github.com/opencv/opencv_contrib/blob/3.4/modules/optflow/src/optical_flow_io.cpp)
 */

/**
 * @brief The function readOpticalFlow loads a flow field from a file and
 * returns it as a single matrix. Resulting Mat has a type CV_32FC2 -
 * floating-point, 2-channel. First channel corresponds to the flow in the
 * horizontal direction (u), second - vertical (v).
 *
 * @param path
 * @return cv::Mat
 */
cv::Mat readOpticalFlow(const std::string& path);

/**
 * @brief The function stores a flow field in a file, returns true on success,
 * false otherwise. The flow field must be a 2-channel, floating-point matrix
 * (CV_32FC2). First channel corresponds to the flow in the horizontal direction
 * (u), second - vertical (v).
 *
 * @param path
 * @param flow
 * @return true
 * @return false
 */
bool writeOpticalFlow(const std::string& path, const cv::Mat& flow);

}  // namespace utils
}  // namespace dyno

#include <yaml-cpp/yaml.h>

namespace YAML {

template <typename T>
struct convert<cv::Size_<T>> {
  static Node encode(const cv::Size_<T>& size) {
    Node node;
    node["width"] = size.width;
    node["height"] = size.height;
    return node;
  }

  static bool decode(const Node& node, cv::Size_<T>& size) {
    if (!node.IsMap()) {
      return false;
    }

    size.width = node["width"].as<T>();
    size.height = node["height"].as<T>();
    return true;
  }
};

}  // namespace YAML
