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

#pragma once

#include <functional>
#include <opencv4/opencv2/opencv.hpp>

#include "dynosam/common/Types.hpp"
#include "dynosam/frontend/anms/NonMaximumSuppression.h"
#include "dynosam/frontend/vision/TrackerParams.hpp"
#include "dynosam/utils/Macros.hpp"
#include "dynosam/utils/OpenCVUtils.hpp"

namespace dyno {

/**
 * @brief Base class for any feature detector
 *
 */
class FeatureDetector {
 public:
  DYNO_POINTER_TYPEDEFS(FeatureDetector)

  FeatureDetector() = default;
  virtual ~FeatureDetector() = default;
  virtual void detect(const cv::Mat& image, KeypointsCV& keypoints,
                      const cv::Mat& detection_mask) = 0;
};

class FunctionalDetector : public FeatureDetector {
 public:
  DYNO_POINTER_TYPEDEFS(FunctionalDetector)
  // Function interface to extract keypoints from an image (arg0) with an
  // optional keypoint mask (arg2)
  using Func =
      std::function<void(const cv::Mat&, KeypointsCV&, const cv::Mat&)>;

  FunctionalDetector(const Func& detection_func)
      : detection_func_(detection_func) {}
  virtual ~FunctionalDetector() = default;

  inline void detect(const cv::Mat& image, KeypointsCV& keypoints,
                     const cv::Mat& detection_mask = cv::Mat()) override {
    detection_func_(image, keypoints, detection_mask);
  }

  /**
   * @brief Factory method that uses the TrackerParams::feature_detector_type to
   * determine the detector
   *
   */
  static FunctionalDetector::Ptr FactoryCreate(
      const TrackerParams& tracker_params);

  // Creation function that can be specalised for the detector to be created
  template <typename DETECTOR>
  static FunctionalDetector::Ptr Create(const TrackerParams& tracker_params);

 protected:
  Func detection_func_;  //! Function that extracts keypoints from the implicit
                         //! (sort of pimpled) detector
};

/**
 * @brief Wrapper on classic sparse feature detection method
 * and integrates the option for adaptive non-maxima supression and sub-pixel
 * refinement
 *
 */
class SparseFeatureDetector {
 public:
  DYNO_POINTER_TYPEDEFS(SparseFeatureDetector)

  SparseFeatureDetector(const TrackerParams& tracker_params,
                        const FeatureDetector::Ptr& feature_detector);

  /**
   * @brief Detects features on the input image using the feature detector.
   *
   * The features are then spaced out using adaptive non-maxima-supression and
   * refined using subpixel refinement (via cornerSubPix), if enabled.
   *
   * The number of tracked points indicates how many currently tracked points we
   * currently have; this is used to calculate how many more features we need to
   * reach the minimum number features per frame.
   *
   * @param image
   * @param keypoints
   * @param number_tracked
   * @param detection_mask
   */
  void detect(const cv::Mat& image, KeypointsCV& keypoints, int number_tracked,
              const cv::Mat& detection_mask = cv::Mat());

 private:
  const TrackerParams tracker_params_;
  FeatureDetector::Ptr feature_detector_;
  cv::Ptr<cv::CLAHE> clahe_;
  NonMaximumSuppression::UniquePtr non_maximum_supression_;
};

}  // namespace dyno
