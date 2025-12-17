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

#include "dynosam/frontend/vision/FeatureDetector.hpp"

#include <glog/logging.h>

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>

#include "dynosam/frontend/vision/ORBextractor.hpp"
#include "dynosam_common/Cuda.hpp"
#include "dynosam_common/utils/TimingStats.hpp"

#ifdef DYNO_CUDA_OPENCV_ENABLED
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif

namespace dyno {

#ifdef DYNO_CUDA_OPENCV_ENABLED

struct GFTTDetectorCUDA {};

template <>
FunctionalDetector::Ptr FunctionalDetector::Create<GFTTDetectorCUDA>(
    const TrackerParams& tracker_params) {
  LOG(INFO) << "Creating cv::cuda::createGoodFeaturesToTrackDetector";

  cv::Ptr<cv::cuda::CornersDetector> detector =
      cv::cuda::createGoodFeaturesToTrackDetector(
          CV_8UC1, tracker_params.max_nr_keypoints_before_anms,
          tracker_params.gfft_params.quality_level,
          tracker_params.min_distance_btw_tracked_and_detected_static_features,
          tracker_params.gfft_params.block_size,
          tracker_params.gfft_params.use_harris_corner_detector,
          tracker_params.gfft_params.k);

  // auto feature_detector_ = cv::GFTTDetector::create(
  //     tracker_params.max_nr_keypoints_before_anms,
  //     tracker_params.gfft_params.quality_level,
  //     tracker_params.min_distance_btw_tracked_and_detected_static_features,
  //     tracker_params.gfft_params.block_size,
  //     tracker_params.gfft_params.use_harris_corner_detector,
  //     tracker_params.gfft_params.k);
  auto functional_detector = [=](const cv::Mat& img, KeypointsCV& keypoints,
                                 const cv::Mat& mask) -> void {
    // Detect keypoints
    // Upload to GPU
    unsigned int width = img.size().width;
    unsigned int height = img.size().height;

    cv::cuda::GpuMat d_img(height, width, img.type());
    d_img.upload(img);  // simple and portable

    cv::cuda::GpuMat d_mask(height, width, mask.type());
    d_mask.upload(mask);  // simple and portable

    cv::cuda::GpuMat keypointsGPU;
    detector->detect(d_img, keypointsGPU, d_mask);

    std::vector<cv::Point2f> points;
    keypointsGPU.download(points);

    cv::KeyPoint::convert(points, keypoints);
  };

  return std::make_shared<FunctionalDetector>(functional_detector);
}
#endif

template <>
FunctionalDetector::Ptr FunctionalDetector::Create<cv::GFTTDetector>(
    const TrackerParams& tracker_params) {
  LOG(INFO) << "Creating cv::GFTTDetector";

  auto feature_detector_ = cv::GFTTDetector::create(
      tracker_params.max_nr_keypoints_before_anms,
      tracker_params.gfft_params.quality_level,
      tracker_params.min_distance_btw_tracked_and_detected_static_features,
      tracker_params.gfft_params.block_size,
      tracker_params.gfft_params.use_harris_corner_detector,
      tracker_params.gfft_params.k);
  auto functional_detector = [=](const cv::Mat& img, KeypointsCV& keypoints,
                                 const cv::Mat& mask) -> void {
    CHECK_NOTNULL(feature_detector_)->detect(img, keypoints, mask);
  };

  return std::make_shared<FunctionalDetector>(functional_detector);
}

template <>
FunctionalDetector::Ptr FunctionalDetector::Create<ORBextractor>(
    const TrackerParams& tracker_params) {
  LOG(INFO) << "Creating dyno::ORBextractor";

  auto orb_detector_ = std::make_shared<ORBextractor>(
      tracker_params.max_nr_keypoints_before_anms,
      tracker_params.orb_params.scale_factor,
      tracker_params.orb_params.n_levels,
      tracker_params.orb_params.init_threshold_fast,
      tracker_params.orb_params.min_threshold_fast);

  // NOTE that the mask is not used in this implementation
  auto functional_detector = [=](const cv::Mat& img, KeypointsCV& keypoints,
                                 const cv::Mat&) -> void {
    CHECK_NOTNULL(orb_detector_);
    // mask and descriptors are empty
    cv::Mat descriptors;
    orb_detector_->operator()(img, cv::Mat(), keypoints, descriptors);
  };

  return std::make_shared<FunctionalDetector>(functional_detector);
}

FunctionalDetector::Ptr FunctionalDetector::FactoryCreate(
    const TrackerParams& tracker_params) {
  using FDT = TrackerParams::FeatureDetectorType;
  switch (tracker_params.feature_detector_type) {
    case FDT::GFTT:
      return FunctionalDetector::Create<cv::GFTTDetector>(tracker_params);
    case FDT::ORB_SLAM_ORB:
      return FunctionalDetector::Create<ORBextractor>(tracker_params);
    case FDT::GFFT_CUDA: {
      // TODO: this should actually be a #ifdef because
      // Create<cv::cuda::FastFeatureDetector> is conditionally compiled
      if (utils::opencvCudaAvailable()) {
        return FunctionalDetector::Create<GFTTDetectorCUDA>(tracker_params);
      } else {
        LOG(WARNING) << "GFFT_CUDA selected but OPENCV CUDA not enabled. "
                        "Falling back to GFFT";
        return FunctionalDetector::Create<cv::GFTTDetector>(tracker_params);
      }
    }
    default:
      LOG(ERROR) << "Unknown Feature detection type!";
      return nullptr;
      break;
  }
}

SparseFeatureDetector::SparseFeatureDetector(
    const TrackerParams& tracker_params,
    const FeatureDetector::Ptr& feature_detector)
    : tracker_params_(tracker_params),
      feature_detector_(CHECK_NOTNULL(feature_detector)),
      clahe_(nullptr),
      non_maximum_supression_(nullptr) {
  if (tracker_params_.use_clahe_filter)
    clahe_ = cv::createCLAHE(2.0, cv::Size(8, 8));  // TODO: make params

  if (tracker_params.use_anms)
    non_maximum_supression_ = std::make_unique<AdaptiveNonMaximumSuppression>(
        tracker_params.anms_params.non_max_suppression_type);
}

void SparseFeatureDetector::detect(const cv::Mat& image, KeypointsCV& keypoints,
                                   int number_tracked,
                                   const cv::Mat& detection_mask) {
  cv::Mat processed_image = image.clone();

  // pre-process image if required
  if (clahe_) clahe_->apply(processed_image, processed_image);

  std::vector<cv::KeyPoint> raw_keypoints;
  {
    utils::ChronoTimingStats timer("feature_detector.detect");
    feature_detector_->detect(processed_image, raw_keypoints, detection_mask);
  }

  std::vector<cv::KeyPoint>& max_keypoints = raw_keypoints;
  if (tracker_params_.use_anms) {
    // calculate number of corners needed
    int nr_corners_needed =
        std::max(tracker_params_.max_features_per_frame - number_tracked, 0);

    static constexpr float tolerance = 0.1;

    const auto& anms_params = tracker_params_.anms_params;
    Eigen::MatrixXd binning_mask = anms_params.binning_mask;

    {
      utils::ChronoTimingStats timer("feature_detector.anms");
      max_keypoints = non_maximum_supression_->suppressNonMax(
          raw_keypoints, nr_corners_needed, tolerance, processed_image.cols,
          processed_image.rows, anms_params.nr_horizontal_bins,
          anms_params.nr_vertical_bins, binning_mask);
    }
  }

  if (tracker_params_.use_subpixel_corner_refinement &&
      max_keypoints.size() > 0u) {
    // Convert keypoints to points
    std::vector<cv::Point2f> points;
    cv::KeyPoint::convert(max_keypoints, points);

    const auto& subpixel_corner_refinement_params =
        tracker_params_.subpixel_corner_refinement_params;
    utils::ChronoTimingStats timer("feature_detector.sub_pix");
    cv::cornerSubPix(processed_image, points,
                     subpixel_corner_refinement_params.window_size,
                     subpixel_corner_refinement_params.zero_zone,
                     subpixel_corner_refinement_params.criteria);

    cv::KeyPoint::convert(points, max_keypoints);
  }

  keypoints = max_keypoints;
}

}  // namespace dyno
