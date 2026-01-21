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

#include <opencv2/cudaoptflow.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "dynosam/frontend/vision/FeatureDetector.hpp"
#include "dynosam/frontend/vision/FeatureTrackerBase.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam/frontend/vision/ORBextractor.hpp"
#include "dynosam/frontend/vision/OccupancyGrid2D.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_cv/Camera.hpp"

namespace dyno {

class StaticFeatureTracker : public FeatureTrackerBase {
 public:
  DYNO_POINTER_TYPEDEFS(StaticFeatureTracker)
  StaticFeatureTracker(const TrackerParams& params, Camera::Ptr camera,
                       ImageDisplayQueue* display_queue);

  virtual ~StaticFeatureTracker() {}

  /**
   * @brief Base function to perform feature tracking on static points.
   *
   * Is not intended to be state-ful (ie. contains a reference to the last
   * frame) and instead we pass in the previous frame everytime.
   *
   * If the previous frame is null this should indicate that this is the first
   * frame to be tracked!
   *
   *
   * @param previous_frame Frame::Ptr. Previous frame (k-1) with filled-out
   * features to be tracked.
   * @param image_container const ImageContainer&. Contains current (k) images
   * which will be tracked from the previous frame.
   * @param tracker_info FeatureTrackerInfo&. Tracking metadata to be filled
   * out.
   * @param detection_mask const cv::Mat& A detection mask in the opencv feature
   * tracking form: CV_8UC1 where white pixels (255) are valid and black pixels
   * (0) should not be detected on
   * @return FeatureContainer Contains all successsfully tracked features.
   */
  virtual FeatureContainer trackStatic(
      Frame::Ptr previous_frame, const ImageContainer& image_container,
      FeatureTrackerInfo& tracker_info, const cv::Mat& detection_mask,
      const std::optional<gtsam::Rot3>& R_km1_k) = 0;
};

// currently assumes flow that is k to k + 1 (gross!!)
class ExternalFlowFeatureTracker : public StaticFeatureTracker {
 public:
  ExternalFlowFeatureTracker(const TrackerParams& params, Camera::Ptr camera,
                             ImageDisplayQueue* display_queue);
  FeatureContainer trackStatic(
      Frame::Ptr previous_frame, const ImageContainer& image_container,
      FeatureTrackerInfo& tracker_info, const cv::Mat& detection_mask,
      const std::optional<gtsam::Rot3>& R_km1_k = {}) override;

 private:
  Feature::Ptr constructStaticFeature(const ImageContainer& image_container,
                                      const Keypoint& kp, size_t age,
                                      TrackletId tracklet_id,
                                      FrameId frame_id) const;

 private:
  OccupandyGrid2D static_grid_;  //! Grid used to feature bin static features
  ORBextractor::UniquePtr orb_detector_{nullptr};

  const size_t static_cell_size =
      15;  // This tracker has practically been depricated so we hardocde in the
           // params ;)
};

/**
 * @brief Static feature tracking that tracks sparse feature set using the
 * iterative Lucas-Kanade method with pyramids.
 *
 */
class KltFeatureTracker : public StaticFeatureTracker {
 public:
  KltFeatureTracker(const TrackerParams& params, Camera::Ptr camera,
                    ImageDisplayQueue* display_queue);

  /**
   * @brief Track features between frames k-1 and the current set of images (at
   * k).
   *
   * General algorithm is:
   * If not previous frame (ie. is null)
   *  - preprocess input images (equalizeImage)
   *  - detect feature in image container (detectFeatures)
   * Else
   *  - preprocess input images (equalizeImage)
   *  - Collect inlier Features from the previous frame
   *  - Track points (trackPoints)
   *      - apply KLT tracking algorithm
   *      - apply geometric verification to all features
   *      - add tracks and check all features are valid (e.g lie on static
   * points etc..)
   *      - if number of tracks < threshold (max_features_per_frame_)
   *          - reapply feature detection NOTE: in a perfect world we would
   * detect on the previous image and then re-track on the current image, but I
   * didnt implement it like this and, in reality, we we will only drop below
   * the threshold for one frame. As long as we're tracking a good number of
   * features this does not effect the accuracy that much :)
   *
   *
   *
   * @param previous_frame Frame::Ptr. Previous frame (k-1) with filled-out
   * features to be tracked.
   * @param image_container const ImageContainer&. Contains current (k) images
   * which will be tracked from the previous frame.
   * @param tracker_info FeatureTrackerInfo&. Tracking metadata to be filled
   * out.
   * @param detection_mask const cv::Mat& A detection mask in the opencv feature
   * tracking form: CV_8UC1 where white pixels (255) are valid and black pixels
   * (0) should not be detected on
   * @return FeatureContainer Contains all successsfully tracked features.
   */
  FeatureContainer trackStatic(
      Frame::Ptr previous_frame, const ImageContainer& image_container,
      FeatureTrackerInfo& tracker_info, const cv::Mat& detection_mask,
      const std::optional<gtsam::Rot3>& R_km1_k = {}) override;

 private:
  /**
   * @brief Outputs a CLAHE equalized greyscale image from the input RGB, which
   * will be used to detect and track features
   *
   * @param image_container
   * @param equialized_greyscale
   */
  void equalizeImage(const ImageContainer& image_container,
                     cv::Mat& equialized_greyscale) const;

  /**
   * @brief Detects features on the input image using the feature detector.
   *
   * The features are then spaced out using adaptive non-maxima-supression and
   * refined using subpixel refinement (via cornerSubPix).
   *
   * The number of tracked points indicates how many currently tracked points we
   * currently have; this is used to calculate how many more features we need to
   * reach the minimum number features per frame.
   *
   * @param processed_img
   * @param number_tracked
   * @param mask
   * @return std::vector<cv::Point2f>
   */
  std::vector<cv::Point2f> detectRawFeatures(const cv::Mat& processed_img,
                                             int number_tracked,
                                             const cv::Mat& mask = cv::Mat());

  // image container associated with the processed image
  bool detectFeatures(const cv::Mat& processed_img,
                      const ImageContainer& image_container,
                      const FeatureContainer& current_features,
                      FeatureContainer& new_features,
                      const cv::Mat& detection_mask);

  /**
   * @brief Tracks features using KLT tracker between the previous image and the
   * current one.
   *
   * Once tracked, geometric outlier rejection is used to sample the inliers
   * and, if not enough features are tracked, new features are detected to be
   * tracked in the previous frame.
   *
   * The input previous_features, is a container of INLIER features tracked from
   * the previous frame and will be used to set the KLT tracker. The new tracks
   * (and detections) are dumped into tracked_features and the features that
   * were poorly tracked from the previous frame are identified in
   * outlier_previous_features; this vector corresponds to features in
   * previous_features.
   *
   * @param current_processed_img
   * @param previous_processed_img
   * @param image_container
   * @param previous_features
   * @param tracked_features
   * @param outlier_previous_features
   * @param tracker_info
   * @param detection_mask
   * @param R_km1_k
   * @return true
   * @return false
   */
  bool trackPoints(const cv::Mat& current_processed_img,
                   const cv::Mat& previous_processed_img,
                   const ImageContainer& image_container,
                   const FeatureContainer& previous_features,
                   FeatureContainer& tracked_features,
                   TrackletIds& outlier_previous_features,
                   FeatureTrackerInfo& tracker_info,
                   const cv::Mat& detection_mask,
                   const std::optional<gtsam::Rot3>& R_km1_k);

  /**
   * @brief Geometric verification using homograph + RANSAC.
   * Input vectors have the same size and are a 1-to-1 of feature
   * correspondences between old (k-1) and new (k) featurs.
   *
   * @param good_old const std::vector<cv::Point2f>&
   * @param good_new const std::vector<cv::Point2f>&
   * @return cv::Mat
   */
  cv::Mat geometricVerification(const std::vector<cv::Point2f>& good_old,
                                const std::vector<cv::Point2f>& good_new) const;

  bool shouldResample(const FeatureContainer& tracked_features) const;

  /**
   * @brief A more concise static feature constructor that the one in
   * ExternalFlowFeatureTracker.
   *
   * Expects the current kp to already be checked (ie. lies within the image,
   * lies on a static pixel etc). This function does the handling of the
   * tracklet ids (ie. incrementing the id in the TrackletIdManager).
   *
   * If the feature tracked age is greater than max_feature_track_age_, we
   * relabel it with a new tracklet id to prevent massive observation growth in
   * the backend.
   *
   * Function will also update the measured flow in the previous feature (silly
   * old API...)
   *
   * As a result should never return nullptr
   *
   * @param kp_current
   * @param previous_feature
   * @param tracklet_id
   * @param frame_id
   * @return Feature::Ptr
   */
  Feature::Ptr constructStaticFeatureFromPrevious(const Keypoint& kp_current,
                                                  Feature::Ptr previous_feature,
                                                  const TrackletId tracklet_id,
                                                  const FrameId frame_id) const;

  /**
   * @brief Construct a new static feature with a unique tracklet id
   *
   * @param kp_current
   * @param frame_id
   * @return Feature::Ptr
   */
  Feature::Ptr constructNewStaticFeature(const Keypoint& kp_current,
                                         const FrameId frame_id) const;

 private:
  SparseFeatureDetector::Ptr detector_;

  // for now!
  cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> lk_cuda_tracker_;
};

}  // namespace dyno
