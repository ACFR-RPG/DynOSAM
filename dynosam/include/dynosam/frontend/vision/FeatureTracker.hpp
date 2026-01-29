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

#include <opencv2/cudaoptflow.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "dynosam/frontend/FrontendParams.hpp"
#include "dynosam/frontend/vision/FeatureTrackerBase.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam/frontend/FrontendModule.hpp"
#include "dynosam/frontend/vision/StaticFeatureTracker.hpp"
#include "dynosam/visualizer/Visualizer-Definitions.hpp"
#include "dynosam_cv/Camera.hpp"
#include "dynosam_cv/Feature.hpp"
#include "dynosam_common/SharedModuleInfo.hpp"

// #include "dynosam_common/DynamicObjects.hpp"
#include "dynosam_nn/ObjectDetector.hpp"

namespace dyno {

/**
 * @brief Feature detector that combines sparse static feature detection and
 * tracking with dense feature detection and tracking on dynamic objects.
 *
 */
class FeatureTracker : public FeatureTrackerBase {
 public:
  DYNO_POINTER_TYPEDEFS(FeatureTracker)

  // and camera?
  // does no processing with any depth
  // if depth is a problem should be handled aftererds and separately
  FeatureTracker(const FrontendParams& params, Camera::Ptr camera,
                 ImageDisplayQueue* display_queue = nullptr);
  virtual ~FeatureTracker() {}

  // note: MOTION MASK!!
  // object keyframes should not be part of the function here. How should we get
  // this data? Put in frame or get as function?
  Frame::Ptr track(FrameId frame_id, Timestamp timestamp,
                   const ImageContainer& image_container,
                   const std::optional<gtsam::Rot3>& R_km1_k = {});

  bool stereoTrack(FeaturePtrs& stereo_features,
                   FeatureContainer& left_features, const cv::Mat& left_image,
                   const cv::Mat& right_image,
                   const double& virtual_baseline) const;

  /**
   * @brief Get the previous frame.
   *
   * Will be null on the first call of track
   *
   * @return Frame::Ptr
   */
  inline Frame::Ptr getPreviousFrame() { return previous_tracked_frame_; }
  inline const Frame::ConstPtr getPreviousFrame() const {
    return previous_tracked_frame_;
  }

  /**
   * @brief Get the most recent frame that has been tracked.
   * After a call to track, this will be the frame that is returned and will
   * track features between getPreviousFrame() to this frame
   *
   * @return Frame::Ptr
   */
  inline Frame::Ptr getCurrentFrame() { return previous_frame_; }

  inline const FeatureTrackerInfo& getTrackerInfo() { return info_; }
  inline const cv::Mat& getBoarderDetectionMask() const {
    return boarder_detection_mask_;
  }

 protected:
  // detection mask is additional mask It must be a 8-bit integer matrix with
  // non-zero values in the region of interest, indicating what featues to not
  // track
  void trackDynamic(
      FrameId frame_id, const ImageContainer& image_container,
      FeatureContainer& dynamic_features, std::set<ObjectId>& object_keyframes,
      const vision_tools::ObjectBoundaryMaskResult& boundary_mask_result);

  void trackDynamicKLT(
      FrameId frame_id, const ImageContainer& image_container,
      FeatureContainer& dynamic_features, std::set<ObjectId>& object_keyframes,
      const vision_tools::ObjectBoundaryMaskResult& boundary_mask_result);

  void sampleDynamic(FrameId frame_id, const ImageContainer& image_container,
                     const std::set<ObjectId>& objects_to_sample,
                     FeatureContainer& dynamic_features,
                     std::set<ObjectId>& objects_sampled,
                     const cv::Mat& detection_mask);
  /**
   * @brief Check which objects require new features to be detected on them
   * based on a set of 'keyframing' criteria.
   *
   * The number of current tracks for each object is encapsualted in
   * FeatureTrackerInfo.
   *
   * @param objects_to_sample  std::set<ObjectId>&
   * @param info const FeatureTrackerInfo&
   * @param image_container const ImageContainer&
   * @param features_per_object const gtsam::FastMap<ObjectId,
   * FeatureContainer>&
   * @param boundary_mask_result const vision_tools::ObjectBoundaryMaskResult&
   * @param dynamic_tracking_mask const cv::Mat&
   */
  void requiresSampling(
      std::set<ObjectId>& objects_to_sample, FeatureTrackerInfo& info,
      const ImageContainer& image_container,
      const gtsam::FastMap<ObjectId, FeatureContainer>& features_per_object,
      const vision_tools::ObjectBoundaryMaskResult& boundary_mask_result,
      const cv::Mat& dynamic_tracking_mask) const;

  void propogateMask(ImageContainer& image_container);

 private:
  // TODO: for now we loose the actual object detection result if inference was
  // run!
  ObjectDetectionResult objectDetection(
      vision_tools::ObjectBoundaryMaskResult& boundary_mask_result,
      ImageContainer& image_container);

  void computeImageBounds(const cv::Size& size, int& min_x, int& max_x,
                          int& min_y, int& max_y) const;

 private:
  const FrontendParams frontend_params_;
  Frame::Ptr previous_frame_{
      nullptr};  //! The frame that will be used as the previous frame next time
                 //! track is called. After track, this is actually the frame
                 //! that track() returns
  Frame::Ptr previous_tracked_frame_{
      nullptr};  //! The frame that has just beed used to track on a new frame
                 //! is created.
  StaticFeatureTracker::UniquePtr static_feature_tracker_;
  //! Dynamic mask detection boarder which indicates valid (255) and invalid (0)
  //! pixles around the boarder of each dynamic object. Mask of type CV_8UC1
  cv::Mat boarder_detection_mask_;

  FeatureTrackerInfo info_;

  // OccupandyGrid2D static_grid_; //! Grid used to feature bin static features
  bool initial_computation_{true};

  // for now!
  cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> lk_cuda_tracker_;
  ObjectDetectionEngine::Ptr object_detection_;
};

}  // namespace dyno
