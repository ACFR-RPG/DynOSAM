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
#include "dynosam/frontend/vision/StaticFeatureTracker.hpp"
#include "dynosam_cv/Camera.hpp"
#include "dynosam_cv/Feature.hpp"

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

  FeatureTracker(const FrontendParams& params, Camera::Ptr camera,
                 ImageDisplayQueue* display_queue = nullptr);
  virtual ~FeatureTracker() {}

  // note: MOTION MASK!!
  // object keyframes should not be part of the function here. How should we get
  // this data? Put in frame or get as function?
  Frame::Ptr track(FrameId frame_id, Timestamp timestamp,
                   const ImageContainer& image_container,
                   const std::optional<gtsam::Rot3>& R_km1_k = {});

  bool stereoTrack(FeatureContainer& left_features,
                   const ImageContainer& image_container) const;

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

  const FrontendParams& frontendParams() const { return frontend_params_; }

 protected:
  // dynamic_detection_mask is for tracking and detection? Which one do we need
  // for re-tracking!? detection mask is additional mask It must be a 8-bit
  // integer matrix with non-zero values in the region of interest, indicating
  // what featues to not track
  void trackDynamic(
      FrameId frame_id, const ImageContainer& image_container,
      FeatureContainer& dynamic_features, ObjectIds& objects_resampled,
      cv::Mat& dynamic_detection_mask,
      const vision_tools::ObjectBoundaryMaskResult& boundary_mask_result);

  void trackDynamicKLT(
      FrameId frame_id, const ImageContainer& image_container,
      FeatureContainer& dynamic_features, ObjectIds& objects_resampled,
      TrackletIds& retroactive_trackslet_ids, cv::Mat& dynamic_detection_mask,
      const vision_tools::ObjectBoundaryMaskResult& boundary_mask_result);

  /* Somewhat depricated. Only used with trackDynamic (ie. non KLT tracking)*/
  void sampleDynamic(FrameId frame_id, const ImageContainer& image_container,
                     const ObjectIds& objects_to_sample,
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
   * @param objects_to_sample  ObjectIds&
   * @param info const FeatureTrackerInfo&
   * @param image_container const ImageContainer&
   * @param dynamic_features_tracked const FeatureContainer&
   * @param boundary_mask_result const vision_tools::ObjectBoundaryMaskResult&
   * @param dynamic_tracking_mask const cv::Mat&
   */
  void requiresSampling(
      ObjectIds& objects_to_sample, FeatureTrackerInfo& info,
      const ImageContainer& image_container,
      const FeatureContainer& dynamic_features_tracked,
      const vision_tools::ObjectBoundaryMaskResult& boundary_mask_result,
      const cv::Mat& dynamic_tracking_mask) const;

  void propogateMask(ImageContainer& image_container);

 private:
  // TODO: for now we loose the actual object detection result if inference was
  // run!
  bool objectDetection(
      vision_tools::ObjectBoundaryMaskResult& boundary_mask_result,
      ImageContainer& image_container);

  void computeImageBounds(const cv::Size& size, int& min_x, int& max_x,
                          int& min_y, int& max_y) const;

 private:
  const FrontendParams frontend_params_;
  //! The frame that will be used as the previous frame next time
  //! track is called. After track, this is actually the frame
  //! that track() returns
  Frame::Ptr previous_frame_{nullptr};

  //! The frame that has just beed used to track on a new frame
  //! is created.
  Frame::Ptr previous_tracked_frame_{nullptr};
  StaticFeatureTracker::UniquePtr static_feature_tracker_;
  //! Dynamic mask detection boarder which indicates valid (255) and invalid (0)
  //! pixles around the boarder of each dynamic object. Mask of type CV_8UC1
  cv::Mat boarder_detection_mask_;

  //! a boolean mask (255 for valid, 0 for invalid) indicating where dynamic
  //! features were detected or tracked on the previous frame
  cv::Mat dynamic_detection_mask_;

  FeatureTrackerInfo info_;

  // OccupandyGrid2D static_grid_; //! Grid used to feature bin static features
  bool initial_computation_{true};

  // for now!
  cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> lk_cuda_tracker_;

  /**
   * @brief Impl LKT dynamic feature tracker.
   * Contains mostly utility functions while most data is stored in the parent
   * tracker.
   *
   */
  struct DynamicTrackerImpl {
    FeatureTracker* parent;
    TrackletIdManager& tracklet_id_manager;
    DynamicTrackerImpl(FeatureTracker* _parent);

    /**
     * @brief Track all dynamic features against the previous frame, filling the
     * feature container with tracked features and marking tracked points on
     * both the detection mask and the labelled detection mask
     *
     * @param previous_frame Frame::Ptr
     * @param current_image_container const ImageContainer&
     * @param dynamic_features FeatureContainer& features tracked against the
     * previous frame
     * @param detection_mask cv::Mat& binary detection mask marked with all
     * valid (255) and invalid (0) pixels
     * @param labelled_detection_mask cv::Mat& pixel level indicator (1...N) of
     * tracked dynamic features
     * @return size_t Number of features tracked
     */
    void trackFromPreviousFrame(Frame::Ptr previous_frame,
                                const ImageContainer& current_image_container,
                                FeatureContainer& dynamic_features,
                                cv::Mat& detection_mask,
                                cv::Mat& labelled_detection_mask);

    void detectNewFeatures(Frame::Ptr previous_frame,
                           const ImageContainer& current_image_container,
                           const ObjectIds& need_new_detections,
                           FeatureContainer& dynamic_features,
                           cv::Mat& detection_mask,
                           TrackletIds& retroactive_tracklets);

    /**
     * @brief
     * Note: keypoints_out and indexed_retroactive_keypoints_out are associated
     * by a common value set in cv::KeyPoint::class_id. This is just some global
     * index and not a tracklet id as this is yet to be construced
     * @param object_id
     * @param current_image_container
     * @param previous_image_container
     * @param detection_mask_previous
     * @param keypoints_out
     * @param indexed_retroactive_keypoints_out
     * @return true
     * @return false
     */
    bool trackRetroactively(
        ObjectId object_id, const std::vector<cv::Point2f>& detected_points,
        const ImageContainer& current_image_container,
        const ImageContainer& previous_image_container,
        const cv::Mat& detection_mask_previous,
        std::vector<KeypointCV>& keypoints_out,
        std::vector<KeypointCV>& indexed_retroactive_keypoints_out);

    /* Create a dynamic feature from a previous feature - inheriting the
     * tracklet id */
    Feature::Ptr featureFromPrevious(const Keypoint& kp_current,
                                     Feature::Ptr previous_feature,
                                     const TrackletId tracklet_id,
                                     const ObjectId object_id,
                                     const FrameId frame_id) const;

    /* Creat a dynamic feature with a new tracklet id  */
    Feature::Ptr newFeature(const Keypoint& kp_current,
                            const ObjectId object_id,
                            const FrameId frame_id) const;
  };
  DynamicTrackerImpl dynamic_lkt_tracker_impl_;

  SparseLKTracker::Ptr lk_tracker_dynamic_;

  const cv::Size klt_window_size_{24, 24};
  const int klt_max_level_ = 4;

  //! Min matches needed for stereo matching. This is minimum needed for outlier
  //! rejection
  static constexpr int kMinStereoMatches = 8;

  ObjectDetectionEngine::Ptr object_detection_;
};

}  // namespace dyno
