#pragma once

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "dynosam/frontend/FrontendParams.hpp"
#include "dynosam/frontend/vision/FeatureTrackerBase.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam/frontend/vision/StaticFeatureTracker.hpp"
#include "dynosam_nn/ObjectDetector.hpp"
#include "dynosam_sensors/Camera.hpp"
#include "dynosam_sensors/Feature.hpp"

namespace dyno {

// Should just be called tracker or something as also does object tracking!
class FeatureTrackerFast {
 public:
  DYNO_POINTER_TYPEDEFS(FeatureTrackerFast)

  FeatureTrackerFast(const FrontendParams& params, Camera::Ptr camera,
                     ImageDisplayQueue* display_queue = nullptr);
  virtual ~FeatureTrackerFast() {}

  Frame::Ptr track(FrameId frame_id, Timestamp timestamp,
                   const ImageContainer& image_container,
                   const std::optional<gtsam::Rot3>& R_km1_k = {});

 private:
  void initDeviceMemory(const cv::Size& size);

  struct ObjectDetectionImpl {
    FeatureTrackerFast* parent;

    ObjectDetectionImpl(FeatureTrackerFast* parent_);

    ObjectDetectionResult detectAndTrack(
        const ImageContainer& image_container) const;
    ObjectDetectionResult detectionViaObjectMask(
        const ImageContainer& image_container) const;
    ObjectDetectionResult detectionViaInference(
        const ImageContainer& image_container) const;
  };

  ObjectDetectionImpl object_detection_impl_;
  ObjectDetectionEngine::Ptr object_detection_engine_;

  // using DetailedCornerTracksPerObject = gtsam::FastMap<ObjectId,
  // DetailedCornerTracks>; DetailedCornerTracksPerObject calcKLTBatch(
  //     const cv::Mat& mono,
  //     const cv::Mat& object_mask);

  // void buildOpticalFlowPyramid(const cv::Mat& mono,
  //                         std::vector<cv::Mat>& pyramid) const;

 private:
  const FrontendParams frontend_params_;
  TrackletIdManager& tracklet_id_manager;

  Frame::Ptr prev_frame_;
  cv::Mat prev_mono_;
  //! Image pyramid for the previous mono frame
  std::vector<cv::Mat> prev_mono_pyr_;
  // //! Binary image used feature detection mask for the previous frame
  // cv::Mat prev_detection_mask_;
  //! Object mask for the previous frame
  cv::Mat prev_object_mask_;

  //! Page locked host allocations which own the image memory. Owns memory
  cv::cuda::HostMem h_mono_;
  cv::cuda::HostMem h_eig_;
  //! cv::Mat headers pointing to the page-locked allocations above. Does not
  //! own memory
  cv::Mat mono_;
  cv::Mat eig_;

  cv::cuda::GpuMat d_mono_;
  cv::cuda::GpuMat d_eig_;

  cv::cuda::Stream stream_;
  cv::Ptr<cv::cuda::CornernessCriteria> detector_;

  //   struct TrackingInfo {
  //     ObjectId object_id{};
  //     FrameId last_detection{};
  //     size_t num_last_detected_features{0};
  //     size_t num_last_tracked{0};
  //   };

  //   gtsam::FastMap<ObjectId, CornerTracks> previous_tracks_;

  struct CornersPerDetectionParams {
    ObjectId object_id;
    //! Binary object/detection mask
    cv::Mat mask;
    cv::Rect bbox;
    // num total corners to extract
    int max_corners;
    // int corners_after_anms;
    float min_distance;
  };

  // optical flow params
  const cv::Size win_size_;
  const int max_level_;
  const cv::TermCriteria criteria_;
};

}  // namespace dyno
