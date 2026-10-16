#pragma once

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "dynosam2/common/Parameters.hpp"
#include "dynosam2/frontend/FeatureBlockContainer.hpp"
#include "dynosam2/frontend/SparseFlow.hpp"
#include "dynosam2/frontend/Utils.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_core/DynamicObjects.hpp"
#include "dynosam_nn/ObjectDetector.hpp"
#include "dynosam_sensors/Camera.hpp"
#include "dynosam_sensors/ImageContainer.hpp"

using namespace dyno;

namespace dyno2 {

class FeatureTracker {
 public:
  DYNO_POINTER_TYPEDEFS(FeatureTracker)

  struct Result {
    ObjectDetectionResult object_detection;
    // Points to data stored in the tracker itself
    FeatureBlockContainer* featues;
  };

  FeatureTracker(const FrontendParameters& parameters, Camera::Ptr camera);

  Result track(ImageContainer& image_container,
               const std::optional<gtsam::Rot3>& R_km1_k = {});

 private:
  struct ObjectDetectionImpl {
    FeatureTracker* parent;

    ObjectDetectionImpl(FeatureTracker* parent_);

    std::pair<ObjectDetectionResult, bool> detectAndTrack(
        const ImageContainer& image_container) const;
    std::pair<ObjectDetectionResult, bool> detectionViaObjectMask(
        const ImageContainer& image_container) const;
    std::pair<ObjectDetectionResult, bool> detectionViaInference(
        const ImageContainer& image_container) const;
  };

  struct GfttDetector {
    struct Param {
      ObjectId object_id;
      //! Binary object/detection mask
      cv::Mat mask;
      cv::Rect bbox;
      // num total corners to extract
      int max_corners;
      // int corners_after_anms;
      float min_distance;
      //! Number of final corners after extraction and pruning
      //! If set to zero, assume same as max_corners
      int num_corners_needed{0};
    };

    struct Params : public std::vector<Param> {
      using Base = std::vector<Param>;
      using Base::Base;

      float quality_level{0.01};
      int block_size{3};
    };

    GfttDetector(const cv::Size& size);
    FeatureBlockContainer calc(const cv::Mat& mono, const cv::Mat& object_mask,
                               const Params& params);

    //! Global id count - since we operate on all tracks at the same time we
    //! dont need this to be thread safe
    TrackletId id_count_;

    /// @brief Get N new tracklet ids and increment the global count by N. Not
    /// thread safe
    /// @param N
    /// @return
    TrackletIds getNewIdsByChunk(size_t N);

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
    // impl detector for min-eigenvalue corner response.
    cv::Ptr<cv::cuda::CornernessCriteria> detector_;
  };

  struct FlowTrackingStats {
    //! Valid features tracked by LKT
    size_t tracked_after_flow{0};
    //! Num tracks after outlier rejection (ie. verification)
    size_t tracked_after_or{0};
    //! Number of tracks in the previous frame
    size_t num_previous_tracks{0};

    float survivalRatio() const {
      if (num_previous_tracks > 0 && tracked_after_or > 0) {
        return (float)tracked_after_or / num_previous_tracks;
      } else {
        return 0.0;
      }
    }
  };
  using FlowTrackingStatsMap = gtsam::FastMap<ObjectId, FlowTrackingStats>;

  std::pair<FeatureBlockContainer, FlowTrackingStatsMap> sparseFeatureTrack(
      const cv::Mat& mono, const cv::Mat& object_mask);

  // R_previous_curr is a rotation from previous to current (ie p_curr = R *
  // p_previous)
  bool predictKeypointsGivenRotation(
      const std::vector<cv::Point2f>& previous_pts,
      const gtsam::Rot3& R_previous_curr,
      std::vector<cv::Point2f>& predicted_pts_curr) const;

  void fillDetectionParam(ObjectId object_id, const cv::Mat& mask,
                          const cv::Rect& bounding_box, int current_tracks,
                          GfttDetector::Param& detection_param) const;

  /**
   * @brief Get the desired minimum distance between features for detection,
   * depending on if the object is static (object_id = 0) or dynamic.
   *
   * @param object_id
   * @return int
   */
  size_t getMinFeatureDistance(ObjectId object_id) const;
  /**
   * @brief Get the maximum desired corners to be extracted depending on if the
   * object is static (object_id = 0) or dynamic.
   *
   * @param object_id
   * @return size_t
   */
  size_t getMaxDetectionCorners(ObjectId object_id) const;
  size_t getMaxTrackingCorners(ObjectId object_id) const;
  size_t getMinAllowableTracks(ObjectId object_id) const;

 private:
  FrontendParameters parameters_;
  Camera::Ptr camera_;
  //! Feature detectior
  GfttDetector feature_detector_;
  ObjectDetectionImpl object_detection_impl_;
  //! Sparse LK tracker
  OpticalFlowLK optical_flow_impl_;

  // Cached camera matrices
  cv::Matx33f K_cv_;
  cv::Matx33f K_inv_cv_;

  ObjectDetectionEngine::Ptr object_detection_engine_;
  std::vector<cv::Mat> curr_mono_pyr_;
  cv::Mat prev_mono_;
  //! Image pyramid for the previous mono frame
  std::vector<cv::Mat> prev_mono_pyr_;
  gtsam::FastMap<ObjectId, cv::Mat> previous_object_detection_masks_;
  //! Object mask for the previous frame
  cv::Mat prev_object_mask_;

  //! Set of previous (inlier) features to track against
  FeatureBlockContainer previous_features_;
};

}  // namespace dyno2
