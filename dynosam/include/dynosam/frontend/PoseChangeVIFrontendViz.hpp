#pragma once

#include "dynosam/frontend/vision/FeatureTrackerBase.hpp"  //ImageTracksParams
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam_common/MotionKeyFrame.hpp"
#include "dynosam_common/Types.hpp"

namespace dyno {
/**
 * @brief Mostly vibe-coded class to do some cool temporal vizusalistions of the
 * objects plus general feature tracking stuff.
 *
 * This essentially replaces the FeatureTrackerBase#computeFeatureTracks
 * function with better visualisation and also detail that is specific to the
 * PoseChangeVIFrontend
 *
 */
class ViTrackingViz {
 public:
  DYNO_POINTER_TYPEDEFS(ViTrackingViz)

  ViTrackingViz(const ImageTracksParams& viz_params);

  struct Data {
    ObjectTrackingStatusMap object_tracking_statuses;
    TrackingQuality camera_tracking_quality;
    StatusLandmarkVector camera_tracking_points;
    KeyframeInfo keyframe_info;
  };

  cv::Mat vizTracking(const Frame& frame_km1, const Frame& frame_k,
                      const Data& data = {});

 private:
  struct TemporalObjectState {
    ObjectId object_id;

    Timestamp first_seen_time = -1.0;
    Timestamp last_seen_time = -1.0;

    float appear_progress = 0.0f;

    mutable std::array<cv::Point2f, 4> filtered_corners;
    mutable bool corners_initialized = false;
  };

  void drawStaticTracks(cv::Mat& img, std::string& info, const Frame& frame_km1,
                        const Frame& frame_k, const Data& data);

  void drawDynamicTracks(cv::Mat& img, std::string& info,
                         const Frame& frame_km1, const Frame& frame_k,
                         const Data& data);

  void drawAnimatedBox(cv::Mat& img, const cv::Rect& bbox,
                       const TemporalObjectState& state) const;

  void writeFrameInfo(cv::Mat& img, const std::string& info_string) const;

  ImageTracksParams viz_params_;
  //! How long (in seconds) for 'lock on'
  float appear_duration_sec_ = 0.6f;
  //! higher = snappier
  float corner_smoothing_alpha_ = 0.6f;
  std::unordered_map<ObjectId, TemporalObjectState> states_;

  // internal camera keyframe count to incremental when a new CKF is made
  // just for display
  // shoulkd shoudl be equivalent to the CKF_index utilised in the backend
  int CKF_count = 0;
  gtsam::FastMap<ObjectId, int> OKF_count_;
};

}  // namespace dyno
