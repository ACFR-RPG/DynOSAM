#pragma once

#include "dynosam/backend/PoseChangeBackendModule.hpp"
#include "dynosam/formulations/KeyFrameHybridEstimator.hpp"
#include "dynosam/frontend/VIFrontend.hpp"
#include "dynosam/frontend/solvers/HybridObjectMotionSolver.hpp"
#include "dynosam_cv/RGBDCamera.hpp"

namespace dyno {

using PoseChangeBackendSink =
    std::function<void(const PoseChangeInput::ConstPtr&)>;

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
    gtsam::FastMap<ObjectId, ObjectTrackingStatus> object_tracking_statuses;
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

class PoseChangeVIFrontend : public VIFrontend {
 public:
  DYNO_POINTER_TYPEDEFS(PoseChangeVIFrontend)
  PoseChangeVIFrontend(const DynoParams& params, Camera::Ptr camera,
                       HybridFormulationKeyFrame::Ptr formulation,
                       ImageDisplayQueue* display_queue = nullptr,
                       const SharedGroundTruth& shared_ground_truth = {});

  ~PoseChangeVIFrontend();

  /** Add sink to send PC data to the backend */
  void addPoseChangeOutputSink(const PoseChangeBackendSink& func) {
    pose_change_backend_sink_ = func;
  };

  /** Callback triggered when the backend has finished a single update */
  void onBackendUpdateComplete(const PoseChangeUpdateComplete& event);

 private:
  SpinReturn boostrapSpin(VIFrontendInput::ConstPtr input) override;
  SpinReturn nominalSpin(VIFrontendInput::ConstPtr input) override;

  bool withBackend() const {
    //! Use existance of backend sink as proxy logicc for "use backend"
    return (bool)pose_change_backend_sink_;
  }

  void solveObjectMotions(MultiObjectTrajectories& trajectories,
                          ObjectIds& object_with_new_motions,
                          ObjectPoseChangeInfoMap& infos, Frame::Ptr frame_k,
                          Frame::Ptr frame_km1);

  bool shouldFrameBeKeyFrame(Frame::Ptr frame_k, Frame::Ptr frame_km1) const;

  /**
   * Contains relative ego motion information
   * between the current frame j, the previous tracked frame i
   * and the (camera) keyframe we are tracking against.
   *
   */
  struct RelEgoPoseInfo {
    //! The keyframe we are tracking against
    FrameId lkf_id;
    //! Previous frame we tracking against (ie. from). Usually j-1
    FrameId i_id;
    //! The current frame (i.e to)
    FrameId j_id;
    //! Frame ptr at current frame (ie. to)
    Frame::Ptr frame_j;
    //! Nav state at current frame (ie. to)
    gtsam::NavState frontend_nav_state_j;
    ImuFrontend::PimPtr pim_lk_j;
    //! Should exist only if PIM is non null
    ImuMeasurements imu_measurements;
    //! Relative camera pose between from -> to frames
    gtsam::Pose3 T_i_j;
    //! Relative camera pose between last keyframe -> to frames
    gtsam::Pose3 T_lkf_j;
  };

  void handleCameraKeyframe(const RelEgoPoseInfo& rel_lkf_k,
                            const UpdateObservationParams& update_params,
                            PostUpdateData& post_update_data,
                            SinglePoseChangeInput::Ptr pc_input);

  void logBestEstimates() const;
  void logRealTimeObjectClouds(const ObjectIds& objects,
                               FrameId frame_id) const;

  bool solveAndRefineEgoMotion(
      Frame::Ptr frame_k, const Frame::Ptr& frame_km1,
      StatusLandmarkVector& points_W_used, TrackingQuality& tracking_quality,
      std::optional<gtsam::NavState> propogated_nav_state_k = std::nullopt,
      std::optional<gtsam::Rot3> R_km1_k = std::nullopt);

 private:
  HybridFormulationKeyFrame::Ptr formulation_;
  KeyFrameMap::Ptr map_;
  HybridObjectMotionSolver::UniquePtr object_motion_solver_;

  gtsam::NavState nav_state_km1_;
  gtsam::NavState nav_state_lkf_;

  //! Last camera keyframe
  Frame::Ptr lCKF_frame_;

  //! Current trajectories. Copied to the DynoState output.
  //! Only contains trajectories for objects observed at the latest frame
  DynoStateTrajectories dyno_state_;
  MultiObjectTrajectories full_object_trajectories_;

  PoseChangeBackendSink pose_change_backend_sink_;

  // Mapping of intermediate relative motions. Stored by to frame.
  gtsam::FastMap<FrameId, RelEgoPoseInfo> rel_egopose_infos_;

  //! Special visualizer ;)
  ViTrackingViz tracking_viz_;
};

}  // namespace dyno
