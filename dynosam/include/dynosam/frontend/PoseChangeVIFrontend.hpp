#pragma once

#include "dynosam/backend/PoseChangeBackendModule.hpp"
#include "dynosam/formulations/KeyFrameHybridEstimator.hpp"
#include "dynosam/frontend/PoseChangeVIFrontendViz.hpp"
#include "dynosam/frontend/VIFrontend.hpp"
#include "dynosam/frontend/solvers/HybridObjectMotionSolver.hpp"
#include "dynosam/frontend/vision/FeatureTrackerFast.hpp"
#include "dynosam_sensors/RGBDCamera.hpp"

namespace dyno {

using PoseChangeBackendSink =
    std::function<void(const PoseChangeInput::ConstPtr&)>;

class PoseChangeVIFrontendFAST : public VIFrontend {
 public:
  DYNO_POINTER_TYPEDEFS(PoseChangeVIFrontendFAST)
  PoseChangeVIFrontendFAST(const DynoParams& params, Camera::Ptr camera,
                           HybridFormulationKeyFrame::Ptr formulation,
                           ImageDisplayQueue* display_queue = nullptr,
                           const SharedGroundTruth& shared_ground_truth = {});

  ~PoseChangeVIFrontendFAST() = default;

  /** Add sink to send PC data to the backend */
  void addPoseChangeOutputSink(const PoseChangeBackendSink& func) {
    pose_change_backend_sink_ = func;
  };

  /** Callback triggered when the backend has finished a single update */
  void onBackendUpdateComplete(const PoseChangeUpdateComplete& event) {}

 private:
  SpinReturn boostrapSpin(VIFrontendInput::ConstPtr input) override;
  SpinReturn nominalSpin(VIFrontendInput::ConstPtr input) override;

  struct GeometrySolveContext {
    std::mutex vo_mutex;
    std::condition_variable vo_cv;
    gtsam::Pose3* vo;
    bool vo_ready{false};
    bool vo_valid{false};
  };

  void solveVisualOdometryByThread(const ImageContainer& image_container,
                                   const LocalLandmarks& reference_geometry,
                                   FrameGeometry& local_geometry,
                                   FeatureBlockContainer& features,
                                   GeometrySolveContext& context,
                                   PoseTrajectory& vo_trajectory,
                                   bool& success);

  // TODO: right now reference geometry must be in k-1!
  void solveObjectOdometryByThread(
      ObjectId object_id, const ImageContainer& image_container,
      const LocalLandmarks& reference_geometry, FrameGeometry& local_geometry,
      FeatureBlockContainer& features, GeometrySolveContext& context,
      PoseWithMotionTrajectory& doo_trajectory, uchar& success);

  bool solve3d2dRansac(gtsam::Pose3& pose, std::vector<bool>& inliers,
                       OpenGVCentralAbsolutePoseAdaptor& adaptor,
                       double ransac_threshold, int max_iterations = 50) const;

 private:
  HybridFormulationKeyFrame::Ptr formulation_;
  HybridFormulationKeyFrameAccessor::Ptr accessor_;
  KeyFrameMap::Ptr map_;
  FeatureTrackerFast feature_tracker_fast_;

  PoseChangeBackendSink pose_change_backend_sink_;

  LocalLandmarksMap local_landmarks_W_;

  // only needed for tracking while we dont use the local landmarks W_
  LocalLandmarksMap local_landmarks_C_km1_;
  // only for points!? Right now need to initalise new landmarks with a knoew
  // geometry eventually need the set of keyframes!
  // //TODO: wrap in frame representation with frame id + timestampe etc!
  // FrameGeometryMap frame_geometry_km1_;

  //! Current trajectories. Copied to the DynoState output.
  //! Only contains trajectories for objects observed at the latest frame
  DynoStateTrajectories dyno_state_;

  // cached parameters for geometric solve
  double vo_ransac_threshold_3d_2d_;
  double doo_ransac_threshold_3d_2d_;

  double computeRansacThreshold(const double repr_error) const;

  // VIOFrame frame_km1_;
  // For now!
  gtsam::Pose3 X_km1_;
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

  /**
   * @brief
   *
   *
   *
   * @param trajectories
   * @param object_with_new_motions All well tracked objects that have a new
   * object estimate this frame.
   * @param object_tracking_status All objects observed this frame but not
   * necessary well tracked (ie. have observations but not an estimate)
   * @param infos
   * @param frame_k
   * @param frame_km1
   */
  void solveObjectMotions(MultiObjectTrajectories& trajectories,
                          ObjectIds& object_with_new_motions,
                          ObjectTrackingStatusMap& object_tracking_status,
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
    //! This is untouched by the refined state
    // TODO: Not needed anymore I think
    gtsam::NavState frontend_nav_state_j;
    ImuFrontend::PimPtr pim_lk_j;
    //! Should exist only if PIM is non null
    ImuMeasurements imu_measurements;
    //! Relative camera pose between from -> to frames
    gtsam::Pose3 T_i_j;
    //! Relative camera pose between last keyframe -> to frames
    gtsam::Pose3 T_lkf_j;

    inline Timestamp timestamp() const { return frame_j->getTimestamp(); }
  };

  struct TemporalNavState {
    FrameId frame_id;
    Timestamp timestamp;
    gtsam::NavState state;
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
      gtsam::Pose3& T_ij,
      std::optional<gtsam::NavState> propogated_nav_state_k = std::nullopt,
      std::optional<gtsam::Rot3> R_km1_k = std::nullopt);

  bool addMeasurementsForObjectKeyframe(FrameId frame_id, ObjectId object_id);

  /** Check if we have an update from the backend and consume the update */
  bool checkAndConsumeUpdate(FrameId frame_id_k);

  /* Refine the full camera trajectory */
  PoseTrajectory refinePerFrameCameraPGO(
      const PoseTrajectory& camera_trajectory,
      dyno::FastSet<FrameId>& frames_in_pgo,
      dyno::FastSet<FrameId>& frames_propogated) const;

 private:
  HybridFormulationKeyFrame::Ptr formulation_;
  HybridFormulationKeyFrameAccessor::Ptr accessor_;
  KeyFrameMap::Ptr map_;
  HybridObjectMotionSolver::UniquePtr object_motion_solver_;

  //! Record of the previous nav state which may be updated via refinement from
  //! the backend
  TemporalNavState nav_state_km1_;
  //! Record of the current keyframe nav state which may be updated via
  //! refinement from the backend
  TemporalNavState nav_state_lkf_;

  // TODO: for now until maybe use propogator
  //  accumulated visual odometry (from direct VO not updated states)
  gtsam::Pose3 T_lkf_j_;

  //! Last camera keyframe
  Frame::Ptr lCKF_frame_;

  //! Current trajectories. Copied to the DynoState output.
  //! Only contains trajectories for objects observed at the latest frame
  DynoStateTrajectories dyno_state_;
  MultiObjectTrajectories full_object_trajectories_;

  PoseChangeBackendSink pose_change_backend_sink_;

  // Mapping of intermediate relative motions. Stored by to frame.
  gtsam::FastMap<FrameId, RelEgoPoseInfo> rel_egopose_infos_;

  //! Records all up to date keyframe infos for each frame
  KeyFrameInfoMap keyframe_infos_;

  //! Mark as having an update from the backend
  std::atomic_bool has_backend_update_{false};

  //! Special visualizer ;)
  ViTrackingViz tracking_viz_;
};

}  // namespace dyno
