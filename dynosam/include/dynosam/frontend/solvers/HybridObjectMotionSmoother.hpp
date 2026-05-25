#pragma once

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/ISAM2UpdateParams.h>
#include <gtsam_unstable/nonlinear/FixedLagSmoother.h>
#include <gtsam_unstable/slam/SmartStereoProjectionPoseFactor.h>

#include <nlohmann/json.hpp>

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/factors/HybridFormulationFactors.hpp"
#include "dynosam/frontend/solvers/HybridObjectMotionSolver-Impl.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam_common/Trajectories.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_opt/ISAM2.hpp"  //FOR TESTING!!!
#include "dynosam_opt/ISAM2Result.hpp"
#include "dynosam_opt/ISAM2UpdateParams.hpp"
#include "dynosam_opt/IncrementalOptimization.hpp"
#include "dynosam_opt/Map.hpp"
#include "dynosam_opt/Symbols.hpp"
#include "dynosam_sensors/RGBDCamera.hpp"

namespace dyno {

/* keyframe tracking metrics per object */
struct OKFTrackingMetrics {
  ObjectTrackingStatus tracking_status;
  double scale_ratio;
  double shape_score;
  double coverage;
  // Error with map
  double repr_error;
  bool is_keyframe{false};
  bool is_reset{false};
};

// TODO: dont need the IMPL class
class HybridObjectMotionSmoother : public HybridObjectMotionSolverImpl,
                                   public gtsam::FixedLagSmoother {
 public:
  DYNO_POINTER_TYPEDEFS(HybridObjectMotionSmoother)

  struct Result {
    bool solver_okay{false};
    gtsam::KeyVector marginalized_keys;
    gtsam::KeyList additional_keys_reeliminate;
    double update_time_ms{0};
    double marginalize_time_ms{0};
    // gtsam::ISAM2Result isam_result;
    dyno::ISAM2Result isam_result;
  };

  enum Solver { Full, Smart, MotionOnly };

  template <typename DERIVED>
  static HybridObjectMotionSmoother::Ptr CreateWithInitialMotion(
      const ObjectId object_id, double smoother_lag,
      const gtsam::Pose3& L_KF_km1, Frame::Ptr frame_km1,
      const TrackletIds& tracklets) {
    auto smoother = std::shared_ptr<DERIVED>(
        new DERIVED(object_id, frame_km1->getCamera(), smoother_lag));

    smoother->resetWithNewKeyedMotion(L_KF_km1, frame_km1, tracklets);
    return smoother;
  }

  ~HybridObjectMotionSmoother();

  // should only be called once a valid resetWithNewKeyedMotion has been called!
  bool update(const gtsam::Pose3& H_w_km1_k_predict, Frame::Ptr frame,
              const TrackletIds& tracklets) override;

  // This is basically reset
  //  What information from the previous state do we propogate over (ie.
  //  points?) if any
  bool resetWithNewKeyedMotion(const gtsam::Pose3& L_KF, Frame::Ptr frame,
                               const TrackletIds& tracklets) override;

  bool setNewKeyframe(Frame::Ptr frame) override;

  // trajectory should with F2F motion!!
  PoseWithMotionTrajectory trajectory() const override;

  /**
   * @brief Construct local trajectory representing the object path since
   * (inclusive) the last keyframe (ie KF_k -> k)
   *
   * @return PoseWithMotionTrajectory
   */
  PoseWithMotionTrajectory localTrajectory() const override;

  // NOTE: this is not actually the current keyframe (this is the start of the
  // active frames!)
  FrameId keyFrameId() const override {
    CHECK(!active_frame_ids_.empty());
    return active_frame_ids_.front();
  }
  FrameId frameId() const override {
    CHECK(!active_frame_ids_.empty());
    return active_frame_ids_.back();
  }
  Timestamp timestamp() const override {
    CHECK(!active_timestamps_.empty());
    return active_timestamps_.back();
  }

  /** Total number of keyframes */
  size_t numKeyframes() const;
  size_t numFramesSinceKeyframe() const;
  /* All involved frames */
  const FrameIds& frameIds() const;
  /* All keyframe ids */
  const FrameIds& keyframeIds() const;
  FrameId firstKeyframe() const;

  // Motion3ReferenceFrame getKeyFramedMotionReference() const override;
  gtsam::Pose3 keyFrameMotion() const override;

  Motion3ReferenceFrame frameToFrameMotionReference() const override;
  gtsam::Pose3 keyFramePose() const override;
  gtsam::Pose3 keyFrameCameraPose() const override;

  gtsam::FastMap<TrackletId, gtsam::Point3> getObjectPoints() const override;

  void receiveUpdate(const PoseChangeUpdateComplete& event) override;

  double reprojectionError(Frame::Ptr frame) const;

  bool shouldBeKeyframe(Frame::Ptr frame, cv::Mat* debug_image = nullptr) const;

  /** Compute an estimate from the incomplete linear delta computed during the
   * last update. This delta is incomplete because it was not updated below
   * wildfire_threshold.  If only a single variable is needed, it is faster to
   * call calculateEstimate(const KEY&).
   */
  gtsam::Values calculateEstimate() const override {
    return isam_.calculateEstimate();
  }

  /** Compute an estimate for a single variable using its incomplete linear
   * delta computed during the last update.  This is faster than calling the
   * no-argument version of calculateEstimate, which operates on all variables.
   * @param key
   * @return
   */
  template <class VALUE>
  VALUE calculateEstimate(gtsam::Key key) const {
    return isam_.calculateEstimate<VALUE>(key);
  }

  /** return the current set of iSAM2 parameters */
  // const gtsam::ISAM2Params& params() const { return isam_.params(); }
  const dyno::ISAM2Params& params() const { return isam_.params(); }

  /** Access the current set of factors */
  const gtsam::NonlinearFactorGraph& getFactors() const {
    return isam_.getFactorsUnsafe();
  }

  /** Access the current linearization point */
  const gtsam::Values& getLinearizationPoint() const {
    return isam_.getLinearizationPoint();
  }

  /** Access the current set of deltas to the linearization point */
  const gtsam::VectorValues& getDelta() const { return isam_.getDelta(); }

  /// Calculate marginal covariance on given variable
  gtsam::Matrix marginalCovariance(gtsam::Key key) const {
    return isam_.marginalCovariance(key);
  }

  /// Get results of latest isam2 update
  // const gtsam::ISAM2Result& getISAM2Result() const { return isamResult_; }
  const dyno::ISAM2Result& getISAM2Result() const { return isamResult_; }

  /// Get the iSAM2 object which is used for the inference internally
  // const gtsam::ISAM2& getISAM2() const { return isam_; }
  const dyno::ISAM2& getISAM2() const { return isam_; }

  struct DebugResult {
    HybridObjectMotionSmoother::Result result;
    ISAM2Stats smoother_stats;
    ObjectId object_id{0};
    FrameId frame_id{0};
    Timestamp timestamp{0};

    // tracking data
    size_t average_feature_age{0};
    size_t num_tracks{0};

    FrameId frame_id_KF{0};
    int num_landmarks_in_smoother{0};
    int num_motions_in_smoother{0};
  };

 protected:
  std::map<gtsam::Key, gtsam::Point3> getObjectPointsFromState(
      const gtsam::Values& values) const;

  inline std::map<gtsam::Key, gtsam::Point3> getObjectPointsFromSmootherState()
      const {
    return getObjectPointsFromState(smoother_state_);
  }

  std::map<gtsam::Key, gtsam::Point3> getObjectPointsFromStateSinceLastKF()
      const {
    return getObjectPointsFromState(state_since_lKF_);
  }

  std::map<gtsam::Key, gtsam::Pose3> getObjectMotionsFromState(
      const gtsam::Values& values) const;

  std::map<gtsam::Key, gtsam::Pose3> getObjectMotionsFromSmootherState() const {
    return getObjectMotionsFromState(smoother_state_);
  }

  std::map<gtsam::Key, gtsam::Pose3> getObjectMotionsFromStateSinceLastKF()
      const {
    return getObjectMotionsFromState(state_since_lKF_);
  }

  Result updateFromInitialMotion(const gtsam::Pose3& H_W_KF_k_initial,
                                 Frame::Ptr frame,
                                 const TrackletIds& tracklets);

  Result updateSmoother(
      const gtsam::NonlinearFactorGraph& newFactors,
      const gtsam::Values& newTheta,
      const KeyTimestampMap& timestamps = KeyTimestampMap(),
      const dyno::ISAM2UpdateParams& update_params = dyno::ISAM2UpdateParams(),
      const gtsam::KeyVector& additional_keys_marginalize = {});

  PoseWithMotionTrajectory localTrajectoryImpl(
      bool include_keyframe = false) const;

  const gtsam::Pose3& getCameraPose(FrameId frame_id) const {
    return camera_poses_.at(frame_id);
  }

  gtsam::Pose3 getObjectPose(FrameId frame_id) const;

  const gtsam::Values& getValuesSinceLastKF() const { return state_since_lKF_; }

  void setTrajectory(const PoseWithMotionTrajectory& past_trajectory) override {
    frozen_trajectory_ = past_trajectory;
  }

 protected:
  HybridObjectMotionSmoother(ObjectId object_id, Camera::Ptr camera,
                             double smootherLag);
  const std::string logger_prefix_;

  virtual Result updateFromInitialMotionImpl(
      gtsam::Values& smoother_state, const gtsam::Pose3& H_W_KF_k_initial,
      Frame::Ptr frame, const TrackletIds& tracklets) = 0;

  virtual gtsam::Pose3 keyFrameMotionImpl(
      FrameId frame_id, const gtsam::Values& values) const = 0;

  virtual void onNewKeyFrameMotion(const dyno::ISAM2& smoother_before_reset,
                                   const gtsam::Pose3 new_L_KF) = 0;

  // if there is a backend update moves ownershup to the input argument
  // and marks the has_update_flag as false
  bool takeBackendUpdate(PoseChangeUpdateComplete& backend_update);

  //! Last (object) keyframe for this object
  // Really this is the tracking keyframe and has nothing to do with the
  // reference frame
  Frame::Ptr lOKF_frame_;
  std::vector<Frame::Ptr> tracking_keyframes_;

  //! Vector of frame ids that correspond to variables current in the smoother
  //! Inlude KF...k
  FrameIds active_frame_ids_;
  //! Vector of timestampsthat correspond to variables current in the smoother
  //! Inlude KF...k
  std::vector<Timestamp> active_timestamps_;

  //! Trajectory up to and including the current KF representing
  //! a trajectory consturcted from variables no longer in the smoother
  PoseWithMotionTrajectory frozen_trajectory_;
  FrameRangeData<gtsam::Pose3> keyframe_range_;

  static dyno::ISAM2Params DefaultISAM2Params() {
    dyno::ISAM2GaussNewtonParams gn_params;
    // gn_params.wildfireThreshold = 0.00001;
    dyno::ISAM2Params params(gn_params);
    params.findUnusedFactorSlots = true;
    // OKAY this seems to be extremely important!
    // when cacheLinearizedFactors = true (default) at least on gtsam 4.2.0
    // then when collecting the factors that are affected by new variables
    // relinearizeAffectedFactors checks if the effected keys are also part of
    // the relinearization keys as provided by both the fluid relin check and as
    // part of the additional params. New affected keys as part of smart factors
    // are NOT part of the relin keys and therefore the cached LINEAR factor
    // will be used (which does not include the new variable!)
    // params.cacheLinearizedFactors = false;
    params.cacheLinearizedFactors = true;
    params.keyFormatter = DynosamKeyFormatter;

    // use with cuation but allows speed ;)
    params.enablePartialRelinearizationCheck = true;
    // params.relinearizeThreshold = 0.01;
    // this value is very important for accuracy
    // and if we want to do multiple update iterations!
    // also if this is not 1 then maybe factors that have a value update may not
    // get relinearized
    params.relinearizeSkip = 1;
    params.evaluateNonlinearError = false;
    return params;
  }

  /** An iSAM2 object used to perform inference. The smoother lag is controlled
   * by what factors are removed each iteration */
  // gtsam::ISAM2 isam_;
  dyno::ISAM2 isam_;

  using SmootherInterface = ISAMInterface<dyno::ISAM2>;
  SmootherInterface smoother_interface_;

  /** Store results of latest isam2 update */
  // gtsam::ISAM2Result isamResult_;
  dyno::ISAM2Result isamResult_;

  std::vector<DebugResult> debug_results_;

  /** Erase any keys associated with timestamps before the provided time */
  void eraseKeysBefore(double timestamp);

  /** Fill in an iSAM2 ConstrainedKeys structure such that the provided keys are
   * eliminated before all others */
  void createOrderingConstraints(
      const gtsam::KeyVector& marginalizableKeys,
      boost::optional<gtsam::FastMap<gtsam::Key, int>>& constrainedKeys) const;

  // for now
  gtsam::Values all_states_;
  //! Best estimate of all values since and including the last KF
  //! May include more values that what is currently in the smoother window
  gtsam::Values state_since_lKF_;

 private:
  //! Updated every update and includes only values in the smoother
  gtsam::Values smoother_state_;

  gtsam::Values all_m_L_points_;

  gtsam::FastMap<FrameId, gtsam::Pose3> camera_poses_;

  mutable CsvWriter kf_decision_logger_;

  // the update could actually be static for all solvers
  // since all object information is contained within the update
  // and is common since we do a joint solve!
  std::mutex backend_update_mutex_;
  std::atomic_bool has_backend_update_{false};
  //! All solvers really share the same update I guess but each one is
  //! responsible for
  // checking if the update contains data relevant for the specific solver
  PoseChangeUpdateComplete backend_update_;

  FrameIds frame_ids_;
  FrameIds keyframe_ids_;

 private:
  inline gtsam::FixedLagSmootherResult update(
      const gtsam::NonlinearFactorGraph&,
      const gtsam::Values&,  //
      const KeyTimestampMap&, const gtsam::FactorIndices&) override {
    throw DynosamException("Not implemented!");
  }
};

class HybridObjectMotionOnlySmoother : public HybridObjectMotionSmoother {
 public:
  using HybridObjectMotionSmoother::Result;

  HybridObjectMotionOnlySmoother(ObjectId object_id, Camera::Ptr camera,
                                 double smootherLag)
      : HybridObjectMotionSmoother(object_id, camera, smootherLag) {}

  Result updateFromInitialMotionImpl(gtsam::Values& smoother_state,
                                     const gtsam::Pose3& H_W_KF_k_initial,
                                     Frame::Ptr frame,
                                     const TrackletIds& tracklets) override;

  gtsam::Pose3 keyFrameMotionImpl(FrameId frame_id,
                                  const gtsam::Values& values) const override;

  void onNewKeyFrameMotion(const dyno::ISAM2& smoother_before_reset,
                           const gtsam::Pose3 new_L_KF) override;

 private:
  // void handleBackendUpdate(const PoseChangeUpdateComplete& backend_update);
 private:
  // gtsam::FastMap<TrackletId, std::vector<std::pair<FrameId,
  // gtsam::StereoPoint2>>> awaiting_measurements_;

  // blah: in the current implementation points are not actually marginalized
  // they are just deleted becuase we remove all factors
  enum PointState { InState, Marginalized };

  // class LandmarkNodeImpl;

  // struct NodeTypesImpl {
  //   using Measurement = gtsam::StereoPoint2;
  //   using FrameNodeT = FrameNodeBase<NodeTypesImpl>;
  //   using ObjectNodeT = ObjectNodeBase<NodeTypesImpl>;
  //   using LandmarkNodeT = LandmarkNodeImpl;
  // };

  // class LandmarkNodeImpl : public LandmarkNodeBase<KeyFrameNodeTypes>

  //! Motion factor constraining H with object point in L
  using StructuredMotionFactor = StereoHybridMotionFactor2::shared_ptr;
  //! Structureless factor constraining N motions with a known object point.
  //! For speed we limit N to 3
  using StructurelessMotionFactor =
      SmartMotionFactor2<3, gtsam::Pose3>::shared_ptr;

  // struct PointStateMap {
  //   private:
  //     typedef std::pair<PointState, Landmark> StateLandmark;
  //     gtsam::FastMap<TrackletId, std::pair<PointState, Landmark>> data_;

  //     size_t num_in_state{0};
  //     size_t num_marginalized{0};

  //   public:
  //     bool insert2(TrackletId tracklet_id, const StateLandmark&
  //     state_lmk_pair) {
  //       // update internal counters
  //       if(data_.exists(tracklet_id)) {

  //       }

  //       return data_.insert2(tracklet_id, state_lmk_pair);
  //     }
  // };

  // shoudl replace m_L_points
  gtsam::FastMap<TrackletId, std::pair<PointState, Landmark>> point_state_;
  gtsam::FastMap<TrackletId, std::vector<StructuredMotionFactor>>
      structured_factors_;

  using StereoSmartFactor = SmartMotionFactor2<3, gtsam::Pose3>;
  gtsam::FastMap<TrackletId, std::vector<StereoSmartFactor::shared_ptr>>
      structureless_factors_;

  // TODO: for now use better data-structure
  gtsam::FastMap<TrackletId, gtsam::FastMap<FrameId, gtsam::StereoPoint2>>
      stereo_measurements_;

  // actually dont think we need this...
  // By last frame, so expected frame-2 and frame-1 to be present
  gtsam::FastMap<FrameId, HybridSmoothingFactor::shared_ptr> smoothing_factors_;

  // // FactorMap<BatchStereoHybridMotionFactor3::shared_ptr> mo_factor_map_;
  // gtsam::FastMap<StereoHybridMotionFactor3::shared_ptr, TrackletFramePair>
  //     mo_factor_to_tracklet_id_;

  // gtsam::FastMap<TrackletId, FrameIds> trackletid_to_frame_ids_;
  // // Object Motion Symbol to observing tracklets
  // // Allows implicit lookup by frame id since ObjectMotionSymbol uses frame
  // id gtsam::FastMap<gtsam::Key, TrackletIds> object_motion_to_tracklets_;

  // For motion only
  // gtsam::FastMap<TrackletId, gtsam::Point3> m_L_points_;
};

class HybridObjectMotionSmartSmoother : public HybridObjectMotionSmoother {
 public:
  using HybridObjectMotionSmoother::Result;

  HybridObjectMotionSmartSmoother(ObjectId object_id, Camera::Ptr camera,
                                  double smootherLag)
      : HybridObjectMotionSmoother(object_id, camera, smootherLag) {}

  Result updateFromInitialMotionImpl(gtsam::Values& smoother_state,
                                     const gtsam::Pose3& H_W_KF_k_initial,
                                     Frame::Ptr frame,
                                     const TrackletIds& tracklets) override;

  gtsam::Pose3 keyFrameMotionImpl(FrameId frame_id,
                                  const gtsam::Values& values) const override;

  void onNewKeyFrameMotion(const dyno::ISAM2& smoother_before_reset,
                           const gtsam::Pose3 new_L_KF) override;

 private:
  /// SmartFactor stuff
  FactorMap<gtsam::SmartStereoProjectionPoseFactor::shared_ptr> factor_map_;
  gtsam::FastMap<gtsam::SmartStereoProjectionPoseFactor::shared_ptr, TrackletId>
      factor_to_tracklet_id_;
};

class HybridObjectMotionFullSmoother : public HybridObjectMotionSmoother {
 public:
  using HybridObjectMotionSmoother::Result;

  HybridObjectMotionFullSmoother(ObjectId object_id, Camera::Ptr camera,
                                 double smootherLag)
      : HybridObjectMotionSmoother(object_id, camera, smootherLag) {}

  Result updateFromInitialMotionImpl(gtsam::Values& smoother_state,
                                     const gtsam::Pose3& H_W_KF_k_initial,
                                     Frame::Ptr frame,
                                     const TrackletIds& tracklets) override;

  gtsam::Pose3 keyFrameMotionImpl(FrameId frame_id,
                                  const gtsam::Values& values) const override;

  void onNewKeyFrameMotion(const dyno::ISAM2& smoother_before_reset,
                           const gtsam::Pose3 new_L_KF) override;
};

using json = nlohmann::json;
void to_json(json& j, const HybridObjectMotionSmoother::Result& result);
void to_json(json& j, const HybridObjectMotionSmoother::DebugResult& result);

}  // namespace dyno
