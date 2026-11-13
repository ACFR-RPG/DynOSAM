#pragma once

#include <condition_variable>

#include "dynosam2/common/Parameters.hpp"
#include "dynosam2/estimator/DoGraph.hpp"
#include "dynosam2/estimator/ImuError.hpp"
#include "dynosam2/estimator/SlamBackend.hpp"
#include "dynosam2/estimator/VioGraph.hpp"
#include "dynosam2/frontend/FeatureTracker.hpp"
#include "dynosam2/frontend/Frame.hpp"
#include "dynosam2/frontend/MatchingAdaptor.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_sensors/Camera.hpp"
#include "dynosam_sensors/ImuMeasurements.hpp"

namespace dyno2 {

struct AyncVO {
  std::mutex vo_mutex;
  std::condition_variable vo_cv;
  gtsam::Pose3* vo;
  bool vo_ready{false};
  bool vo_valid{false};

  void markReadyButInvalidAndNotify();
  void markReadyAndValidAndNotify(const gtsam::Pose3& pose);
};

/// @brief Helper struct for motion propagation
struct VioMotionPropagation {
  bool success{false};
  PoseVelocity3 predicted_pv;
  RelativePoseErrorBase::Ptr relative_motion_error;

  enum Source { PRIOR_MOTION = 0, IMU = 1 };
  Source source;
};

std::ostream& operator<<(std::ostream& os,
                         const VioMotionPropagation::Source& source);

class Frontend {
 public:
  DYNO_POINTER_TYPEDEFS(Frontend)

  Frontend(const FrontendParameters& parameters, dyno::Camera::Ptr camera);
  ~Frontend() = default;

  // prpogate IMU

  // TODO: eventually R_k_km1 should become a full prediction for all objects!
  void detectAndTrack(FeatureBlockContainer*& features_out,
                      dyno::ObjectDetectionResult& object_detection_out,
                      dyno::ImageContainer::Ptr container_in_out,
                      const gtsam::Rot3& R_k_km1);

  void solveVisualOdometry(VioGraphEstimator& estimator,
                           FeatureBlockContainer& features,
                           FeatureBlockGeometry& geometry,
                           AyncVO& async_context,
                           std::vector<FrameId>& frames_updated,
                           std::vector<KeypointIdentifier>& new_measurements,
                           std::vector<TrackletId>& correspondences,
                           bool& success, const ImageContainer& image_container,
                           const VioMotionPropagation& T_k_km1_propagated);

  void solveObjectOdometry(
      ObjectId object_id, const VioGraphEstimator& vo_estimator,
      const EstimationParameters& params, const ImageContainer& image_container,
      DoGraphEstimator& estimator, FeatureBlockContainer& features,
      FeatureBlockGeometry& geometry, AyncVO& async_context,
      std::vector<FrameId>& frames_updated,
      std::vector<TrackletId>& correspondences, uchar& success);

  inline bool isVioInitalised() const {
    return vi_tracking_condition_.is_initalized;
  }

  /**
   * @brief Attempt to the next pose (k) given the current motion.
   * The motion may either come from IMU measurements (between the last state
   * and the current state k) or a constant motion model will be applied.
   *
   * @param frame_id
   * @param timestamp
   * @param estimator
   * @param imu_measurements
   * @param constant_motion_max_dt dyno::Timestamp If using constant motion
   * propagation, what is the maximum dt we accept to do the propagation
   * @return VioMotionPropagation
   */
  static VioMotionPropagation predictFromMotion(
      const dyno::FrameId frame_id, const dyno::Timestamp timestamp,
      const VioGraphEstimator& estimator,
      const ImuMeasurements& imu_measurements,
      const dyno::Timestamp constant_motion_max_dt = 2);

 private:
  // frame should have descriptors but not yet be added to the Graph as a
  // keyframe
  void matchToVioMap(
      VioGraphEstimator& estimator, const ImageContainer& container,
      ViFrame::Ptr frame,
      std::unordered_map<dyno::TrackletId, dyno::TrackletId>& merged_lmks);

  /**
   * @brief Initalise new points in the estimator by triangulation
   *
   * @param estimator
   * @param frames
   * @param frame'
   * @param new_measurements
   * @return size_t
   */
  size_t temporalVioTriangultion(
      VioGraphEstimator& estimator, ViFrame::Ptr frame,
      std::unordered_set<KeypointIdentifier>& new_measurements);

  bool solve3d2dRansac(gtsam::Pose3& pose, std::vector<bool>& inliers,
                       FrameAbsoluteAdaptor& adaptor, double ransac_threshold,
                       int max_iterations = 50) const;

  bool shouldBeViKeyFrame(const VioGraphEstimator& estimator,
                          ViFrame::Ptr frame) const;

  struct ObjectKeyframeMetrics {
    static void computeCovariance(const std::vector<Eigen::Vector2d>& pts,
                                  Eigen::Matrix2d& cov, Eigen::Vector2d& mean);

    static double computeCoverage(
        const std::vector<Eigen::Vector2d>& pts_ref,
        const std::vector<Eigen::Vector2d>& pts_target, double radius = 10.0);
  };

  bool shouldBeDoKeyFrame(const DoGraphEstimator& estimator,
                          DoFrame::Ptr frame) const;

 private:
  dyno::Camera::Ptr camera_;
  FeatureTracker feature_tracker_;
  double vio_ransac_threshold_3d_2d_;
  double do_ransac_threshold_3d_2d_;

  struct MatchInfo {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    gtsam::Point3 m_W;
    TrackletId id{0};
    bool is_valid{false};
    double quality{0.0};
  };

  /// @brief Internal state of the ViFrontend
  struct ViTrackingCondition {
    bool tracking_lost{false};
    bool is_initalized{false};
  };
  ViTrackingCondition vi_tracking_condition_;

  cv::Ptr<cv::Feature2D> describer_;
};

}  // namespace dyno2
