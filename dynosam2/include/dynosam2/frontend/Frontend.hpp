#pragma once

#include <condition_variable>

#include "dynosam2/common/Parameters.hpp"
#include "dynosam2/estimator/DoGraph.hpp"
#include "dynosam2/estimator/VioGraph.hpp"
#include "dynosam2/frontend/FeatureTracker.hpp"
#include "dynosam2/frontend/Frame.hpp"
#include "dynosam2/frontend/MatchingAdaptor.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_sensors/Camera.hpp"

namespace dyno2 {

struct AyncVO {
  std::mutex vo_mutex;
  std::condition_variable vo_cv;
  gtsam::Pose3* vo;
  bool vo_ready{false};
  bool vo_valid{false};
};

class Frontend {
 public:
  DYNO_POINTER_TYPEDEFS(Frontend)

  Frontend(const FrontendParameters& parameters, dyno::Camera::Ptr camera);
  ~Frontend() = default;

  // prpogate IMU

  void detectAndTrack(FeatureBlockContainer*& features_out,
                      dyno::ObjectDetectionResult& object_detection_out,
                      ImageContainer& container_in_out);

  void solveVisualOdometry(VioGraphEstimator& estimator,
                           FeatureBlockContainer& features,
                           FeatureBlockGeometry& geometry, ViFrame& frame,
                           AyncVO& async_context,
                           std::vector<FrameId>& frames_updated, bool& success,
                           const ViFrames& frames,
                           const ImageContainer& image_container);

  void solveObjectOdometry(
      ObjectId object_id, const VioGraphEstimator& vo_estimator,
      const ImageContainer& image_container, const DoFrames& frames,
      DoGraphEstimator& estimator, FeatureBlockContainer& features,
      FeatureBlockGeometry& geometry, DoFrame& frame, AyncVO& async_context,
      std::vector<FrameId>& frames_updated, uchar& success);

  inline bool isVioInitalised() const {
    return vi_tracking_condition_.is_initalized;
  }

 private:
  /**
   * @brief Initalise new points in the estimator by triangulation
   *
   * @param estimator
   * @param frames
   * @param frame
   * @return size_t
   */
  size_t matchToVioMap(VioGraphEstimator& estimator, const ViFrames& frames,
                       ViFrame& frame);

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
                          const DoFrame& frame, const DoFrames& frames) const;

 private:
  FrontendParameters parameters_;
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
};

}  // namespace dyno2
