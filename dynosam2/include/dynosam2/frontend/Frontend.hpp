#pragma once

#include <condition_variable>

#include "dynosam2/common/Parameters.hpp"
#include "dynosam2/estimator/SlamBackend.hpp"
#include "dynosam2/frontend/FeatureTracker.hpp"
#include "dynosam2/frontend/Frame.hpp"
#include "dynosam2/frontend/MatchingAdaptor.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_sensors/Camera.hpp"

namespace dyno2 {

struct UpdatedStates {
  std::vector<FrameId> vo_states;
  std::vector<TemporalObjectIdentifier> do_stats;
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

  void solve(SlamBackend& estimator, FeatureBlockContainer& features,
             const ImageContainer& container, UpdatedStates& updated_states,
             ObjectIds& solved_objects);

 private:
  bool solve3d2dRansac(gtsam::Pose3& pose, std::vector<bool>& inliers,
                       FrameAbsoluteAdaptor& adaptor, double ransac_threshold,
                       int max_iterations = 50) const;

 private:
  FrontendParameters parameters_;
  dyno::Camera::Ptr camera_;
  FeatureTracker feature_tracker_;

  struct MatchInfo {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    gtsam::Point3 m_W;
    TrackletId id{0};
    bool is_valid{false};
    double quality{0.0};
  };

  struct AyncVO {
    std::mutex vo_mutex;
    std::condition_variable vo_cv;
    gtsam::Pose3* vo;
    bool vo_ready{false};
    bool vo_valid{false};
  };

  /// @brief Internal state of the ViFrontend
  struct ViTrackingCondition {
    bool tracking_lost{false};
    bool is_initalized{false};
  };
  ViTrackingCondition vi_tracking_condition_;

  //! All Visual Inertial Frames with detections
  ViFrames vi_frames_;
  //! All Dyamic Object Frames with detections
  std::map<ObjectId, DoFrames> do_frames_;
};

}  // namespace dyno2
