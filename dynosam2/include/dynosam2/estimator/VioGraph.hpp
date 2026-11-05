
#pragma once

#include <gtsam/navigation/NavState.h>

#include "dynosam2/estimator/BaGraph.hpp"
#include "dynosam2/frontend/Frame.hpp"
#include "dynosam_opt/NonlinearOptimizer.hpp"
#include "dynosam_sensors/Camera.hpp"
#include "dynosam_sensors/ImuFrontend.hpp"
#include "dynosam_sensors/RGBDCamera.hpp"

namespace dyno2 {

// TODO: hold factors so we dont need to rebuild them!?
struct VioGraphState {
  dyno::Timestamp timestamp{0.0};
  dyno::FrameId frame_id{0};
  // Viostate!
  gtsam::Pose3 pose_W;
  //! Velocity in world frame
  Velocity3 velocity_W;
  // IMU link
  bool is_keyframe{false};

  inline gtsam::NavState navState() const {
    return gtsam::NavState{pose_W, velocity_W.linear};
  }
};

class VioGraph : public BaGraphStates<VioGraphState> {
 public:
  typedef BaGraphStates<VioGraphState> Base;
  typedef VioGraphState State;

  DYNO_POINTER_TYPEDEFS(VioGraph)
  VioGraph(dyno::Camera::Ptr camera);
  virtual ~VioGraph() = default;

  // only set of IMU is to be used
  void addImu(dyno::CombinedPIMParamPtr imu_config);
  dyno::CombinedPIMParamPtr getImuConfig() const { return imu_config_; }
  inline bool usingImu() const { return imu_config_ != nullptr; }

  // get landmark in estimated reference frame not map frame
  using LandmarkMap::getLandmark;

  void initaliseStates(ViFrame::Ptr frame, const gtsam::Pose3& pose);
  void initaliseStates(ViFrame::Ptr frame, const PoseVelocity3& state_W);
  void initaliseStates(ViFrame::Ptr frame, const gtsam::Pose3& pose,
                       const Velocity3& velocity_W);

  inline ViFrame::Ptr getFramePtr(dyno::FrameId frame_id) const {
    return this->getFrameAs<ViFrame>(
        dyno::TemporalObjectId{frame_id, dyno::background_label});
  }

  /// @brief Sets the camera pose X_W at k. Is thread safe.
  /// @param frame_id
  /// @param pose
  void setPose(dyno::FrameId frame_id, const gtsam::Pose3& pose);
  void setVelocity(dyno::FrameId frame_id, const Velocity3& velocity_W);
  void setVelocityFromPose(dyno::FrameId frame_id);

  const VioGraphState& getVioState(dyno::FrameId frame_id) const;

  /// @brief Get the camera pose X_W at k. Thread safe.
  /// @param frame_id
  /// @return
  const gtsam::Pose3& getPose(dyno::FrameId frame_id) const;

  /// @brief Get velocity in the world frame
  /// @param frame_id
  /// @return
  const Velocity3& getVelocity(dyno::FrameId frame_id) const;

  /// @brief Get velocity in the body frame of the camera
  /// @param frame_id
  /// @return
  // const Velocity3& getBodyVelocity(dyno::FrameId frame_id) const;

  bool getMapPoint(dyno::TrackletId id, MapPoint& map_point) const;
  size_t getMapPoints(MapPointVector& map_points) const;
  size_t getMapPoints(MapPoints& map_points) const;

  bool poseExists(dyno::FrameId frame_id) const;
  // currently no removal of keyframe
  bool setKeyframe(dyno::FrameId id, bool is_keyframe);
  bool isKeyframe(dyno::FrameId id) const;

  // active keyframes
  const std::set<dyno::FrameId>& keyFrames() const;

  bool pruneKeyframes(size_t num_keyframes, size_t num_motion_frames);

 private:
  VioGraphState& initaliseStatesWithPose(ViFrame::Ptr frame,
                                         const gtsam::Pose3& pose);

  void setStateVelocityFromPose(VioGraphState& current_state);

 protected:
  CombinedPIMParamPtr imu_config_{nullptr};
  //! Active keyframes used for selection and bundle adjustmement!
  std::set<dyno::FrameId> keyframes_;
  //! Mutex lock for the few functions that we expect to be accessed by thread
  //! This is mostly the get/set camera pose function which is needed by the
  //! DoGraph Otherwise we try and keep multi-thread access to a minimum for
  //! safety and speed
  mutable std::mutex mutex_;
};

class VioGraphEstimator : public VioGraph, public BaProblemInterface {
 public:
  DYNO_POINTER_TYPEDEFS(VioGraphEstimator)
  VioGraphEstimator(dyno::Camera::Ptr camera) : VioGraph(camera) {}

  void optimize(dyno::FrameId frame_id,
                std::vector<dyno::FrameId>& frames_affected,
                bool only_states = false);

 private:
};

}  // namespace dyno2
