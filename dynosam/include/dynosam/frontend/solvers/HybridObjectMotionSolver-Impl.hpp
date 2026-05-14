#pragma once

#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam_common/MotionKeyFrame.hpp"
#include "dynosam_common/Trajectories.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_cv/RGBDCamera.hpp"

namespace dyno {

// RENAME for keyframe?
class HybridObjectMotionSolverImpl {
 public:
  DYNO_POINTER_TYPEDEFS(HybridObjectMotionSolverImpl)

  HybridObjectMotionSolverImpl(ObjectId object_id, Camera::Ptr camera)
      : object_id_(object_id) {
    rgbd_camera_ = CHECK_NOTNULL(camera)->safeGetRGBDCamera();
    CHECK(rgbd_camera_);
    stereo_calibration_ = rgbd_camera_->getFakeStereoCalib();
  }

  ~HybridObjectMotionSolverImpl() = default;

  ObjectId objectId() const { return object_id_; }

  virtual FrameId keyFrameId() const = 0;
  virtual FrameId frameId() const = 0;
  virtual Timestamp timestamp() const = 0;

  virtual gtsam::Pose3 keyFrameMotion() const = 0;
  virtual gtsam::Pose3 keyFramePose() const = 0;
  virtual Motion3ReferenceFrame frameToFrameMotionReference() const = 0;
  virtual gtsam::Pose3 keyFrameCameraPose() const = 0;

  /* Relative motion of the object in Lkf */
  virtual gtsam::Pose3 relativeTransform() const {
    const gtsam::Pose3 L_kf = keyFramePose();
    return L_kf.inverse() * keyFrameMotion() * L_kf;
  }

  virtual Motion3ReferenceFrame keyFrameMotionReference() const {
    return Motion3ReferenceFrame(
        keyFrameMotion(), Motion3ReferenceFrame::Style::KF,
        ReferenceFrame::GLOBAL, keyFrameId(), frameId());
  }

  virtual gtsam::Pose3 frameToFrameMotion() const {
    // Motion3ReferenceFrame auto casts to pose
    return frameToFrameMotionReference();
  }

  virtual gtsam::Pose3 pose() const {
    return keyFrameMotion() * keyFramePose();
  }

  virtual bool update(const gtsam::Pose3& H_w_km1_k_predict, Frame::Ptr frame,
                      const TrackletIds& tracklets) = 0;

  virtual bool resetWithNewKeyedMotion(const gtsam::Pose3& L_KF,
                                       Frame::Ptr frame,
                                       const TrackletIds& tracklets) = 0;

  virtual bool setNewKeyframe(Frame::Ptr) { return false; }

  // points in L
  virtual gtsam::FastMap<TrackletId, gtsam::Point3> getObjectPoints() const = 0;
  virtual PoseWithMotionTrajectory trajectory() const = 0;
  /**
   * @brief Construct local trajectory representing the object path since
   * (inclusive) the last keyframe (ie KF_k -> k)
   *
   * @return PoseWithMotionTrajectory
   */
  virtual PoseWithMotionTrajectory localTrajectory() const = 0;

  virtual void receiveUpdate(const PoseChangeUpdateComplete&){};

 protected:
  friend class HybridObjectMotionSolver;
  virtual void setTrajectory(
      const PoseWithMotionTrajectory& past_trajectory) = 0;

  const ObjectId object_id_;
  std::shared_ptr<RGBDCamera> rgbd_camera_;
  gtsam::Cal3_S2Stereo::shared_ptr stereo_calibration_;
};

}  // namespace dyno
