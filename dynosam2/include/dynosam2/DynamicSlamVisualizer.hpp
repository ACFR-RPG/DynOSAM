#pragma once

#include <glog/logging.h>

#include "dynosam2/common/DynamicSlamInterface.hpp"
#include "dynosam2/estimator/SlamBackend.hpp"
#include "dynosam_common/Trajectories.hpp"
#include "dynosam_common/Transforms.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_processing/ImageViewer.hpp"
#include "dynosam_sensors/SensorRig.hpp"

namespace dyno2 {

class DynamicSlamVisualizer {
 public:
  DynamicSlamVisualizer(dyno::CanonicalSensorRig::Ptr sensor_rig,
                        const SlamBackend* estimator);
  virtual ~DynamicSlamVisualizer() {}

  // viz data AND estimator update!
  // this function is called asynchronously!
  virtual void publishEstimatorUpdate(
      const PublicationData& publication_data) = 0;

  /// @brief Set external image display queue
  /// @param display_queue
  void setDisplayQueue(dyno::ImageDisplayQueue* display_queue);

  /// @brief Adds an image display to the ImageBatchDisplayer. Must be called
  /// after setDisplayQueue.
  /// @param display
  void addImageDisplay(dyno::ImageDisplay::Ptr display);

  /// @brief Draw images from the image displayer (if set).
  void displayImages();

 protected:
  inline gtsam::Pose3 cameraToRobotFrame(const gtsam::Pose3& X_W_C) {
    return dyno::changeBasis(T_RC_, X_W_C);
  }

  inline dyno::PoseWithMotion cameraToRobotFrame(
      const dyno::PoseWithMotion& entry_W_C) {
    dyno::PoseWithMotion entry_W_R = entry_W_C;
    entry_W_R.pose = dyno::changeBasis(T_RC_, entry_W_C.pose);

    // explicit conversion to pose3
    const gtsam::Pose3& H_WC = entry_W_C.motion;
    entry_W_R.motion.estimate() = dyno::changeBasis(T_RC_, H_WC);
    return entry_W_R;
  }

 protected:
  dyno::CanonicalSensorRig::Ptr sensor_rig_{nullptr};
  const SlamBackend* estimator_{nullptr};
  //! Cached extrinsics transform (camera -> robot frame)
  gtsam::Pose3 T_RC_;
  //! ROS style frame ids for output frames
  dyno::ReferenceFrames reference_frames_;

  dyno::ImageBatchDisplayer::UniquePtr image_displayer_{nullptr};

  // keep the camera in robot coordinate frame, not estimated (camera) frame
  dyno::PoseTrajectory camera_trajectory_;
  dyno::MultiObjectTrajectories object_trajectories_;
};

}  // namespace dyno2
