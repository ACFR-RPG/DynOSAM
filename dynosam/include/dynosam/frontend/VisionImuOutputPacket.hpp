/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a
 copy
 *   of this software and associated documentation files (the "Software"), to
 deal
 *   in the Software without restriction, including without limitation the
 rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in
 all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE
 *   SOFTWARE.
 */

#pragma once

#include "dynosam/common/Camera.hpp"
#include "dynosam/common/GroundTruthPacket.hpp"
#include "dynosam/common/PointCloudProcess.hpp"
#include "dynosam/common/SensorModels.hpp"
#include "dynosam/common/Types.hpp"
#include "dynosam/frontend/Frontend-Definitions.hpp"
#include "dynosam/frontend/FrontendInputPacket.hpp"
#include "dynosam/frontend/imu/ImuFrontend.hpp"

namespace dyno {

/**
 * @brief Visual Tracking and Ego-motion information constructed by the DynoSAM
 * frontend.
 *
 * Contains feature tracking information capable of representing information
 * from monocular/RGB-D/Stereo cameras and ego-motion information from Visual
 * Odometry and Preintegrated IMU data.
 *
 */
class VisionImuPacket {
 public:
  DYNO_POINTER_TYPEDEFS(VisionImuPacket)

  /// @brief Basic track structure representing a tracking status and visual
  /// measurements
  struct Tracks {
    TrackingStatus status;
    CameraMeasurementStatusVector measurements;
    bool is_keyframe{false};

    inline bool valid() const { return status == TrackingStatus::VALID; }
  };

  /**
   * @brief Ego-motion/camera tracks representing visual measurements on the
   * static background and ego-motion information.
   *
   */
  struct CameraTracks : public Tracks {
    //! Camera pose in world frame
    gtsam::Pose3 X_W_k;
    //! Relative camera pose from k-1 to k
    gtsam::Pose3 T_k_1_k;
  };

  /**
   * @brief Object track information epresenting visual measurements for a
   * single object as well as frame-to-frame (and possibly other) motion/pose
   * information.
   *
   */
  struct ObjectTracks : public Tracks {
    //! Object motion from k-1 to k in W
    Motion3ReferenceFrame H_W_k_1_k;
    //! Object pose at k in W
    gtsam::Pose3 L_W_k;
  };
  //! Map of object id's to ObjectTracks
  using ObjectTrackMap = gtsam::FastMap<ObjectId, ObjectTracks>;

  Timestamp timestamp() const;
  FrameId frameId() const;
  ImuFrontend::PimPtr pim() const;
  Camera::ConstPtr camera() const;
  PointCloudLabelRGB::Ptr denseLabelledCloud() const;

  const CameraTracks& cameraTracks() const;
  const gtsam::Pose3& cameraPose() const;
  /**
   * @brief Returns the relative camera motion T_k_1_k, representing the motion
   * of the camera from k-1 to k in the camera local frame (at k-1)
   *
   * @return const gtsam::Pose3&
   */
  const gtsam::Pose3& relativeCameraTransform() const;

  const ObjectTrackMap& objectTracks() const;
  /**
   * @brief Object poses for this frame matching the set of objects available in
   * ObjectTracks
   *
   * @return const PoseEstimateMap&
   */
  const PoseEstimateMap& objectPoses() const;
  const ObjectIds& getObjectIds() const;
  const MotionEstimateMap& objectMotions() const;

  const GroundTruthInputPacket::Optional& groundTruthPacket() const;
  const DebugImagery::Optional& debugImagery() const;

  const CameraMeasurementStatusVector& objectMeasurements() const;
  const CameraMeasurementStatusVector& staticMeasurements() const;

  // Gets static landmark measurements (if any)
  StatusLandmarkVector staticLandmarkMeasurements() const;
  // Gets dynamic landmark measurements (if any)
  StatusLandmarkVector dynamicLandmarkMeasurements() const;

  VisionImuPacket& timestamp(Timestamp ts);
  VisionImuPacket& frameId(FrameId id);
  VisionImuPacket& pim(const ImuFrontend::PimPtr& pim);
  VisionImuPacket& camera(const Camera::Ptr& cam);
  VisionImuPacket& denseLabelledCloud(const PointCloudLabelRGB::Ptr& cloud);
  VisionImuPacket& cameraTracks(const CameraTracks& camera_tracks);
  VisionImuPacket& objectTracks(const ObjectTrackMap& object_tracks);
  VisionImuPacket& objectTracks(const ObjectTracks& object_track,
                                ObjectId object_id);
  VisionImuPacket& groundTruthPacket(
      const GroundTruthInputPacket::Optional& gt);
  VisionImuPacket& debugImagery(const DebugImagery::Optional& dbg);

  bool operator==(const VisionImuPacket& other) const {
    return frame_id_ == other.frame_id_ && timestamp_ == other.timestamp_;
    // TODO: minimal operator
  }

 protected:
  //! Timestamp
  Timestamp timestamp_;
  //! Frame Id
  FrameId frame_id_;

  //! Possible PIM going from last frame to this frame
  ImuFrontend::PimPtr pim_;

  //! Possible camera
  Camera::Ptr camera_;

  //! Possible dense point cloud (with label and RGB) in camera frame
  PointCloudLabelRGB::Ptr dense_labelled_cloud_;
  //! Optional ground truth information for this frame
  GroundTruthInputPacket::Optional ground_truth_;
  //! Optional debug/visualiation imagery for this frame
  DebugImagery::Optional debug_imagery_;

 protected:
  void updateObjectTrackCaches();

 private:
  //! Static point tracks
  CameraTracks camera_tracks_;
  //! Dynamic point tracks associated to each object
  ObjectTrackMap object_tracks_;

  //! Object poses for this frame (cached when object tracks are set)
  PoseEstimateMap cached_object_poses_;
  //! Object motions for this frame (cached when object tracks are set)
  MotionEstimateMap cached_object_motions_;
  //! Object ids for this frame (cached when object tracks are set)
  ObjectIds cached_object_ids_;
  //! All object measurements for this frame
  CameraMeasurementStatusVector cached_object_measurements_;

 private:
  static void fillLandmarkMeasurements(
      StatusLandmarkVector& landmarks,
      const CameraMeasurementStatusVector& camera_measurements);
};

}  // namespace dyno
