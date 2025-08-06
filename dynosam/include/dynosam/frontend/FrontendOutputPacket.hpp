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

struct VisionImuPacket {
  DYNO_POINTER_TYPEDEFS(VisionImuPacket)

  struct Tracks {
    TrackingStatus status;
    CameraMeasurementStatusVector measurements;
    bool is_keyframe{false};

    bool valid() const { return status == TrackingStatus::VALID; }
  };

  struct CameraTracks : public Tracks {
    //! Camera pose in world frame
    gtsam::Pose3 X_W_k;
    //! Relative camera pose from k-1 to k
    gtsam::Pose3 T_k_1_k;
  };

  struct ObjectTracks : public Tracks {
    //! Object motion from k-1 to k in W
    Motion3ReferenceFrame H_W_k_1_k;
  };

  //! Timestamp
  Timestamp timestamp;
  //! Frame Id
  FrameId frame_id;

  //! Possible PIM going from last frame to this frame
  ImuFrontend::PimPtr pim;

  //! Static point tracks
  CameraTracks static_tracks;
  //! Dynamic point tracks associated to each object
  gtsam::FastMap<ObjectId, ObjectTracks> object_tracks;

  //! Possible camera
  Camera::Ptr camera;

  //! Possible dense point cloud (with label and RGB) in camera frame
  PointCloudLabelRGB::Ptr dense_labelled_cloud;

  //! Trajectory of all objects from the frontend (mostly used for
  //! visualisation)
  ObjectPoseMap object_poses;
  //! Trajectory of camera from the frontend (mostly used for visualisation)
  gtsam::Pose3Vector camera_poses;

  //! Optional ground truth information for this frame
  GroundTruthInputPacket::Optional ground_truth;
  //! Optional debug/visualiation imagery for this frame
  DebugImagery::Optional debug_imagery;
};

struct FrontendOutputPacketBase {
 public:
  DYNO_POINTER_TYPEDEFS(FrontendOutputPacketBase)

 public:
  const FrontendType frontend_type_;
  const StatusKeypointVector static_keypoint_measurements_;
  const StatusKeypointVector dynamic_keypoint_measurements_;
  const gtsam::Pose3 T_world_camera_;
  const Timestamp timestamp_;
  const FrameId frame_id_;
  const Camera::Ptr camera_;
  const GroundTruthInputPacket::Optional gt_packet_;
  const DebugImagery::Optional debug_imagery_;

  FrontendOutputPacketBase(
      const FrontendType frontend_type,
      const StatusKeypointVector& static_keypoint_measurements,
      const StatusKeypointVector& dynamic_keypoint_measurements,
      const gtsam::Pose3& T_world_camera, const Timestamp timestamp,
      const FrameId frame_id, const Camera::Ptr camera = nullptr,
      const GroundTruthInputPacket::Optional& gt_packet = std::nullopt,
      const DebugImagery::Optional& debug_imagery = std::nullopt)
      : frontend_type_(frontend_type),
        static_keypoint_measurements_(static_keypoint_measurements),
        dynamic_keypoint_measurements_(dynamic_keypoint_measurements),
        T_world_camera_(T_world_camera),
        timestamp_(timestamp),
        frame_id_(frame_id),
        camera_(camera),
        gt_packet_(gt_packet),
        debug_imagery_(debug_imagery) {}

  virtual ~FrontendOutputPacketBase() {}

  inline bool hasCamera() const { return (bool)camera_; }
  inline Timestamp getTimestamp() const { return timestamp_; }
  inline FrameId getFrameId() const { return frame_id_; }

  bool operator==(const FrontendOutputPacketBase& other) const {
    return frontend_type_ == other.frontend_type_ &&
           static_keypoint_measurements_ ==
               other.static_keypoint_measurements_ &&
           dynamic_keypoint_measurements_ ==
               other.dynamic_keypoint_measurements_ &&
           gtsam::traits<gtsam::Pose3>::Equals(T_world_camera_,
                                               other.T_world_camera_) &&
           timestamp_ == other.timestamp_ && frame_id_ == other.timestamp_;
    // TODO: no camera or gt packket
  }
};

}  // namespace dyno
