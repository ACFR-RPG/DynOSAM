/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
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

#include <glog/logging.h>

#include <pcl/impl/point_types.hpp>

#include "dynosam/common/PointCloudProcess.hpp"
#include "dynosam/common/Types.hpp"
#include "dynosam/frontend/VisionImuOutputPacket.hpp"
#include "dynosam/frontend/imu/ImuFrontend.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam/logger/Logger.hpp"
#include "dynosam/utils/OpenCVUtils.hpp"

namespace dyno {

// forward from Tracker
struct FeatureTrackerInfo;

// TODO: for now just resend ALL history every time
// TODO: eventually object motions and poses could be a map of new/changes
// values unsure yet how we want to encapsulate incremental updates!!
struct RGBDInstanceOutputPacket : public FrontendOutputPacketBase {
 public:
  DYNO_POINTER_TYPEDEFS(RGBDInstanceOutputPacket)

  const StatusLandmarkVector static_landmarks_;   //! in the camera frame
  const StatusLandmarkVector dynamic_landmarks_;  //! in the camera frame
  const ObjectMotionMap object_motions_;
  // const MotionEstimateMap
  //     estimated_motions_;  //! Estimated motions in the world frame
  const ObjectPoseMap propogated_object_poses_;  //! Propogated poses using the
                                                 //! esimtate from the frontend
  const gtsam::Pose3Vector
      camera_poses_;  //! Vector of ego-motion poses (drawn everytime)
  const PointCloudLabelRGB::Ptr
      dense_labelled_cloud_;  //! Dense point cloud (with label and RGB) in
                              //! camera frame
  const FrameIdTimestampMap involved_timestamps_;

  ImuFrontend::PimPtr pim_;

  // relative pose of the camera (ie the increment) from k-1 to k, expressed in
  // the camera frame (at k-1)
  gtsam::Pose3 T_k_1_k_;

  std::set<ObjectId> is_keyframe_;

  RGBDInstanceOutputPacket(
      const StatusKeypointVector& static_keypoint_measurements,
      const StatusKeypointVector& dynamic_keypoint_measurements,
      const StatusLandmarkVector& static_landmarks,
      const StatusLandmarkVector& dynamic_landmarks,
      const gtsam::Pose3 T_world_camera, const Timestamp timestamp,
      const FrameId frame_id, const ObjectMotionMap& object_motions,
      const ObjectPoseMap propogated_object_poses = {},
      const gtsam::Pose3Vector camera_poses = {},
      const Camera::Ptr camera = nullptr,
      const GroundTruthInputPacket::Optional& gt_packet = std::nullopt,
      const DebugImagery::Optional& debug_imagery = std::nullopt,
      const PointCloudLabelRGB::Ptr dense_labelled_cloud = nullptr,
      const FrameIdTimestampMap& involved_timestamps = {})
      : FrontendOutputPacketBase(
            FrontendType::kRGBD, static_keypoint_measurements,
            dynamic_keypoint_measurements, T_world_camera, timestamp, frame_id,
            camera, gt_packet, debug_imagery),
        static_landmarks_(static_landmarks),
        dynamic_landmarks_(dynamic_landmarks),
        object_motions_(object_motions),
        propogated_object_poses_(propogated_object_poses),
        camera_poses_(camera_poses),
        dense_labelled_cloud_(dense_labelled_cloud),
        involved_timestamps_(involved_timestamps) {
    // they need to be the same size as we expect a 1-to-1 relation between the
    // keypoint and the landmark (which acts as an initalisation point)
    CHECK_EQ(static_landmarks_.size(), static_keypoint_measurements_.size());
    CHECK_EQ(dynamic_landmarks_.size(), dynamic_keypoint_measurements_.size());

    object_clouds_ = groupObjectCloud(dynamic_landmarks_, T_world_camera_);
    for (const auto& [object_id, this_object_cloud] : object_clouds_) {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr obj_cloud_ptr =
          pcl::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>(
              this_object_cloud);

      // TODO: currently axis-aligned bounding box (not orientated bbx)
      ObjectBBX this_object_bbx =
          findAABBFromCloud<pcl::PointXYZRGB>(obj_cloud_ptr);
      object_bbxes_.insert2(object_id, this_object_bbx);
      object_ids_.push_back(object_id);
    }
  }

  GenericTrackedStatusVector<LandmarkKeypointStatus>
  collectStaticLandmarkKeypointMeasurements() const;
  GenericTrackedStatusVector<LandmarkKeypointStatus>
  collectDynamicLandmarkKeypointMeasurements() const;

  /**
   * @brief Checks if an object id exists within the frontend output
   *
   * Actually looks in the map of objectids -> object_clouds_, does not check
   * the status keypoints/landmarks for this id.
   *
   * @param object_id
   * @return true
   * @return false
   */
  inline bool hasObject(ObjectId object_id) const {
    return object_clouds_.exists(object_id);
  }

  inline const ObjectIds& getObjectIds() const { return object_ids_; }

  const pcl::PointCloud<pcl::PointXYZRGB>& getDynamicObjectPointCloud(
      ObjectId object_id) const {
    CHECK(hasObject(object_id))
        << "Cannot get dynamic object as id missing: " << object_id;
    return object_clouds_.at(object_id);
  }

  const ObjectBBX& getDynamicObjectBBX(ObjectId object_id) const {
    CHECK(hasObject(object_id))
        << "Cannot get dynamic object bbx as id missing: " << object_id;
    return object_bbxes_.at(object_id);
  }

  const CloudPerObject& getObjectClouds() const { return object_clouds_; }

  const BbxPerObject& getObjectBbxes() const { return object_bbxes_; }

  bool operator==(const RGBDInstanceOutputPacket& other) const {
    return static_cast<const FrontendOutputPacketBase&>(*this) ==
               static_cast<const FrontendOutputPacketBase&>(other) &&
           static_landmarks_ == other.static_landmarks_ &&
           dynamic_landmarks_ == other.dynamic_landmarks_;
    // TODO: no estimated_motions_, propogated_object_poses_... becuase of
    // gtsam...
  }

 private:
  ObjectIds object_ids_;
  CloudPerObject object_clouds_;  //! Point clouds per object extracted from
                                  //! dynamic_landmarks_ in the world frame
  BbxPerObject object_bbxes_;  //! Bounding boxes per object extracted from the
                               //! object_clouds_
};

// write to file on destructor
class RGBDFrontendLogger : public EstimationModuleLogger {
 public:
  DYNO_POINTER_TYPEDEFS(RGBDFrontendLogger)
  RGBDFrontendLogger();
  virtual ~RGBDFrontendLogger();

  void logTrackingLengthHistogram(const Frame::Ptr frame);

 private:
  std::string tracking_length_hist_file_name_;
  json tracklet_length_json_;
};

}  // namespace dyno
