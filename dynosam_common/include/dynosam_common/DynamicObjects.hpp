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

#include <opencv4/opencv2/core.hpp>

#include "dynosam_common/GroundTruthPacket.hpp"
#include "dynosam_common/Types.hpp"

namespace dyno {

// TODO: eventually this should be the same as from the object-track module!
// right now detected by frontend
enum class ObjectTrackingStatus {
  New = 0,
  WellTracked = 1,
  PoorlyTracked = 2,
  ReTracked = 3,
  Lost = 4
};

// kind of repeatition of TrackingStatus (which is not really used anywhere!!!)
enum class ObjectFeatureTrackingStatus {
  Valid = 0,
  //! New features were detected not that the object itself is retracked
  Resampled = 1
  // TODO: maybe too few etc
};

template <>
inline std::string to_string(const ObjectTrackingStatus& status) {
  std::string status_str = "";
  switch (status) {
    case ObjectTrackingStatus::New: {
      status_str = "New";
      break;
    }
    case ObjectTrackingStatus::WellTracked: {
      status_str = "WellTracked";
      break;
    }
    case ObjectTrackingStatus::PoorlyTracked: {
      status_str = "PoorlyTracked";
      break;
    }
    case ObjectTrackingStatus::ReTracked: {
      status_str = "ReTracked";
      break;
    }
    case ObjectTrackingStatus::Lost: {
      status_str = "Lost";
      break;
    }
  }
  return status_str;
}

// should probably go in SingleDetecionResult!
struct ObjectStatus {
  ObjectTrackingStatus tracking_status;
  ObjectFeatureTrackingStatus feature_status;
};

class ObjectStatusMap : public gtsam::FastMap<ObjectId, ObjectStatus> {
 public:
  using Base = gtsam::FastMap<ObjectId, ObjectStatus>;
  using Base::Base;

  // ObjectIds getResampledObjects() const;
};

/**
 * @brief The result of an object detection from a detection network
 *
 */
struct ObjectDetection {
  //! Binary object mask of type CV_8U
  cv::Mat mask;
  cv::Rect bounding_box{};
  std::string class_name;
  float confidence;
};

struct SingleDetectionResult : public ObjectDetection {
  //! Indicates the source of the result
  //! If detection it means inference (likely a NN was run on an imput image)
  //! If mask, it means some pre-processing was done and only a tracking mask
  //! was used to generate this detection result. In this case only the object
  //! id and bounding box should be used (ie. class_name and confidence will
  //! likely not be set)
  enum Source { DETECTION, MASK };

  ObjectId object_id{-1};
  bool well_tracked{false};
  Source source = Source::MASK;

  bool isValid() const {
    // This is implicitly checks that object_id is not -1 (ie not set)
    // AND is a valiid object label (i.e. is non-zero which is the background
    // label)
    return (object_id > background_label) && well_tracked;
  }
};

/**
 * @brief Holds object detection/tracking result
 *
 */
struct ObjectDetectionResult {
  std::vector<SingleDetectionResult> detections;
  cv::Mat labelled_mask;
  cv::Mat input_image;  // Should be a 3 channel RGB image. Should always be set

  cv::Mat colouredMask() const;
  //! number of detections
  inline size_t num() const { return detections.size(); }
  ObjectIds objectIds() const;

  friend std::ostream& operator<<(std::ostream& os,
                                  const dyno::ObjectDetectionResult& res);
};

/**
 * @brief Calculate the local body velocity given the motion in world from k-1
 * to k and the object pose at k-1. This returns the body velocity at k-1.
 *
 * @param w_k_1_H_k const gtsam::Pose3&
 * @param w_L_k_1 const gtsam::Pose3&
 * @return gtsam::Vector3
 */
gtsam::Vector3 calculateBodyMotion(const gtsam::Pose3& w_k_1_H_k,
                                   const gtsam::Pose3& w_L_k_1);

enum PropogateType {
  InitGT,
  InitCentroid,
  Propogate,    // Propogated via a motion
  Interpolate,  // Interpolated via a motion
  Reinit        // Reinitalisaed via a centroid
};

using PropogatePoseResult = TemporalObjectCentricMap<PropogateType>;

/**
 * @brief Propogated a map of object poses via their motions or otherwise.
 *
 * At its core, take a map of object poses and a current frame, as well as a map
 * of estimation motions, H, at the current frame and propogates the object
 * poses according to: ^wH_k = ^w_{k-1}H_K * ^wL_{k-1} which is then added to
 * the object_poses.
 *
 * If ^wL_{k-1} is not available in the ObjectPoseMap but  ^w_{k-1}H_K is in the
 * MotionEstimateMap then EITHER object_centroids_k_1 is used as the translation
 * component of ^wL_{k-1} or the ground truth is - if GroundTruthPacketMap is
 * provided, then initalisation with ground truth is preferred.
 *
 * If ^wL_{k-1} and ^w_{k-1}H_K is not available the function will attempt to
 * interpolate between the the last object pose available in the map and the
 * current pose (initalised with object_centroids_k). If the last object pose is
 * too far away, the function will just re-enit with the current centroid.
 *
 * @param object_poses ObjectPoseMap& map of object poses and their appearing
 * frames
 * @param object_motions_k const MotionEstimateMap& object motions from k-1 to k
 * (size N)
 * @param object_centroids_k_1 const gtsam::Point3Vector& estimated object
 * centroids at k-1, must be of size N
 * @param object_centroids_k const gtsam::Point3Vector& estimated object
 * centroids at k,. must be of size N
 * @param frame_id_k FrameId current frame id (k)
 * @param gt_packet_map std::optional<GroundTruthPacketMap> optionally provided
 * gt map
 * @param result PropogatePoseResult* result map. If not null, will be populated
 * with how each object new object pose was calculated
 */
void propogateObjectPoses(
    ObjectPoseMap& object_poses, const MotionEstimateMap& object_motions_k,
    const gtsam::Point3Vector& object_centroids_k_1,
    const gtsam::Point3Vector& object_centroids_k, FrameId frame_id_k,
    std::optional<GroundTruthPacketMap> gt_packet_map = {},
    PropogatePoseResult* result = nullptr);

}  // namespace dyno
