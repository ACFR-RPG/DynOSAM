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

#include "dynosam/common/GroundTruthPacket.hpp"
#include "dynosam/common/Types.hpp"

namespace dyno {

// TODO: this data structure is really unncessary as it REALLY is only used to
// indicates which objects we have detected and their bounding boxes which are
// ONLY used for visualisation!!
struct DynamicObjectObservation {
  ObjectId object_id;
  TrackletIds object_features;  //! Tracklet id's of object features within the
                                //! frame. Does not indicate usability

  cv::Rect bounding_box{};  // reconstructed from the object mask and not
                            // directly from the object features, although all
                            // features should lie in this cv::Rect

  inline size_t numFeatures() const { return object_features.size(); }
  inline bool hasBoundingBox() const { return !bounding_box.empty(); }
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

using DynamicObjectObservations = std::vector<DynamicObjectObservation>;

enum PropogateType {
  InitGT,
  InitCentroid,
  Propogate,    // Propogated via a motion
  Interpolate,  // Interpolated via a motion
  Reinit        // Reinitalisaed via a centroid
};

using PropogatePoseResult = GenericObjectCentricMap<PropogateType>;

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
