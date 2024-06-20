/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 */

#include "dynosam/common/DynamicObjects.hpp"
#include "dynosam/common/GroundTruthPacket.hpp"

namespace dyno {

void propogateObjectPoses(
    ObjectPoseMap& object_poses,
    const MotionEstimateMap& object_motions_k,
    const gtsam::Point3Vector& object_centroids_k_1,
    const gtsam::Point3Vector& object_centroids_k,
    FrameId frame_id_k,
    std::optional<GroundTruthPacketMap> gt_packet_map
) {

    CHECK_EQ(object_motions_k.size(), object_centroids_k_1.size());
    CHECK_EQ(object_centroids_k.size(), object_centroids_k_1.size());
    const FrameId frame_id_k_1 = frame_id_k - 1;

    size_t i = 0; //used to index the object centroid vectors
    for(const auto&[object_id, motion] : object_motions_k) {
        const auto centroid_k = object_centroids_k.at(i);
        const auto centroid_k_1 = object_centroids_k_1.at(i);
        const gtsam::Pose3 prev_H_world_curr = motion;
        //new object - so we need to add at k-1 and k
        if(!object_poses.exists(object_id)) {
            bool initalised_with_gt = false;
            gtsam::Pose3 pose_k_1;

            //if gt packet exists for this frame, use that as the rotation
            if(gt_packet_map) {
                if(gt_packet_map->exists(frame_id_k_1)) {
                    const GroundTruthInputPacket& gt_packet_k_1 = gt_packet_map->at(frame_id_k_1);

                    ObjectPoseGT object_pose_gt_k;
                    if(!gt_packet_k_1.getObject(object_id, object_pose_gt_k)) {
                        pose_k_1 = object_pose_gt_k.L_world_;
                        initalised_with_gt = true;
                    }
                }
            }

            if(!initalised_with_gt) {
                //could not init with gt, use identity rotation and centroid
                pose_k_1 = gtsam::Pose3(gtsam::Rot3::Identity(), centroid_k_1);
            }

            object_poses.insert2(object_id , gtsam::FastMap<FrameId, gtsam::Pose3>{});
            object_poses.at(object_id).insert2(frame_id_k_1, pose_k_1);
        }

        auto& per_frame_poses = object_poses.at(object_id);
        //if we have a pose at the previous frame, simply apply motion
        if(per_frame_poses.exists(frame_id_k_1)) {
            const gtsam::Pose3& object_pose_k_1 = per_frame_poses.at(frame_id_k_1);
            //assuming in world
            gtsam::Pose3 object_pose_k = prev_H_world_curr * object_pose_k_1;
            per_frame_poses.insert2(frame_id_k, object_pose_k);
        }
        else {
            //no motion at the previous frame - if close, interpolate between last pose and this pose
            //no motion used
            const size_t min_diff_frames = 4;

            //last frame SHOULD be the largest frame (as we use a std::map with std::less)
            auto last_record_itr = per_frame_poses.rbegin();
            const FrameId last_frame = last_record_itr->first;
            const gtsam::Pose3 last_recorded_pose = last_record_itr->second;

            //construct current pose using last poses rotation (I guess?)
            gtsam::Pose3 current_pose = gtsam::Pose3(
                last_recorded_pose.rotation(),
                object_centroids_k.at(i)
            );

            // CHECK_LT(last_frame, frame_id_k_1);
            // if(frame_id_k - last_frame < min_diff_frames) {
            //     //apply interpolation
            //     //need to map [last_frame:frame_id_k] -> [0,1] for the interpolation function
            //     //with N values such that frame_id_k - last_frame + 1= N (to be inclusive)
            //     const size_t N = frame_id_k - last_frame + 1;
            //     const double divisor = (double)(frame_id_k - last_frame);
            //     for(size_t j = 0; j < N; j++) {
            //         double t = (double)j/divisor;
            //         gtsam::Pose3 interpolated_pose = last_recorded_pose.slerp(t, current_pose, boost::none, boost::none);

            //         FrameId frame = last_frame + j;
            //         per_frame_poses.insert2(frame, interpolated_pose);
            //     }

            // }
            // else {
            //     //last frame too far away - reinitalise with centroid!
            //     LOG(ERROR) << "Frames too far away - current frame is " << frame_id_k << " previous frame is " << last_frame;
            // }


        }
        i++;
    }


}


}
