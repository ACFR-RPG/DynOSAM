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

#include "dynosam_common/DynamicObjects.hpp"

#include "dynosam_common/GroundTruthPacket.hpp"
#include "dynosam_common/utils/OpenCVUtils.hpp"

namespace dyno {

cv::Mat ObjectDetectionResult::colouredMask() const {
  // input image should never be empty
  if (num() == 0 || labelled_mask.empty()) {
    return input_image;
  }
  return utils::labelMaskToRGB(labelled_mask,
                               background_label,  // from dynosam_common
                               input_image);
}

ObjectIds ObjectDetectionResult::objectIds() const {
  ObjectIds object_ids;
  std::transform(
      detections.begin(), detections.end(), std::back_inserter(object_ids),
      [](const SingleDetectionResult& result) { return result.object_id; });
  return object_ids;
}

std::ostream& operator<<(std::ostream& os,
                         const dyno::ObjectDetectionResult& res) {
  os << "ObjectDetectionResult:\n";
  os << "  detections (" << res.detections.size() << "):\n";
  for (const auto& det : res.detections) {
    os << "    object id=" << det.object_id << ", class=" << det.class_name
       << ", confidence=" << det.confidence << ", bbox=(" << det.bounding_box.x
       << "," << det.bounding_box.y << "," << det.bounding_box.width << ","
       << det.bounding_box.height << ")\n";
  }
  // if (!res.labelled_mask.empty()) {
  //     os << "  labelled_mask: (" << res.labelled_mask.rows
  //        << "x" << res.labelled_mask.cols
  //        << ", channels=" << res.labelled_mask.channels() << ")\n";
  // } else {
  //     os << "  labelled_mask: empty\n";
  // }
  return os;
}

gtsam::Vector3 calculateBodyMotion(const gtsam::Pose3& w_k_1_H_k,
                                   const gtsam::Pose3& w_L_k_1) {
  const gtsam::Point3& t_motion = w_k_1_H_k.translation();
  const gtsam::Rot3& R_motion = w_k_1_H_k.rotation();
  const gtsam::Point3& t_pose = w_L_k_1.translation();

  static const gtsam::Rot3 I = gtsam::Rot3::Identity();

  return t_motion - (gtsam::Rot3(I.matrix() - R_motion.matrix())) * t_pose;
}

void propogateObjectPoses(ObjectPoseMap& object_poses,
                          const MotionEstimateMap& object_motions_k,
                          const gtsam::Point3Vector& object_centroids_k_1,
                          const gtsam::Point3Vector& object_centroids_k,
                          FrameId frame_id_k,
                          std::optional<GroundTruthPacketMap> gt_packet_map,
                          PropogatePoseResult* result) {
  CHECK_EQ(object_motions_k.size(), object_centroids_k_1.size());
  CHECK_EQ(object_centroids_k.size(), object_centroids_k_1.size());
  const FrameId frame_id_k_1 = frame_id_k - 1;

  // get centroid for object at k-1 using the gt pose if available, or the
  // centroid if not
  auto get_centroid = [=](ObjectId object_id, const gtsam::Point3& centroid_k_1,
                          gtsam::Pose3& pose_k_1) -> PropogateType {
    bool initalised_with_gt = false;
    // if gt packet exists for this frame, use that as the rotation
    if (gt_packet_map) {
      if (gt_packet_map->exists(frame_id_k_1)) {
        const GroundTruthInputPacket& gt_packet_k_1 =
            gt_packet_map->at(frame_id_k_1);

        ObjectPoseGT object_pose_gt_k_1;
        if (gt_packet_k_1.getObject(object_id, object_pose_gt_k_1)) {
          pose_k_1 = object_pose_gt_k_1.L_world_;
          initalised_with_gt = true;
        }
      }
    }

    if (!initalised_with_gt) {
      // could not init with gt, use identity rotation and centroid
      pose_k_1 = gtsam::Pose3(gtsam::Rot3::Identity(), centroid_k_1);

      return PropogateType::InitCentroid;
    } else {
      return PropogateType::InitGT;
    }
  };

  size_t i = 0;  // used to index the object centroid vectors
  for (const auto& [object_id, motion] : object_motions_k) {
    const auto centroid_k_1 = object_centroids_k_1.at(i);
    const gtsam::Pose3 prev_H_world_curr = motion;
    // new object - so we need to add at k-1 and k
    if (!object_poses.exists(object_id)) {
      gtsam::Pose3 pose_k_1;
      auto propgate_result = get_centroid(object_id, centroid_k_1, pose_k_1);
      if (result) result->insert22(object_id, frame_id_k_1, propgate_result);

      object_poses.insert2(object_id, gtsam::FastMap<FrameId, gtsam::Pose3>{});
      object_poses.at(object_id).insert2(frame_id_k_1, pose_k_1);
    }

    auto& per_frame_poses = object_poses.at(object_id);
    // if we have a pose at the previous frame, simply apply motion
    if (per_frame_poses.exists(frame_id_k_1)) {
      const gtsam::Pose3& object_pose_k_1 = per_frame_poses.at(frame_id_k_1);
      // assuming in world
      gtsam::Pose3 object_pose_k = prev_H_world_curr * object_pose_k_1;
      per_frame_poses.insert2(frame_id_k, object_pose_k);

      // update result map
      if (result)
        result->insert22(object_id, frame_id_k, PropogateType::Propogate);
    } else {
      // no motion at the previous frame - if close, interpolate between last
      // pose and this pose no motion used
      const size_t min_diff_frames = 3;

      // last frame SHOULD be the largest frame (as we use a std::map with
      // std::less)
      auto last_record_itr = per_frame_poses.rbegin();
      const FrameId last_frame = last_record_itr->first;
      const gtsam::Pose3 last_recorded_pose = last_record_itr->second;

      const gtsam::Pose3& centroid_k = object_centroids_k.at(i);
      // construct current pose using last poses rotation (I guess?)
      gtsam::Pose3 current_pose =
          gtsam::Pose3(last_recorded_pose.rotation(), centroid_k);

      CHECK_LT(last_frame, frame_id_k_1);
      if (frame_id_k - last_frame < min_diff_frames) {
        // apply interpolation
        // need to map [last_frame:frame_id_k] -> [0,1] for the interpolation
        // function with N values such that frame_id_k - last_frame + 1= N (to
        // be inclusive)
        const size_t N = frame_id_k - last_frame + 1;
        const double divisor = (double)(frame_id_k - last_frame);
        for (size_t j = 0; j < N; j++) {
          double t = (double)j / divisor;

          gtsam::Pose3 interpolated_pose = last_recorded_pose.slerp(
              t, current_pose, boost::none, boost::none);

          FrameId frame = last_frame + j;
          per_frame_poses.insert2(frame, interpolated_pose);

          // update result map
          if (result)
            result->insert22(object_id, frame, PropogateType::Interpolate);
        }

      } else {
        gtsam::Pose3 pose_k_1;
        // last frame too far away - reinitalise with centroid!
        VLOG(20) << "Frames too far away - current frame is " << frame_id_k
                 << " previous frame is " << last_frame << " for object "
                 << object_id;
        auto propogate_result = get_centroid(object_id, centroid_k_1, pose_k_1);
        object_poses.at(object_id).insert2(frame_id_k_1, pose_k_1);

        if (result) result->insert22(object_id, frame_id_k_1, propogate_result);

        gtsam::Pose3 object_pose_k = prev_H_world_curr * pose_k_1;
        per_frame_poses.insert2(frame_id_k, object_pose_k);

        // update result map
        if (result)
          result->insert22(object_id, frame_id_k, PropogateType::Propogate);
      }
    }
    i++;
  }
}

}  // namespace dyno
