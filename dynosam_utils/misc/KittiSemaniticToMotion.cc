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

#include <glog/logging.h>
#include <gtsam/geometry/Pose3.h>

#include <dynosam/common/GroundTruthPacket.hpp>
#include <dynosam/dataprovider/KittiDataProvider.hpp>
#include <fstream>
#include <iostream>
#include <opencv4/opencv2/core.hpp>
#include <ostream>

using namespace dyno;

std::vector<int> findMovingObject(
    const GroundTruthInputPacket& prev_packet_vector,
    const GroundTruthInputPacket& curr_packet_vector, double tol_m = 0.4) {
  // find object in previous
  std::vector<int> moving_labels;

  for (const ObjectPoseGT& object_pose_gt : curr_packet_vector.object_poses_) {
    const ObjectId object_id = object_pose_gt.object_id_;
    auto it = std::find_if(prev_packet_vector.object_poses_.begin(),
                           prev_packet_vector.object_poses_.end(),
                           [=](const ObjectPoseGT& gt_object) {
                             return gt_object.object_id_ == object_id;
                           });
    if (it == prev_packet_vector.object_poses_.end()) {
      continue;
    }

    const ObjectPoseGT& object_pose_gt_in_previous = *it;
    const gtsam::Pose3 prev_L_world =
        prev_packet_vector.X_world_ * object_pose_gt_in_previous.L_camera_;

    const gtsam::Pose3 curr_L_world =
        curr_packet_vector.X_world_ * object_pose_gt.L_camera_;

    gtsam::Vector3 t_diff =
        curr_L_world.translation() - prev_L_world.translation();

    // const gtsam::Pose3 pose_diff = prev_L_world.inverse() * curr_L_world;
    // const gtsam::Point3& translation_error = pose_diff.translation();
    // L2 norm - ie. magnitude
    double t_error = t_diff.norm();

    if (t_error > tol_m) {
      LOG(INFO) << "Object " << object_id << " is moving!";
      moving_labels.push_back(object_id);
    } else {
      LOG(INFO) << "Object " << object_id << " is NOT moving!";
    }
  }

  return moving_labels;
}

cv::Mat constructMotionMask(const cv::Mat& instance_mask,
                            const std::vector<int>& moving_labels) {
  cv::Mat motion_mask;
  instance_mask.copyTo(motion_mask);

  for (int i = 0; i < motion_mask.rows; i++) {
    for (int j = 0; j < motion_mask.cols; j++) {
      int label = motion_mask.at<int>(i, j);
      if (label == 0) {
        continue;
      }

      // check if label is in moving labels. if not, make zero!
      auto it = std::find(moving_labels.begin(), moving_labels.end(), label);
      if (it == moving_labels.end()) {
        motion_mask.at<int>(i, j) = 0;
      }
    }
  }
  return motion_mask;
}

void writeMask(const std::string& path, const cv::Mat& motion_mask) {
  std::ofstream file;
  file.open(path);

  for (int i = 0; i < motion_mask.rows; i++) {
    for (int j = 0; j < motion_mask.cols; j++) {
      int label = motion_mask.at<int>(i, j);
      file << std::to_string(label) << " ";
    }
    file << '\n';
  }

  file.close();
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  FLAGS_log_prefix = 1;

  GroundTruthInputPacket prev_packet_vector;
  GroundTruthInputPacket curr_packet_vector;
  cv::Mat mask_cur;
  cv::Mat rgb_cur;

  const std::string path = "/root/data/vdo_slam/kitti/kitti/0020";
  // becuase this uses the KittiDataLoader, this will use whatever
  // semantic/instance laoder is in
  KittiDataLoader::Params params;
  params.base_line = 0;  // dont care
  params.depth_scale_factor = 0;
  params.mask_type = MaskType::SEMANTIC_INSTANCE;
  KittiDataLoader loader(path, params);

  size_t index = 0;
  auto callback = [&](size_t, Timestamp timestamp, cv::Mat rgb, cv::Mat,
                      cv::Mat, cv::Mat instance_mask,
                      gtsam::Pose3 camera_pose_gt,
                      GroundTruthInputPacket gt_object_pose_gt) -> bool {
    CHECK(camera_pose_gt.equals(gt_object_pose_gt.X_world_));
    CHECK(timestamp == gt_object_pose_gt.timestamp_);

    // //start with all background
    // cv::Mat motion_mask = cv::Mat(instance_mask.size(), instance_mask.type(),
    // 0);
    cv::Mat motion_mask = instance_mask;

    std::stringstream ss;
    ss << std::setfill('0') << std::setw(6) << index;
    const std::string file_path = path + "/motion/" + ss.str() + ".txt";
    if (index != 0) {
      mask_cur = instance_mask;
      rgb_cur = rgb;

      std::vector<int> moving_labels =
          findMovingObject(prev_packet_vector, gt_object_pose_gt, 0.2);
      motion_mask = constructMotionMask(instance_mask, moving_labels);

      cv::Mat motion_mask_viz =
          utils::labelMaskToRGB(motion_mask, background_label, rgb_cur);
      cv::Mat instance_mask_viz =
          utils::labelMaskToRGB(instance_mask, background_label, rgb_cur);

      // cv::Mat disp = utils::concatenateImagesVertically(instance_mask_viz,
      // motion_mask_viz); cv::imshow("Masks", disp); cv::waitKey(1);
    }

    LOG(INFO) << "Writing file to " << file_path;
    writeMask(file_path, motion_mask);

    index++;
    prev_packet_vector = gt_object_pose_gt;
    return true;
  };
  loader.setCallback(callback);

  while (loader.spin()) {
  }
}
