/*
 *   Copyright (c) 2025 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam_ros/displays/dynamic_slam_displays/BackendDSDRos.hpp"

#include "dynosam_common/utils/Timing.hpp"

namespace dyno {

BackendDSDRos::BackendDSDRos(const DisplayParams params,
                             rclcpp::Node::SharedPtr node)
    : BackendDisplay(), DSDRos(params, node) {
  temporal_dynamic_points_pub_ =
      node->create_publisher<sensor_msgs::msg::PointCloud2>(
          "temporal_dynamic_cloud", 1);
}

void BackendDSDRos::spinOnceImpl(
    const BackendOutputPacket::ConstPtr& backend_output) {
  // publish vo and path
  auto tic = utils::Timer::tic();
  constexpr static bool kPublishOdomAsTf = false;
  this->publishVisualOdometry(backend_output->pose(),
                              backend_output->getTimestamp(), kPublishOdomAsTf);
  this->publishVisualOdometryPath(backend_output->optimized_camera_poses,
                                  backend_output->getTimestamp());

  auto odom_toc = utils::Timer::toc<std::chrono::nanoseconds>(tic);

  // publish static cloud
  this->publishStaticPointCloud(backend_output->static_landmarks,
                                backend_output->pose());

  // publish dynamic cloud
  this->publishDynamicPointCloud(backend_output->dynamic_landmarks,
                                 backend_output->pose());

  // publishTemporalDynamicMaps(backend_output);
  auto clouds_toc = utils::Timer::toc<std::chrono::nanoseconds>(tic);

  const auto& object_motions = backend_output->optimized_object_motions;
  const auto& object_poses = backend_output->optimized_object_poses;
  const auto& timestamp_map = this->shared_module_info.getTimestampMap();

  ObjectMotionMap keyframed_motions;
  ObjectMotionMap keyframed_poses;

  // this is awful - hack to make the frame ids consequative (not sure if this
  // is the final way of doing this)
  //  only now for when camera keyframe every pose but objects do not!
  //   for(const auto& [object_id, per_frame_motion] : object_motions) {
  //    // this is awful
  //    // hack for now to make frames consecutaive. may not be the same for
  //    each object/pose!! FrameId key_frame_id = 0; for(const auto& [frame_id,
  //    motion] : per_frame_motion) {

  //   }
  //   ss << "\n";
  // }

  // this is awful
  // FrameId key_frame_id = 0;
  std::stringstream ss;
  for (const auto& [object_id, per_frame_motion] : object_poses) {
    ss << "j= " << object_id << " appeared at ";
    for (const auto& [frame_id, motion] : per_frame_motion) {
      ss << frame_id << " ";
    }
    ss << "\n";
  }
  LOG(INFO) << ss.str();

  // // publish objects
  DSDTransport::Publisher object_poses_publisher = dsd_transport_.addObjectInfo(
      object_motions, object_poses, params_.world_frame_id, timestamp_map,
      backend_output->getFrameId(), backend_output->getTimestamp());
  object_poses_publisher.publishObjectOdometry();
  object_poses_publisher.publishObjectPaths();
  auto objects_toc = utils::Timer::toc<std::chrono::nanoseconds>(tic);

  LOG(INFO)
      << "Published odom: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(odom_toc).count()
      << "[ms], Clouds: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(clouds_toc)
             .count()
      << "[ms], Objects: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(objects_toc)
             .count();
}

void BackendDSDRos::publishTemporalDynamicMaps(
    const BackendOutputPacket::ConstPtr& latest_backend_output) {
  const auto& dynamic_landmarks = latest_backend_output->dynamic_landmarks;
  const auto& current_frame_id = latest_backend_output->getFrameId();
  const auto& temporal_object_data =
      latest_backend_output->temporal_object_data;

  // used to set the colour decay for active/inactive objects
  static gtsam::FastMap<ObjectId, int> decay_count;
  // we assume we insert new clouds into the end each dequeue contained in
  // temporal_clouds_ so this index actual is references against the end of the
  // deque so the position to access = end() - index by adding values to the
  // vector this enables use to iterate in reverse order since we start at the
  // end
  static gtsam::FastMap<ObjectId, std::vector<int>> pos_to_viz;

  CloudPerObject clouds_per_obj =
      groupObjectCloud(dynamic_landmarks, latest_backend_output->pose());

  // add to temporal structure
  for (const auto& [object_id, cloud] : clouds_per_obj) {
    if (!temporal_clouds_.exists(object_id)) {
      temporal_clouds_.insert2(object_id,
                               std::deque<pcl::PointCloud<pcl::PointXYZRGB>>{});
    }
    temporal_clouds_.at(object_id).push_back(cloud);

    // initalise delay count for new objects
    if (!decay_count.exists(object_id)) decay_count[object_id] = 0;
    // 1 reference aginast the end of the dequeue to always include the most
    // recent cloud
    if (!pos_to_viz.exists(object_id))
      pos_to_viz[object_id] = std::vector<int>{1};
  }

  int skip = 5;  // skip frames
  float active_saturation_increment = 0.08;
  float inactive_saturation_increment = 0.02;

  // accumulated temporal cloud of all the objects over time
  pcl::PointCloud<pcl::PointXYZRGB> temporal_cloud;
  for (const auto& [object_id, dynamic_clouds] : temporal_clouds_) {
    // may not have 1-to-1 temporal match as sometimes a few frames are needed
    // before we have a cloud
    auto temporal_info_it =
        std::find_if(temporal_object_data.begin(), temporal_object_data.end(),
                     [&object_id](const TemporalObjectMetaData& tom) {
                       return tom.object_id == object_id;
                     });

    if (temporal_info_it == temporal_object_data.end()) {
      continue;
    }

    const auto& last_seen = temporal_info_it->last_seen;
    // if object is currently seen
    const bool is_active = last_seen == current_frame_id;
    // if object is active, reset the count to zero so that we start the colour
    // from original colour ie. the colour without any decay as would be used to
    // ordinarily colour the object if the object is not active, then the count
    // will continue to increase, thereby continuing to decay the objects colour
    // when it is not seen.
    float saturation_increment = inactive_saturation_increment;
    if (is_active) {
      decay_count.at(object_id) = 0;
      saturation_increment = active_saturation_increment;
    }
    // current count used to increment the saturation value
    // note reference
    int& count = decay_count.at(object_id);

    const int num_dynamic_clouds = static_cast<int>(dynamic_clouds.size());
    // calculate or update new indicies which will be used to ensure the same
    // temporal cloud e.g slice at timestep gets visualised each time
    std::vector<int>& positions = pos_to_viz.at(object_id);
    const int last_position = positions.back();
    int potential_new_position = last_position + skip;
    // check if we have skipped this object n times
    // if we have, grow the index vector to display the new slice
    if (potential_new_position == num_dynamic_clouds) {
      // minus 1 from the actual index to account for length
      positions.push_back(potential_new_position - 1u);
    }

    for (int reverse_index : positions) {
      int i = num_dynamic_clouds - reverse_index;
      CHECK_GE(i, 0);
      CHECK_LT(i, num_dynamic_clouds);

      // decay HSV saturation over time via the value increment
      // start at the original colour value (ie. with defaults)
      float colour_saturation =
          std::max(Color::unique_id_default_saturation -
                       static_cast<float>(count) * saturation_increment,
                   0.0f);
      Color color = Color::uniqueId(object_id, colour_saturation,
                                    Color::unique_id_default_value);

      const pcl::PointCloud<pcl::PointXYZRGB>& dynamic_cloud =
          dynamic_clouds[i];
      for (pcl::PointXYZRGB pt : dynamic_cloud.points) {
        pt.r = static_cast<std::uint8_t>(color.r);
        pt.g = static_cast<std::uint8_t>(color.g);
        pt.b = static_cast<std::uint8_t>(color.b);
        temporal_cloud.points.push_back(pt);
      }
      count++;
    }
  }

  sensor_msgs::msg::PointCloud2 pc2_msg;
  pcl::toROSMsg(temporal_cloud, pc2_msg);
  pc2_msg.header.frame_id = params_.world_frame_id;
  temporal_dynamic_points_pub_->publish(pc2_msg);
}

}  // namespace dyno
