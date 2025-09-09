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

#include "dynosam_ros/DataProviderRos.hpp"

#include <dynosam/common/ImageTypes.hpp>
#include <dynosam/utils/Numerical.hpp>

#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace dyno {

void GroundTruthInputHandler::add(const GroundTruthRosWrapper& input,
                                  FrameId frame_id) {
  // already have a ground truth for this frame. Assume nothing will change in
  // the past (becuase ground truth!!)
  if (ground_truths_.exists(frame_id)) {
    return;
  }

  gtsam::Pose3 X_W;
  convert(input.X_W.pose.pose, X_W);

  // desired timestamp
  Timestamp timestamp = utils::fromRosTime(input.timestamp);

  gtsam::FastMap<ObjectId, ObjectPoseGT> ground_truth_object_map;
  const dynamic_slam_interfaces::msg::MultiObjectOdometryPath& object_paths =
      input.object_paths;
  for (const auto& path : object_paths.paths) {
    // only for this frame id which is represented in the
    // MultiObjectOdometryPath by the sequence id this requires a nested for
    // loop which is slow and annoying due to the (potential) presense of
    // multiple path segments we iterate from the end becuase it is likely the
    // entry we want is the last one if we recieve the input ground truth in
    // order (which we should!)
    for (auto it = path.object_odometries.rbegin();
         it != path.object_odometries.rend(); ++it) {
      const auto& object_odom = *it;

      Timestamp object_timestamp =
          utils::fromRosTime(object_odom.odom.header.stamp);
      // match based on timestamp as frame id is internal to dynosam and cannot
      // be used for global association!
      if (fpEqual(timestamp, object_timestamp)) {
        ObjectId object_id = path.object_id;
        // sanity check that each object is unique that this frame
        CHECK(!ground_truth_object_map.exists(object_id))
            << "Failure when constructing ground truth from ROS "
            << info_string(frame_id, object_id);
        ObjectPoseGT gt_object;
        gt_object.frame_id_ = frame_id;
        gt_object.object_id_ = object_id;

        // TODO: NOTE: no checking of timestamp!!!?
        //  We only need to set L_W and L_camera in the input packet since the
        //  calculateAndSetMotions will do everything else for us!!
        convert(object_odom.odom.pose.pose, gt_object.L_world_);
        gt_object.L_camera_ = X_W.inverse() * gt_object.L_world_;
        ground_truth_object_map.insert2(object_id, gt_object);
        VLOG(20) << "Added ground truth object "
                 << info_string(frame_id, object_id);
        break;
      }
    }
  }

  std::vector<ObjectPoseGT> gt_object_poses;
  gt_object_poses.reserve(ground_truth_object_map.size());

  std::transform(ground_truth_object_map.begin(), ground_truth_object_map.end(),
                 std::back_inserter(gt_object_poses),
                 [](auto const& kv) { return kv.second; });

  // finally add to internal data-structure
  GroundTruthInputPacket ground_truth_packet_k(timestamp, frame_id, X_W,
                                               gt_object_poses);

  GroundTruthInputPacket ground_truth_packet_km1;
  if (query(&ground_truth_packet_km1, frame_id - 1)) {
    ground_truth_packet_k.calculateAndSetMotions(ground_truth_packet_km1);
  }

  ground_truths_.insert2(frame_id, ground_truth_packet_k);
}

bool GroundTruthInputHandler::query(GroundTruthInputPacket* ground_truth,
                                    FrameId query_frame) const {
  if (!ground_truths_.exists(query_frame)) {
    return false;
  }

  CHECK_NOTNULL(ground_truth);
  *ground_truth = ground_truths_.at(query_frame);
  return true;
}

GroundTruthInputHandlerROS::GroundTruthInputHandlerROS(
    rclcpp::Node::SharedPtr node,
    DataProvider::GroundTruthPacketCallback* ground_truth_packet_callback)
    : node_(node), ground_truth_packet_callback_(ground_truth_packet_callback) {
  rclcpp::Node* node_ptr = node_.get();
  CHECK_NOTNULL(node_ptr);
  robot_odometry_sub_.subscribe(node_ptr, "ground_truth/odom");
  object_odometry_sub_.subscribe(node_ptr, "ground_truth/object_odometries");

  static constexpr size_t kQueueSize = 1000u;
  sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
      SyncPolicy(kQueueSize), robot_odometry_sub_, object_odometry_sub_);
  sync_->registerCallback(std::bind(&GroundTruthInputHandlerROS::callback, this,
                                    std::placeholders::_1,
                                    std::placeholders::_2));

  RCLCPP_INFO_STREAM(
      node_->get_logger(),
      "GroundTruthInputHandlerROS has been connected. Subscribed to ground "
      "truth topics: "
          << robot_odometry_sub_.getSubscriber()->get_topic_name() << " "
          << object_odometry_sub_.getSubscriber()->get_topic_name() << ".");
}

void GroundTruthInputHandlerROS::emit(Timestamp timestamp, FrameId frame_id) {
  GroundTruthRosWrapper ground_truth_wrapper;
  if (cache_.getValueAtTime(timestamp, &ground_truth_wrapper)) {
    this->add(ground_truth_wrapper, frame_id);

    GroundTruthInputPacket ground_truth;
    CHECK(this->query(&ground_truth, frame_id));
    // the callback is only set in the DynosamPipeline so may not be valid at
    // the time of construction it should be valid by the time we emit since
    // callbacks should only start after everything is constructed
    CHECK(ground_truth_packet_callback_);
    ground_truth_packet_callback_->operator()(ground_truth);
    // send ground truth to the data-interface

    // TODO: once weve sent to the data-interface we should delete all earlier
    // timestamps from the cache!!
  } else {
    VLOG(20) << "Failed to emit ground truth packet at t=" << timestamp
             << " k=" << frame_id << ". Exact timestamp was not available!";
  }
}

void GroundTruthInputHandlerROS::callback(
    const nav_msgs::msg::Odometry::ConstSharedPtr& X_W,
    const dynamic_slam_interfaces::msg::MultiObjectOdometryPath::ConstSharedPtr&
        object_paths) {
  auto ros_time = X_W->header.stamp;
  Timestamp timestamp = utils::fromRosTime(ros_time);
  VLOG(20) << "Gotten ground truth callback t= " << timestamp;
  cache_.addValue(timestamp,
                  GroundTruthRosWrapper(ros_time, *X_W, *object_paths));
}

DataProviderRos::DataProviderRos(rclcpp::Node::SharedPtr node)
    : DataProvider(), node_(node) {
  bool use_ground_truth = ParameterConstructor(node_, "use_ground_truth", false)
                              .description(
                                  "If the online DataProvider should try and "
                                  "subscribe to ground truth topics "
                                  " for odometry and objects")
                              .finish()
                              .get<bool>();
  if (use_ground_truth) {
    RCLCPP_INFO_STREAM(node_->get_logger(),
                       "Creating ground truth handler ROS");
    ground_truth_handler_ = std::make_unique<GroundTruthInputHandlerROS>(
        node_, &this->ground_truth_packet_callback_);
  }
}

// specalise conversion function for depth image to handle the case we are given
// float32 bit images
template <>
const cv::Mat DataProviderRos::convertRosImage<ImageType::Depth>(
    const sensor_msgs::msg::Image::ConstSharedPtr& img_msg) const {
  const cv_bridge::CvImageConstPtr cvb_image = readRosImage(img_msg);
  const cv::Mat img = cvb_image->image;

  try {
    image_traits<ImageType::Depth>::validate(img);
    return img;

  } catch (const InvalidImageTypeException& exception) {
    // handle the case its a float32
    if (img.type() == CV_32FC1) {
      cv::Mat depth64;
      img.convertTo(depth64, CV_64FC1);
      image_traits<ImageType::Depth>::validate(depth64);
      return depth64;
    }

    RCLCPP_FATAL_STREAM(node_->get_logger(),
                        image_traits<ImageType::Depth>::name()
                            << " Image msg was of the wrong type (validate "
                               "failed with exception "
                            << exception.what() << "). "
                            << "ROS encoding type used was "
                            << cvb_image->encoding);

    rclcpp::shutdown();
    return cv::Mat();
  }
}

const cv::Mat DataProviderRos::readRgbRosImage(
    const sensor_msgs::msg::Image::ConstSharedPtr& img_msg) const {
  cv::Mat rgb_mono_image = convertRosImage<ImageType::RGBMono>(img_msg);
  if (img_msg->encoding == "rgb8") {
    cv::cvtColor(rgb_mono_image, rgb_mono_image, cv::COLOR_RGB2BGR);
  }
  return rgb_mono_image;
}

const cv::Mat DataProviderRos::readDepthRosImage(
    const sensor_msgs::msg::Image::ConstSharedPtr& img_msg) const {
  return convertRosImage<ImageType::Depth>(img_msg);
}

const cv::Mat DataProviderRos::readFlowRosImage(
    const sensor_msgs::msg::Image::ConstSharedPtr& img_msg) const {
  return convertRosImage<ImageType::OpticalFlow>(img_msg);
}

const cv::Mat DataProviderRos::readMaskRosImage(
    const sensor_msgs::msg::Image::ConstSharedPtr& img_msg) const {
  return convertRosImage<ImageType::MotionMask>(img_msg);
}

const cv_bridge::CvImageConstPtr DataProviderRos::readRosImage(
    const sensor_msgs::msg::Image::ConstSharedPtr& img_msg) const {
  CHECK(img_msg);
  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    // important to copy to ensure that memory does not go out of scope (which
    // it seems to !!!)
    cv_ptr = cv_bridge::toCvCopy(img_msg);
  } catch (cv_bridge::Exception& exception) {
    RCLCPP_FATAL(node_->get_logger(), "cv_bridge exception: %s",
                 exception.what());
    rclcpp::shutdown();
  }
  return cv_ptr;
}

}  // namespace dyno
