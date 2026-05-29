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

#include <cv_bridge/cv_bridge.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <dynosam/common/Camera.hpp>
#include <dynosam/dataprovider/KittiDataProvider.hpp>
#include <dynosam/dataprovider/OMDDataProvider.hpp>
#include <dynosam/frontend/vision/FeatureTracker.hpp>
#include <dynosam/frontend/vision/VisionTools.hpp>
#include <dynosam/utils/OpenCVUtils.hpp>

#include "dynosam_ros/Utils.hpp"
#include "image_transport/image_transport.hpp"
#include "rclcpp/executor.hpp"
#include "rclcpp/node.hpp"
#include "rclcpp/node_options.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char* argv[]) {
  using namespace dyno;
  auto non_ros_args = dyno::initRosAndLogging(argc, argv);
  return 0;

  rclcpp::NodeOptions options;
  options.arguments(non_ros_args);
  options.use_intra_process_comms(true);

  auto node =
      std::make_shared<rclcpp::Node>("dynosam_dataset_example", options);

  image_transport::Publisher input_images_pub =
      image_transport::create_publisher(node.get(),
                                        "dynosam_dataset/input_images");
  image_transport::Publisher tracking_images_pub =
      image_transport::create_publisher(node.get(),
                                        "dynosam_dataset/tracking_image");

  // KittiDataLoader::Params params;
  // KittiDataLoader loader("/root/data/vdo_slam/kitti/kitti/0004/", params);

  OMDDataLoader loader(
      "/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/");

  auto camera = std::make_shared<Camera>(*loader.getCameraParams());
  auto tracker = std::make_shared<FeatureTracker>(FrontendParams(), camera);

  loader.setCallback([&](dyno::FrameId frame_id, dyno::Timestamp timestamp,
                         cv::Mat rgb, cv::Mat optical_flow, cv::Mat depth,
                         cv::Mat motion, GroundTruthInputPacket) -> bool {
    // loader.setCallback([&](dyno::FrameId frame_id, dyno::Timestamp timestamp,
    //     cv::Mat rgb, cv::Mat optical_flow, cv::Mat depth, cv::Mat motion,
    //     gtsam::Pose3, GroundTruthInputPacket) -> bool {

    LOG(INFO) << frame_id << " " << timestamp;

    cv::Mat flow_viz, mask_viz, depth_viz;
    flow_viz = ImageType::OpticalFlow::toRGB(optical_flow);
    mask_viz = ImageType::MotionMask::toRGB(motion);
    depth_viz = ImageType::Depth::toRGB(depth);

    ImageContainer image_container(frame_id, timestamp);
    image_container.rgb(rgb)
        .depth(depth)
        .opticalFlow(optical_flow)
        .objectMotionMask(motion);

    std::set<ObjectId> object_keyframes;
    auto frame =
        tracker->track(frame_id, timestamp, image_container, object_keyframes);
    Frame::Ptr previous_frame = tracker->getPreviousFrame();

    cv::Mat tracking;
    if (previous_frame) {
      tracking = tracker->computeFeatureTracks(*previous_frame, *frame, false);
      std_msgs::msg::Header hdr;
      sensor_msgs::msg::Image::SharedPtr msg =
          cv_bridge::CvImage(hdr, "bgr8", tracking).toImageMsg();

      tracking_images_pub.publish(msg);
    }

    cv::Mat combined_image = utils::concatenateImagesVertically(
        utils::concatenateImagesHorizontally(rgb, depth_viz),
        utils::concatenateImagesHorizontally(flow_viz, mask_viz));

    std_msgs::msg::Header hdr;
    sensor_msgs::msg::Image::SharedPtr msg =
        cv_bridge::CvImage(hdr, "bgr8", combined_image).toImageMsg();

    input_images_pub.publish(msg);

    return true;
  });

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);

  while (rclcpp::ok()) {
    if (!loader.spin()) {
      break;
    }
    exec.spin_some();
  }
}
