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

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <Eigen/Dense>
#include <png++/png.hpp>

#include "dynosam/dataprovider/ClusterSlamDataProvider.hpp"
#include "dynosam/dataprovider/KittiDataProvider.hpp"
#include "dynosam/dataprovider/TartanAirShibuya.hpp"
#include "dynosam/dataprovider/ViodeDataProvider.hpp"
#include "dynosam/dataprovider/VirtualKittiDataProvider.hpp"
#include "dynosam/frontend/vision/FeatureTracker.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam/pipeline/PipelineManager.hpp"
#include "dynosam/pipeline/PipelineParams.hpp"
#include "dynosam/visualizer/OpenCVFrontendDisplay.hpp"
#include "dynosam_common/viz/Colour.hpp"
#include "dynosam_cv/Camera.hpp"
#include "dynosam_cv/ImageContainer.hpp"
#include "dynosam_nn/PyObjectDetector.hpp"

DEFINE_string(path_to_kitti, "/root/data/kitti", "Path to KITTI dataset");
// TODO: (jesse) many better ways to do this with ros - just for now
DEFINE_string(
    params_folder_path, "dynosam/params",
    "Path to the folder containing the yaml files with the VIO parameters.");

#include "dynosam/dataprovider/KittiDataProvider.hpp"
#include "dynosam/dataprovider/OMDDataProvider.hpp"
#include "dynosam/frontend/vision/VisionTools.hpp"

int main(int argc, char* argv[]) {
  using namespace dyno;
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  FLAGS_log_prefix = 1;

  KittiDataLoader::Params params;
  KittiDataLoader loader("/root/data/vdo_slam/kitti/kitti/0018/", params);
  // ClusterSlamDataLoader loader("/root/data/cluster_slam/CARLA-L1");
  // loader.setStartingFrame(600);
  // OMDDataLoader loader(
  //     "/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/");

  // TartanAirShibuyaLoader
  // loader("/root/data/TartanAir_shibuya/RoadCrossing07/");
  // ViodeLoader loader("/root/data/VIODE/city_day/mid");

  auto detector = dyno::PyObjectDetectorWrapper::CreateYoloDetector();
  CHECK_NOTNULL(detector);

  FrontendParams fp;
  fp.tracker_params.feature_detector_type =
      TrackerParams::FeatureDetectorType::GFFT_CUDA;
  fp.tracker_params.max_dynamic_features_per_frame = 300;
  // fp.tracker_params.feature_detector_type =
  // TrackerParams::FeatureDetectorType::ORB_SLAM_ORB;

  auto camera = std::make_shared<Camera>(*loader.getCameraParams());
  auto tracker = std::make_shared<FeatureTracker>(fp, camera);

  loader.setCallback([&](dyno::FrameId frame_id, dyno::Timestamp timestamp,
                         cv::Mat rgb, cv::Mat optical_flow, cv::Mat depth,
                         cv::Mat motion, gtsam::Pose3,
                         GroundTruthInputPacket) -> bool {
    // loader.setCallback([&](dyno::FrameId frame_id, dyno::Timestamp timestamp,
    //                        cv::Mat rgb, cv::Mat optical_flow, cv::Mat depth,
    //                        cv::Mat motion, GroundTruthInputPacket,
    //                        std::optional<ImuMeasurements> imu_measurements,
    //                        std::optional<cv::Mat>) -> bool {
    // loader.setCallback([&](dyno::FrameId frame_id, dyno::Timestamp timestamp,
    //                        cv::Mat rgb, cv::Mat optical_flow, cv::Mat depth,
    //                        cv::Mat motion, GroundTruthInputPacket,
    //                        std::optional<cv::Mat>) -> bool {
    LOG(INFO) << frame_id << " " << timestamp;

    cv::Mat of_viz, motion_viz, depth_viz;
    of_viz = ImageType::OpticalFlow::toRGB(optical_flow);
    motion_viz = ImageType::MotionMask::toRGB(motion);
    depth_viz = ImageType::Depth::toRGB(depth);

    // ImageContainerDeprecate::Ptr container = ImageContainerDeprecate::Create(
    //     timestamp, frame_id, ImageWrapper<ImageType::RGBMono>(rgb),
    //     ImageWrapper<ImageType::Depth>(depth),
    //     ImageWrapper<ImageType::OpticalFlow>(optical_flow),
    //     ImageWrapper<ImageType::MotionMask>(motion));

    // cv::Mat boarder_mask;
    // vision_tools::computeObjectMaskBoundaryMask(
    //     motion,
    //     boarder_mask,
    //     8
    // );

    // cv::Scalar red = dyno::Color::red();

    // const ObjectIds instance_labels = vision_tools::getObjectLabels(motion);
    // for(const auto object_id : instance_labels) {
    //     std::vector<std::vector<cv::Point>> detected_contours;
    //     vision_tools::findObjectBoundingBox(motion,
    //     object_id,detected_contours);

    //     cv::drawContours(boarder_mask, detected_contours, -1, red, 8);
    // }

    // cv::imshow("Mask with boarder", boarder_mask);

    // cv::imshow("RGB", rgb);
    // cv::imshow("OF", of_viz);
    // cv::imshow("Motion", motion_viz);
    // // cv::waitKey(1);
    // cv::imshow("Depth", depth_viz);

    auto object_detection_result = detector->process(rgb);
    cv::imshow("Detection Result", object_detection_result.colouredMask());

    ImageContainer image_container(frame_id, timestamp);
    image_container.rgb(rgb)
        .depth(depth)
        .opticalFlow(optical_flow)
        .objectMotionMask(object_detection_result.labelled_mask);

    auto frame = tracker->track(frame_id, timestamp, image_container);
    Frame::Ptr previous_frame = tracker->getPreviousFrame();

    // if(frame_id == 605) {
    //   auto all_tracks = frame->static_features_.collectTracklets();
    //   frame->static_features_.markOutliers(all_tracks);
    // }

    // // motion_viz =
    // ImageType::MotionMask::toRGB(frame->image_container_.get<ImageType::MotionMask>());
    // // // cv::imshow("Motion", motion_viz);

    cv::Mat tracking;
    if (previous_frame) {
      ImageTracksParams track_viz_params(true);
      track_viz_params.show_intermediate_tracking = true;
      tracking = tracker->computeImageTracks(*previous_frame, *frame,
                                             track_viz_params);

      // if (imu_measurements) {
      //   const auto previous_timestamp = previous_frame->getTimestamp();

      //   CHECK_GE(imu_measurements->timestamps_[0], previous_timestamp);
      //   CHECK_LT(imu_measurements
      //                ->timestamps_[imu_measurements->timestamps_.cols() - 1],
      //            timestamp);

      //   LOG(INFO) << "Gotten imu messages!";

      //   CHECK(imu_measurements->synchronised_frame_id);
      //   CHECK_EQ(imu_measurements->synchronised_frame_id.value(),
      //            frame->getFrameId());
      // }
    }
    if (!tracking.empty()) cv::imshow("Tracking", tracking);

    LOG(INFO) << to_string(tracker->getTrackerInfo());
    const std::string path = "/root/results/misc/";
    // if (previous_frame && (char)cv::waitKey(0) == 's') {
    //   LOG(INFO) << "Saving...";
    //   // cv::imwrite(path + "omd_su4_rgb.png", rgb);
    //   // cv::imwrite(path + "omd_su4_of.png", of_viz);
    //   // cv::imwrite(path + "omd_su4_motion.png", motion_viz);
    //   // cv::imwrite(path + "omd_su4_depth.png", depth_viz);
    //   // cv::imwrite(
    //   //     path + "cluster_tracking_new" + std::to_string(frame_id) +
    //   ".png",
    //   //     tracking);
    // }
    cv::waitKey(1);

    return true;
  });

  while (loader.spin()) {
  }
}

// #include "dynosam/dataprovider/ProjectAriaDataProvider.hpp"
// #include "dynosam/frontend/vision/VisionTools.hpp"

// int main(int argc, char* argv[]) {

//     using namespace dyno;
//     google::ParseCommandLineFlags(&argc, &argv, true);
//     google::InitGoogleLogging(argv[0]);
//     FLAGS_logtostderr = 1;
//     FLAGS_colorlogtostderr = 1;
//     FLAGS_log_prefix = 1;

//     // ClusterSlamDataLoader loader("/root/data/cluster_slam/CARLA-S1");
//     ProjectARIADataLoader loader("/root/data/zed/acfr_3_moving_medium/");

//     loader.setCallback([&](dyno::FrameId frame_id, dyno::Timestamp timestamp,
//     cv::Mat rgb, cv::Mat optical_flow, cv::Mat depth, cv::Mat motion) -> bool
//     {

//         LOG(INFO) << frame_id << " " << timestamp;

//         cv::imshow("RGB", rgb);
//         cv::imshow("OF", ImageType::OpticalFlow::toRGB(optical_flow));
//         cv::imshow("Motion", ImageType::MotionMask::toRGB(motion));
//         cv::imshow("Depth", ImageType::Depth::toRGB(depth));

//         cv::Mat shrunk_mask;
//         vision_tools::shrinkMask(motion, shrunk_mask, 20);
//         cv::imshow("Shrunk Motion",
//         ImageType::MotionMask::toRGB(shrunk_mask));

//         cv::waitKey(1);
//         return true;
//     });

//     while(loader.spin()) {}

// }
