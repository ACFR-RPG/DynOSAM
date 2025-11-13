#include <opencv4/opencv2/opencv.hpp>

#include "cv_bridge/cv_bridge.hpp"
#include "dynosam/frontend/vision/FeatureTracker.hpp"
#include "dynosam_cv/Camera.hpp"
#include "dynosam_cv/ImageContainer.hpp"
#include "dynosam_ros/DataProviderRos.hpp"
#include "dynosam_ros/RosUtils.hpp"
#include "image_transport/image_transport.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/image.hpp"

using namespace dyno;

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  FLAGS_log_prefix = 1;

  auto node = rclcpp::Node::make_shared("dynosam_tracking_node");

  FrontendParams fp;
  fp.tracker_params.feature_detector_type =
      TrackerParams::FeatureDetectorType::GFFT_CUDA;
  fp.tracker_params.max_dynamic_features_per_frame = 300;
  fp.tracker_params.prefer_provided_optical_flow = false;
  fp.tracker_params.prefer_provided_object_detection = false;

  const CameraParams original_camera_params =
      waitAndSetCameraParams(node, "camera_info");

  const auto original_size = original_camera_params.imageSize();
  cv::Size rescale_size = original_size;
  cv::Mat original_K = original_camera_params.getCameraMatrix();
  cv::Mat distortion = original_camera_params.getDistortionCoeffs();

  cv::Mat new_K = cv::getOptimalNewCameraMatrix(
      original_K, distortion, original_size, 1.0, rescale_size);

  cv::Mat mapx, mapy;
  LOG(INFO) << CV_32FC1;
  cv::initUndistortRectifyMap(original_K, distortion, cv::Mat(), new_K,
                              rescale_size, CV_32FC1, mapx, mapy);

  dyno::CameraParams::IntrinsicsCoeffs intrinsics;
  cv::Mat K_double;
  new_K.convertTo(K_double, CV_64F);
  // TODO: should make this a constructor...
  dyno::CameraParams::convertKMatrixToIntrinsicsCoeffs(K_double, intrinsics);
  dyno::CameraParams::DistortionCoeffs zero_distortion(4, 0);

  CameraParams new_camera_params(intrinsics, zero_distortion, rescale_size,
                                 original_camera_params.getDistortionModel(),
                                 original_camera_params.getExtrinsics());

  LOG(INFO) << "New camera " << new_camera_params.toString();

  auto camera = std::make_shared<Camera>(new_camera_params);
  auto tracker = std::make_shared<FeatureTracker>(fp, camera);

  FrameId frame_id{0};

  auto subscription = node->create_subscription<sensor_msgs::msg::Image>(
      "image_raw", 100,
      [&tracker, &frame_id, &mapx,
       &mapy](const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg);
        const cv::Mat rgb = cv_ptr->image;

        cv::Mat rgb_undistort;
        cv::remap(rgb, rgb_undistort, mapx, mapy, cv::INTER_LINEAR);

        // output will have the same type as mapx/y so covnert back to required
        // type
        rgb_undistort.convertTo(rgb_undistort, rgb.type());

        const Timestamp timestamp = utils::fromRosTime(msg->header.stamp);

        ImageContainer image_container(frame_id, timestamp);
        image_container.rgb(rgb);

        frame_id++;

        auto frame = tracker->track(frame_id, timestamp, image_container);
        Frame::Ptr previous_frame = tracker->getPreviousFrame();

        if (previous_frame) {
          ImageTracksParams track_viz_params(true);
          track_viz_params.show_intermediate_tracking = true;
          cv::Mat tracking = tracker->computeImageTracks(
              *previous_frame, *frame, track_viz_params);

          if (!tracking.empty()) cv::imshow("Tracking", tracking);
          cv::waitKey(1);
        }
      });

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
