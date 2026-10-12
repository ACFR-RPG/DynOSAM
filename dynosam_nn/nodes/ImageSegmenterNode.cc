#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <boost/python.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "cv_bridge/cv_bridge.hpp"
#include "dynosam_common/utils/Statistics.hpp"
#include "dynosam_nn/ModelConfig.hpp"
#include "dynosam_nn/PyObjectDetector.hpp"
#include "dynosam_nn/TrtUtilities.hpp"
#include "dynosam_nn/YoloV8ObjectDetector.hpp"
#include "image_transport/image_transport.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/image.hpp"

class ImageSegmenterNode : public rclcpp::Node {
 public:
  ImageSegmenterNode()
      : Node("image_subscriber_node"),
        topic_(declare_parameter<std::string>(
            "image_topic", "/camera/color/image_rect_color")),
        model_name_(
            declare_parameter<std::string>("model_name", "yolov8n-seg.pt")) {
    dyno::YoloConfig yolo_config;
    dyno::ModelConfig model_config;

    RCLCPP_INFO_STREAM(this->get_logger(), "Using model: " << model_name_);
    model_config.model_file = model_name_;
    engine_ =
        std::make_unique<dyno::YoloV8ObjectDetector>(model_config, yolo_config);

    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
        topic_, 100,
        std::bind(&ImageSegmenterNode::imageCallback, this,
                  std::placeholders::_1));

    RCLCPP_INFO_STREAM(
        this->get_logger(),
        "Image subscriber initialized and listening on " << topic_);
  }

 private:
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    try {
      // Convert to OpenCV image (BGR8)
      cv::Mat frame = cv_bridge::toCvShare(msg, "bgr8")->image;

      // cv::Mat resized;
      // cv::resize(frame, resized, cv::Size(640, 640), 0, 0, cv::INTER_LINEAR);

      // if (model_) model_->infer(frame);
      auto result = engine_->process(frame);
      //   yolov11_->preprocess(resized);

      // // Perform inference
      // yolov11_->infer();

      // // Postprocess to get detections
      // std::vector<Detection> detections;
      // yolov11_->postprocess(detections);

      // // Draw detections on the frame
      // yolov11_->draw(resized, detections);

      // // Display the frame (optional)
      // cv::imshow("Inference", resized);

      // // Print image info
      // RCLCPP_INFO(this->get_logger(), "Received image %dx%d", resized.cols,
      //             resized.rows);
      // auto r = engine_->process(resized);

      // LOG(INFO) << result;

      // // // // Optional: visualize (disable in headless mode)
      cv::imshow("View", result.colouredMask());
      cv::waitKey(1);

    } catch (const cv_bridge::Exception& e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }

    RCLCPP_INFO_STREAM_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                dyno::utils::Statistics::Print());
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  dyno::ObjectDetectionEngine::UniquePtr engine_;

  std::string topic_;
  std::string model_name_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  FLAGS_log_prefix = 1;
  FLAGS_v = 10;

  rclcpp::spin(std::make_shared<ImageSegmenterNode>());
  rclcpp::shutdown();
  return 0;
}
