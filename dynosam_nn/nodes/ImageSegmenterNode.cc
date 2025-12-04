#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <boost/python.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "cv_bridge/cv_bridge.hpp"
#include "dynosam_nn/ModelConfig.hpp"
#include "dynosam_nn/PyObjectDetector.hpp"
#include "dynosam_nn/TrtUtilities.hpp"
#include "dynosam_nn/YoloObjectDetector.hpp"
#include "image_transport/image_transport.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/image.hpp"

class ImageSegmenterNode : public rclcpp::Node {
 public:
  ImageSegmenterNode() : Node("image_subscriber_node") {
    // engine_ = dyno::PyObjectDetectorWrapper::CreateYoloDetector();
    dyno::YoloConfig yolo_config;
    dyno::ModelConfig model_config;
    model_config.model_file = "yolov8n-seg.pt";
    engine_ =
        std::make_unique<dyno::YoloV8ObjectDetector>(model_config, yolo_config);
    // model_ = std::make_unique<dyno::Model>(config);
    // Use image_transport for efficiency (handles compressed images too)
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/camera/color/image_rect_color", 10,
        std::bind(&ImageSegmenterNode::imageCallback, this,
                  std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(),
                "Image subscriber initialized and listening on "
                "/camera/color/image_rect_color");
  }

 private:
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    try {
      // Convert to OpenCV image (BGR8)
      cv::Mat frame = cv_bridge::toCvShare(msg, "bgr8")->image;

      // if (model_) model_->infer(frame);
      auto result = engine_->process(frame);

      // cv::Mat resized;
      // cv::resize(frame, resized, cv::Size(640, 480), 0, 0, cv::INTER_LINEAR);

      // // Print image info
      // RCLCPP_INFO(this->get_logger(), "Received image %dx%d", resized.cols,
      //             resized.rows);
      // auto r = engine_->process(resized);

      LOG(INFO) << result;

      // // // // Optional: visualize (disable in headless mode)
      // cv::imshow("View", result.colouredMask());
      // cv::waitKey(1);

    } catch (const cv_bridge::Exception& e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  dyno::ObjectDetectionEngine::UniquePtr engine_;
  // std::unique_ptr<dyno::Model> model_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  FLAGS_log_prefix = 1;

  // const std::string enginePath = dyno::getNNWeightsPath() /
  // "yolov8n-seg.engine";

  // dyno::ModelConfig config;
  // config.model_file = "yolov8n-seg.pt";
  // dyno::Model model(config);

  // Py_Initialize();
  // {
  auto node = std::make_shared<ImageSegmenterNode>();
  //   FLAGS_logtostderr = 1;
  //   FLAGS_colorlogtostderr = 1;
  //   FLAGS_log_prefix = 1;

  rclcpp::spin(node);
  // }
  rclcpp::shutdown();
  // Finalize the Python interpreter.
  // Py_FinalizeEx();
  return 0;
}
