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
#include "dynosam_nn/YOLOV11.hpp"
#include "dynosam_nn/YoloObjectDetector.hpp"
#include "image_transport/image_transport.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/image.hpp"

/**
 * @brief Setting up Tensorrt logger
 */
class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    // Only output logs with severity greater than warning
    LOG(INFO) << msg;
    // if (severity <= Severity::kWARNING)
    //     std::cout << msg << std::endl;
  }
};

class ImageSegmenterNode : public rclcpp::Node {
 public:
  ImageSegmenterNode() : Node("image_subscriber_node") {
    // engine_ = dyno::PyObjectDetectorWrapper::CreateYoloDetector();
    dyno::YoloConfig yolo_config;
    dyno::ModelConfig model_config;
    model_config.model_file = "yolov8n-seg.pt";
    // model_config.model_file = "yolo11s.pt";
    engine_ =
        std::make_unique<dyno::YoloV8ObjectDetector>(model_config, yolo_config);
    // model_ = std::make_unique<dyno::Model>(config);
    // yolov11_ =
    // std::make_unique<YOLOv11>(model_config.modelPath().replace_extension(".engine"),
    // nv_logger_); yolov11_ =
    // std::make_unique<YOLOv11>(model_config.onnxPath(), nv_logger_); Use
    // image_transport for efficiency (handles compressed images too)
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

      LOG(INFO) << result;

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
  // std::unique_ptr<YOLOv11> yolov11_;
  // std::unique_ptr<dyno::Model> model_;
  Logger nv_logger_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  FLAGS_log_prefix = 1;
  FLAGS_v = 5;

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
  // // }
  rclcpp::shutdown();
  // // Finalize the Python interpreter.
  // Py_FinalizeEx();
  return 0;
}
