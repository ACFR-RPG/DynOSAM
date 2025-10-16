#include <boost/python.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "cv_bridge/cv_bridge.hpp"
#include "dynosam_nn/PyObjectDetector.hpp"
#include "image_transport/image_transport.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/image.hpp"

class ImageSegmenterNode : public rclcpp::Node {
 public:
  ImageSegmenterNode() : Node("image_subscriber_node") {
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
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
    try {
      // Convert to OpenCV image (BGR8)
      cv::Mat frame = cv_bridge::toCvShare(msg, "bgr8")->image;

      // Print image info
      RCLCPP_INFO(this->get_logger(), "Received image %dx%d", frame.cols,
                  frame.rows);
      auto r = engine_.process(frame);

      LOG(INFO) << r;

      // Optional: visualize (disable in headless mode)
      // cv::imshow("View", frame);
      // cv::waitKey(1);

    } catch (const cv_bridge::Exception &e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  dyno::PyObjectDetectorWrapper engine_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  Py_Initialize();
  {
    auto node = std::make_shared<ImageSegmenterNode>();
    FLAGS_logtostderr = 1;
    FLAGS_colorlogtostderr = 1;
    FLAGS_log_prefix = 1;

    rclcpp::spin(node);
  }
  rclcpp::shutdown();
  // Finalize the Python interpreter.
  Py_FinalizeEx();
  return 0;
}
