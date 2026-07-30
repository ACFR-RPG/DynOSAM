#pragma once

#include <dynosam/dataprovider/DataProvider.hpp>
#include <mutex>

#include "cv_bridge/cv_bridge.hpp"
#include "dynamic_slam_interfaces/msg/groundtruth_packet.hpp"
#include "dynosam_ros/CameraSystem.hpp"
#include "dynosam_ros/adaptors/GroundtuthPacketAdaptor.hpp"
#include "dynosam_ros/adaptors/ImuMeasurementAdaptor.hpp"
#include "image_transport/image_transport.hpp"
#include "rclcpp/node.hpp"
#include "rclcpp/node_options.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/imu.hpp"

namespace dyno {

typedef sensor_msgs::msg::Image::ConstSharedPtr ImageMsgPtr;

typedef dynamic_slam_interfaces::msg::GroundtruthPacket GroundtruthMsg;

/**
 * @brief
 * Heavily inspiried by the OKVIS2 implementation:
 * https://github.com/ethz-mrl/okvis2/blob/main/okvis_ros2/include/okvis/ros2/Subscriber.hpp
 */
class Subscriber : public DataProvider {
 public:
  DYNO_POINTER_TYPEDEFS(Subscriber)

  // NOTE: ground truth subscription handled via ros param gotten through node
  Subscriber(SensorSystem::Ptr sensor_system,
             std::shared_ptr<rclcpp::Node> node);
  // For two rigs exactly
  Subscriber(SensorSystem::Ptr sensor_system_l,
             SensorSystem::Ptr sensor_system_f,
             std::shared_ptr<rclcpp::Node> node);
  ~Subscriber() = default;

  /** No end to the dataset */
  int datasetSize() const override { return -1; }
  /* True while not shutdown */
  bool spin() override;
  /* Disconnects all subscriber */
  void shutdown() override;

  CanonicalSensorRig::Ptr sensorRig() const override;

 private:
  void imageCallback(const ImageMsgPtr& msg, unsigned int stream_index);

  bool addImages(Timestamp timestamp,
                 const std::map<size_t, ImageMsgPtr>& image_msgs);

  // Currently only account for having two rigs
  void imageMultiRigCallback(const ImageMsgPtr& msg, unsigned int stream_index);
  bool addImagesMultiRig(Timestamp timestamp,
                 const std::map<size_t, ImageMsgPtr>& image_msgs);

 private:
  /**
   * @brief Convers a sensor_msgs::msg::Image to a cv::Mat while testing that
   * the input has the correct datatype for an RGB image (as defined by
   * ImageType::RGBMono).
   *
   * @param img_msg const ImageMsgPtr&
   * @return const cv::Mat
   */
  const cv::Mat readRgbRosImage(const ImageMsgPtr& img_msg) const;

  /**
   * @brief Convers a sensor_msgs::msg::Image to a cv::Mat while testing that
   * the input has the correct datatype for an Depth image (as defined by
   * ImageType::Depth).
   *
   * @param img_msg const ImageMsgPtr&
   * @return const cv::Mat
   */
  const cv::Mat readDepthRosImage(const ImageMsgPtr& img_msg) const;

  /**
   * @brief Convers a sensor_msgs::msg::Image to a cv::Mat while testing that
   * the input has the correct datatype for an Optical Flow image (as defined by
   * ImageType::OpticalFlow).
   *
   * @param img_msg const ImageMsgPtr&
   * @return const cv::Mat
   */
  const cv::Mat readFlowRosImage(const ImageMsgPtr& img_msg) const;

  /**
   * @brief Convers a sensor_msgs::msg::Image to a cv::Mat while testing that
   * the input has the correct datatype for an Motion Mask image (as defined by
   * ImageType::MotionMask).
   *
   * @param img_msg const ImageMsgPtr&
   * @return const cv::Mat
   */
  const cv::Mat readMaskRosImage(const ImageMsgPtr& img_msg) const;

  /**
   * @brief Helper function to convert a ROS Image message to a CvImageConstPtr
   * via the cv bridge.
   *
   * @param img_msg const ImageMsgPtr&
   * @return const cv_bridge::CvImageConstPtr
   */
  const cv_bridge::CvImageConstPtr readRosImage(
      const ImageMsgPtr& img_msg) const;

  /**
   * @brief Helper function to convert a
   * sensor_msgs::msg::Image::ConstSharedPtr& to a cv::Mat with the right
   * datatype.
   *
   * The datatype is specified from the template IMAGETYPE::OpenCVType and
   * ensures the passed in image has the correct datatype for the desired
   * IMAGETYPE.
   *
   * ROS will be shutdown if the incoming image has an incorrect type.
   *
   * @tparam IMAGETYPE
   * @param img_msg  const ImageMsgPtr&
   * @return const cv::Mat
   */
  template <typename IMAGETYPE>
  const cv::Mat convertRosImage(const ImageMsgPtr& img_msg) const {
    const cv_bridge::CvImageConstPtr cvb_image = readRosImage(img_msg);
    try {
      const cv::Mat img = cvb_image->image;
      image_traits<IMAGETYPE>::validate(img);
      return img;

    } catch (const InvalidImageTypeException& exception) {
      RCLCPP_FATAL_STREAM(node_->get_logger(),
                          image_traits<IMAGETYPE>::name()
                              << " Image msg was of the wrong type (validate "
                                 "failed with exception "
                              << exception.what() << "). "
                              << "ROS encoding type used was "
                              << cvb_image->encoding);
      rclcpp::shutdown();
      return cv::Mat();
    }
  }

 private:
  SensorSystem::Ptr sensor_system_;
  SensorSystem::Ptr sensor_system_f_;
  std::shared_ptr<rclcpp::Node> node_;

  /// @}
  /// @name Node and subscriber related
  /// @{
  std::shared_ptr<image_transport::ImageTransport> img_transport_;
  std::vector<image_transport::Subscriber> image_subscribers_;

  rclcpp::CallbackGroup::SharedPtr imu_callback_group_;
  using ImuAdaptedType =
      rclcpp::adapt_type<dyno::ImuMeasurement>::as<sensor_msgs::msg::Imu>;
  rclcpp::Subscription<ImuAdaptedType>::SharedPtr imu_sub_;
  rclcpp::Subscription<ImuAdaptedType>::SharedPtr imu_sub_f_; // if there is a second rig
  std::mutex time_mutex_;  ///< Lock when accessing time

  using GroundTruthAdaptedType =
      rclcpp::adapt_type<dyno::GroundTruthInputPacket>::as<GroundtruthMsg>;
  rclcpp::Subscription<GroundTruthAdaptedType>::SharedPtr ground_truth_sub_;

  /// @}

  typedef std::function<cv::Mat(ImageMsgPtr)> ReadImageFunc;
  std::atomic<FrameId> driving_frame_id_{0};

  std::mutex images_received_mutex_;  ///< Lock when accessing buffer.
  std::vector<std::map<uint64_t, ImageMsgPtr>>
      images_received_;  ///< Images obtained&buffered (to sync).

  // Images types to function that loads and processes the image correctly
  // according to the expected stream type
  std::map<StreamConfig::Types, ReadImageFunc> read_image_functions_;
};

}  // namespace dyno
