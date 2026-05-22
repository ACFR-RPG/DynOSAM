#pragma once

#include <mutex>

#include "dynosam_ros/CameraSystem.hpp"
#include "dynosam_ros/DataProviderRos.hpp"
#include "dynosam_ros/adaptors/ImuMeasurementAdaptor.hpp"
#include "image_transport/image_transport.hpp"
#include "rclcpp/node.hpp"
#include "rclcpp/node_options.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/imu.hpp"

namespace dyno {

typedef sensor_msgs::msg::Image::ConstSharedPtr ImageMsgPtr;

/**
 * @brief
 * Heavily inspiried by the OKVIS2 implementation:
 * https://github.com/ethz-mrl/okvis2/blob/main/okvis_ros2/include/okvis/ros2/Subscriber.hpp
 */
class Subscriber : public DataProviderRos {
 public:
  DYNO_POINTER_TYPEDEFS(Subscriber)

  Subscriber(SensorSystem::Ptr sensor_system,
             std::shared_ptr<rclcpp::Node> node);
  ~Subscriber() = default;

  /** No end to the dataset */
  int datasetSize() const override { return -1; }
  /* True while not shutdown */
  bool spin() override;
  /* Disconnects all subscriber */
  void shutdown() override;

  // TODO: change to not being optional (the optional is only for realdata)
  //  as all datasets shoudl load the camera params
  //  now we handle the optional in the CameraRig!
  /* Returns canonical params */
  CameraParams::Optional getCameraParams() const override;

  void imageCallback(const ImageMsgPtr& msg, unsigned int stream_index);

  /// @brief The IMU callback.
  void imuCallback(const sensor_msgs::msg::Imu& msg);

 private:
  void addImages(Timestamp timestamp,
                 const std::map<size_t, ImageMsgPtr>& image_msgs);

 private:
  SensorSystem::Ptr sensor_system_;

  /// @}
  /// @name Node and subscriber related
  /// @{
  std::shared_ptr<image_transport::ImageTransport> img_transport_;
  std::vector<image_transport::Subscriber> image_subscribers_;

  rclcpp::CallbackGroup::SharedPtr imu_callback_group_;
  using ImuAdaptedType =
      rclcpp::adapt_type<dyno::ImuMeasurement>::as<sensor_msgs::msg::Imu>;
  rclcpp::Subscription<ImuAdaptedType>::SharedPtr imu_sub_;
  std::mutex time_mutex_;  ///< Lock when accessing time

  /// @}

  typedef std::function<cv::Mat(ImageMsgPtr)> ReadImageFunc;

  std::mutex images_received_mutex_;  ///< Lock when accessing buffer.
  std::vector<std::map<uint64_t, ImageMsgPtr>>
      images_received_;  ///< Images obtained&buffered (to sync).

  // Images types to function that loads and processes the image correctly
  // according to the expected stream type
  std::map<StreamConfig::Types, ReadImageFunc> read_image_functions_;
};

}  // namespace dyno
