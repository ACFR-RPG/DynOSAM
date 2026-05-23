#include "dynosam_ros/Subscriber.hpp"

namespace dyno {

Subscriber::Subscriber(SensorSystem::Ptr sensor_system,
                       std::shared_ptr<rclcpp::Node> node)
    : DataProviderRos(node), sensor_system_(sensor_system) {
  if (!sensor_system->isInitalised()) {
    throw DynosamException(
        "dyno::Subscriber provided a SensorSystem that is not initalised!");
  }

  image_subscribers_.resize(sensor_system->numCameraStreams());
  images_received_.resize(sensor_system->numCameraStreams());

  // set up image reception
  img_transport_.reset(new image_transport::ImageTransport(node));

  read_image_functions_ = {
      {StreamConfig::Types::RGBMono,
       [&](ImageMsgPtr msg) -> cv::Mat { return this->readRgbRosImage(msg); }},
      {StreamConfig::Types::Depth,
       [&](ImageMsgPtr msg) -> cv::Mat {
         return this->readDepthRosImage(msg);
       }},
      {StreamConfig::Types::OpticalFlow,
       [&](ImageMsgPtr msg) -> cv::Mat { return this->readFlowRosImage(msg); }},
      {StreamConfig::Types::Mask, [&](ImageMsgPtr msg) -> cv::Mat {
         return this->readMaskRosImage(msg);
       }}};

  int queue_size = ParameterConstructor(node_.get(), "image_queue_size", 1000)
                       .description("Queue size for the image subscriber(s)")
                       .finish()
                       .get<int>();

  auto image_qos = rclcpp::SensorDataQoS()
                       .keep_last(static_cast<size_t>(queue_size))
                       .reliable();

  // Need to explicitly pass VoidPtr, transport options (nullptr) and subscriber
  // options to subscribe to avoid ambiguous overloading specifically in the
  // case when we specify a rmw_qos_profile (which we want to), rahter than just
  // a queue size likely this is only a problem with ROS kilted
  rclcpp::SubscriptionOptions subscriber_options;
  // set up callbacks
  for (size_t i = 0; i < sensor_system->numCameraStreams(); ++i) {
    const std::string stream_name = sensor_system->streamName(i);
    image_subscribers_[i] = img_transport_->subscribe(
        "/dynosam/" + stream_name + "/image_raw",
        image_qos.get_rmw_qos_profile(),
        // 30 * sensor_system->numCameraStreams(),
        std::bind(&Subscriber::imageCallback, this, std::placeholders::_1, i),
        image_transport::ImageTransport::VoidPtr(), nullptr,
        subscriber_options);
  }

  //  imu_callback_group_ =
  //     node_->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

  // rclcpp::SubscriptionOptions imu_sub_options;
  // imu_sub_options.callback_group = imu_callback_group_;

  // imu_sub_ = node_->create_subscription<ImuAdaptedType>(
  //     "imu", rclcpp::SensorDataQoS(),
  //     [&](const dyno::ImuMeasurement& imu) -> void {
  //       if (!imu_single_input_callback_) {
  //         RCLCPP_ERROR_THROTTLE(
  //             node_->get_logger(), *node_->get_clock(), 1000,
  //             "Imu callback triggered but "
  //             "imu_single_input_callback_ is not registered!");
  //         return;
  //       }
  //       imu_single_input_callback_(imu);
  //     },
  //     imu_sub_options);
}

bool Subscriber::spin() { return !shutdown_; }
void Subscriber::shutdown() {
  shutdown_ = true;
  // stop callbacks
  for (size_t i = 0; i < sensor_system_->numCameraStreams(); ++i) {
    image_subscribers_[i].shutdown();
  }
  //   subImu_.reset();
}

SensorRigBase::Ptr Subscriber::sensorRig() const { return sensor_system_; }

void Subscriber::imageCallback(const ImageMsgPtr& msg,
                               unsigned int stream_index) {
  static constexpr Timestamp kDynoThresholdSync = 0.01;

  // //TODO: we will decripcate a lot of the readRosImage stuff likely in the
  // base function... const cv::Mat image = this->readRosImage(msg)->image;

  const Timestamp timestamp = utils::fromRosTime(msg->header.stamp);
  images_received_.at(stream_index)[toNSec(timestamp)] = msg;

  // try sync
  std::lock_guard<std::mutex> lock(time_mutex_);
  std::set<uint64_t> all_times;
  const int num_streams = images_received_.size();
  for (int i = 0; i < num_streams; ++i) {
    for (const auto& entry : images_received_.at(i)) {
      all_times.insert(entry.first);
    }
  }
  for (const auto& time : all_times) {
    // note: ordered old to new
    std::vector<uint64_t> syncedTimes(num_streams, 0);
    std::map<size_t, ImageMsgPtr> images;
    Timestamp tcheck = fromNSec(time);

    bool synced = true;
    for (int i = 0; i < num_streams; ++i) {
      bool syncedi = false;
      for (const auto& entry : images_received_.at(i)) {
        Timestamp ti = fromNSec(entry.first);
        if (fabs((tcheck - ti)) < kDynoThresholdSync) {
          syncedTimes.at(i) = entry.first;
          images[i] = images_received_.at(i).at(entry.first);
          syncedi = true;
          break;
        }
      }
      if (!syncedi) {
        synced = false;
        break;
      }
    }
    if (synced) {
      // probably bad to do image processing in here...?
      //  add
      std::cout << "add images" << time << " " << images.size() << std::endl;
      addImages(tcheck, images);

      // if (!viInterface_->addImages(tcheck, images)) {
      //     LOG(WARNING) << "Frame not added at t="<< tcheck;
      // }
      // remove all the older stuff from buffer
      for (int i = 0; i < num_streams; ++i) {
        const int size0 = images_received_.at(i).size();
        auto end = images_received_.at(i).find(syncedTimes.at(i));
        if (end != images_received_.at(i).end()) {
          ++end;
        }
        images_received_.at(i).erase(images_received_.at(i).begin(), end);
        const int size1 = images_received_.at(i).size();
        if (size0 - size1 > 1) {
          LOG(WARNING) << "dropped " << size0 - size1 - 1
                       << " unsyncable frame(s) of camera " << i
                       << " before t=" << tcheck;
        }
      }
    }
  }
}

void Subscriber::imuCallback(const sensor_msgs::msg::Imu& msg) {}

bool Subscriber::addImages(Timestamp timestamp,
                           const std::map<size_t, ImageMsgPtr>& image_msgs) {
  if (!image_container_callback_) {
    return false;
  }

  std::map<size_t, cv::Mat> images;
  for (const auto& [i, msg] : image_msgs) {
    // read and process image based on config type
    cv::Mat image = read_image_functions_[sensor_system_->streamType(i)](msg);
    images[i] = image;
  }

  CHECK_GE(images.size(), 2);
  const cv::Mat depth_rig0 = images.at(0);
  const cv::Mat depth_rig1 = images.at(1);

  cv::Mat depth_rig0_processed, depth_rig1_processed;
  sensor_system_->calibrateDetphRig(depth_rig0, depth_rig1,
                                    depth_rig0_processed, depth_rig1_processed);

  auto image_container =
      std::make_shared<ImageContainer>(driving_frame_id_, timestamp);
  driving_frame_id_++;

  // for now only support rgb+depth/stereo
  // TODO: it would be MUCH better to map the image stream names to the names
  // expected by the image container...
  image_container->rgb(depth_rig0_processed);
  if (sensor_system_->depthRigType() == DepthRigType::RGBD) {
    image_container->depth(depth_rig1_processed);
  } else if (sensor_system_->depthRigType() == DepthRigType::Stereo) {
    image_container->rightRgb(depth_rig1_processed);
  } else {
    throw DynosamException("Unknown DepthRigType!");
  }

  LOG(INFO) << image_container->toString();
  // image_container_callback_(image_container);
  return true;
}

}  // namespace dyno
