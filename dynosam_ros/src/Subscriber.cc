#include "dynosam_ros/Subscriber.hpp"

namespace dyno {

Subscriber::Subscriber(SensorSystem::Ptr sensor_system,
                       std::shared_ptr<rclcpp::Node> node)
    : DataProviderRos(node), sensor_system_(sensor_system) {
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

  // TODO: make param
  static constexpr size_t queue_size = 1000;
  auto image_qos = rclcpp::SensorDataQoS().keep_last(queue_size).reliable();

  // set up callbacks
  for (size_t i = 0; i < sensor_system->numCameraStreams(); ++i) {
    const std::string stream_name = sensor_system->streamName(i);
    image_subscribers_[i] = img_transport_->subscribe(
        "/dynosam/" + stream_name + "/image_raw", queue_size,
        // 30 * sensor_system->numCameraStreams(),
        std::bind(&Subscriber::imageCallback, this, std::placeholders::_1, i));
  }
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

CameraParams::Optional Subscriber::getCameraParams() const {
  return sensor_system_->getCanonicalParams();
}

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

void Subscriber::addImages(Timestamp timestamp,
                           const std::map<size_t, ImageMsgPtr>& image_msgs) {
  std::map<size_t, cv::Mat> images;
  for (const auto& [i, msg] : image_msgs) {
    // read and process image based on config type
    cv::Mat image = read_image_functions_[sensor_system_->streamType(i)](msg);
    images[i] = image;
  }

  CHECK_GE(images.size(), 2);
}

}  // namespace dyno
