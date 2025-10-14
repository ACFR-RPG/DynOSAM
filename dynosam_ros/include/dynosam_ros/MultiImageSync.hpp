#pragma once

#include "dynosam/dataprovider/DataProvider.hpp"  // for ImageContainerCallback
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/Tuple.hpp"
#include "message_filters/subscriber.hpp"
#include "message_filters/sync_policies/exact_time.hpp"
#include "message_filters/synchronizer.hpp"
#include "rclcpp/node.hpp"
#include "rclcpp/node_interfaces/node_interfaces.hpp"
#include "rclcpp/node_options.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace dyno {

namespace {

template <typename Msg, std::size_t... Is>
auto exact_time_policy_helper_impl(std::index_sequence<Is...>)
    -> message_filters::sync_policies::ExactTime<
        std::remove_reference_t<decltype((void(Is), std::declval<Msg>()))>...>;
template <typename Msg, std::size_t N>
using exact_time_policy_helper =
    decltype(exact_time_policy_helper_impl<Msg>(std::make_index_sequence<N>{}));

// //dereeference tuple of pointers and return result
// template <typename... Ptrs>
// auto tuple_of_pointee_refs(const std::tuple<Ptrs...>& t)
// {
//     return std::apply([](auto const&... ptrs) {
//         return std::forward_as_tuple((*ptrs)...);
//     }, t);
// }

}  // namespace

class MultiImageSyncBase {
 public:
  DYNO_POINTER_TYPEDEFS(MultiImageSyncBase)

  MultiImageSyncBase() = default;
  virtual ~MultiImageSyncBase() = default;

  virtual void connect() = 0;
  virtual void shutdown() = 0;
};

template <typename Derived, size_t N>
class MultiImageSync : public MultiImageSyncBase {
 public:
  // In ROS Kilted curent version the message filter subscriber base requires a
  // node interface to the patramters and topics not the node itself. See:
  // https://docs.ros.org/en/humble/Tutorials/Intermediate/Using-Node-Interfaces-Template-Class.html
  // and
  // https://github.com/ros2/message_filters/blob/kilted/include/message_filters/subscriber.hpp
  using NodeParametersInterface =
      rclcpp::node_interfaces::NodeParametersInterface;
  using NodeTopicsInterface = rclcpp::node_interfaces::NodeTopicsInterface;
  using RequiredInterfaces =
      rclcpp::node_interfaces::NodeInterfaces<NodeParametersInterface,
                                              NodeTopicsInterface>;

  template <typename Msg>
  using ExactTimePolicyN = exact_time_policy_helper<Msg, N>;

  using SyncPolicy = ExactTimePolicyN<sensor_msgs::msg::Image>;
  using SyncType = message_filters::Synchronizer<SyncPolicy>;

  // tuple of pointers must be explicitly initalised
  MultiImageSync(rclcpp::Node& node, const std::array<std::string, N>& topics,
                 uint32_t queue_size)
      : node_(node),
        topics_(topics),
        queue_size_(queue_size),
        subscriber_qos_(rclcpp::QoS(
            rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data))) {
  }

  void connect() override {
    if (sync_) sync_.reset();

    subscribe();
    createSync();
  }

  void shutdown() override {
    if (sync_) sync_.reset();
    unsubscribe();
  }

 private:
  void subscribe() {
    std::stringstream ss;
    ss << type_name<Derived>()
       << " has been connected. Subscribing to image topics: ";
    RequiredInterfaces interface(node_);
    for (size_t i = 0; i < N; i++) {
      internal::select_apply<N>(i, [&](auto I) {
        std::get<I>(subs_) = std::make_shared<Subscriber>(
            interface, topics_.at(i), subscriber_qos_, subscriber_options_);
        ss << std::get<I>(subs_)->getSubscriber()->get_topic_name() << " ";
      });
    }
    ss << "\n";

    RCLCPP_INFO_STREAM(node_.get_logger(), ss.str());
  }

  void unsubscribe() {
    for (size_t i = 0; i < N; i++) {
      internal::select_apply<N>(i, [&](auto I) { std::get<I>(subs_).reset(); });
    }
  }

  template <size_t... Is>
  void createSyncImpl(std::index_sequence<Is...>) {
    // Create synchronizer with N subscribers
    sync_ = std::make_unique<SyncType>(SyncPolicy(queue_size_),
                                       *std::get<Is>(subs_)...);

    // Use lambda to forward messages to callDerived
    sync_->registerCallback([this](auto&&... args) {
      callDerived(std::forward<decltype(args)>(args)...);
    });
  }

  void createSync() {
    createSyncImpl(std::make_index_sequence<N>{});  // expand 0..N-1
  }

  template <typename... Args>
  void callDerived(Args&&... args) {
    static_assert(sizeof...(Args) == N, "Number of arguments must match N");
    // Forward N arguments to derived class “virtual-like” function
    static_cast<Derived*>(this)->imageSyncCallback(std::forward<Args>(args)...);
  }

 protected:
  using Subscriber =
      message_filters::Subscriber<sensor_msgs::msg::Image, RequiredInterfaces>;
  /// @brief a tuple of Subscribers of size N
  using SubscriberTuple =
      typename internal::repeat_type<std::shared_ptr<Subscriber>, N>::type;

  rclcpp::Node& node_;
  std::array<std::string, N> topics_;
  uint32_t queue_size_;
  SubscriberTuple subs_{};
  //! Initalised with SensorDataQoS
  rclcpp::QoS subscriber_qos_;
  rclcpp::SubscriptionOptions subscriber_options_;

  std::shared_ptr<SyncType> sync_;
};

using ImageContainerCallback = DataProvider::ImageContainerCallback;

struct ImageContainerCallbackWrapper {
  ImageContainerCallback image_callback;
  ImageContainerCallbackWrapper(const ImageContainerCallback& cb)
      : image_callback(CHECK_NOTNULL(cb)) {}
};

class MultiImageSync2 : public MultiImageSync<MultiImageSync2, 2>,
                        public ImageContainerCallbackWrapper {
 public:
  using SyncBase = MultiImageSync<MultiImageSync2, 2>;

  MultiImageSync2(rclcpp::Node& node, const std::array<std::string, 2>& topics,
                  uint32_t queue_size, const ImageContainerCallback& cb);

  void imageSyncCallback(
      const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg,
      const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg);
};

class MultiImageSync3 : public MultiImageSync<MultiImageSync3, 3>,
                        public ImageContainerCallbackWrapper {
 public:
  using SyncBase = MultiImageSync<MultiImageSync3, 3>;

  MultiImageSync3(rclcpp::Node& node, const std::array<std::string, 3>& topics,
                  uint32_t queue_size, const ImageContainerCallback& cb);

  void imageSyncCallback(
      const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg,
      const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg,
      const sensor_msgs::msg::Image::ConstSharedPtr& flow_msg);
};

class MultiImageSync4 : public MultiImageSync<MultiImageSync4, 4>,
                        public ImageContainerCallbackWrapper {
 public:
  using SyncBase = MultiImageSync<MultiImageSync4, 4>;

  MultiImageSync4(rclcpp::Node& node, const std::array<std::string, 4>& topics,
                  uint32_t queue_size, const ImageContainerCallback& cb);

  void imageSyncCallback(
      const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg,
      const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg,
      const sensor_msgs::msg::Image::ConstSharedPtr& flow_msg,
      const sensor_msgs::msg::Image::ConstSharedPtr& mask_msg);
};

}  // namespace dyno
