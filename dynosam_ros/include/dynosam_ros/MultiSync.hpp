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

/**
 * @brief Helper to define a function definition based on many messages type
 *
 * Msg will be the raw msg type (ie sensor_msgs::msg::Image) but this will NOT
 * be the same as the function signature which will take const
 * Msg::ConstSharedPtr&...
 *
 * @tparam Msg
 */
template <typename... Msg>
struct CallbackTypeHelpers {
  using Callback = std::function<void(const std::shared_ptr<const Msg>&...)>;
};

namespace {

template <typename Msg, std::size_t... Is>
auto exact_time_policy_helper_impl(std::index_sequence<Is...>)
    -> message_filters::sync_policies::ExactTime<
        std::remove_reference_t<decltype((void(Is), std::declval<Msg>()))>...>;
template <typename Msg, std::size_t N>
using exact_time_policy_helper =
    decltype(exact_time_policy_helper_impl<Msg>(std::make_index_sequence<N>{}));

template <typename Msg, std::size_t... Is>
auto callback_type_helper_impl(std::index_sequence<Is...>)
    -> CallbackTypeHelpers<
        std::remove_reference_t<decltype((void(Is), std::declval<Msg>()))>...>;
template <typename Msg, std::size_t N>
using callback_type_helper =
    decltype(callback_type_helper_impl<Msg>(std::make_index_sequence<N>{}));

}  // namespace

class MultiSyncBase {
 public:
  DYNO_POINTER_TYPEDEFS(MultiSyncBase)

  MultiSyncBase() = default;
  virtual ~MultiSyncBase() = default;

  virtual bool connect() = 0;
  virtual void shutdown() = 0;
};

/**
 * @brief Wrapper for a message_filters::Synchronizer that encapsualtes
 * subscribing to N topics of type Msg.
 *
 * Some limitations:
 *  - only supports ExactTime sync policy
 *  - only supports one message type for all N subscribers
 *
 * @tparam Msg
 * @tparam N
 */
template <typename Msg, size_t N>
class MultiSync : public MultiSyncBase {
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

  using MessageType = Msg;

  using SyncPolicy = exact_time_policy_helper<MessageType, N>;
  using SyncType = message_filters::Synchronizer<SyncPolicy>;

  ///! Callback in the form sensor::msg::Image.... repeatead N times
  using Callback = typename callback_type_helper<MessageType, N>::Callback;

  // tuple of pointers must be explicitly initalised
  MultiSync(rclcpp::Node& node, const std::array<std::string, N>& topics,
            uint32_t queue_size)
      : node_(node),
        topics_(topics),
        queue_size_(queue_size),
        subscriber_qos_(rclcpp::QoS(
            rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data))) {
  }

  bool connect() override {
    if (sync_) sync_.reset();

    subscribe();
    return createSync();
  }

  void shutdown() override {
    if (sync_) sync_.reset();
    unsubscribe();
  }

  void registerCallback(const Callback& cb) { callback_ = cb; }

 private:
  void subscribe() {
    static const auto msg_name = type_name<Msg>();
    std::stringstream ss;
    ss << "MultiSync of type " << msg_name << " and size " << N
       << " is subscribing to topics: ";
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

  bool createSync() {
    return createSyncImpl(std::make_index_sequence<N>{});  // expand 0..N-1
  }

  template <size_t... Is>
  bool createSyncImpl(std::index_sequence<Is...>) {
    if (callback_) {
      // Create synchronizer with N subscribers
      sync_ = std::make_unique<SyncType>(SyncPolicy(queue_size_),
                                         *std::get<Is>(subs_)...);

      // Use lambda to forward messages to callDerived
      sync_->registerCallback(callback_);
      RCLCPP_INFO_STREAM(node_.get_logger(),
                         "MultiSync connected and subscribed");
      return true;
    } else {
      RCLCPP_ERROR_STREAM(
          node_.get_logger(),
          "MultiSync failed to connect as callback was not"
          "registered (did you forget to call registerCallback?). "
          "Unsubscribing!");
      shutdown();
      return false;
    }
  }

 protected:
  using Subscriber =
      message_filters::Subscriber<MessageType, RequiredInterfaces>;
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
  Callback callback_;
};

///@brief A MultiSync templated on sensor_msgs::msg::Image
template <size_t N>
using MultiImageSync = MultiSync<sensor_msgs::msg::Image, N>;

/// @brief Some common MultiImageSync typedefs
typedef MultiImageSync<1> MultiImageSync1;
typedef MultiImageSync<2> MultiImageSync2;
typedef MultiImageSync<3> MultiImageSync3;
typedef MultiImageSync<4> MultiImageSync4;

}  // namespace dyno
