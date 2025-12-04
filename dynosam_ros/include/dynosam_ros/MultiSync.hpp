#pragma once

#include <type_traits>

#include "dynosam/dataprovider/DataProvider.hpp"  // for ImageContainerCallback
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/Tuple.hpp"
#include "message_filters/subscriber.hpp"
#include "message_filters/sync_policies/exact_time.hpp"
#include "message_filters/synchronizer.hpp"
#include "rclcpp/node.hpp"
#include "rclcpp/node_interfaces/node_interfaces.hpp"
#include "rclcpp/node_interfaces/node_topics.hpp"
#include "rclcpp/node_options.hpp"
#include "sensor_msgs/msg/image.hpp"

// namespace  {

// using namespace message_filters;
// in the message_filters interface changes between Jazzy and Kilted
template <typename, typename = void>
struct message_filters_has_required_interfaces : std::false_type {};

template <typename T>
struct message_filters_has_required_interfaces<
    T, std::void_t<typename T::NodeParametersInterface>> : std::true_type {};

constexpr static bool message_filters_uses_node_interface =
    message_filters_has_required_interfaces<
        message_filters::SubscriberBase<>>::value;

#if message_filters_uses_node_interface
// for Kilted and above
#define MESSAGE_FILTERS_USES_NODE_INTERFACE 1
#pragma message("ROS2 message filters version >=Kilted detected")
#else
// for at least Jazzy
#define MESSAGE_FILTERS_USES_NODE_INTERFACE 0
#pragma message("ROS2 message filters version <=Jazzy detected")
#endif

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

template <typename Msg, std::size_t N>
struct ExactTimePolicyHelperImpl {
// in Jazzy ExactTime filter requires at least two message types to be defined
// (ie. does not support N==1)
#if !MESSAGE_FILTERS_USES_NODE_INTERFACE
  static_assert(N > 1,
                "Message filters (in at least Jazzy) does not support N==1. "
                "MultiSync with N>=2 must be used!");
#endif

  template <std::size_t... Is>
  static auto get_policy(std::index_sequence<Is...>)
      -> message_filters::sync_policies::ExactTime<std::remove_reference_t<
          decltype((void(Is), std::declval<Msg>()))>...>;

  using Type = decltype(get_policy(std::make_index_sequence<N>{}));
};

template <typename Msg, std::size_t N>
using exact_time_policy_helper =
    typename ExactTimePolicyHelperImpl<Msg, N>::Type;

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

struct MultiSyncConfig {
  uint32_t queue_size = 20u;
  //! Initalised with SensorDataQoS
  rclcpp::QoS subscriber_qos = rclcpp::QoS(
      rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data));
  rclcpp::SubscriptionOptions subscriber_options{};

  MultiSyncConfig() = default;
  MultiSyncConfig(uint32_t _queue_size) : queue_size(_queue_size) {}
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
  struct Config {
    uint32_t queue_size = 20u;
    //! Initalised with SensorDataQoS
    rclcpp::QoS subscriber_qos = rclcpp::QoS(
        rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data));
    rclcpp::SubscriptionOptions subscriber_options{};

    Config() = default;
    Config(uint32_t _queue_size) : queue_size(_queue_size) {}
  };

#if MESSAGE_FILTERS_USES_NODE_INTERFACE
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
#endif

  using MessageType = Msg;

  using SyncPolicy = exact_time_policy_helper<MessageType, N>;
  using SyncType = message_filters::Synchronizer<SyncPolicy>;

  ///! Callback in the form sensor::msg::Image.... repeatead N times
  using Callback = typename callback_type_helper<MessageType, N>::Callback;

  // tuple of pointers must be explicitly initalised
  MultiSync(rclcpp::Node& node, const std::array<std::string, N>& topics,
            const MultiSyncConfig& config = MultiSyncConfig())
      : node_(node), topics_(topics), config_(config) {}

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
#if MESSAGE_FILTERS_USES_NODE_INTERFACE
    RequiredInterfaces interface(node_);
    auto make_subscriber =
        [&](const std::string& topic) -> std::shared_ptr<Subscriber> {
      return std::make_shared<Subscriber>(
          interface, topic, config_.subscriber_qos, config_.subscriber_options);
    };
#else
    rclcpp::Node* interface = &node_;
    CHECK_NOTNULL(interface);
    auto make_subscriber =
        [&](const std::string& topic) -> std::shared_ptr<Subscriber> {
      auto subscriber = std::make_shared<Subscriber>();
      subscriber->subscribe(interface, topic,
                            config_.subscriber_qos.get_rmw_qos_profile(),
                            config_.subscriber_options);
      return subscriber;
    };
#endif

    auto node_topics = node_.get_node_topics_interface();
    CHECK_NOTNULL(node_topics);

    for (size_t i = 0; i < N; i++) {
      internal::select_apply<N>(i, [&](auto I) {
        // for some reason using the message_filter::Subscriber does not adhere
        // to remapping manually resolve the topic node
        // false is for only_expand
        const std::string resolved_topic =
            node_topics->resolve_topic_name(topics_.at(i), false);
        std::get<I>(subs_) = make_subscriber(resolved_topic);
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
      sync_ = std::make_unique<SyncType>(SyncPolicy(config_.queue_size),
                                         *std::get<Is>(subs_)...);

      // Use lambda to forward messages to callDerived
#if MESSAGE_FILTERS_USES_NODE_INTERFACE
      sync_->registerCallback(callback_);
#else
      // 1. Lambda accepts all 9 arguments (as auto).
      // 2. The arguments are passed to the slice_and_call helper, along with
      //    the compile-time index sequence (Is...), which dictates how many to
      //    keep (N).
      sync_->registerCallback([this](const auto&... args) {
        // Is... is visible here as a compile-time constant parameter pack
        this->slice_and_call(std::index_sequence<Is...>(), args...);
      });
#endif
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

 private:
#if !MESSAGE_FILTERS_USES_NODE_INTERFACE
  /**
   * @brief Helper function to slice the 9-argument signal down to N for the
   * callback_.
   * @tparam Is Indices 0 to N-1 (provided by the outer createSyncImpl)
   * @tparam Args The full 9 arguments from the Signal9
   */
  template <size_t... Is, typename... Args>
  void slice_and_call(std::index_sequence<Is...>, const Args&... args) {
    // Create a tuple containing the 9 incoming arguments (by const reference)
    auto args_tuple = std::forward_as_tuple(args...);
    // Call the user callback, expanding the tuple only for indices 0 to N-1
    this->callback_(std::get<Is>(args_tuple)...);
  }
#endif

 protected:
#if MESSAGE_FILTERS_USES_NODE_INTERFACE
  using Subscriber =
      message_filters::Subscriber<MessageType, RequiredInterfaces>;
#else
  using Subscriber = message_filters::Subscriber<MessageType>;
#endif
  /// @brief a tuple of Subscribers of size N
  using SubscriberTuple =
      typename internal::repeat_type<std::shared_ptr<Subscriber>, N>::type;

  rclcpp::Node& node_;
  std::array<std::string, N> topics_;
  MultiSyncConfig config_;
  SubscriberTuple subs_{};

  std::shared_ptr<SyncType> sync_;
  Callback callback_;
};

///@brief A MultiSync templated on sensor_msgs::msg::Image
template <size_t N>
using MultiImageSync = MultiSync<sensor_msgs::msg::Image, N>;

/// @brief Some common MultiImageSync typedefs
// typedef MultiImageSync<1> MultiImageSync1;
typedef MultiImageSync<2> MultiImageSync2;
typedef MultiImageSync<3> MultiImageSync3;
typedef MultiImageSync<4> MultiImageSync4;

}  // namespace dyno
