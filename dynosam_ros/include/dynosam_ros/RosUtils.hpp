/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a
 copy
 *   of this software and associated documentation files (the "Software"), to
 deal
 *   in the Software without restriction, including without limitation the
 rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in
 all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE
 *   SOFTWARE.
 */

#pragma once

#include <type_traits>

#include "dynosam_common/Exceptions.hpp"
#include "dynosam_common/Types.hpp"
#include "rcl_interfaces/msg/parameter.hpp"
#include "rclcpp/node.hpp"
#include "rclcpp/parameter.hpp"
#include "rclcpp/time.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace dyno {

/**
 * @brief A generic type trait with the dyno namespace. Here we define compile
 * time conversions between c++ types and their corresponding
 * rcl_interfaces::msg::ParameterType. The following properies are defined:
 *
 * ::traits<double>::ros_parameter_type, where T=double is any primitive type
 * that has a parameter type versions, and,
 * ::traits<rcl_interfaces::msg::ParameterType::PARAMETER_DOUBLE>::cpp_type
 *
 *
 * @tparam T
 */
template <typename T>
struct traits;

/**
 * @brief Type trait to retrieve a c++ type from the corresponding
 * rcl_interfaces::msg::ParameterType.
 * ros_param_traits<rcl_interfaces::msg::ParameterType::PARAMETER_DOUBLE>::cpp_type
 * == decltype(double)
 *
 *
 * @tparam Type rcl_interfaces::msg::ParameterType
 */
template <uint8_t Type>
struct ros_param_traits;

namespace internal {

// Taken from https://en.cppreference.com/w/cpp/types/type_identity
// since ROS2 does not support C++20 yet
template <class T>
struct type_identity {
  using type = T;
};

template <class T>
using type_identity_t = typename type_identity<T>::type;

template <typename Msg, typename = int>
struct HasMsgHeader : std::false_type {};

template <typename Msg>
struct HasMsgHeader<Msg, decltype((void)Msg::header, 0)> : std::true_type {};

template <typename Msg, typename = int>
struct HasChildFrame : std::false_type {};

template <typename Msg>
struct HasChildFrame<Msg, decltype((void)Msg::child_frame_id, 0)>
    : std::true_type {};

template <uint8_t Type>
struct RosParameterType {
  // as of the time of writing rcl_interfaces::msg::ParameterType:: is simply a
  // uint8 type for the sake of type safetly we will just check this as I dont
  // see version changes happening that much
  static_assert(std::is_same<decltype(Type), uint8_t>::value,
                "rcl_interfaces::msg::ParameterType:: is expected to be a "
                "uint8_t - check version!");
  static constexpr uint8_t ros_parameter_type = Type;
};

/**
 * @brief Type trait to retrieve a
 *
 * @tparam CPP_TYPE
 */
template <typename CPP_TYPE>
struct CppParameterType {
  using cpp_type = typename internal::type_identity_t<CPP_TYPE>;
};

}  // namespace internal

/**
 * @brief Exception thrown when a parameter is missing a default parameter and
 * MUST be set by the user at runtime (ie. no default value allowed)
 *
 */
class InvalidDefaultParameter : public DynosamException {
 public:
  InvalidDefaultParameter(const std::string &missing_param,
                          const std::string &custom_message = std::string())
      : DynosamException(
            "Missing required param " + missing_param +
            " which must set at runtime by the user (e.g. via a config file or "
            "via launch configurations)" +
            (custom_message.empty() ? "" : std::string(": " + custom_message))),
        missing_param_(missing_param),
        custom_message_(custom_message) {}

  const std::string &missingParam() const { return missing_param_; }
  const std::string &customMessage() const { return custom_message_; }

 private:
  const std::string missing_param_;
  const std::string custom_message_;
};

class ParameterDetails {
 public:
  friend class ParameterConstructor;

  /**
   * @brief Name of the paramter
   *
   * @return const std::string&
   */
  const std::string &name() const;

  /**
   * @brief Name of the associated node
   *
   * @return std::string
   */
  std::string node_name() const;

  /**
   * @brief Get the rclcpp::Parameter from the node
   *
   * @return rclcpp::Parameter
   */
  rclcpp::Parameter get() const;

  /**
   * @brief Get the c++ value from the node.
   *
   * rclcpp::exceptions::InvalidParameterTypeException	if the type doesn't
   * match
   *
   * @tparam ValueTypeT c++ type
   * @return decltype(auto)
   */
  template <typename ValueTypeT>
  decltype(auto) get() const;

  //   /**
  //    * @brief Registeres a calback to be triggered when a paramter is
  //    updated.
  //    *
  //    * This uses the /paramter_events and ParameterCallbackHandle mechanism
  //    internally.
  //    *
  //    * @tparam ValueTypeT
  //    * @param callback
  //    */
  //   template <typename ValueTypeT>
  //   void registerParamCallback(const std::function<void(ValueTypeT)>&
  //   callback);

  /**
   * @brief Get the c++ value from the node or the default value if param is not
   * set in the node. If the default value is used, the param in the node will
   * NOT be updated.
   *
   * This default value will only be used if there is NO value in the node, i.e
   * setting a default value from the ParameterConstructor or using paramter
   * overwrites using NodeOptions or the launchfile will set value in the node
   * and the default_value argument will not be used.
   *
   * @tparam ValueTypeT
   * @param default_value
   * @return ValueTypeT
   */
  template <typename ValueTypeT>
  ValueTypeT get(ValueTypeT default_value) const;

  /**
   * @brief Specalisation for getting the value from the node as a std::string
   * or a default string.
   *
   * See template <typename ValueTypeT> ValueTypeT get(ValueTypeT) for details
   * on how the default value is used.
   *
   * @param default_value const char *
   * @return std::string
   */
  std::string get(const char *default_value) const;

  /**
   * @brief Specalisation for getting the value from the node as a std::string
   * or a default string.
   *
   * See template <typename ValueTypeT> ValueTypeT get(ValueTypeT) for details
   * on how the default value is used.
   *
   * @param default_value const std::string
   * @return std::string
   */
  std::string get(const std::string &default_value) const;

  inline bool isSet() const {
    return node_->get_parameter(this->name())
               .get_parameter_value()
               .get_type() != rclcpp::PARAMETER_NOT_SET;
  }

  /**
   * @brief Casting operator and returns the Paramter type from the node
   *
   * @return rclcpp::Parameter
   */
  operator rclcpp::Parameter() const { return this->get(); }

  /**
   * @brief Casting operator for the parameter value from the node
   *
   * @return rclcpp::ParameterValue
   */
  operator rclcpp::ParameterValue() const {
    return this->get().get_parameter_value();
  }

  operator std::string() const;

 private:
  ParameterDetails(rclcpp::Node *node, const rclcpp::Parameter &parameter,
                   const rcl_interfaces::msg::ParameterDescriptor &description);

  rclcpp::Parameter get_param(const rclcpp::Parameter &default_param) const;

  void declare();

  template <typename ValueTypeT>
  decltype(auto) get_param(const rclcpp::Parameter &default_param) const;

 private:
  mutable rclcpp::Node *node_;
  const rclcpp::Parameter
      default_parameter_;  //! the original parameter, prior to declaration with
                           //! the node, as provided by ParameterConstructor
  const rcl_interfaces::msg::ParameterDescriptor description_;

  //   mutable std::shared_ptr<rclcpp::ParameterEventHandler> param_subscriber_;
  //   //! monitors paramter changes mutable
  //   std::shared_ptr<rclcpp::ParameterCallbackHandle> cb_handle_; //! Callback
  //   handler - must remain in scope for the callbacks to remain

  //   mutable PropertyHandler property_handler_; //! Handler for param specific
  //   callbacks
};

/**
 * @brief Wrapper class to define paramter details. Using these details a
 * ParameterDetails object can be created which declares the paramter with its
 * associated node automatically and can be used to directly access the latest
 * value.
 *
 */
class ParameterConstructor {
 public:
  /**
   * @brief Constructor with a node and param name. Default value is set to type
   * rclcpp::PARAMETER_NOT_SET and the resulting ParameterDetails will throw an
   * 'InvalidDefaultParameter' exception upon construction unless the user
   * provides a default argument in the parameter overwrites.
   *
   * @param node
   * @param name
   */
  ParameterConstructor(rclcpp::Node::SharedPtr node, const std::string &name);
  ParameterConstructor(rclcpp::Node *node, const std::string &name);

  /**
   * @brief Construct with a node, param name and default type. The he value
   * type is determined by traits<ValueTypeT>::ros_parameter_type; and will be
   * used as the default value if a parameter overwrite is not provided
   *
   * @tparam ValueTypeT
   * @param node
   * @param name
   * @param value
   */
  template <typename ValueTypeT>
  ParameterConstructor(rclcpp::Node::SharedPtr node, const std::string &name,
                       ValueTypeT value);
  template <typename ValueTypeT>
  ParameterConstructor(rclcpp::Node *node, const std::string &name,
                       ValueTypeT value);

  /**
   * @brief Constructs a ParameterDetails object from the details in this.
   *
   * As per the ParameterDetails comments, upon constructio the parameter is
   * declared in the associated node (from this) and is kept in the
   * ParameterDetails as a shared pointer.
   *
   * @return ParameterDetails
   */
  ParameterDetails finish() const;

  /**
   * @brief Updates the paramters description and a returns a reference to this.
   *
   * Internally, updates parameter_descriptor_.
   *
   * @param description const std::string
   * @return ParameterConstructor&
   */
  ParameterConstructor &description(const std::string &description);

  /**
   * @brief Updates the read only flag and a returns a reference to this.
   *
   * Internally, updates parameter_descriptor_.
   *
   * @param read_only bool
   * @return ParameterConstructor&
   */
  ParameterConstructor &read_only(bool read_only);

  /**
   * @brief Updates the full internal paramter description and returns a
   * reference to this.
   *
   * @param parameter_description const rcl_interfaces::msg::ParameterDescriptor
   * &
   * @return ParameterConstructor&
   */
  ParameterConstructor &parameter_description(
      const rcl_interfaces::msg::ParameterDescriptor &parameter_description);

  /**
   * @brief Name of the paramter
   *
   * @return const std::string&
   */
  inline const std::string &name() const { return parameter_.get_name(); }
  /**
   * @brief Casting operator and the internal storage for the parameter value
   *
   * @return rclcpp::ParameterValue
   */
  operator rclcpp::ParameterValue() const {
    return parameter_.get_parameter_value();
  }

  operator rcl_interfaces::msg::ParameterDescriptor() const {
    return parameter_descriptor_;
  }

 private:
  rclcpp::Node *node_;
  rclcpp::Parameter parameter_;
  rcl_interfaces::msg::ParameterDescriptor parameter_descriptor_;
};

namespace utils {

/**
 * @brief Convert ROS time to timestamp
 *
 * @param time
 * @return Timestamp time in seconds
 */
Timestamp fromRosTime(const rclcpp::Time &time);

/**
 * @brief Converts Timestamp (in seconds) to ROS time
 *
 * @param timestamp
 * @return rclcpp::Time
 */
rclcpp::Time toRosTime(Timestamp timestamp);

/**
 * @brief Converts from input type to output type where output type is a ROS
 * message with a header. OUTPUT type MUST be a ROS message type
 *
 * Checks if the OUTPUT type has a variable called header and then sets the
 * timestamp and frame id Checks if the OUTPUT type has a variable called
 * child_frame_id and, if child_frame_id provided, sets the value
 *
 */
template <typename INPUT, typename OUTPUT,
          typename = std::enable_if_t<
              rosidl_generator_traits::is_message<OUTPUT>::value>>
bool convertWithHeader(
    const INPUT &input, OUTPUT &output, Timestamp timestamp,
    const std::string &frame_id,
    std::optional<std::string> child_frame_id = std::nullopt) {
  if (!dyno::convert<INPUT, OUTPUT>(input, output)) return false;

  // does actually have a header
  if constexpr (internal::HasMsgHeader<OUTPUT>::value) {
    output.header.stamp = toRosTime(timestamp);
    output.header.frame_id = frame_id;
  }

  // if has child frame id
  if constexpr (internal::HasChildFrame<OUTPUT>::value) {
    if (child_frame_id) {
      output.child_frame_id = child_frame_id.value();
    }
  }

  return true;
}

}  // namespace utils
}  // namespace dyno

#define DECLARE_ROS_PARAM_TYPE_CONVERSIONS(CPP_TYPE, PARAMETER_TYPE) \
  template <>                                                        \
  struct dyno::traits<CPP_TYPE>                                      \
      : dyno::internal::RosParameterType<PARAMETER_TYPE> {};         \
  template <>                                                        \
  struct dyno::ros_param_traits<PARAMETER_TYPE>                      \
      : dyno::internal::CppParameterType<CPP_TYPE> {};

DECLARE_ROS_PARAM_TYPE_CONVERSIONS(
    bool, rcl_interfaces::msg::ParameterType::PARAMETER_BOOL)
DECLARE_ROS_PARAM_TYPE_CONVERSIONS(
    int, rcl_interfaces::msg::ParameterType::PARAMETER_INTEGER)
DECLARE_ROS_PARAM_TYPE_CONVERSIONS(
    double, rcl_interfaces::msg::ParameterType::PARAMETER_DOUBLE)
DECLARE_ROS_PARAM_TYPE_CONVERSIONS(
    const char *, rcl_interfaces::msg::ParameterType::PARAMETER_STRING)
DECLARE_ROS_PARAM_TYPE_CONVERSIONS(
    std::vector<bool>, rcl_interfaces::msg::ParameterType::PARAMETER_BOOL_ARRAY)
DECLARE_ROS_PARAM_TYPE_CONVERSIONS(
    std::vector<int>,
    rcl_interfaces::msg::ParameterType::PARAMETER_INTEGER_ARRAY)
DECLARE_ROS_PARAM_TYPE_CONVERSIONS(
    std::vector<double>,
    rcl_interfaces::msg::ParameterType::PARAMETER_DOUBLE_ARRAY)
DECLARE_ROS_PARAM_TYPE_CONVERSIONS(
    std::vector<const char *>,
    rcl_interfaces::msg::ParameterType::PARAMETER_STRING_ARRAY)

// specalise for the non-specific rclcpp::ParameterValue type (which is a c++
// object type) as rcl_interfaces::msg::ParameterType::PARAMETER_NOT_SET when a
// Paramter is not set we can only retrieve its ParameterValue and not cast it
// to a type since it does not know which type it is!
template <>
struct dyno::traits<rclcpp::ParameterValue>
    : dyno::internal::RosParameterType<
          rcl_interfaces::msg::ParameterType::PARAMETER_NOT_SET> {};

template <>
struct dyno::traits<std::string>
    : dyno::internal::RosParameterType<
          rcl_interfaces::msg::ParameterType::PARAMETER_STRING> {};

#include "dynosam_ros/RosUtils-inl.hpp"
