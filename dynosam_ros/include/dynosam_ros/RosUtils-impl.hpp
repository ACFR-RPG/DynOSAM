/*
 *   Copyright (c) 2025 ACFR-RPG, University of Sydney, Jesse Morris
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

#include <glog/logging.h>

#include "dynosam_ros/RosUtils.hpp"
#include "dynosam_ros/adaptors/CameraParamsAdaptor.hpp"
#include "rclcpp/wait_for_message.hpp"

namespace dyno {

template <typename Msg, class Rep, class Period>
inline void waitAndGetMessage(
    Msg& msg, std::shared_ptr<rclcpp::Node> node, const std::string& topic,
    const std::chrono::duration<Rep, Period>& time_to_wait) {
  RCLCPP_INFO_STREAM(node->get_logger(), "Waiting for message "
                                             << type_name<Msg>()
                                             << "on topic: " << topic);

  // Only create status thread if we may wait a while
  const bool enable_wait_logging =
      time_to_wait.count() < 0 ||
      std::chrono::duration_cast<std::chrono::seconds>(time_to_wait) >=
          std::chrono::seconds(1);

  std::atomic_bool done = false;
  std::thread wait_thread;

  if (enable_wait_logging) {
    wait_thread = std::thread([node, topic, &done]() {
      rclcpp::Clock::SharedPtr clock = node->get_clock();

      const auto start_time = clock->now();
      rclcpp::Rate rate(1, clock);

      while (rclcpp::ok() && !done.load()) {
        const auto elapsed = (clock->now() - start_time).seconds();

        RCLCPP_INFO_STREAM(node->get_logger(),
                           "Still waiting for msg on topic '"
                               << topic << "' after " << std::fixed
                               << std::setprecision(1) << elapsed
                               << " seconds");

        rate.sleep();
      }
    });
  }

  if (!rclcpp::wait_for_message<Msg, Rep, Period>(msg, node, topic,
                                                  time_to_wait)) {
    done = true;

    if (wait_thread.joinable()) {
      wait_thread.join();
    }

    const auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(time_to_wait);
    DYNO_THROW_MSG(DynosamException)
        << "Failed to receive ROS msg " << type_name<Msg>() << " on topic "
        << topic << " (waited with timeout "
        << std::to_string(milliseconds.count()) << " ms).";
    throw;
  }

  done = true;

  if (wait_thread.joinable()) {
    wait_thread.join();
  }

  RCLCPP_INFO_STREAM(node->get_logger(), "Received msg on topic: " << topic);
}

template <typename Adaptor, class Rep, class Period>
inline typename Adaptor::custom_type waitAndGetMessageViaAdaptor(
    std::shared_ptr<rclcpp::Node> node, const std::string& topic,
    const std::chrono::duration<Rep, Period>& time_to_wait) {
  using RosMsgType = typename Adaptor::ros_message_type;
  using CustomMsgType = typename Adaptor::custom_type;

  RosMsgType ros_msg;
  waitAndGetMessage<RosMsgType, Rep, Period>(ros_msg, node, topic,
                                             time_to_wait);

  CustomMsgType custom_msg;
  Adaptor::convert_to_custom(ros_msg, custom_msg);
  return custom_msg;
}

template <class Rep, class Period>
inline CameraParams waitAndSetCameraParams(
    std::shared_ptr<rclcpp::Node> node, const std::string& topic,
    const std::chrono::duration<Rep, Period>& time_to_wait) {
  using Adaptor =
      rclcpp::TypeAdapter<dyno::CameraParams, sensor_msgs::msg::CameraInfo>;
  return waitAndGetMessageViaAdaptor<Adaptor, Rep, Period>(node, topic,
                                                           time_to_wait);
}

template <typename ValueTypeT>
decltype(auto) ParameterDetails::get() const {
  return this->get_param<ValueTypeT>(this->default_parameter_);
}

template <typename ValueTypeT>
ValueTypeT ParameterDetails::get(ValueTypeT default_value) const {
  return this->get_param<ValueTypeT>(
      rclcpp::Parameter(this->name(), default_value));
}

// template <typename ValueTypeT>
// void ParameterDetails::registerParamCallback(const
// std::function<void(ValueTypeT)>& callback) {
//   PropertyHandler::OnChangeFunction<rclcpp::Parameter> wrapper =
//   [=](rclcpp::Parameter, rclcpp::Parameter new_parameter) -> void {
//     try {
//       //this type does not necessarily match with the internal paramter type
//       //as this gets updated in too many places
//       ValueTypeT value = new_parameter.get_value<ValueTypeT>();
//       //call the actual user defined callback
//       callback(value);
//     }
//     catch(rclcpp::exceptions::InvalidParameterTypeException& e) {
//       LOG(ERROR) << "Failed to emit callback for parameter change with param:
//       " << this->name()
//         << " requested type " << type_name<ValueTypeT>() << " but actual type
//         was " << new_parameter.get_type_name();
//     }
//   };

//   //always return true so that the callback is always triggered from the
//   handler static const HasChangedValue<rclcpp::Parameter> has_changed =
//   [](rclcpp::Parameter, rclcpp::Parameter) -> bool { return true; };

//   property_handler_.registerVariable<rclcpp::Parameter>(
//     this->name(),
//     /// Default value that currently exists
//     /// Might be a problem if this is used prior to declare param and this
//     does not have a default!! this->get(), wrapper, has_changed
//   );
// }

template <typename ValueTypeT>
decltype(auto) ParameterDetails::get_param(
    const rclcpp::Parameter& default_param) const {
  const rclcpp::Parameter param = get_param(default_param);
  return param.get_value<ValueTypeT>();
}

template <typename ValueTypeT>
ParameterConstructor::ParameterConstructor(rclcpp::Node::SharedPtr node,
                                           const std::string& name,
                                           ValueTypeT value)
    : ParameterConstructor(node.get(), name, value) {}

template <typename ValueTypeT>
ParameterConstructor::ParameterConstructor(rclcpp::Node* node,
                                           const std::string& name,
                                           ValueTypeT value)
    : node_(node), parameter_(name, value) {
  CHECK_NOTNULL(node_);
  parameter_descriptor_.name = name;
  parameter_descriptor_.type = traits<ValueTypeT>::ros_parameter_type;
}

}  // namespace dyno
