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
#include "dynosam_ros/RosUtils.hpp"

#include <glog/logging.h>
#include <gtsam/geometry/Pose3.h>

#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/FileSystem.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/time.hpp"

/**
 * @brief Constructs a c-style pointer array from a vector of strings.
 *
 * Must be freed
 *
 * @param args
 * @return char**
 */
char** constructArgvC(const std::vector<std::string>& args) {
  char** argv = new char*[args.size()];

  for (size_t i = 0; i < args.size(); i++) {
    const std::string& arg = args.at(i);
    argv[i] = new char[arg.size() + 1];
    strcpy(argv[i], arg.c_str());
  }

  return argv;
}

namespace dyno::ros {

std::vector<std::string> initRosAndLogging(int argc, char* argv[]) {
  // google::ParseCommandLineFlags(&argc, &argv, true);
  auto non_ros_args = rclcpp::init_and_remove_ros_arguments(argc, argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  FLAGS_log_prefix = 1;

  int non_ros_argc = non_ros_args.size();
  char** non_ros_argv_c = constructArgvC(non_ros_args);
  // non_ros_argv_c is heap allocated but attempting to free it after usage in
  // the ParseCommandLineFlags function results in a "double free or corruption"
  // error. I think this is because ParseCommandLineFlags modifies it in place
  // and then free it itself somehow unsure, and may result in a minor memory
  // leak
  google::ParseCommandLineFlags(&non_ros_argc, &non_ros_argv_c, true);

  return non_ros_args;
}

void initGlog(const std::string& params_folder,
              const std::string& executable_name) {
  static bool is_init = false;
  if (!is_init) {
    google::InitGoogleLogging(executable_name.c_str());
    FLAGS_logtostderr = 1;
    FLAGS_colorlogtostderr = 1;
    FLAGS_log_prefix = 1;

    auto formatFilePath = [](const std::string& file_path) -> std::string {
      return "--flagfile=" + file_path;
    };

    std::vector<std::filesystem::path> files =
        utils::getAllFilesInDir(params_folder);
    // discover .flags
    std::vector<std::string> non_ros_args;
    // NOTE: while not confirmed via the documentation it seems
    // ParseCommandLineFlags expects the first value to be the executable name
    // (like for regular argv) and therefore we should prepend the args with
    // some value
    non_ros_args.push_back(executable_name);
    for (const auto& file_path : files) {
      if (static_cast<std::string>(file_path.extension()) == ".flags") {
        non_ros_args.push_back(formatFilePath(file_path));
      }
    }

    int non_ros_argc = non_ros_args.size();
    char** non_ros_argv_c = constructArgvC(non_ros_args);
    // non_ros_argv_c is heap allocated but attempting to free it after usage in
    // the ParseCommandLineFlags function results in a "double free or
    // corruption" error. I think this is because ParseCommandLineFlags modifies
    // it in place and then free it itself somehow unsure, and may result in a
    // minor memory leak.
    google::ParseCommandLineFlags(&non_ros_argc, &non_ros_argv_c, true);
    // is_init = true;
  }
}

rclcpp::QoS addQosParameter(rclcpp::Node& node, std::string default_qos,
                            std::string parameter_name, const int default_depth,
                            const bool use_node_namespace) {
  std::string name = parameter_name;
  if (use_node_namespace) {
    name = node.get_effective_namespace() + "_" + parameter_name;
  }

  std::string qos_str =
      Parameter::Builder(&node, name, default_qos)
          .description("QoS profile specifier for param " + name +
                       " (default: " + default_qos + ")")
          .finish()
          .get<std::string>();

  const std::string depth_param = name + "_depth";
  const int depth =
      Parameter::Builder(&node, depth_param, default_depth)
          .description("QoS depth (keep N) specifier for param " + name +
                       " (default: " + std::to_string(default_depth) + ")")
          .finish()
          .get<int>();

  return parseQosString(qos_str, depth);
}

rclcpp::QoS parseQosString(const std::string& str, const int depth) {
  std::string profile = str;
  // Convert to upper case.
  std::transform(profile.begin(), profile.end(), profile.begin(), ::toupper);

  rmw_qos_profile_t rmw_qos = rmw_qos_profile_default;

  if (profile == "SYSTEM_DEFAULT") {
    rmw_qos = rmw_qos_profile_system_default;
  } else if (profile == "DEFAULT") {
    rmw_qos = rmw_qos_profile_default;
  } else if (profile == "PARAMETER_EVENTS") {
    rmw_qos = rmw_qos_profile_parameter_events;
  } else if (profile == "SERVICES_DEFAULT") {
    rmw_qos = rmw_qos_profile_services_default;
  } else if (profile == "PARAMETERS") {
    rmw_qos = rmw_qos_profile_parameters;
  } else if (profile == "SENSOR_DATA") {
    rmw_qos = rmw_qos_profile_sensor_data;
  } else {
    RCLCPP_WARN_STREAM(
        rclcpp::get_logger("parseQoSString"),
        "Unknown QoS profile: " << profile << ". Returning profile: DEFAULT");
  }
  auto qos_init =
      depth == 0 ? rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default)
                 : rclcpp::KeepLast(depth);
  return rclcpp::QoS(qos_init, rmw_qos);
}

std::ostream& operator<<(std::ostream& stream, const Parameter& param) {
  stream << (std::string)param;
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const rclcpp::Parameter& param) {
  stream << param.get_name() << ": " << param.value_to_string() << " ("
         << param.get_type_name() << ")";
  return stream;
}

const std::string& Parameter::name() const {
  return default_parameter_.get_name();
}

std::string Parameter::node_name() const { return node_->get_name(); }

rclcpp::Parameter Parameter::get() const {
  return this->get_param(this->default_parameter_);
}

std::string Parameter::get(const char* default_value) const {
  return this->get_param<std::string>(
      rclcpp::Parameter(this->name(), std::string(default_value)));
}

std::string Parameter::get(const std::string& default_value) const {
  return this->get_param<std::string>(
      rclcpp::Parameter(this->name(), default_value));
}

Parameter::operator std::string() const {
  std::stringstream ss;
  ss << "[ name: " << this->name();
  ss << " value: " << this->get().value_to_string();
  ss << " description: " << description_.description << "]";
  return ss.str();
}

Parameter::Parameter(
    rclcpp::Node* node, const rclcpp::Parameter& parameter,
    const rcl_interfaces::msg::ParameterDescriptor& description)
    : node_(node), default_parameter_(parameter), description_(description) {
  declare();
}

rclcpp::Parameter Parameter::get_param(
    const rclcpp::Parameter& default_param) const {
  const bool is_set = isSet();
  bool has_default = default_param.get_type() != rclcpp::PARAMETER_NOT_SET;

  if (!is_set) {
    // if no default value we treat a not set parameter as a error as the use
    // MUST override it via runtime configuration
    if (!has_default) {
      throw InvalidDefaultParameter(this->name());
    } else {
      return default_param;
    }
  }
  return node_->get_parameter(this->name());
}

void Parameter::declare() {
  // only declare if needed
  if (!node_->has_parameter(this->name())) {
    const rclcpp::ParameterValue default_value =
        default_parameter_.get_parameter_value();
    const rclcpp::ParameterValue effective_value =
        node_->declare_parameter(this->name(), default_value, description_);
    (void)effective_value;
  }

  // add the param subscriber if we dont have one!!
  // if(!param_subscriber_) {
  //   param_subscriber_ =
  //   std::make_shared<rclcpp::ParameterEventHandler>(node_);

  //   auto cb = [&](const rclcpp::Parameter& new_parameter) {
  //       node_->set_parameter(new_parameter);
  //       //should check same type?
  //       CHECK_EQ(new_parameter.get_name(), this->name());

  //       property_handler_.update(this->name(),new_parameter);
  //   };
  //   //callback handler must be set and remain in scope for the cb's to
  //   trigger cb_handle_ =
  //   param_subscriber_->add_parameter_callback(this->name(), cb);
  // }
}

Parameter::Builder::Builder(rclcpp::Node::SharedPtr node,
                            const std::string& name)
    : Builder(node.get(), name) {}

Parameter::Builder::Builder(rclcpp::Node* node, const std::string& name)
    : node_(node), parameter_(name) {
  CHECK_NOTNULL(node_);
  parameter_descriptor_.name = name;
  parameter_descriptor_.dynamic_typing = true;
}

Parameter Parameter::Builder::finish() const {
  return Parameter(node_, parameter_, parameter_descriptor_);
}

Parameter::Builder& Parameter::Builder::description(
    const std::string& description) {
  parameter_descriptor_.description = description;
  return *this;
}

Parameter::Builder& Parameter::Builder::read_only(bool read_only) {
  parameter_descriptor_.read_only = read_only;
  return *this;
}

Parameter::Builder& Parameter::Builder::parameter_description(
    const rcl_interfaces::msg::ParameterDescriptor& parameter_description) {
  parameter_descriptor_ = parameter_description;
  return *this;
}

Timestamp fromRosTime(const rclcpp::Time& time) {
  Timestamp timestamp;
  convert(time, timestamp);
  return timestamp;
}

rclcpp::Time toRosTime(Timestamp timestamp) {
  rclcpp::Time time;
  convert(timestamp, time);
  return time;
}

}  // namespace dyno::ros
