#include <glog/logging.h>

#include <iostream>

#include "dynosam_ros/RosUtils.hpp"
#include "rclcpp/rclcpp.hpp"

// DEFINE_string(output_path, "/bad/path" ,"Test variable");
DECLARE_string(output_path);

// char** constructArgvC(const std::vector<std::string>& args) {
//   char** argv = new char*[args.size()];

//   for (size_t i = 0; i < args.size(); i++) {
//     const std::string& arg = args.at(i);
//     argv[i] = new char[arg.size() + 1];
//     strcpy(argv[i], arg.c_str());
//   }

//   return argv;
// }

namespace dyno {

class TestComponent : public rclcpp::Node {
 public:
  TestComponent(const rclcpp::NodeOptions& options)
      : Node("dyno_test_component", options) {
    RCLCPP_INFO_STREAM(this->get_logger(), "In test component");

    std::string param_folder =
        this->declare_parameter("params_path", "/path/to/dyno/params");
    RCLCPP_INFO_STREAM(this->get_logger(), param_folder);

    // load all gflags via flag files auto-discovered in the params folder
    ros::initGlog(param_folder, "dyno_test_component");
    // ros::initGlog(param_folder, "dyno_test_component1");

    // std::string pipeline_flags = param_folder + "pipeline.flags";
    // google::InitGoogleLogging("dyno_test_component");
    // FLAGS_logtostderr = 1;
    // FLAGS_colorlogtostderr = 1;
    // FLAGS_log_prefix = 1;

    // std::vector<std::string> non_ros_args;
    // std::stringstream ss;
    // ss << "--flagfile=" << pipeline_flags;
    // non_ros_args.push_back("fake_executable");
    // non_ros_args.push_back(ss.str());
    // // f"--flagfile={os.path.join(params_path, f)}"
    // RCLCPP_INFO_STREAM(this->get_logger(), non_ros_args.back());

    // int non_ros_argc = non_ros_args.size();
    // char** non_ros_argv_c = constructArgvC(non_ros_args);
    // // non_ros_argv_c is heap allocated but attempting to free it after usage
    // in
    // // the ParseCommandLineFlags function results in a "double free or
    // corruption"
    // // error. I think this is because ParseCommandLineFlags modifies it in
    // place
    // // and then free it itself somehow unsure, and may result in a minor
    // memory
    // // leak
    // google::ParseCommandLineFlags(&non_ros_argc, &non_ros_argv_c, true);

    RCLCPP_INFO_STREAM(this->get_logger(), FLAGS_output_path);
  }
};

}  // namespace dyno
