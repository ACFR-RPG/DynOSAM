#include <gflags/gflags.h>
#include <glog/logging.h>

#include "dynosam2_ros/Dynosam2Node.hpp"
#include "dynosam_ros_common/RosUtils.hpp"
#include "rclcpp/executor.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char* argv[]) {
  auto non_ros_args = dyno::ros::initRosAndLogging(argc, argv);

  rclcpp::NodeOptions options;
  options.arguments(non_ros_args);
  options.use_intra_process_comms(true);

  rclcpp::executors::MultiThreadedExecutor exec;

  // at least in kilted we need to create the node sepately and assign to some
  // variable othewise if we add directly to exec, it passed by reference and
  // the object is immediately deallocated
  auto node = std::make_shared<dyno2::Dynosam2Node>(options);
  exec.add_node(node);
  exec.spin();
  return 0;
}
