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

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "dynosam_ros/PipelineRos.hpp"
#include "dynosam_ros/RosUtils.hpp"
#include "rclcpp/executor.hpp"
#include "rclcpp/rclcpp.hpp"

DEFINE_bool(show_dyno_args, false,
            "Show all loaded DynoSAM args (YAML and gflag) and exit");

int main(int argc, char* argv[]) {
  auto non_ros_args = dyno::ros::initRosAndLogging(argc, argv);

  rclcpp::NodeOptions options;
  options.arguments(non_ros_args);
  options.use_intra_process_comms(true);

  rclcpp::executors::MultiThreadedExecutor exec;
  auto ros_pipeline = std::make_shared<dyno::DynoPipelineManagerRos>();

  if (FLAGS_show_dyno_args) {
    const dyno::DynoParams& params = ros_pipeline->getDynoParams();
    params.printAllParams(true);
    rclcpp::shutdown();
    return 0;
  } else {
    ros_pipeline->initalisePipeline();
  }

  exec.add_node(ros_pipeline);
  while (rclcpp::ok()) {
    if (!ros_pipeline->spinOnce()) {
      break;
    }
    exec.spin_some();
  }

  ros_pipeline.reset();
  return 0;
}
