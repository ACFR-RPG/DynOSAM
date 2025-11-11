/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam_ros/Utils.hpp"

#include <glog/logging.h>

#include <string>

#include "rclcpp/rclcpp.hpp"

namespace dyno {

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

}  // namespace dyno
