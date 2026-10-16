#pragma once

#include <memory>

#include "rclcpp/node.hpp"
#include "rclcpp/node_options.hpp"

namespace dyno2 {

class Dynosam2Node : public rclcpp::Node {
 public:
  explicit Dynosam2Node(
      const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
  ~Dynosam2Node();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace dyno2
