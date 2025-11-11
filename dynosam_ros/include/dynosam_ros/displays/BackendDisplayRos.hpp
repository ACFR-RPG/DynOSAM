#pragma once

#include "dynosam/visualizer/VisualizerPipelines.hpp"  //for BackendModuleDisplay
#include "dynosam_ros/Display-Definitions.hpp"
#include "rclcpp/node.hpp"
#include "rclcpp/node_options.hpp"

namespace dyno {

class BackendModuleDisplayRos : public BackendModuleDisplay {
 public:
  DYNO_POINTER_TYPEDEFS(BackendModuleDisplayRos)

  BackendModuleDisplayRos(const DisplayParams& params, rclcpp::Node* node)
      : params_(params), node_(CHECK_NOTNULL(node)) {}
  virtual ~BackendModuleDisplayRos() = default;

 protected:
  DisplayParams params_;
  rclcpp::Node* node_;
};

}  // namespace dyno
