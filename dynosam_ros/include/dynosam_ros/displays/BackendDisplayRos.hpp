#pragma once

#include "dynosam/visualizer/VisualizerPipelines.hpp"  //for BackendModuleDisplay
#include "dynosam_cv/SensorRig.hpp"
#include "rclcpp/node.hpp"
#include "rclcpp/node_options.hpp"

namespace dyno {

class BackendModuleDisplayRos : public BackendModuleDisplay {
 public:
  DYNO_POINTER_TYPEDEFS(BackendModuleDisplayRos)

  BackendModuleDisplayRos(const ReferenceFrames& params, rclcpp::Node* node)
      : params_(params), node_(CHECK_NOTNULL(node)) {}
  virtual ~BackendModuleDisplayRos() = default;

 protected:
  ReferenceFrames params_;
  rclcpp::Node* node_;
};

}  // namespace dyno
