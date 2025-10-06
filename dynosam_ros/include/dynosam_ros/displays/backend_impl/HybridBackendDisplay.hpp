#pragma once

#include "dynosam_ros/displays/BackendDisplayRos.hpp"
// #include "dynosam_ros/BackendDisplayPolicyRos.hpp" //for
// BackendModuleDisplayTraits

#include "dynosam/backend/ParallelHybridBackendModule.hpp"
#include "dynosam/backend/rgbd/HybridEstimator.hpp"

namespace dyno {

template <typename T>
struct BackendModuleDisplayTraits;

class ParalleHybridModuleDisplay : public BackendModuleDisplayRos {
 public:
  ParalleHybridModuleDisplay(
      const DisplayParams& params, rclcpp::Node::SharedPtr node,
      std::shared_ptr<ParallelHybridBackendModule> module)
      : BackendModuleDisplayRos(params, node) {
    CHECK_NOTNULL(module);
  }
};

class RegularHybridFormulationDisplay : public BackendModuleDisplayRos {
 public:
  RegularHybridFormulationDisplay(
      const DisplayParams& params, rclcpp::Node::SharedPtr node,
      std::shared_ptr<RegularHybridFormulation> module)
      : BackendModuleDisplayRos(params, node) {
    CHECK_NOTNULL(module);
  }
};

/// @brief Register ParalleHybridModuleDisplay as the acting backend display for
/// the Factory policy
template <>
struct BackendModuleDisplayTraits<ParallelHybridBackendModule> {
  using type = ParalleHybridModuleDisplay;
};

/// @brief Register RegularHybridFormulationDisplay as the acting backend
/// display for the Factory policy
template <>
struct BackendModuleDisplayTraits<RegularHybridFormulation> {
  using type = RegularHybridFormulationDisplay;
};

}  // namespace dyno
