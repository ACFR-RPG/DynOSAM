#pragma once

#include "dynosam/backend/ParallelHybridBackendModule.hpp"
#include "dynosam/backend/rgbd/HybridEstimator.hpp"
#include "dynosam_ros/displays/BackendDisplayRos.hpp"
#include "dynosam_ros/displays/DisplaysCommon.hpp"

namespace dyno {

template <typename T>
struct BackendModuleDisplayTraits;

class HybridModuleDisplayCommon : public BackendModuleDisplayRos {
 public:
  HybridModuleDisplayCommon(const DisplayParams& params, rclcpp::Node* node);

  void publishObjectBoundingBoxes(const BackendOutputPacket::ConstPtr& outpu);
  void publishObjectKeyFrames();

 private:
  MarkerArrayPub::SharedPtr object_bounding_box_pub_;
};

class ParalleHybridModuleDisplay : public HybridModuleDisplayCommon {
 public:
  ParalleHybridModuleDisplay(
      const DisplayParams& params, rclcpp::Node* node,
      std::shared_ptr<ParallelHybridBackendModule> module)
      : HybridModuleDisplayCommon(params, node),
        module_(CHECK_NOTNULL(module)) {}

  void spin(const BackendOutputPacket::ConstPtr& output) override;

 private:
  std::shared_ptr<ParallelHybridBackendModule> module_;
};

class RegularHybridFormulationDisplay : public HybridModuleDisplayCommon {
 public:
  RegularHybridFormulationDisplay(
      const DisplayParams& params, rclcpp::Node* node,
      std::shared_ptr<RegularHybridFormulation> module)
      : HybridModuleDisplayCommon(params, node),
        module_(CHECK_NOTNULL(module)) {
    CHECK_NOTNULL(module);
  }

  void spin(const BackendOutputPacket::ConstPtr& output) override;

 private:
  std::shared_ptr<RegularHybridFormulation> module_;
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
