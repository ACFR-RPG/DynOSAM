#pragma once

#include "dynosam/backend/ParallelHybridBackendModule.hpp"
#include "dynosam/backend/rgbd/HybridEstimator.hpp"
#include "dynosam_ros/displays/BackendDisplayRos.hpp"
#include "dynosam_ros/displays/DisplaysCommon.hpp"

namespace dyno {

template <typename T>
struct BackendModuleDisplayTraits;

static HybridAccessorCommon::Ptr hybridAccessorCommonHelper(
    Accessor::Ptr accessor) {
  CHECK(accessor);
  HybridAccessorCommon::Ptr hybrid_accessor_common =
      std::dynamic_pointer_cast<HybridAccessorCommon>(accessor);
  if (!hybrid_accessor_common) {
    throw DynosamException(
        "Accessor could not be cast to HybridAccessorCommon");
  }
  return hybrid_accessor_common;
}

class HybridModuleDisplayCommon : public BackendModuleDisplayRos {
 public:
  HybridModuleDisplayCommon(const DisplayParams& params, rclcpp::Node* node,
                            HybridAccessorCommon::Ptr hybrid_accessor);

  void publishObjectBoundingBoxes(const BackendOutputPacket::ConstPtr& output);
  void publishObjectKeyFrames(const FrameId frame_id,
                              const Timestamp timestamp);

 private:
  HybridAccessorCommon::Ptr hybrid_accessor_;
  MarkerArrayPub::SharedPtr object_bounding_box_pub_;
  MarkerArrayPub::SharedPtr object_key_frame_pub_;
};

class ParalleHybridModuleDisplay : public HybridModuleDisplayCommon {
 public:
  ParalleHybridModuleDisplay(
      const DisplayParams& params, rclcpp::Node* node,
      std::shared_ptr<ParallelHybridBackendModule> module)
      : HybridModuleDisplayCommon(
            params, node, hybridAccessorCommonHelper(module->getAccessor())),
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
      : HybridModuleDisplayCommon(
            params, node, module->derivedAccessor<HybridAccessorCommon>()),
        module_(CHECK_NOTNULL(module)) {
    CHECK_NOTNULL(module);
  }

  void spin(const BackendOutputPacket::ConstPtr& output) override;

 private:
  std::shared_ptr<RegularHybridFormulation> module_;
};

class HybridKeyFrameFormulationDisplay : public HybridModuleDisplayCommon {
 public:
  HybridKeyFrameFormulationDisplay(
      const DisplayParams& params, rclcpp::Node* node,
      std::shared_ptr<HybridFormulationKeyFrame> module)
      : HybridModuleDisplayCommon(
            params, node, module->derivedAccessor<HybridAccessorCommon>()),
        module_(CHECK_NOTNULL(module)) {
    CHECK_NOTNULL(module);
  }

  void spin(const BackendOutputPacket::ConstPtr& output) override;

 private:
  std::shared_ptr<HybridFormulationKeyFrame> module_;
};

// TODO: in light of now having many Hybrid formulations we should template on
// HybridFormulation not on RegularHybridFormulation as the display should work
// for all!!

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

template <>
struct BackendModuleDisplayTraits<HybridFormulationKeyFrame> {
  using type = HybridKeyFrameFormulationDisplay;
};

}  // namespace dyno
