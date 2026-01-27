#pragma once

#include "dynosam_ros/FormulationFactoryPlugin.hpp"
#include "dynosam_ros/displays/BackendDisplayRos.hpp"

using namespace dyno;

namespace dyno_testing {

class DummyFormulation : public Formulation<MapVision> {
 public:
  using Base = Formulation<MapVision>;
  using Base::AccessorTypePointer;
  using Base::ConstructorParams;
  using Base::MapTraitsType;
  using Base::ObjectUpdateContextType;
  using Base::PointUpdateContextType;

  DummyFormulation(const ConstructorParams& constructor_params)
      : Base(constructor_params) {}

  void dynamicPointUpdateCallback(const PointUpdateContextType&,
                                  UpdateObservationResult&, gtsam::Values&,
                                  gtsam::NonlinearFactorGraph&) override {}

  void objectUpdateContext(const ObjectUpdateContextType&,
                           UpdateObservationResult&, gtsam::Values&,
                           gtsam::NonlinearFactorGraph&) override {}

  bool isDynamicTrackletInMap(
      const typename MapTraitsType::LandmarkNodePtr&) const override {
    return false;
  }

 protected:
  AccessorTypePointer createAccessor(
      const SharedFormulationData&) const override {
    return nullptr;
  }

  std::string loggerPrefix() const override { return "dummy_formulation"; }
};

class DummyFormulationDisplay : public BackendModuleDisplayRos {
 public:
  DummyFormulationDisplay(const DisplayParams& params,
                          rclcpp::Node::SharedPtr node,
                          std::shared_ptr<DummyFormulation> formulation)
      : BackendModuleDisplayRos(params, node), formulation_(formulation) {}

  inline void spin(const BackendOutputPacket::ConstPtr& output) override {}

 private:
  std::shared_ptr<DummyFormulation> formulation_;
};

class TestFormulationFactoryPlugin
    : public FormulationFactoryPluginT<MapVision> {
 public:
  TestFormulationFactoryPlugin() = default;

  FormulationVizWrapper<MapVision> create(
      rclcpp::Node::SharedPtr node, const DisplayParams& display_params,
      const FormulationConstructorParams<MapVision>& constructor_params)
      override {
    LOG(INFO) << "In TestFormulationFactoryPlugin";
    FormulationVizWrapper<MapVision> result;

    auto dummy_formulation =
        std::make_shared<DummyFormulation>(constructor_params);

    result.formulation = dummy_formulation;
    result.display = std::make_shared<DummyFormulationDisplay>(
        display_params, node, dummy_formulation);

    return result;
  }
};

}  // namespace dyno_testing
