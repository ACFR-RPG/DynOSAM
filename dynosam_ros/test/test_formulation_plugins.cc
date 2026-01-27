#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "dynosam_ros/FormulationFactoryPlugin.hpp"
#include "rclcpp/node.hpp"

using namespace dyno;

TEST(FormulationPlugins, testLoadDummyPlugin) {
  FormulationFactoryPluginLoader loader;

  FormulationConstructorParams<MapVision> params;
  params.map = MapVision::create();

  auto node = std::make_shared<rclcpp::Node>("dyno_plugin");
  auto result = loader.loadFormulation<MapVision>(
      "dyno_testing::TestFormulationFactoryPlugin", node, DisplayParams{},
      params);

  auto formulation = result.formulation;
  EXPECT_TRUE(formulation != nullptr);
  EXPECT_EQ(formulation->getFullyQualifiedName(), "dummy_formulation");
  EXPECT_TRUE(result.hasDisplay());
}
