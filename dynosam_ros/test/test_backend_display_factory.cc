
#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <dynosam/test/helpers.hpp>

// #include "dynosam_ros/BackendModuleFactory.hpp"

// using namespace dyno;

// class TestModule {};
// class TestModule1 {};

// class BackendDisplayRosTest : public BackendModuleDisplayRos {
//     public:
//     BackendDisplayRosTest(const DisplayParams& params,
//     rclcpp::Node::SharedPtr node, std::shared_ptr<TestModule> module)
//         : BackendModuleDisplayRos(params, node) {}
// };

// template<>
// struct dyno::BackendModuleDisplayTraits<TestModule> { using type =
// BackendDisplayRosTest; };

// REGISTER_VIZ_AUTONAME(TestModule, BackendDisplayRosTest);

// TEST(BackendDisplayFactory, basic) {
//     DisplayParams params;
//      auto node = std::make_shared<rclcpp::Node>("display");

//     BackendModuleDisplayFactory factory(params, node);
//     auto display = factory.create("TestModule");

//     EXPECT_TRUE(display != nullptr);

//     auto display_ros_test =
//     dynamic_cast<BackendDisplayRosTest*>(display.get());
//     EXPECT_TRUE(display_ros_test != nullptr);

// }

// TEST(BackendDisplayFactory, basic) {
//     DisplayParams params;
//      auto node = std::make_shared<rclcpp::Node>("display");

//     BackendModuleFactoryRos factory(params, node);

//     auto module = std::make_shared<TestModule>();

//     // auto display = factory.create("TestModule");
//     auto display = factory.makeBackendModuleDisplay(module);

//     EXPECT_TRUE(display != nullptr);

//     auto display_ros_test =
//     dynamic_cast<BackendDisplayRosTest*>(display.get());
//     EXPECT_TRUE(display_ros_test != nullptr);

// }

TEST(BackendDisplayFactory, policy) {
  DisplayParams params;
  auto node = std::make_shared<rclcpp::Node>("display");

  // BackendModuleFactoryRos factory(params, node);
  // RegularBackendModuleFactory<BackendModulePolicyRos> factory()
  // BackendModuleFactory<BackendModulePolicyRos> factory(params, node);
  // factory.testPolicy();

  // auto module = std::make_shared<TestModule>();

  // // auto display = factory.create("TestModule");
  // auto display = factory.makeBackendModuleDisplay(module);

  // EXPECT_TRUE(display != nullptr);

  // auto display_ros_test =
  // dynamic_cast<BackendDisplayRosTest*>(display.get());
  // EXPECT_TRUE(display_ros_test != nullptr);
}
