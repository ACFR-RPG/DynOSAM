#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "dynosam_ros/CameraSystem.hpp"

using namespace dyno;

TEST(CameraSystem, SensorModeRGBD) {
  SensorMode mode("rgb+depth");
  EXPECT_EQ(mode.depthRigMode(), DepthRigType::RGBD);
  EXPECT_FALSE(mode.useImu());
  EXPECT_EQ(mode.configs().size(), 2);
}

TEST(CameraSystem, SensorModeRGBDIMUMisc) {
  SensorMode mode("rgb+depth+imu+mask");
  EXPECT_EQ(mode.depthRigMode(), DepthRigType::RGBD);
  EXPECT_TRUE(mode.useImu());
  EXPECT_EQ(mode.configs().size(), 3);
}

TEST(CameraSystem, SensorModeStereo) {
  SensorMode mode("stereo+imu");
  EXPECT_EQ(mode.depthRigMode(), DepthRigType::Stereo);
  EXPECT_TRUE(mode.useImu());
  EXPECT_EQ(mode.configs().size(), 2);
}
