
#include "test_derived_map.hpp"

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "dynosam_opt/Map.hpp"

using namespace dyno;

TEST(DerivedMap, test) {
  auto test_map = dyno_testing::Map<>::create();
  test_map->numFrames();

  auto frame_interface = test_map->asFrameInterface();
}
