#include "dynosam2/frontend/Frame.hpp"
#include "dynosam_common/Types.hpp"

using namespace dyno2;

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <unordered_map>

struct KeypointIdentifierHash {
  std::size_t operator()(const dyno::KeypointIdentifier& id) const {
    return dyno::hashTriplet(id.frameId(), id.trackletId(), id.objectId());
  }
};

TEST(BaGraph, test) {
  // std::unordered_map<dyno::KeypointIdentifier, int> frames;
  dyno::FastUnorderedMap<dyno::TemporalObjectId, Frame::Ptr> frames;
}
