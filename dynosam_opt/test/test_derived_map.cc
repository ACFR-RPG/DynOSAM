
#include "test_derived_map.hpp"

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

// #include "dynosam_opt/Map.hpp"

#include "dynosam_common/Types.hpp"
// #include "internal/helpers.hpp"

using namespace dyno;

// namespace dyno_testing {

// inline KeypointStatus makeStatusKeypointMeasurement(
//     TrackletId tracklet_id, ObjectId object_id, FrameId frame_id,
//     const Keypoint& keypoint = Keypoint(), double sigma = 2.0) {
//   gtsam::Vector2 kp_sigmas;
//   kp_sigmas << sigma, sigma;
//   MeasurementWithCovariance<Keypoint> kp_measurement =
//       MeasurementWithCovariance<Keypoint>::FromSigmas(keypoint, kp_sigmas);
//   return KeypointStatus(kp_measurement, frame_id, 0.0, tracklet_id,
//   object_id,
//                         ReferenceFrame::LOCAL);
// }

// }

TEST(DerivedMap, testNotNullInterfaces) {
  auto test_map = dyno_testing::Map<>::create();

  auto frame_interface = test_map->asFrameInterface();
  auto lmk_interface = test_map->asFrameInterface();
  auto obj_interface = test_map->asFrameInterface();

  EXPECT_TRUE(frame_interface != nullptr);
  EXPECT_TRUE(lmk_interface != nullptr);
  EXPECT_TRUE(obj_interface != nullptr);
}

// TEST(DerivedMap, basicAddOnlyStatic) {

//   StatusKeypointVector measurements;

//   TrackletIds expected_tracklets;
//   // 10 measurements with unique tracklets at frame 0
//   for (size_t i = 0; i < 10; i++) {
//     measurements.push_back(
//         dyno_testing::makeStatusKeypointMeasurement(i, background_label, 0));
//     expected_tracklets.push_back(i);
//   }

//   using TestingMap = dyno_testing::Map<>;
//   using SharedLandmarkNode = TestingMap::SharedLandmarkNodeT;

//   auto map = TestingMap::create();

//   map->updateObservations(measurements);

//   EXPECT_TRUE(map->frameExists(0));
//   EXPECT_FALSE(map->frameExists(1));

//   EXPECT_TRUE(map->landmarkExists(0));
//   EXPECT_TRUE(map->landmarkExists(9));
//   EXPECT_FALSE(map->landmarkExists(10));

//   EXPECT_EQ(map->staticTrackletsByFrame(0), expected_tracklets);

//   // expected tracklets in frame 0
//   TrackletIds expected_tracklets_f0 = expected_tracklets;

//   TrackletIds expected_tracklets_f1;
//   // add another 5 points at frame 1
//   measurements.clear();
//   for (size_t i = 0; i < 5; i++) {
//     measurements.push_back(
//         dyno_testing::makeStatusKeypointMeasurement(i, background_label, 1));

//     expected_tracklets.push_back(i);
//     expected_tracklets_f1.push_back(i);
//   }

//   // apply update
//   map->updateObservations(measurements);

//   EXPECT_EQ(map->staticTrackletsByFrame(0), expected_tracklets_f0);
//   EXPECT_EQ(map->staticTrackletsByFrame(1), expected_tracklets_f1);

//   // check for frames in some landmarks
//   // should be seen in frames 0 and 1
//   SharedLandmarkNode lmk1 = map->getLandmark(0);
//   std::vector<FrameId> lmk_1_seen_frames =
//       lmk1->getSeenFrames().collectIds<FrameId>();
//   std::vector<FrameId> lmk_1_seen_frames_expected = {0, 1};
//   EXPECT_EQ(lmk_1_seen_frames, lmk_1_seen_frames_expected);

//   // should be seen in frames 0
//   SharedLandmarkNode lmk6 = map->getLandmark(6);
//   std::vector<FrameId> lmk_6_seen_frames =
//       lmk6->getSeenFrames().collectIds<FrameId>();
//   std::vector<FrameId> lmk_6_seen_frames_expected = {0};
//   EXPECT_EQ(lmk_6_seen_frames, lmk_6_seen_frames_expected);

//   // check that the frames here are the ones in the map
//   EXPECT_EQ(map->getFrame(lmk_1_seen_frames.at(0)),
//             map->getFrame(lmk_6_seen_frames.at(0)));

//   // finally check that there are no objects
//   EXPECT_EQ(map->getFrame(lmk_1_seen_frames.at(0))->objects_seen.size(), 0);
//   EXPECT_EQ(map->numObjectsSeen(), 0u);
// }
