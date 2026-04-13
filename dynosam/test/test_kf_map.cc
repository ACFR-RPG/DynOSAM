// #include <glog/logging.h>
// #include <gtest/gtest.h>

// #include <exception>

// #include "dynosam_common/Types.hpp"
// #include "dynosam_common/utils/GtsamUtils.hpp"
// #include "dynosam/formulations/KeyFrameHybridMap.hpp"
// #include "dynosam_opt/Map.hpp"
// #include "internal/helpers.hpp"

// using namespace dyno;

// GenericValueTrack<CameraMeasurement> cameraTrackMeasurement(
//   FrameId frame_id, Timestamp timestamp, TrackletId tracklet_id,
//   ObjectId object_id
// ) {
//   return GenericValueTrack<CameraMeasurement>(
//     CameraMeasurement(Keypoint()),
//     frame_id,
//     timestamp,
//     tracklet_id,
//     object_id,
//     ReferenceFrame::LOCAL
//   );
// }

// TEST(KeyFrameMap, SetAndQueryCameraKeyFrame) {
//   auto map = KeyFrameMap::create();

//   FrameId k = 1;
//   Timestamp t = 0.1;

//   map->updateObservations(cameraTrackMeasurement(k, t, 1, background_label));

//   EXPECT_FALSE(map->isCameraKeyFrame(k));

//   map->setCameraKeyFrame(k);

//   EXPECT_TRUE(map->isCameraKeyFrame(k));
//   EXPECT_FALSE(map->isCameraKeyFrame(k+1));
//   EXPECT_TRUE(map->isAnyKeyFrame(k));
// }

// TEST(KeyFrameMap, SetAndQueryObjectKeyFrame) {
//   auto map = KeyFrameMap::create();

//   FrameId k = 1;
//   ObjectId j = 42;

//   map->updateObservations(cameraTrackMeasurement(k, 0.0, 1, j));

//   map->setObjectKeyFrame(k, j);

//   EXPECT_TRUE(map->isObjectKeyFrame(k, j));
//   EXPECT_FALSE(map->isObjectKeyFrame(k, j+1));
//   EXPECT_FALSE(map->isObjectKeyFrame(k+1, j));
//   EXPECT_FALSE(map->isCameraKeyFrame(k));
//   EXPECT_TRUE(map->isAnyKeyFrame(k));
// }

// TEST(KeyFrameMap, StaticLandmarkOnlyUsesCameraKeyFrames) {
//   auto map = KeyFrameMap::create();

//   FrameId k1 = 1, k2 = 2;
//   Timestamp t = 0.0;
//   TrackletId l = 10;

//   auto m1 = cameraTrackMeasurement(k1, t, l, background_label);
//   auto m2 = cameraTrackMeasurement(k2, t, l, background_label);

//   map->updateObservations(m1);
//   map->updateObservations(m2);

//   map->setCameraKeyFrame(k2);

//   auto lmk = map->asLandmarkInterface()->getLandmark(l);
//   EXPECT_TRUE(lmk->isStatic());

//   auto seen = lmk->getSeenFrameIds();

//   EXPECT_EQ(seen.size(), 1);
//   EXPECT_EQ(seen[0], k2);

//   EXPECT_TRUE(map->setCameraKeyFrame(k1));
//   seen = lmk->getSeenFrameIds();

//   EXPECT_EQ(seen.size(), 2);
//   EXPECT_EQ(seen, FrameIds({k1, k2}));
// }

// TEST(KeyFrameMap, DynamicLandmarkUsesObjectKeyFrames) {
//   auto map = KeyFrameMap::create();

//   FrameId k1 = 1, k2 = 2;
//   ObjectId j = 5;
//   Timestamp t = 0.0;
//   TrackletId l = 20;

//   auto m1 = cameraTrackMeasurement(k1, t, l, j);
//   auto m2 = cameraTrackMeasurement(k2, t, l, j);

//   map->updateObservations(m1);
//   map->updateObservations(m2);

//   auto lmk = map->asLandmarkInterface()->getLandmark(l);

//   auto seen = lmk->getSeenFrameIds();

//   EXPECT_EQ(seen.size(), 0);

//   // mark k1 as object keyframe
//   map->setObjectKeyFrame(k1, j);
//   seen = lmk->getSeenFrameIds();

//   EXPECT_EQ(seen.size(), 1);
//   EXPECT_EQ(seen, FrameIds({k1}));

//   // mark k1 as object keyframe
//   map->setObjectKeyFrame(k2, j);
//   seen = lmk->getSeenFrameIds();

//   EXPECT_EQ(seen.size(), 2);
//   EXPECT_EQ(seen, FrameIds({k1, k2}));
// }

// TEST(KeyFrameMap, ObjectSeenFramesRespectKeyFrames) {
//   auto map = KeyFrameMap::create();

//   FrameId k1 = 1, k2 = 2;
//   ObjectId j = 3;
//   Timestamp t = 0.0;
//   TrackletId l = 30;

//   auto m1 = cameraTrackMeasurement(k1, t, l, j);
//   auto m2 = cameraTrackMeasurement(k2, t, l, j);

//   map->updateObservations(m1);
//   map->updateObservations(m2);

//   map->setCameraKeyFrame(k2);

//   auto object_interface = map->asObjectInterface();
//   auto obj = object_interface->getObject(j);

//   auto seen_frames = obj->getSeenFrameIds();

//   EXPECT_EQ(seen_frames.size(), 2);
// }

// TEST(KeyFrameMap, RetroactiveCameraKeyFrameUpdatesLandmarks) {
//   auto map = KeyFrameMap::create();

//   FrameId k = 1;
//   Timestamp t = 0.0;
//   TrackletId l = 10;

//   auto m1 = cameraTrackMeasurement(k, t, l, background_label);
//   map->updateObservations(m1);

//   auto frame_interface = map->asFrameInterface();
//   auto landmark_interface = map->asLandmarkInterface();

//   auto frame = frame_interface->getFrame(k);
//   auto lmk = landmark_interface->getLandmark(l);

//   // Sanity: not a keyframe yet
//   EXPECT_FALSE(frame->isCameraKeyFrame());
//   EXPECT_EQ(lmk->getSeenFrameIds().size(), 0);

//   // Now promote to keyframe
//   map->setCameraKeyFrame(k);

//   EXPECT_TRUE(frame->isCameraKeyFrame());

//   auto seen = lmk->getSeenFrameIds();
//   ASSERT_EQ(seen.size(), 1);
//   EXPECT_EQ(seen[0], k);
// }

// TEST(KeyFrameMap, AddAfterKeyFrameGoesDirectlyToKeyframes) {
//   auto map = KeyFrameMap::create();

//   FrameId k = 1;
//   TrackletId l = 20;
//   Timestamp t = 0.0;

//   auto m = cameraTrackMeasurement(k, t, l, background_label);
//   map->updateObservations(m);
//   map->setCameraKeyFrame(k);

//   auto lmk = map->asLandmarkInterface()->getLandmark(l);

//   auto seen = lmk->getSeenFrameIds();

//   ASSERT_EQ(seen.size(), 1);
//   EXPECT_EQ(seen[0], k);
// }

// TEST(KeyFrameMap, DynamicLandmarksIgnoreCameraKeyframeFiltering) {
//   auto map = KeyFrameMap::create();

//   FrameId k1 = 1, k2 = 2;
//   TrackletId l = 40;
//   ObjectId j = 5;
//   Timestamp t = 0.0;

//   map->updateObservations(cameraTrackMeasurement(k1, t, l, j));
//   map->updateObservations(cameraTrackMeasurement(k2, t, l, j));

//   map->setCameraKeyFrame(k1);

//   auto lmk = map->asLandmarkInterface()->getLandmark(l);

//   auto seen = lmk->getSeenFrameIds();

//   EXPECT_EQ(seen.size(), 2);
// }

// TEST(KeyFrameMap, SettingKeyFrameTwiceIsSafe) {
//   auto map = KeyFrameMap::create();

//   FrameId k = 1;
//   TrackletId l = 50;
//   Timestamp t = 0.0;

//   map->updateObservations(cameraTrackMeasurement(k, t, l, background_label));

//   map->setCameraKeyFrame(k);
//   map->setCameraKeyFrame(k);

//   auto lm = map->asLandmarkInterface()->getLandmark(l);

//   EXPECT_EQ(lm->getSeenFrameIds().size(), 1);
// }

// TEST(KeyFrameMap, LandmarkKeyFrameSubsetOfAllFrames) {
//   auto map = KeyFrameMap::create();

//   FrameId k1 = 1, k2 = 2;
//   TrackletId l = 60;
//   Timestamp t = 0.0;

//   map->updateObservations(cameraTrackMeasurement(k1, t, l,
//   background_label)); map->updateObservations(cameraTrackMeasurement(k2, t,
//   l, background_label));

//   map->setCameraKeyFrame(k2);

//   auto lmk = map->asLandmarkInterface()->getLandmark(l);

//   // full set (unfiltered)
//   auto all_frames = lmk->getAllSeenFrames();

//   // filtered set (keyframes only)
//   auto keyframes = lmk->getSeenFrames();

//   // --- actual subset test ---
//   EXPECT_TRUE(all_frames.exists(k1));
//   EXPECT_TRUE(all_frames.exists(k2));

//   EXPECT_FALSE(keyframes.exists(k1));
//   EXPECT_TRUE(keyframes.exists(k2));
// }
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <exception>

#include "dynosam/formulations/KeyFrameHybridMap.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/GtsamUtils.hpp"
#include "dynosam_opt/Map.hpp"
#include "internal/helpers.hpp"

using namespace dyno;

// --------------------------------------------------------
// Helper
// --------------------------------------------------------

GenericValueTrack<CameraMeasurement> cameraTrackMeasurement(
    FrameId frame_id, Timestamp timestamp, TrackletId tracklet_id,
    ObjectId object_id) {
  return GenericValueTrack<CameraMeasurement>(CameraMeasurement(Keypoint()),
                                              frame_id, timestamp, tracklet_id,
                                              object_id, ReferenceFrame::LOCAL);
}

// --------------------------------------------------------
// Basic KeyFrame Queries
// --------------------------------------------------------

TEST(KeyFrameMap, SetAndQueryCameraKeyFrame) {
  auto map = KeyFrameMap::create();

  FrameId k = 1;
  Timestamp t = 0.1;

  map->updateObservations(cameraTrackMeasurement(k, t, 1, background_label));

  EXPECT_FALSE(map->isCameraKeyFrame(k));

  map->setCameraKeyFrame(k);

  EXPECT_TRUE(map->isCameraKeyFrame(k));
  EXPECT_FALSE(map->isCameraKeyFrame(k + 1));
  EXPECT_TRUE(map->isAnyKeyFrame(k));
}

TEST(KeyFrameMap, SetAndQueryObjectKeyFrame) {
  auto map = KeyFrameMap::create();

  FrameId k = 1;
  ObjectId j = 42;

  map->updateObservations(cameraTrackMeasurement(k, 0.0, 1, j));

  map->setObjectKeyFrame(k, j);

  EXPECT_TRUE(map->isObjectKeyFrame(k, j));
  EXPECT_FALSE(map->isObjectKeyFrame(k, j + 1));
  EXPECT_FALSE(map->isObjectKeyFrame(k + 1, j));
  EXPECT_FALSE(map->isCameraKeyFrame(k));
  EXPECT_TRUE(map->isAnyKeyFrame(k));
}

// --------------------------------------------------------
// Static Landmark Behaviour
// --------------------------------------------------------

TEST(KeyFrameMap, StaticLandmarkOnlyUsesCameraKeyFrames) {
  auto map = KeyFrameMap::create();

  FrameId k1 = 1, k2 = 2;
  Timestamp t = 0.0;
  TrackletId l = 10;

  map->updateObservations(cameraTrackMeasurement(k1, t, l, background_label));
  map->updateObservations(cameraTrackMeasurement(k2, t, l, background_label));

  map->setCameraKeyFrame(k2);

  auto lmk = map->asLandmarkInterface()->getLandmark(l);
  EXPECT_TRUE(lmk->isStatic());

  auto seen = lmk->getSeenFrameIds();

  EXPECT_EQ(seen.size(), 1);
  EXPECT_EQ(seen[0], k2);

  map->setCameraKeyFrame(k1);

  seen = lmk->getSeenFrameIds();
  EXPECT_EQ(seen, FrameIds({k1, k2}));
}

TEST(KeyFrameMap, RetroactiveCameraKeyFrameUpdatesStaticLandmark) {
  auto map = KeyFrameMap::create();

  FrameId k = 1;
  Timestamp t = 0.0;
  TrackletId l = 11;

  map->updateObservations(cameraTrackMeasurement(k, t, l, background_label));

  auto lmk = map->asLandmarkInterface()->getLandmark(l);

  EXPECT_EQ(lmk->getSeenFrameIds().size(), 0);

  map->setCameraKeyFrame(k);

  EXPECT_EQ(lmk->getSeenFrameIds(), FrameIds({k}));
}

TEST(KeyFrameMap, StaticLandmarkSubsetInvariant) {
  auto map = KeyFrameMap::create();

  FrameId k1 = 1, k2 = 2;
  Timestamp t = 0.0;
  TrackletId l = 12;

  map->updateObservations(cameraTrackMeasurement(k1, t, l, background_label));
  map->updateObservations(cameraTrackMeasurement(k2, t, l, background_label));

  map->setCameraKeyFrame(k2);

  auto lmk = map->asLandmarkInterface()->getLandmark(l);

  auto all = lmk->getAllSeenFrames();
  auto filtered = lmk->getSeenFrames();

  EXPECT_TRUE(all.exists(k1));
  EXPECT_TRUE(all.exists(k2));

  EXPECT_FALSE(filtered.exists(k1));
  EXPECT_TRUE(filtered.exists(k2));
}

// --------------------------------------------------------
// Dynamic Landmark Behaviour (Object KeyFrames)
// --------------------------------------------------------

TEST(KeyFrameMap, DynamicLandmarkUsesObjectKeyFrames) {
  auto map = KeyFrameMap::create();

  FrameId k1 = 1, k2 = 2;
  ObjectId j = 5;
  Timestamp t = 0.0;
  TrackletId l = 20;

  map->updateObservations(cameraTrackMeasurement(k1, t, l, j));
  map->updateObservations(cameraTrackMeasurement(k2, t, l, j));

  auto lmk = map->asLandmarkInterface()->getLandmark(l);

  EXPECT_EQ(lmk->getSeenFrameIds().size(), 0);

  map->setObjectKeyFrame(k1, j);
  EXPECT_EQ(lmk->getSeenFrameIds(), FrameIds({k1}));

  map->setObjectKeyFrame(k2, j);
  EXPECT_EQ(lmk->getSeenFrameIds(), FrameIds({k1, k2}));
}

TEST(KeyFrameMap, RetroactiveObjectKeyFrameUpdatesLandmark) {
  auto map = KeyFrameMap::create();

  FrameId k = 1;
  Timestamp t = 0.0;
  TrackletId l = 21;
  ObjectId j = 7;

  map->updateObservations(cameraTrackMeasurement(k, t, l, j));

  auto lmk = map->asLandmarkInterface()->getLandmark(l);
  EXPECT_EQ(lmk->getSeenFrameIds().size(), 0);

  map->setObjectKeyFrame(k, j);

  EXPECT_EQ(lmk->getSeenFrameIds(), FrameIds({k}));
}

TEST(KeyFrameMap, DynamicLandmarkIgnoresCameraKeyFrames) {
  auto map = KeyFrameMap::create();

  FrameId k1 = 1, k2 = 2;
  Timestamp t = 0.0;
  TrackletId l = 22;
  ObjectId j = 3;

  map->updateObservations(cameraTrackMeasurement(k1, t, l, j));
  map->updateObservations(cameraTrackMeasurement(k2, t, l, j));

  map->setCameraKeyFrame(k1);

  auto lmk = map->asLandmarkInterface()->getLandmark(l);

  EXPECT_EQ(lmk->getSeenFrameIds().size(), 0);
}

TEST(KeyFrameMap, DynamicLandmarkSubsetInvariant) {
  auto map = KeyFrameMap::create();

  FrameId k1 = 1, k2 = 2;
  Timestamp t = 0.0;
  TrackletId l = 23;
  ObjectId j = 4;

  map->updateObservations(cameraTrackMeasurement(k1, t, l, j));
  map->updateObservations(cameraTrackMeasurement(k2, t, l, j));

  map->setObjectKeyFrame(k2, j);

  auto lmk = map->asLandmarkInterface()->getLandmark(l);

  auto all = lmk->getAllSeenFrames();
  auto filtered = lmk->getSeenFrames();

  EXPECT_TRUE(all.exists(k1));
  EXPECT_TRUE(all.exists(k2));

  EXPECT_FALSE(filtered.exists(k1));
  EXPECT_TRUE(filtered.exists(k2));
}

// --------------------------------------------------------
// Object-Level Behaviour
// --------------------------------------------------------

TEST(KeyFrameMap, ObjectSeenFramesRespectObjectKeyFrames) {
  auto map = KeyFrameMap::create();

  FrameId k1 = 1, k2 = 2;
  Timestamp t = 0.0;
  ObjectId j = 6;
  TrackletId l = 30;

  map->updateObservations(cameraTrackMeasurement(k1, t, l, j));
  map->updateObservations(cameraTrackMeasurement(k2, t, l, j));

  auto obj = map->asObjectInterface()->getObject(j);

  EXPECT_EQ(obj->getSeenFrameIds().size(), 0);

  map->setObjectKeyFrame(k2, j);

  EXPECT_EQ(obj->getSeenFrameIds(), FrameIds({k2}));
}

// --------------------------------------------------------
// Isolation Tests
// --------------------------------------------------------

TEST(KeyFrameMap, StaticAndDynamicFilteringAreIndependent) {
  auto map = KeyFrameMap::create();

  FrameId k1 = 1, k2 = 2;
  Timestamp t = 0.0;

  map->updateObservations(cameraTrackMeasurement(k1, t, 1, background_label));
  map->updateObservations(cameraTrackMeasurement(k2, t, 1, background_label));

  map->updateObservations(cameraTrackMeasurement(k1, t, 2, 10));
  map->updateObservations(cameraTrackMeasurement(k2, t, 2, 10));

  map->setCameraKeyFrame(k2);
  map->setObjectKeyFrame(k1, 10);

  auto l_static = map->asLandmarkInterface()->getLandmark(1);
  auto l_dyn = map->asLandmarkInterface()->getLandmark(2);

  EXPECT_EQ(l_static->getSeenFrameIds(), FrameIds({k2}));
  EXPECT_EQ(l_dyn->getSeenFrameIds(), FrameIds({k1}));
}

TEST(KeyFrameMap, ObjectKeyFramesArePerObject) {
  auto map = KeyFrameMap::create();

  FrameId k = 1;
  Timestamp t = 0.0;

  map->updateObservations(cameraTrackMeasurement(k, t, 1, 10));
  map->updateObservations(cameraTrackMeasurement(k, t, 2, 20));

  map->setObjectKeyFrame(k, 10);

  auto l1 = map->asLandmarkInterface()->getLandmark(1);
  auto l2 = map->asLandmarkInterface()->getLandmark(2);

  EXPECT_EQ(l1->getSeenFrameIds(), FrameIds({k}));
  EXPECT_EQ(l2->getSeenFrameIds().size(), 0);
}

// --------------------------------------------------------
// Robustness / Idempotence
// --------------------------------------------------------

TEST(KeyFrameMap, SettingCameraKeyFrameTwiceIsSafe) {
  auto map = KeyFrameMap::create();

  FrameId k = 1;
  Timestamp t = 0.0;

  map->updateObservations(cameraTrackMeasurement(k, t, 1, background_label));

  map->setCameraKeyFrame(k);
  map->setCameraKeyFrame(k);

  auto lmk = map->asLandmarkInterface()->getLandmark(1);

  EXPECT_EQ(lmk->getSeenFrameIds().size(), 1);
}

TEST(KeyFrameMap, SettingObjectKeyFrameTwiceIsSafe) {
  auto map = KeyFrameMap::create();

  FrameId k = 1;
  Timestamp t = 0.0;
  ObjectId j = 5;

  map->updateObservations(cameraTrackMeasurement(k, t, 1, j));

  map->setObjectKeyFrame(k, j);
  map->setObjectKeyFrame(k, j);

  auto lmk = map->asLandmarkInterface()->getLandmark(1);

  EXPECT_EQ(lmk->getSeenFrameIds().size(), 1);
}

TEST(KeyFrameMap, StaticLandmarkAddedAfterExistingKeyFrameAppears) {
  auto map = KeyFrameMap::create();

  FrameId k = 1;
  Timestamp t = 0.0;
  TrackletId l = 100;

  // First create frame via dummy measurement (different tracklet)
  map->updateObservations(cameraTrackMeasurement(k, t, 999, background_label));

  // Promote to keyframe BEFORE landmark exists
  map->setCameraKeyFrame(k);

  // Now add the actual landmark
  map->updateObservations(cameraTrackMeasurement(k, t, l, background_label));

  auto lmk = map->asLandmarkInterface()->getLandmark(l);

  auto seen = lmk->getSeenFrameIds();

  ASSERT_EQ(seen.size(), 1);
  EXPECT_EQ(seen[0], k);
}

TEST(KeyFrameMap, StaticLandmarkAddedToNonKeyFrameDoesNotAppear) {
  auto map = KeyFrameMap::create();

  FrameId k = 1;
  Timestamp t = 0.0;
  TrackletId l = 101;

  map->updateObservations(cameraTrackMeasurement(k, t, l, background_label));

  auto lmk = map->asLandmarkInterface()->getLandmark(l);

  EXPECT_TRUE(lmk->getSeenFrameIds().empty());
}

TEST(KeyFrameMap, StaticLandmarkMixedKeyframeInsertion) {
  auto map = KeyFrameMap::create();

  FrameId k1 = 1, k2 = 2, k3 = 3;
  Timestamp t = 0.0;
  TrackletId l = 102;

  // Create all frames first
  map->updateObservations(cameraTrackMeasurement(k1, t, 999, background_label));
  map->updateObservations(cameraTrackMeasurement(k2, t, 999, background_label));
  map->updateObservations(cameraTrackMeasurement(k3, t, 999, background_label));

  // Mark only k2 as keyframe
  map->setCameraKeyFrame(k2);

  // Now add landmark to ALL frames
  map->updateObservations(cameraTrackMeasurement(k1, t, l, background_label));
  map->updateObservations(cameraTrackMeasurement(k2, t, l, background_label));
  map->updateObservations(cameraTrackMeasurement(k3, t, l, background_label));

  auto lmk = map->asLandmarkInterface()->getLandmark(l);

  auto seen = lmk->getSeenFrameIds();

  ASSERT_EQ(seen.size(), 1);
  EXPECT_EQ(seen[0], k2);
}

TEST(KeyFrameMap, StaticLandmarkOrderIndependenceStrong) {
  auto map1 = KeyFrameMap::create();
  auto map2 = KeyFrameMap::create();

  FrameId k = 1;
  Timestamp t = 0.0;
  TrackletId l = 103;

  // CASE 1: add then keyframe
  map1->updateObservations(cameraTrackMeasurement(k, t, l, background_label));
  map1->setCameraKeyFrame(k);

  // CASE 2: keyframe then add
  map2->updateObservations(cameraTrackMeasurement(k, t, 999, background_label));
  map2->setCameraKeyFrame(k);
  map2->updateObservations(cameraTrackMeasurement(k, t, l, background_label));

  auto seen1 = map1->asLandmarkInterface()->getLandmark(l)->getSeenFrameIds();
  auto seen2 = map2->asLandmarkInterface()->getLandmark(l)->getSeenFrameIds();

  EXPECT_EQ(seen1, seen2);
}

TEST(KeyFrameMap, DynamicLandmarkAddedAfterObjectKeyFrameAppears) {
  auto map = KeyFrameMap::create();

  FrameId k = 1;
  ObjectId j = 5;
  Timestamp t = 0.0;
  TrackletId l = 200;

  // Create frame first
  map->updateObservations(cameraTrackMeasurement(k, t, 999, j));

  // Set object keyframe BEFORE adding landmark
  map->setObjectKeyFrame(k, j);

  // Add landmark
  map->updateObservations(cameraTrackMeasurement(k, t, l, j));

  auto lmk = map->asLandmarkInterface()->getLandmark(l);

  auto seen = lmk->getSeenFrameIds();

  ASSERT_EQ(seen.size(), 1);
  EXPECT_EQ(seen[0], k);
}

TEST(KeyFrameMap, DynamicLandmarkWithoutObjectKeyFrameDoesNotAppear) {
  auto map = KeyFrameMap::create();

  FrameId k = 1;
  ObjectId j = 6;
  Timestamp t = 0.0;
  TrackletId l = 201;

  map->updateObservations(cameraTrackMeasurement(k, t, l, j));

  auto lmk = map->asLandmarkInterface()->getLandmark(l);

  EXPECT_TRUE(lmk->getSeenFrameIds().empty());
}

TEST(KeyFrameMap, DynamicLandmarkMixedObjectKeyframes) {
  auto map = KeyFrameMap::create();

  FrameId k1 = 1, k2 = 2;
  ObjectId j = 7;
  Timestamp t = 0.0;
  TrackletId l = 202;

  map->updateObservations(cameraTrackMeasurement(k1, t, l, j));
  map->updateObservations(cameraTrackMeasurement(k2, t, l, j));

  map->setObjectKeyFrame(k2, j);

  auto lmk = map->asLandmarkInterface()->getLandmark(l);

  auto seen = lmk->getSeenFrameIds();

  ASSERT_EQ(seen.size(), 1);
  EXPECT_EQ(seen[0], k2);
}

// --------------------------------------------------------
// Property-Based / Randomized Tests
// --------------------------------------------------------

#include <random>

// Utility RNG
struct TestRNG {
  std::mt19937 gen;
  TestRNG() : gen(0) {}  // deterministic seed

  int uniformInt(int a, int b) {
    std::uniform_int_distribution<> d(a, b);
    return d(gen);
  }

  bool coinFlip(double p = 0.5) {
    std::bernoulli_distribution d(p);
    return d(gen);
  }
};

struct ConsistentTrackletGenerator : public TestRNG {
  std::unordered_map<TrackletId, ObjectId> tracklet_to_object;

  ObjectId getObjectId(TrackletId l) {
    auto it = tracklet_to_object.find(l);
    if (it != tracklet_to_object.end()) {
      return it->second;
    }

    ObjectId j = this->coinFlip() ? background_label : this->uniformInt(1, 3);

    tracklet_to_object[l] = j;
    return j;
  }
};

// --------------------------------------------------------
// Property: Filtered frames are always subset of all frames
// --------------------------------------------------------

// TEST(KeyFrameMapProperty, FilteredIsSubsetOfAllFrames) {
//   ConsistentTrackletGenerator gen;

//   for (int trial = 0; trial < 50; ++trial) {
//     auto map = KeyFrameMap::create();

//     const int num_frames = gen.uniformInt(3, 10);
//     const int num_landmarks = gen.uniformInt(3, 10);

//     Timestamp t = 0.0;

//     // random measurements
//     for (int i = 0; i < num_landmarks; ++i) {
//       FrameId k = gen.uniformInt(0, 5);
//       TrackletId l = gen.uniformInt(0, 5);
//       ObjectId j = gen.getObjectId(l);

//       for (int k = 0; k < num_frames; ++k) {
//         if (gen.coinFlip(0.7)) {
//           map->updateObservations(cameraTrackMeasurement(k, t, l, j));
//         }
//       }
//     }

//     // random keyframes
//     for (int k = 0; k < num_frames; ++k) {
//       if (gen.coinFlip(0.5)) map->setCameraKeyFrame(k);

//       for (int j = 1; j <= 3; ++j) {
//         if (gen.coinFlip(0.3)) {
//           map->setObjectKeyFrame(k, j);
//         }
//       }
//     }

//     // check invariant
//     auto li = map->asLandmarkInterface();

//     for (auto [key, lmk] : li->getLandmarks()) {
//       // auto lmk = li->getLandmark(key);

//       auto all = lmk->getAllSeenFrames();
//       auto filtered = lmk->getSeenFrames();

//       for (auto f_id : filtered.collectKeys()) {
//         EXPECT_TRUE(all.exists(f_id));
//       }
//     }
//   }
// }

// // --------------------------------------------------------
// // Property: Order independence (shuffle updates)
// // --------------------------------------------------------

// TEST(KeyFrameMapProperty, OrderIndependence) {
//   ConsistentTrackletGenerator gen;

//   for (int trial = 0; trial < 50; ++trial) {
//     std::vector<GenericValueTrack<CameraMeasurement>> measurements;

//     Timestamp t = 0.0;

//     for (int i = 0; i < 10; ++i) {
//       FrameId k = gen.uniformInt(0, 5);
//       TrackletId l = gen.uniformInt(0, 5);
//       ObjectId j = gen.getObjectId(l);

//       measurements.push_back(cameraTrackMeasurement(k, t, l, j));
//     }

//     auto map1 = KeyFrameMap::create();
//     auto map2 = KeyFrameMap::create();

//     // original order
//     for (auto& m : measurements) map1->updateObservations(m);

//     // shuffled order
//     std::shuffle(measurements.begin(), measurements.end(), gen.gen);
//     for (auto& m : measurements) map2->updateObservations(m);

//     // same keyframe assignment
//     for (int k = 0; k < 6; ++k) {
//       if (gen.coinFlip(0.5)) {
//         map1->setCameraKeyFrame(k);
//         map2->setCameraKeyFrame(k);
//       }

//       for (int j = 1; j <= 3; ++j) {
//         if (gen.coinFlip(0.3)) {
//           map1->setObjectKeyFrame(k, j);
//           map2->setObjectKeyFrame(k, j);
//         }
//       }
//     }

//     auto li1 = map1->asLandmarkInterface();
//     auto li2 = map2->asLandmarkInterface();

//     for (auto [key, lmk] : li1->getLandmarks()) {
//       auto l1 = li1->getLandmark(key);
//       auto l2 = li2->getLandmark(key);

//       EXPECT_EQ(l1->getSeenFrameIds(), l2->getSeenFrameIds());
//     }
//   }
// }

// // --------------------------------------------------------
// // Property: Adding keyframes only increases visibility (monotonic)
// // --------------------------------------------------------

// TEST(KeyFrameMapProperty, KeyFrameMonotonicity) {
//   TestRNG rng;

//   for (int trial = 0; trial < 50; ++trial) {
//     auto map = KeyFrameMap::create();

//     Timestamp t = 0.0;

//     for (int k = 0; k < 5; ++k) {
//       map->updateObservations(cameraTrackMeasurement(k, t, k,
//       background_label));
//     }

//     auto lmk = map->asLandmarkInterface()->getLandmark(0);

//     size_t prev_size = 0;

//     for (int k = 0; k < 5; ++k) {
//       map->setCameraKeyFrame(k);

//       auto seen = lmk->getSeenFrameIds();

//       EXPECT_GE(seen.size(), prev_size);
//       prev_size = seen.size();
//     }
//   }
// }

// // --------------------------------------------------------
// // Property: Object keyframes only affect matching object
// // --------------------------------------------------------

// TEST(KeyFrameMapProperty, ObjectIsolationInvariant) {
//   TestRNG rng;

//   for (int trial = 0; trial < 50; ++trial) {
//     auto map = KeyFrameMap::create();

//     Timestamp t = 0.0;

//     // two objects
//     map->updateObservations(cameraTrackMeasurement(0, t, 1, 10));
//     map->updateObservations(cameraTrackMeasurement(0, t, 2, 20));

//     map->setObjectKeyFrame(0, 10);

//     auto l1 = map->asLandmarkInterface()->getLandmark(1);
//     auto l2 = map->asLandmarkInterface()->getLandmark(2);

//     EXPECT_EQ(l1->getSeenFrameIds(), FrameIds({0}));
//     EXPECT_EQ(l2->getSeenFrameIds().size(), 0);
//   }
// }

// // --------------------------------------------------------
// // Property: No duplication in filtered sets
// // --------------------------------------------------------

// TEST(KeyFrameMapProperty, NoDuplicateFrames) {
//   auto map = KeyFrameMap::create();

//   Timestamp t = 0.0;

//   for (int i = 0; i < 5; ++i) {
//     map->updateObservations(cameraTrackMeasurement(0, t, 1,
//     background_label));
//   }

//   map->setCameraKeyFrame(0);

//   auto lmk = map->asLandmarkInterface()->getLandmark(1);

//   auto seen = lmk->getSeenFrameIds();

//   EXPECT_EQ(seen.size(), 1);
// }
