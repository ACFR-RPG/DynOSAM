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

#include "dynosam/formulations/KeyFrameHybridEstimator.hpp"
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

gtsam::Pose3 makePose(double rx, double ry, double rz, double tx, double ty,
                      double tz) {
  gtsam::Rot3 R = gtsam::Rot3::RzRyRx(rx, ry, rz);
  gtsam::Point3 t(tx, ty, tz);
  return gtsam::Pose3(R, t);
}
using namespace gtsam;

TEST(BetweenMotionsWithRelativeTransform, ZeroError) {
  Pose3 Lo = makePose(0.1, -0.05, 0.03, 1.0, 2.0, -0.3);

  Pose3 oHi = makePose(0.2, 0.1, -0.1, 0.5, -1.0, 2.0);

  Pose3 oHj = makePose(-0.1, 0.05, 0.2, 1.5, 0.3, 2.5);

  Pose3 Li = oHi.compose(Lo);
  Pose3 Lj = oHj.compose(Lo);

  Pose3 T_meas = Li.between(Lj);

  auto noise = noiseModel::Diagonal::Sigmas(Vector6::Ones());

  BetweenMotionsWithRelativeTransform factor(Symbol('x', 0), Symbol('x', 1),
                                             T_meas, Lo, noise);

  Vector error = factor.evaluateError(oHi, oHj);

  EXPECT_TRUE(assert_equal(Vector6::Zero(), error, 1e-9));
}

TEST(BetweenMotionsWithRelativeTransform, NonZeroError) {
  Pose3 Lo = makePose(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);

  Pose3 oHi = makePose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

  Pose3 oHj = makePose(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);

  Pose3 wrong_measurement = makePose(0.0, 0.0, 0.0, 5.0, 0.0, 0.0);

  auto noise = noiseModel::Diagonal::Sigmas(Vector6::Ones());

  BetweenMotionsWithRelativeTransform factor(Symbol('x', 0), Symbol('x', 1),
                                             wrong_measurement, Lo, noise);

  Vector error = factor.evaluateError(oHi, oHj);

  EXPECT_GT(error.norm(), 1e-3);
}

TEST(BetweenMotionsWithRelativeTransform, JacobianWrtFirstPose) {
  Pose3 Lo = makePose(0.1, -0.2, 0.05, 0.4, -0.3, 1.2);

  Pose3 oHi = makePose(0.2, 0.1, -0.3, 1.0, 2.0, 3.0);

  Pose3 oHj = makePose(-0.1, 0.4, 0.2, -2.0, 0.5, 1.0);

  Pose3 T_meas = oHi.compose(Lo).between(oHj.compose(Lo));

  auto noise = noiseModel::Diagonal::Sigmas(Vector6::Ones());

  BetweenMotionsWithRelativeTransform factor(Symbol('x', 0), Symbol('x', 1),
                                             T_meas, Lo, noise);

  Matrix H1_analytic;
  Matrix H2_dummy;

  factor.evaluateError(oHi, oHj, H1_analytic, H2_dummy);

  Matrix H1_numeric = numericalDerivative11<Vector, Pose3>(
      [&](const Pose3& x1) { return factor.evaluateError(x1, oHj); }, oHi,
      1e-6);

  EXPECT_TRUE(assert_equal(H1_numeric, H1_analytic, 1e-6));
}

TEST(BetweenMotionsWithRelativeTransform, JacobianWrtSecondPose) {
  Pose3 Lo = makePose(-0.1, 0.15, 0.07, 0.8, 1.2, -0.4);

  Pose3 oHi = makePose(0.3, -0.1, 0.2, -1.0, 2.5, 0.3);

  Pose3 oHj = makePose(-0.2, 0.5, -0.4, 1.2, -0.7, 2.2);

  Pose3 T_meas = oHi.compose(Lo).between(oHj.compose(Lo));

  auto noise = noiseModel::Diagonal::Sigmas(Vector6::Ones());

  BetweenMotionsWithRelativeTransform factor(Symbol('x', 0), Symbol('x', 1),
                                             T_meas, Lo, noise);

  Matrix H1_dummy;
  Matrix H2_analytic;

  factor.evaluateError(oHi, oHj, H1_dummy, H2_analytic);

  Matrix H2_numeric = numericalDerivative11<Vector, Pose3>(
      [&](const Pose3& x2) { return factor.evaluateError(oHi, x2); }, oHj,
      1e-6);

  EXPECT_TRUE(assert_equal(H2_numeric, H2_analytic, 1e-6));
}

TEST(BetweenMotionsWithRelativeTransform, BothJacobians) {
  Pose3 Lo = makePose(0.12, -0.08, 0.03, 0.5, 0.2, -1.0);

  Pose3 oHi = makePose(0.4, -0.2, 0.1, 1.0, -1.0, 0.5);

  Pose3 oHj = makePose(-0.3, 0.25, -0.15, -0.5, 2.0, 1.5);

  Pose3 T_meas = oHi.compose(Lo).between(oHj.compose(Lo));

  auto noise = noiseModel::Diagonal::Sigmas(Vector6::Ones());

  BetweenMotionsWithRelativeTransform factor(Symbol('x', 0), Symbol('x', 1),
                                             T_meas, Lo, noise);

  Matrix H1_analytic;
  Matrix H2_analytic;

  factor.evaluateError(oHi, oHj, H1_analytic, H2_analytic);

  Matrix H1_numeric = numericalDerivative11<Vector, Pose3>(
      [&](const Pose3& x1) { return factor.evaluateError(x1, oHj); }, oHi,
      1e-6);

  Matrix H2_numeric = numericalDerivative11<Vector, Pose3>(
      [&](const Pose3& x2) { return factor.evaluateError(oHi, x2); }, oHj,
      1e-6);

  EXPECT_TRUE(assert_equal(H1_numeric, H1_analytic, 1e-6));

  EXPECT_TRUE(assert_equal(H2_numeric, H2_analytic, 1e-6));
}

TEST(SmartMotionFactor, testZeroErrorWithIdenties) {
  using SmartFactor = dyno::SmartMotionFactor<3, gtsam::Pose3, gtsam::Pose3>;

  gtsam::Pose3 pose = gtsam::Pose3::Identity();
  gtsam::Pose3 motion = gtsam::Pose3::Identity();
  gtsam::Pose3 L_e = gtsam::Pose3::Identity();
  gtsam::Point3 point(1.0, 2.0, 3.0);
  gtsam::Point3 noise(0.0, 0.0, 0.0);
  gtsam::Point3 measured = point + noise;

  gtsam::Key pose_key(1);
  gtsam::Key motion_key(2);
  gtsam::Values values;
  values.insert(pose_key, pose);
  values.insert(motion_key, motion);

  SmartFactor factor(L_e, gtsam::noiseModel::Isotropic::Sigma(3, 0.05),
                     SmartMotionFactorParams{}, point);
  factor.add(measured, motion_key, pose_key);

  gtsam::Vector expectedError = gtsam::Vector3(0.0, 0.0, 0);
  gtsam::Vector actualReproError = factor.reprojectionError(values);
  // Vector actualError = factor.unwhitenedError(values);
  EXPECT_TRUE(gtsam::assert_equal(expectedError, actualReproError, 1E-5));
}

TEST(SmartMotionFactor, testZeroErrorWithL0AndCamera) {
  using SmartFactor = dyno::SmartMotionFactor<3, gtsam::Pose3, gtsam::Pose3>;

  gtsam::Pose3 pose(gtsam::Rot3::Rodrigues(-0.1, 0.2, 0.25),
                    gtsam::Point3(0.05, -0.10, 0.20));
  gtsam::Pose3 motion = gtsam::Pose3::Identity();
  gtsam::Pose3 L_e = gtsam::Pose3(gtsam::Rot3::Ypr(-M_PI / 10, 0., -M_PI / 10),
                                  gtsam::Point3(0.5, 0.1, 0.3));
  gtsam::Point3 point_l(1.0, 2.0, 3.0);
  gtsam::Point3 noise(0.0, 0.0, 0.0);
  gtsam::Point3 measured_c = pose.inverse() * L_e * (point_l + noise);

  gtsam::Key pose_key(1);
  gtsam::Key motion_key(2);
  gtsam::Values values;
  values.insert(pose_key, pose);
  values.insert(motion_key, motion);

  SmartFactor factor(L_e, gtsam::noiseModel::Isotropic::Sigma(3, 0.05),
                     SmartMotionFactorParams{}, point_l);
  factor.add(measured_c, motion_key, pose_key);

  gtsam::Vector expectedError = gtsam::Vector3(0.0, 0.0, 0);
  gtsam::Vector actualReproError = factor.reprojectionError(values);
  // Vector actualError = factor.unwhitenedError(values);
  EXPECT_TRUE(gtsam::assert_equal(expectedError, actualReproError, 1E-5));
}

TEST(SmartMotionFactor, testNoiseWithIdentity) {
  using SmartFactor = dyno::SmartMotionFactor<3, gtsam::Pose3, gtsam::Pose3>;

  gtsam::Pose3 pose = gtsam::Pose3::Identity();
  gtsam::Pose3 motion = gtsam::Pose3::Identity();
  gtsam::Pose3 L_e = gtsam::Pose3::Identity();
  gtsam::Point3 point_l(1.0, 2.0, 3.0);
  gtsam::Point3 noise(0.4, 1.0, 2.0);
  gtsam::Point3 measured_c = pose.inverse() * L_e * (point_l + noise);

  gtsam::Key pose_key(1);
  gtsam::Key motion_key(2);
  gtsam::Values values;
  values.insert(pose_key, pose);
  values.insert(motion_key, motion);

  SmartFactor factor(L_e, gtsam::noiseModel::Isotropic::Sigma(3, 0.05),
                     SmartMotionFactorParams{}, point_l);
  factor.add(measured_c, motion_key, pose_key);

  gtsam::Vector expectedError = -noise;
  gtsam::Vector actualReproError = factor.reprojectionError(values);
  // Vector actualError = factor.unwhitenedError(values);
  EXPECT_TRUE(gtsam::assert_equal(expectedError, actualReproError, 1E-5));
}

TEST(SmartMotionFactor, testBasicSchurCompliment) {
  using SmartFactor = dyno::SmartMotionFactor<3, gtsam::Pose3, gtsam::Pose3>;

  gtsam::Pose3 pose1 = gtsam::Pose3::Identity();
  gtsam::Pose3 pose2(gtsam::Rot3::Rodrigues(-0.1, 0.2, 0.25),
                     gtsam::Point3(0.05, -0.10, 0.20));
  // gtsam::Pose3 pose(gtsam::Rot3::Identity(), gtsam::Point3(1.4, 0, 0));
  gtsam::Pose3 motion = gtsam::Pose3::Identity();
  gtsam::Pose3 L_e = gtsam::Pose3::Identity();
  gtsam::Point3 point(1.0, 1.4, 2);
  gtsam::Point3 noise(10.0, 0.0, 7.0);
  gtsam::Point3 measured = point + noise;

  // with two sets of emasurements (2 * (1 pose + 1 motion) the number of blocks
  // is 3
  // motion + pose form a single block currently + 1 block for error

  gtsam::Key pose_key(1);
  gtsam::Key motion_key(2);

  gtsam::Key pose_key1(3);
  gtsam::Key motion_key1(4);

  gtsam::Key pose_key2(5);
  gtsam::Key motion_key2(6);
  gtsam::Values values;
  values.insert(pose_key, pose1);
  values.insert(motion_key, motion);
  values.insert(pose_key1, pose2);
  values.insert(motion_key1, motion);
  // values.insert(pose_key2, pose2);
  // values.insert(motion_key2, motion);

  SmartFactor factor(L_e, gtsam::noiseModel::Isotropic::Sigma(3, 0.05),
                     SmartMotionFactorParams{}, point);
  factor.add(measured, motion_key, pose_key);
  factor.add(measured + 2 * noise, motion_key1, pose_key1);
  // factor.add(measured + 3 * noise, motion_key2, pose_key2);

  LOG(INFO) << "createReducedMatrix";

  gtsam::SymmetricBlockMatrix actualReduced =
      factor.createReducedMatrix(values);

  LOG(INFO) << "N blocks " << actualReduced.nBlocks()
            << " rows=" << actualReduced.rows()
            << " cols=" << actualReduced.cols();

  gtsam::Matrix adjoint_view = actualReduced.selfadjointView();
  // LOG(INFO) << "adjoint_view " << adjoint_view;

  SmartFactor::GBlocks Gs;
  SmartFactor::FBlocks Fs;
  SmartFactor::EBlocks Es;

  // not negative...?
  gtsam::Vector b = -factor.reprojectionError(values, Gs, Fs, Es);
  // factor.whitenJacobians(Gs, Fs, Es, b);

  gtsam::Matrix E;
  SmartFactor::EVectorToMatrix(Es, E);
  EXPECT_EQ(E.rows(), 6);
  EXPECT_EQ(E.cols(), 3);

  gtsam::Matrix Et = E.transpose();
  const Eigen::Matrix<double, 3, 3> P = (Et * E).inverse();

  SmartFactor::GFBlocks GFs;
  SmartFactor::GFVectorsToGFBlocks(Gs, Fs, GFs);

  /**
   * for one measurement we should have 1 motion and 1 pose (in that order)
      the A matrix should then iniially look like
          key(h_1)  key(x_1) m
      A = [G1       F1       E]

      with the regular schur compliment we expect
      g = Ft * (b - E * P * Et * b);
      G = Ft * F - Ft * E * P * Et * F
      Schur = G, g, g.transpose(), b.squaredNorm()

      we treat the GF blocks like the F block in the original schur compliment
   *
   */
  EXPECT_EQ(GFs.size(), 2u);
  Eigen::Matrix<double, 3, 12> F1 = GFs.at(0);
  Eigen::Matrix<double, 3, 12> F2 = GFs.at(1);
  Eigen::Matrix<double, 6, 24> F;

  // F is diagonal
  F << F1, Eigen::Matrix<double, 3, 12>::Zero(),
      Eigen::Matrix<double, 3, 12>::Zero(), F2;
  gtsam::Matrix Ft = F.transpose();

  LOG(INFO) << "F=" << F;

  Eigen::Matrix<double, 24, 1> g = Ft * (b - E * P * Et * b);
  Eigen::Matrix<double, 24, 24> G = Ft * F - Ft * E * P * Et * F;

  gtsam::Matrix schur(25, 25);
  schur << G, g, g.transpose(), b.squaredNorm();

  // test programatic construction
  {
    constexpr static auto HDim = 6;
    constexpr static auto XDim = 6;
    constexpr static auto HessianDim = 12;
    size_t m = 2;  // measurements
    std::vector<Eigen::DenseIndex> f_block_dims(2 * m);
    std::fill(f_block_dims.begin(), f_block_dims.end(),
              HDim);  // assuming HDim and Xdim are the same size
    // dims.back() = 1;

    // Make full stock block matrix
    gtsam::Matrix F_block_matrix(m * 3, m * HessianDim);
    F_block_matrix.setZero();
    // gtsam::SymmetricBlockMatrix F_block_matrix(f_block_dims);
    // size_t block_idx = 0;
    for (size_t i = 0; i < m; i++) {
      const Eigen::Matrix<double, 3, HessianDim>& GFblock = GFs.at(i);
      LOG(INFO) << GFblock;

      // Eigen::Matrix<double, 3, HDim> gblock = GFblock.leftCols(HDim);
      // Eigen::Matrix<double, 3, HDim> fblock = GFblock.rightCols(XDim);

      // Eigen::Matrix<double, 3, HessianDim> GF
      // set along diagonals i, j, p,q
      LOG(INFO) << 3 * i << " " << HessianDim * i;
      // F_block_matrix.block( 3*i, HessianDim*i, 3, HessianDim) = GFblock;
      F_block_matrix.block<3, HessianDim>(3 * i, HessianDim * i) = GFblock;

      // F_block_matrix.setDiagonalBlock(2*i, gblock);
      // F_block_matrix.setDiagonalBlock(2*i+1, fblock);
    }

    gtsam::Matrix F = F_block_matrix;
    gtsam::Matrix Ft = F.transpose();
    // LOG(INFO) << "F=" << F;

    gtsam::Matrix g = Ft * (b - E * P * Et * b);
    gtsam::Matrix G = Ft * F - Ft * E * P * Et * F;

    // size of schur = num measurements * Hessian size + 1
    size_t aug_hessian_size = m * HessianDim + 1;
    gtsam::Matrix other_schur(aug_hessian_size, aug_hessian_size);

    other_schur << G, g, g.transpose(), b.squaredNorm();

    EXPECT_TRUE(gtsam::assert_equal(schur, other_schur, 1E-5));

    std::vector<Eigen::DenseIndex> dims(2 * m + 1);  // includes b term
    std::fill(dims.begin(), dims.end() - 1,
              HDim);  // assuming HDim and Xdim are the same size
    dims.back() = 1;

    gtsam::SymmetricBlockMatrix augmented_hessian(dims, other_schur);
    gtsam::RegularHessianFactor<XDim> hessian_factor(factor.keys(),
                                                     augmented_hessian);

    // boost::make_shared<gtsam::RegularHessianFactor<HessianDim>>(
    // this->keys_, augmented_hessian
  }
  // LOG(INFO) << "g Matrix size: " << g.rows() << " x " << g.cols();
  // LOG(INFO) << "G Matrix size: " << G.rows() << " x " << G.cols();
  // LOG(INFO) << "adjoint_view Matrix size: " << adjoint_view.rows() << " x "
  // << adjoint_view.cols();

  // LOG(INFO) << "g:\n " << g;
  // LOG(INFO) << "G:\n " << G;
  // LOG(INFO) << "schur:\n" << schur;
  // LOG(INFO) << "adjoint_view:\n" << adjoint_view;
  EXPECT_TRUE(gtsam::assert_equal(schur, adjoint_view, 1E-5));
}

TEST(SmartMotionFactor, testSimpleOptimise) {
  using SmartFactor = dyno::SmartMotionFactor<3, gtsam::Pose3, gtsam::Pose3>;

  gtsam::Pose3 pose1 = gtsam::Pose3::Identity();
  gtsam::Pose3 pose2(gtsam::Rot3::Rodrigues(-0.1, 0.2, 0.25),
                     gtsam::Point3(0.05, -0.10, 0.20));
  gtsam::Pose3 L_e = utils::createRandomAroundIdentity<gtsam::Pose3>(2.0);
  gtsam::Pose3 motion1 = gtsam::Pose3::Identity();
  gtsam::Pose3 motion2 = utils::createRandomAroundIdentity<gtsam::Pose3>(3.0);
  gtsam::Point3 point1(3.0, 0, 1.2);
  gtsam::Point3 point2(2.0, -5, 3.2);

  double sigma = 0.001;

  gtsam::Key pose_key1(1);
  gtsam::Key motion_key1(2);

  gtsam::Key pose_key2(3);
  gtsam::Key motion_key2(4);

  auto create_smart_factor = [=](const gtsam::Point3& point,
                                 double sigma) -> SmartFactor::shared_ptr {
    // start with actual point ie no linearization error here
    SmartFactor::shared_ptr factor(
        new SmartFactor(L_e, gtsam::noiseModel::Isotropic::Sigma(3, sigma),
                        SmartMotionFactorParams{}, point));

    auto Z1 = HybridObjectMotion::projectToCamera3(pose1, motion1, L_e, point);
    Z1 = utils::perturbWithNoise(Z1, sigma);

    auto Z2 = HybridObjectMotion::projectToCamera3(pose2, motion2, L_e, point);
    Z2 = utils::perturbWithNoise(Z2, sigma);

    factor->add(Z1, motion_key1, pose_key1);
    factor->add(Z2, motion_key2, pose_key2);

    return factor;
  };

  // measurements have explicity noisy since we puerturb both motions and poses
  // however the noisy is not propogated correctly since we do not add sigma
  // noisy to the measurements
  gtsam::Pose3 pose1_peturb = utils::perturbWithNoise(pose1, sigma);
  gtsam::Pose3 pose2_peturb = utils::perturbWithNoise(pose2, sigma);
  gtsam::Pose3 motion1_peturb = utils::perturbWithNoise(motion1, sigma);
  gtsam::Pose3 motion2_peturb = utils::perturbWithNoise(motion2, sigma);

  // gtsam::Point3 measurement1 =
  //     perturbCameraAndMotion(point, L_e, motion1_peturb, pose1_peturb,
  //     sigma);
  // gtsam::Point3 measurement2 =
  //     perturbCameraAndMotion(point, L_e, motion2_peturb, pose2_peturb,
  //     sigma);

  gtsam::Values values;
  values.insert(pose_key1, pose1_peturb);
  values.insert(motion_key1, motion1_peturb);
  values.insert(pose_key2, pose2_peturb);
  values.insert(motion_key2, motion2_peturb);

  // SmartFactor::shared_ptr factor(new SmartFactor(
  //     L_e, gtsam::noiseModel::Isotropic::Sigma(3, sigma), point));

  // factor->add(measurement1, motion_key1, pose_key1);
  // factor->add(measurement2, motion_key2, pose_key2);

  gtsam::NonlinearFactorGraph graph;
  graph.addPrior<gtsam::Pose3>(motion_key1, gtsam::Pose3::Identity(),
                               gtsam::noiseModel::Isotropic::Sigma(6, 1e-5));
  graph.add(create_smart_factor(point1, sigma));
  graph.add(create_smart_factor(point2, sigma));

  gtsam::LevenbergMarquardtParams lmParams;
  lmParams.relativeErrorTol = 1e-8;
  lmParams.absoluteErrorTol = 0;
  lmParams.maxIterations = 20;

  gtsam::Values result;
  values.print("Before ");
  gtsam::LevenbergMarquardtOptimizer optimizer(graph, values, lmParams);
  result = optimizer.optimize();
  result.print("After ");

  EXPECT_TRUE(
      gtsam::assert_equal(result.at<gtsam::Pose3>(pose_key1), pose1, 1E-5));
  EXPECT_TRUE(
      gtsam::assert_equal(result.at<gtsam::Pose3>(pose_key2), pose2, 1E-5));
  EXPECT_TRUE(
      gtsam::assert_equal(result.at<gtsam::Pose3>(motion_key1), motion1, 1E-5));
  EXPECT_TRUE(
      gtsam::assert_equal(result.at<gtsam::Pose3>(motion_key2), motion2, 1E-5));

  LOG(INFO) << graph.error(values);
  LOG(INFO) << graph.error(result);
}
TEST(SmartMotionFactor, computeTriangulationBasic) {
  using SmartFactor = dyno::SmartMotionFactor<3, gtsam::Pose3, gtsam::Pose3>;
  gtsam::Pose3 pose1 = gtsam::Pose3::Identity();
  gtsam::Pose3 pose2(gtsam::Rot3::Rodrigues(-0.1, 0.2, 0.25),
                     gtsam::Point3(0.05, -0.10, 0.20));
  gtsam::Pose3 L_e = utils::createRandomAroundIdentity<gtsam::Pose3>(2.0);
  gtsam::Pose3 motion1 = gtsam::Pose3::Identity();
  gtsam::Pose3 motion2 = utils::createRandomAroundIdentity<gtsam::Pose3>(3.0);
  gtsam::Point3 point(3.0, 0, 1.2);

  // no noise!!!
  const double sigma = 0.00;

  gtsam::Values values;
  gtsam::Key pose_key1(1);
  gtsam::Key motion_key1(2);

  gtsam::Key pose_key2(3);
  gtsam::Key motion_key2(4);

  values.insert(pose_key1, pose1);
  values.insert(motion_key1, motion1);
  values.insert(pose_key2, pose2);
  values.insert(motion_key2, motion2);

  SmartFactor::shared_ptr factor(
      new SmartFactor(L_e, gtsam::noiseModel::Isotropic::Sigma(3, sigma),
                      SmartMotionFactorParams{}));

  auto Z1 = HybridObjectMotion::projectToCamera3(pose1, motion1, L_e, point);
  Z1 = utils::perturbWithNoise(Z1, sigma);

  auto Z2 = HybridObjectMotion::projectToCamera3(pose2, motion2, L_e, point);
  Z2 = utils::perturbWithNoise(Z2, sigma);

  EXPECT_FALSE(factor->point(values));
  EXPECT_FALSE(factor->point());

  // should now be degenerate
  factor->add(Z1, motion_key1, pose_key1);
  EXPECT_FALSE(factor->point(values));
  EXPECT_TRUE(factor->point(values).degenerate());

  factor->add(Z2, motion_key2, pose_key2);

  EXPECT_EQ(factor->numMeasurements(), 2u);

  // given erroneous measurements but accurayte motion/pose
  // actually should only be okay given the params!!
  EXPECT_TRUE(factor->point(values));

  LOG(INFO) << "Triangulated point = " << factor->point();
  EXPECT_TRUE(gtsam::assert_equal(point, factor->point().value(), 1E-5));
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
