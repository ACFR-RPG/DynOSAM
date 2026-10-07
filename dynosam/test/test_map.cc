/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a
 copy
 *   of this software and associated documentation files (the "Software"), to
 deal
 *   in the Software without restriction, including without limitation the
 rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in
 all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE
 *   SOFTWARE.
 */

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <exception>

#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/GtsamUtils.hpp"
#include "dynosam_opt/Map.hpp"
#include "internal/helpers.hpp"

using namespace dyno;

template <typename V>
struct SimpleNode {
  int id;

  SimpleNode(int id_) : id(id_) {}
  int getId() const { return id; }
};

using SimpleFrameNode = SimpleNode<struct F>;

TEST(Map, SharedNodeSet) {
  using Traits = NodeTraits<int, SimpleFrameNode>;
  using SharedNodeSet = Traits::SharedNodeSet;

  SharedNodeSet set;
  set.insert(std::make_shared<SimpleFrameNode>(0));
  EXPECT_EQ(set.size(), 1);
  EXPECT_EQ(set.collectKeys(), std::vector<int>({0}));

  set.insert(std::make_shared<SimpleFrameNode>(1));
  EXPECT_EQ(set.size(), 2);
  EXPECT_EQ(set.collectKeys(), std::vector<int>({0, 1}));

  set.insert(std::make_shared<SimpleFrameNode>(1));
  EXPECT_EQ(set.size(), 2);
  EXPECT_EQ(set.collectKeys(), std::vector<int>({0, 1}));
}

TEST(Map, basicAddOnlyStatic) {
  //   GenericTrackedStatusVector<VisualMeasurementStatus<Keypoint>>
  //   measurements;

  StatusKeypointVector measurements;

  TrackletIds expected_tracklets;
  // 10 measurements with unique tracklets at frame 0
  for (size_t i = 0; i < 10; i++) {
    measurements.push_back(
        dyno_testing::makeStatusKeypointMeasurement(i, background_label, 0));
    expected_tracklets.push_back(i);
  }

  Map2d::Ptr map = Map2d::create();

  map->updateObservations(measurements);

  EXPECT_TRUE(map->frameExists(0));
  EXPECT_FALSE(map->frameExists(1));

  EXPECT_TRUE(map->landmarkExists(0));
  EXPECT_TRUE(map->landmarkExists(9));
  EXPECT_FALSE(map->landmarkExists(10));

  EXPECT_EQ(map->staticTrackletsByFrame(0), expected_tracklets);

  // expected tracklets in frame 0
  TrackletIds expected_tracklets_f0 = expected_tracklets;

  TrackletIds expected_tracklets_f1;
  // add another 5 points at frame 1
  measurements.clear();
  for (size_t i = 0; i < 5; i++) {
    measurements.push_back(
        dyno_testing::makeStatusKeypointMeasurement(i, background_label, 1));

    expected_tracklets.push_back(i);
    expected_tracklets_f1.push_back(i);
  }

  // apply update
  map->updateObservations(measurements);

  EXPECT_EQ(map->staticTrackletsByFrame(0), expected_tracklets_f0);
  EXPECT_EQ(map->staticTrackletsByFrame(1), expected_tracklets_f1);

  // check for frames in some landmarks
  // should be seen in frames 0 and 1
  auto lmk1 = map->getLandmark(0);
  std::vector<FrameId> lmk_1_seen_frames = lmk1->getSeenFrameIds();
  std::vector<FrameId> lmk_1_seen_frames_expected = {0, 1};
  EXPECT_EQ(lmk_1_seen_frames, lmk_1_seen_frames_expected);

  // should be seen in frames 0
  auto lmk6 = map->getLandmark(6);
  std::vector<FrameId> lmk_6_seen_frames = lmk6->getSeenFrameIds();
  std::vector<FrameId> lmk_6_seen_frames_expected = {0};
  EXPECT_EQ(lmk_6_seen_frames, lmk_6_seen_frames_expected);

  // check that the frames here are the ones in the map
  EXPECT_EQ(map->getFrame(lmk_1_seen_frames.at(0)),
            map->getFrame(lmk_6_seen_frames.at(0)));

  // finally check that there are no objects
  EXPECT_EQ(map->getFrame(lmk_1_seen_frames.at(0))->objectsSeen().size(), 0);
  EXPECT_EQ(map->numObjectsSeen(), 0u);
}

TEST(Map, setStaticOrdering) {
  // add frames out of order
  Map2d::Ptr map = Map2d::create();

  StatusKeypointVector measurements;
  // frame 0
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(1, 0, 0));
  // frame 2
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(1, 0, 2));
  // frame 1
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(1, 0, 1));
  // frame 3
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(1, 0, 3));
  map->updateObservations(measurements);

  auto lmk = map->getLandmark(1);
  EXPECT_TRUE(lmk != nullptr);

  FrameIds expected_frame_ids = {0, 1, 2, 3};
  EXPECT_EQ(lmk->getSeenFrameIds(), expected_frame_ids);

  EXPECT_EQ(lmk->getSeenFrameIds().front(), 0u);
  EXPECT_EQ(lmk->getSeenFrameIds().back(), 3u);
}

TEST(Map, basicObjectAdd) {
  Map2d::Ptr map = Map2d::create();
  StatusKeypointVector measurements;
  // add two dynamic points on object 1 and frames 0 and 1
  // tracklet 0, object 1 frame 0
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(0, 1, 0));
  // tracklet 0, object 1 frame 1
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(0, 1, 1));

  map->updateObservations(measurements);
  EXPECT_EQ(map->numObjectsSeen(), 1u);
  EXPECT_TRUE(map->objectExists(1));

  auto object1 = map->getObject(1);
  // object 1 should have 1 point seen at frames 0 and 1
  EXPECT_EQ(object1->trackletIds(), TrackletIds{0});
  EXPECT_EQ(object1->landmarks().size(), 1u);

  // now check that the frames also have these measurements
  auto frame_0 = map->getFrame(0);
  auto frame_1 = map->getFrame(1);
  EXPECT_TRUE(frame_0 != nullptr);
  EXPECT_TRUE(frame_1 != nullptr);

  EXPECT_EQ(frame_0->dynamicLandmarks().collectKeys(), TrackletIds{0});
  EXPECT_EQ(frame_1->dynamicLandmarks().collectKeys(), TrackletIds{0});

  // sanity check that there are no static points
  EXPECT_EQ(frame_0->staticLandmarks().size(), 0u);
  EXPECT_EQ(frame_1->staticLandmarks().size(), 0u);

  // check object id and seen frames
  auto lmk_0 = map->getLandmark(0);
  EXPECT_EQ(lmk_0->objectId(), 1);                        // object Id 1;
  EXPECT_EQ(lmk_0->getSeenFrameIds(), FrameIds({0, 1}));  // seen frames

  // finally check that the landmark referred to by the frames are the same one
  // as getLandmark(0) this also implicitly tests FastMapNodeSet::find(index)
  auto frame_0_dynamic_lmks = frame_0->dynamicLandmarks();
  auto frame_1_dynamic_lmks = frame_1->dynamicLandmarks();

  // look from the lmk with id 0
  auto lmk_itr_frame_0 = frame_0_dynamic_lmks.find(0);
  auto lmk_itr_frame_1 = frame_1_dynamic_lmks.find(0);
  // should not be at the end as we have this landmark
  EXPECT_FALSE(lmk_itr_frame_0 == frame_0_dynamic_lmks.end());
  EXPECT_FALSE(lmk_itr_frame_1 == frame_1_dynamic_lmks.end());

  // check the lmk is the one we got from the map
  EXPECT_EQ(lmk_0, *lmk_itr_frame_0);
  EXPECT_EQ(lmk_0, *lmk_itr_frame_1);
}

TEST(Map, framesSeenDuplicates) {
  Map2d::Ptr map = Map2d::create();
  Map2d::SharedLandmarkNodeT landmark_node =
      std::make_shared<Map2d::LandmarkNodeT>(0, 0);

  EXPECT_EQ(landmark_node->numObservations(), 0);

  Map2d::SharedFrameNodeT frame_node =
      std::make_shared<Map2d::FrameNodeT>(0, 0.0);

  landmark_node->add(frame_node, Keypoint());

  EXPECT_EQ(landmark_node->numObservations(), 1);
  EXPECT_EQ(*landmark_node->getSeenFrames().begin(), frame_node);
  EXPECT_EQ(landmark_node->getMeasurements().size(), 1);

  // now add the same frame again
  EXPECT_THROW({ landmark_node->add(frame_node, Keypoint()); },
               DynosamException);
}

TEST(Map, objectSeenFrames) {
  Map2d::Ptr map = Map2d::create();
  StatusKeypointVector measurements;

  // add 2 objects
  // object 1 seen at frames 0 and 1
  // tracklet 0, object 1 frame 0
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(0, 1, 0));
  // tracklet 0, object 1 frame 1
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(0, 1, 1));
  // tracklet 0 has two observations

  // object 2 seen at frames 1 and 2
  // tracklet 1, object 2 frame 1
  // tracklet 1 has 1 observations
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(1, 2, 1));
  // tracklet 2, object 2 frame 2
  // tracklet 2 has 1 observations
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(2, 2, 2));

  // object 3 seen at frames 0, 1, 2
  // tracklet 3, object 3 frame 0
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(3, 3, 0));
  // tracklet 3, object 3 frame 1
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(3, 3, 1));
  // tracklet 3, object 3 frame 2
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(3, 3, 2));
  // tracklet 3 has 3 observations

  map->updateObservations(measurements);
  EXPECT_EQ(map->numObjectsSeen(), 3u);

  auto object1 = map->getObject(1);
  auto object2 = map->getObject(2);
  auto object3 = map->getObject(3);

  using SharedFrameSet = Map2d::SharedFrameSet;
  using SharedObjectSet = Map2d::SharedObjectSet;

  SharedFrameSet expected_frame_set_object1;
  expected_frame_set_object1.insert(CHECK_NOTNULL(map->getFrame(0)));
  expected_frame_set_object1.insert(CHECK_NOTNULL(map->getFrame(1)));

  SharedFrameSet expected_frame_set_object2;
  expected_frame_set_object2.insert(CHECK_NOTNULL(map->getFrame(1)));
  expected_frame_set_object2.insert(CHECK_NOTNULL(map->getFrame(2)));

  SharedFrameSet expected_frame_set_object3;
  expected_frame_set_object3.insert(CHECK_NOTNULL(map->getFrame(0)));
  expected_frame_set_object3.insert(CHECK_NOTNULL(map->getFrame(1)));
  expected_frame_set_object3.insert(CHECK_NOTNULL(map->getFrame(2)));

  EXPECT_EQ(object1->getSeenFrames(), expected_frame_set_object1);
  EXPECT_EQ(object2->getSeenFrames(), expected_frame_set_object2);
  EXPECT_EQ(object3->getSeenFrames(), expected_frame_set_object3);

  // frame 0 has seen object 1 and 3 -the reverse of the above testt
  EXPECT_EQ(map->getFrame(0)->objectsSeen(),
            SharedObjectSet({object1, object3}));
  // frame 1 has seen object 1, 2 and 3
  EXPECT_EQ(map->getFrame(1)->objectsSeen(),
            SharedObjectSet({object1, object3, object2}));
  // frame 2 has seen object 2 and 3
  EXPECT_EQ(map->getFrame(2)->objectsSeen(),
            SharedObjectSet({object2, object3}));

  // check object observation functions
  auto frame0 = map->getFrame(0);
  auto frame1 = map->getFrame(1);
  auto frame2 = map->getFrame(2);
  // check object observed frame 0
  EXPECT_TRUE(frame0->objectObserved(1));
  EXPECT_TRUE(frame0->objectObserved(3));
  EXPECT_FALSE(frame0->objectObserved(2));

  // check object observed frame 1 (all)
  EXPECT_TRUE(frame1->objectObserved(1));
  EXPECT_TRUE(frame1->objectObserved(3));
  EXPECT_TRUE(frame1->objectObserved(2));
  // check object observed frame 2
  EXPECT_TRUE(frame2->objectObserved(2));
  EXPECT_TRUE(frame2->objectObserved(3));
  EXPECT_FALSE(frame2->objectObserved(1));

  // check observed in previous (in frame 0, there is no previous so all
  // false!!)
  // EXPECT_FALSE(frame0->objectObservedInPrevious(1));
  // EXPECT_FALSE(frame0->objectObservedInPrevious(3));

  // // both object1 and 3 appear in frame 0 but not object 2
  // EXPECT_TRUE(frame1->objectObservedInPrevious(1));
  // EXPECT_TRUE(frame1->objectObservedInPrevious(3));
  // EXPECT_FALSE(frame1->objectObservedInPrevious(2));

  // // all objects are observed at frame 1
  // EXPECT_TRUE(frame2->objectObservedInPrevious(1));
  // EXPECT_TRUE(frame2->objectObservedInPrevious(3));
  // EXPECT_TRUE(frame2->objectObservedInPrevious(2));

  // // check objectMotionExpected (i.e objects are observed at both frames)
  // EXPECT_FALSE(frame0->objectMotionExpected(1));
  // EXPECT_FALSE(frame0->objectMotionExpected(3));

  // // object 1 and 3 seen at frames 0 and 1, but not object 2
  // EXPECT_TRUE(frame1->objectMotionExpected(1));
  // EXPECT_TRUE(frame1->objectMotionExpected(3));
  // EXPECT_FALSE(frame1->objectMotionExpected(2));
  // // object 2 and 3 seen at frames 1 and 2, but not object 1
  // EXPECT_FALSE(frame2->objectMotionExpected(1));
  // EXPECT_TRUE(frame2->objectMotionExpected(3));
  // EXPECT_TRUE(frame2->objectMotionExpected(2));
}

TEST(Map, landmarksSeenAtFrame) {
  Map2d::Ptr map = Map2d::create();
  StatusKeypointVector measurements;

  // add 2 objects
  // object 1 seen at frames 0 and 1
  // tracklet 0, object 1 frame 0
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(0, 1, 0));
  // tracklet 0, object 1 frame 1
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(0, 1, 1));

  // object 2 seen at frames 1 and 2
  // tracklet 1, object 2 frame 1
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(1, 2, 1));
  // tracklet 2, object 2 frame 1
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(2, 2, 1));

  // object 3 seen at frames 0, 1, 2
  // tracklet 3, object 3 frame 0
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(3, 3, 0));
  // tracklet 3, object 3 frame 1
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(3, 3, 1));
  // tracklet 3, object 3 frame 1
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(4, 3, 1));
  // tracklet 3 has 3 observations

  map->updateObservations(measurements);
  EXPECT_EQ(map->numObjectsSeen(), 3u);

  auto object1 = map->getObject(1);
  auto object2 = map->getObject(2);
  auto object3 = map->getObject(3);

  using SharedFrameSet = Map2d::SharedFrameSet;
  using SharedObjectSet = Map2d::SharedObjectSet;
  using SharedLandmarkSet = Map2d::SharedLandmarkSet;

  SharedLandmarkSet expected_lmk_set_object1_frame0;
  expected_lmk_set_object1_frame0.insert(CHECK_NOTNULL(map->getLandmark(0)));

  SharedLandmarkSet expected_lmk_set_object1_frame1;
  expected_lmk_set_object1_frame1.insert(CHECK_NOTNULL(map->getLandmark(0)));

  SharedLandmarkSet expected_lmk_set_object2_frame1;
  expected_lmk_set_object2_frame1.insert(CHECK_NOTNULL(map->getLandmark(1)));
  expected_lmk_set_object2_frame1.insert(CHECK_NOTNULL(map->getLandmark(2)));

  SharedLandmarkSet expected_lmk_set_object3_frame0;
  expected_lmk_set_object3_frame0.insert(CHECK_NOTNULL(map->getLandmark(3)));

  SharedLandmarkSet expected_lmk_set_object3_frame1;
  expected_lmk_set_object3_frame1.insert(CHECK_NOTNULL(map->getLandmark(3)));
  expected_lmk_set_object3_frame1.insert(CHECK_NOTNULL(map->getLandmark(4)));

  EXPECT_EQ(object1->landmarksSeenAtFrame(0), expected_lmk_set_object1_frame0);
  EXPECT_EQ(object1->landmarksSeenAtFrame(1), expected_lmk_set_object1_frame1);
  EXPECT_EQ(object1->landmarksSeenAtFrame(2), SharedLandmarkSet{});
  EXPECT_EQ(object2->landmarksSeenAtFrame(1), expected_lmk_set_object2_frame1);
  EXPECT_EQ(object3->landmarksSeenAtFrame(0), expected_lmk_set_object3_frame0);
  EXPECT_EQ(object3->landmarksSeenAtFrame(1), expected_lmk_set_object3_frame1);
}

// Even if derived classes override logic, they must still obey ordering +
// uniqueness.
TEST(Map, seenFramesAreSortedAndUniqueContract) {
  Map2d::Ptr map = Map2d::create();
  StatusKeypointVector measurements;

  // deliberately unordered + duplicates
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(0, 1, 2));
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(0, 1, 0));
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(0, 1, 1));
  // measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(0, 2,
  // 1));

  map->updateObservations(measurements);

  auto lmk = map->getLandmark(0);
  auto frames = lmk->getSeenFrameIds();

  // must be sorted + unique regardless of implementation
  EXPECT_EQ(frames, FrameIds({0, 1, 2}));
}

// If an object reports it has seen a frame → that frame must report the object.
TEST(Map, frameObjectSymmetryContract) {
  Map2d::Ptr map = Map2d::create();
  StatusKeypointVector measurements;

  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(0, 1, 0));
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(1, 1, 1));

  map->updateObservations(measurements);

  auto object = map->getObject(1);

  for (const auto& frame : object->getSeenFrames()) {
    EXPECT_TRUE(frame->objectObserved(object->getId()));
  }
}

// Landmark ↔ Frame Bidirectional Consistency
TEST(Map, landmarkFrameBidirectionalConsistency) {
  Map2d::Ptr map = Map2d::create();
  StatusKeypointVector measurements;

  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(0, 1, 0));
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(0, 1, 1));

  map->updateObservations(measurements);

  auto lmk = map->getLandmark(0);

  for (FrameId fid : lmk->getSeenFrameIds()) {
    auto frame = map->getFrame(fid);

    auto it = frame->dynamicLandmarks().find(0);
    EXPECT_FALSE(it == frame->dynamicLandmarks().end());
    EXPECT_EQ(*it, lmk);
  }
}

struct FilteredFrameNode;
struct NodeTypesWithFilteredFrameNode {
  using Measurement = Keypoint;
  using FrameNodeT = FilteredFrameNode;
  using ObjectNodeT = RegularObjectNode<NodeTypesWithFilteredFrameNode>;
  using LandmarkNodeT = RegularLandmarkNode<NodeTypesWithFilteredFrameNode>;
};

struct FilteredFrameNode
    : public RegularFrameNode<NodeTypesWithFilteredFrameNode> {
  using Base = RegularFrameNode<NodeTypesWithFilteredFrameNode>;

  FilteredFrameNode(FrameId id, Timestamp ts) : Base(id, ts) {}

  bool objectObserved(ObjectId obj_id) const {
    // pretend we ignore object 2
    if (obj_id == 2) return false;
    return Base::objectObserved(obj_id);
  }
};

using FilteredFrameMap = Map<NodeTypesWithFilteredFrameNode>;

TEST(Map, derivedFrameOverridesObservationLogic) {
  auto map = std::make_shared<FilteredFrameMap>();

  StatusKeypointVector measurements;
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(0, 2, 0));

  map->updateObservations(measurements);

  std::shared_ptr<FilteredFrameNode> frame0 = map->getFrame(0);

  // base map still thinks object exists
  EXPECT_TRUE(map->objectExists(2));

  // but frame override should hide it
  EXPECT_FALSE(frame0->objectObserved(2));
}

// // Simulate an object that only considers frames with ≥2 landmarks:
// struct MinObservationObjectNode : public Map2d::ObjectNodeT {
//   using Base = Map2d::ObjectNodeT;

//   using Base::Base;

//   SharedFrameSet getSeenFrames() const override {
//     SharedFrameSet filtered;
//     for (const auto& frame : Base::getSeenFrames()) {
//       if (landmarksSeenAtFrame(frame->getId()).size() >= 2) {
//         filtered.insert(frame);
//       }
//     }
//     return filtered;
//   }
// };

// TEST(Map, derivedObjectFiltersSeenFrames) {
//   Map2d::Ptr map = Map2d::create();
//   StatusKeypointVector measurements;

//   // frame 0: 1 landmark
//   measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(0, 1,
//   0));

//   // frame 1: 2 landmarks
//   measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(1, 1,
//   1)); measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(2,
//   1, 1));

//   map->updateObservations(measurements);

//   auto obj = map->getObject(1);

//   auto frames = obj->getSeenFrames();

//   EXPECT_EQ(frames.size(), 1);
//   EXPECT_TRUE((*frames.begin())->getId() == 1);
// }

// TEST(Map, repeatedUpdatesDoNotDuplicateState) {
//   Map2d::Ptr map = Map2d::create();
//   StatusKeypointVector measurements;

//   for (int i = 0; i < 3; i++) {
//     measurements.clear();
//     measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(0, 1,
//     0)); map->updateObservations(measurements);
//   }

//   auto obj = map->getObject(1);

//   EXPECT_EQ(obj->trackletIds(), TrackletIds({0}));
//   EXPECT_EQ(obj->getSeenFrames().size(), 1);
// }

TEST(Map, objectSeenFramesDuplicateFrameFromMultipleLandmarks) {
  Map2d::Ptr map = Map2d::create();
  StatusKeypointVector measurements;

  // Same object, same frame, different tracklets
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(0, 1, 0));
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(1, 1, 0));
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(2, 1, 0));

  map->updateObservations(measurements);

  auto object = map->getObject(1);

  // Should ONLY have frame 0 once
  EXPECT_EQ(object->getSeenFrameIds(), FrameIds({0}));
}

TEST(Map, landmarksSeenAtFrameMultiFrameConsistency) {
  Map2d::Ptr map = Map2d::create();
  StatusKeypointVector measurements;

  // Tracklet 0 seen in 0,1
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(0, 1, 0));
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(0, 1, 1));

  // Tracklet 1 seen only in 1
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(1, 1, 1));

  map->updateObservations(measurements);

  auto object = map->getObject(1);

  // Frame 0: only tracklet 0
  EXPECT_EQ(object->landmarksSeenAtFrame(0).collectKeys(), TrackletIds({0}));

  // Frame 1: both
  EXPECT_EQ(object->landmarksSeenAtFrame(1).collectKeys(), TrackletIds({0, 1}));
}

TEST(Map, setAndGetInitialSensorPose) {
  Map3d::Ptr map = Map3d::create();

  FrameId k = 0;
  Timestamp t = 1.0;
  Pose3Measurement X = Pose3Measurement::Random();

  map->setInitialSensorPose(k, t, X);

  EXPECT_TRUE(map->frameExists(k));

  auto frame = map->getFrame(k);
  ASSERT_TRUE(frame != nullptr);

  Pose3Measurement X_out;
  EXPECT_TRUE(frame->getInitialSensorPose(X_out));

  EXPECT_TRUE(gtsam::assert_equal(X_out, X, 1e-4));
  EXPECT_TRUE(gtsam::assert_equal(frame->initialSensorPose(), X, 1e-4));
}

TEST(Map, overwriteInitialSensorPose) {
  Map3d::Ptr map = Map3d::create();

  FrameId k = 0;
  Timestamp t = 1.0;

  Pose3Measurement X1 = Pose3Measurement::Random();
  Pose3Measurement X2 = Pose3Measurement::Random();

  map->setInitialSensorPose(k, t, X1);
  map->setInitialSensorPose(k, t, X2);

  auto frame = map->getFrame(k);

  EXPECT_TRUE(gtsam::assert_equal(frame->initialSensorPose(), X2, 1e-4));
}

TEST(FrameNode, missingInitialSensorPose) {
  auto frame = std::make_shared<Map3d::FrameNodeT>(0, 0.0);

  Pose3Measurement X;
  EXPECT_FALSE(frame->getInitialSensorPose(X));

  EXPECT_THROW({ frame->initialSensorPose(); }, DynosamException);
}

TEST(Map, setAndGetInitialObjectMotions) {
  Map2d::Ptr map = Map2d::create();

  FrameId k = 2;
  Timestamp t = 1.0;

  // must create frame first
  map->setInitialSensorPose(k, t, Pose3Measurement());

  MotionEstimateMap motions;
  ObjectId j = 1;
  Motion3ReferenceFrame H(utils::createRandomAroundIdentity<gtsam::Pose3>(3.2),
                          MotionRepresentationStyle::F2F,
                          ReferenceFrame::GLOBAL, k - 1, k);

  motions.insert2(j, H);

  // simulate object observation (required!)
  StatusKeypointVector measurements;
  measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(0, j, k));
  map->updateObservations(measurements);

  map->setInitialObjectMotions(k, motions);

  Motion3ReferenceFrame H_out;
  EXPECT_TRUE(map->getInitialObjectMotion(k, j, H_out));
  EXPECT_EQ(H_out, H);

  EXPECT_FALSE(map->getInitialObjectMotion(k + 1, j, H_out));
  EXPECT_FALSE(map->getInitialObjectMotion(k, j + 1, H_out));

  EXPECT_EQ(map->initialObjectMotion(k, j), H);
}

// test Motion exists BUT object not observed → returns false
TEST(Map, objectMotionIgnoredIfObjectNotObserved) {
  Map3d::Ptr map = Map3d::create();

  FrameId k = 0;
  Timestamp t = 1.0;

  map->setInitialSensorPose(k, t, Pose3Measurement());

  MotionEstimateMap motions;
  ObjectId j = 1;
  Motion3ReferenceFrame H;

  motions.insert2(j, H);

  // NO observation added!
  map->setInitialObjectMotions(k, motions);

  Motion3ReferenceFrame H_out;
  EXPECT_FALSE(map->getInitialObjectMotion(k, j, H_out));
}

TEST(Map, getInitialObjectMotionMissingFrame) {
  Map3d::Ptr map = Map3d::create();

  Motion3ReferenceFrame H;
  EXPECT_FALSE(map->getInitialObjectMotion(999, 1, H));

  EXPECT_THROW({ map->initialObjectMotion(999, 1); }, DynosamException);
}

// TODO: bring back!!
//  TEST(Map, testSimpleEstimateAccessWithPose) {
//      Map2d::Ptr map = Map2d::create();
//      StatusKeypointMeasurements measurements;

//     //TODO:frame cannot be 0? what is invalid frame then?
//     EXPECT_EQ(map->lastEstimateUpdate(), 0u);

//     //tracklet 0, static, frame
//     measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(0, 0,
//     0)); map->updateObservations(measurements);

//     auto frame_0 = map->getFrame(0);
//     auto pose_0_query = frame_0->getPoseEstimate();
//     EXPECT_FALSE(pose_0_query);
//     EXPECT_FALSE(pose_0_query.isValid());

//     gtsam::Values estimate;

//     //some random pose
//     gtsam::Rot3 R = gtsam::Rot3::Rodrigues(0.3,0.4,-0.5);
//     gtsam::Point3 t(3.5,-8.2,4.2);
//     gtsam::Pose3 pose_0_actual(R,t);
//     gtsam::Key pose_0_key = CameraPoseSymbol(0);
//     estimate.insert(pose_0_key, pose_0_actual);

//     map->updateEstimates(estimate, gtsam::NonlinearFactorGraph{}, 0);
//     pose_0_query = frame_0->getPoseEstimate();

//     EXPECT_TRUE(pose_0_query);
//     EXPECT_TRUE(pose_0_query.isValid());
//     EXPECT_TRUE(gtsam::assert_equal(pose_0_query.get(), pose_0_actual));
//     EXPECT_EQ(pose_0_query.key_, pose_0_key);

// }

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <unordered_map>
#include <vector>

// ============================================================================
// Dummy value
// ============================================================================

struct Dummy {
  std::uint64_t id = 0;
  std::uint64_t payload = 0;

  bool operator<(const Dummy& other) const noexcept { return id < other.id; }

  bool operator==(const Dummy& other) const noexcept {
    return id == other.id && payload == other.payload;
  }

  void print(const std::string& s = "") const {}
  bool equals(const Dummy& q, double tol = 1e-9) const { return false; }
};

// needed for use in gtsam::FastSet
template <>
struct gtsam::traits<Dummy> : public gtsam::Testable<Dummy> {};

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <iterator>
#include <unordered_map>
#include <utility>
#include <vector>

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <iterator>
#include <unordered_map>
#include <utility>
#include <vector>

// =============================================================================
// DenseSetStorage
//
// Common storage for DenseSet and DenseOrderedSet.
//
// Invariants:
//   values_.size() == number of unique keys
//   index_[key] == index of the corresponding value in values_
//
// values_ is never reordered after insertion.
// =============================================================================

template <typename T, typename Key, typename KeyOfValue,
          typename Hash = std::hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
class DenseSetStorage {
 public:
  using value_type = T;
  using key_type = Key;
  using size_type = std::size_t;

 protected:
  using IndexMap = std::unordered_map<Key, size_type, Hash, KeyEqual>;

  DenseSetStorage() = default;

  explicit DenseSetStorage(const KeyOfValue& key_of_value)
      : key_of_value_(key_of_value) {}

  void reserve(size_type n) {
    values_.reserve(n);
    index_.reserve(n);
  }

  bool empty() const noexcept { return values_.empty(); }

  size_type size() const noexcept { return values_.size(); }

  bool contains(const Key& key) const {
    return index_.find(key) != index_.end();
  }

  const T& at(const Key& key) const {
    const auto it = index_.find(key);

    assert(it != index_.end());

    return values_[it->second];
  }

  // Inserts into the contiguous storage and hash index.
  //
  // Returns false if the key already exists.
  //
  // On success:
  //   storage_index = index of the new value in values_
  //
  // Exception safety:
  //   If index_.emplace() throws, values_ is rolled back.
  template <typename U>
  bool emplaceValue(U&& value, size_type& storage_index) {
    const Key key = key_of_value_(value);

    // Uniqueness check before modifying storage.
    if (index_.find(key) != index_.end()) {
      return false;
    }

    storage_index = values_.size();

    values_.emplace_back(std::forward<U>(value));

    try {
      index_.emplace(key, storage_index);
    } catch (...) {
      values_.pop_back();
      throw;
    }

    return true;
  }

  std::vector<T> values_;
  IndexMap index_;
  KeyOfValue key_of_value_;
};

// =============================================================================
// DenseSet
//
// Unique keys.
// Contiguous storage.
// Iteration order is insertion/storage order.
//
// Does NOT guarantee sorted ordering.
// =============================================================================

template <typename T, typename Key, typename KeyOfValue,
          typename Hash = std::hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
class DenseSet : private DenseSetStorage<T, Key, KeyOfValue, Hash, KeyEqual> {
 private:
  using Base = DenseSetStorage<T, Key, KeyOfValue, Hash, KeyEqual>;

 public:
  using value_type = typename Base::value_type;
  using key_type = typename Base::key_type;
  using size_type = typename Base::size_type;

  // --------------------------------------------------------------------------
  // Iterator
  // --------------------------------------------------------------------------

  class const_iterator {
   public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = const T;
    using difference_type = std::ptrdiff_t;
    using pointer = const T*;
    using reference = const T&;

    const_iterator() = default;

    reference operator*() const noexcept { return owner_->values_[position_]; }

    pointer operator->() const noexcept { return &owner_->values_[position_]; }

    const_iterator& operator++() noexcept {
      ++position_;
      return *this;
    }

    const_iterator operator++(int) noexcept {
      const_iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    const_iterator& operator--() noexcept {
      --position_;
      return *this;
    }

    const_iterator operator--(int) noexcept {
      const_iterator tmp = *this;
      --(*this);
      return tmp;
    }

    const_iterator& operator+=(difference_type n) noexcept {
      position_ += n;
      return *this;
    }

    const_iterator& operator-=(difference_type n) noexcept {
      position_ -= n;
      return *this;
    }

    const_iterator operator+(difference_type n) const noexcept {
      const_iterator result = *this;
      result += n;
      return result;
    }

    const_iterator operator-(difference_type n) const noexcept {
      const_iterator result = *this;
      result -= n;
      return result;
    }

    difference_type operator-(const const_iterator& other) const noexcept {
      return static_cast<difference_type>(position_) -
             static_cast<difference_type>(other.position_);
    }

    reference operator[](difference_type n) const noexcept {
      return owner_->values_[position_ + n];
    }

    bool operator==(const const_iterator& other) const noexcept {
      return owner_ == other.owner_ && position_ == other.position_;
    }

    bool operator!=(const const_iterator& other) const noexcept {
      return !(*this == other);
    }

    bool operator<(const const_iterator& other) const noexcept {
      return position_ < other.position_;
    }

    bool operator>(const const_iterator& other) const noexcept {
      return other < *this;
    }

    bool operator<=(const const_iterator& other) const noexcept {
      return !(other < *this);
    }

    bool operator>=(const const_iterator& other) const noexcept {
      return !(*this < other);
    }

   private:
    friend class DenseSet;

    const_iterator(const DenseSet* owner, size_type position) noexcept
        : owner_(owner), position_(position) {}

    const DenseSet* owner_ = nullptr;
    size_type position_ = 0;
  };

  // --------------------------------------------------------------------------
  // Construction
  // --------------------------------------------------------------------------

  DenseSet() = default;

  explicit DenseSet(const KeyOfValue& key_of_value) : Base(key_of_value) {}

  // --------------------------------------------------------------------------
  // Capacity
  // --------------------------------------------------------------------------

  using Base::empty;
  using Base::size;

  void reserve(size_type n) { Base::reserve(n); }

  // --------------------------------------------------------------------------
  // Lookup
  // --------------------------------------------------------------------------

  using Base::at;
  using Base::contains;

  const_iterator find(const Key& key) const noexcept {
    const auto it = Base::index_.find(key);

    if (it == Base::index_.end()) {
      return end();
    }

    return const_iterator{this, it->second};
  }

  // --------------------------------------------------------------------------
  // Iteration
  // --------------------------------------------------------------------------

  const_iterator begin() const noexcept { return const_iterator{this, 0}; }

  const_iterator end() const noexcept {
    return const_iterator{this, Base::values_.size()};
  }

  // --------------------------------------------------------------------------
  // Access
  // --------------------------------------------------------------------------

  const T& front() const {
    assert(!empty());
    return Base::values_.front();
  }

  const T& back() const {
    assert(!empty());
    return Base::values_.back();
  }

  // --------------------------------------------------------------------------
  // Insertion
  // --------------------------------------------------------------------------

  bool insert(const T& value) {
    size_type storage_index;
    return Base::emplaceValue(value, storage_index);
  }

  bool insert(T&& value) {
    size_type storage_index;
    return Base::emplaceValue(std::move(value), storage_index);
  }
};

// =============================================================================
// DenseOrderedSet
//
// Unique keys.
// Contiguous storage.
// Sorted iteration.
//
// Data layout:
//
//   values_
//       [ T ][ T ][ T ][ T ][ T ]
//         0    1    2    3    4
//
//   ordered_keys_
//       [ k0 ][ k1 ][ k2 ][ k3 ][ k4 ]
//
//   ordered_indices_
//       [  3  ][  0  ][  4  ][  1  ][ 2 ]
//
// Thus:
//
//   ordered_keys_[i]
//   ordered_indices_[i]
//
// together describe the i-th element in sorted order.
//
// Insertion:
//
//   increasing:
//       O(1) amortised
//
//   arbitrary:
//       O(log N) lower_bound
//       O(N) vector shift
// =============================================================================

template <typename T, typename Key, typename KeyOfValue,
          typename Hash = std::hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
class DenseOrderedSet
    : private DenseSetStorage<T, Key, KeyOfValue, Hash, KeyEqual> {
 private:
  using Base = DenseSetStorage<T, Key, KeyOfValue, Hash, KeyEqual>;

 public:
  using value_type = typename Base::value_type;
  using key_type = typename Base::key_type;
  using size_type = typename Base::size_type;

  // --------------------------------------------------------------------------
  // Iterator
  // --------------------------------------------------------------------------

  class const_iterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = const T;
    using difference_type = std::ptrdiff_t;
    using pointer = const T*;
    using reference = const T&;

    const_iterator() = default;

    reference operator*() const noexcept {
      return owner_->values_[owner_->ordered_indices_[position_]];
    }

    pointer operator->() const noexcept { return &operator*(); }

    const_iterator& operator++() noexcept {
      ++position_;
      return *this;
    }

    const_iterator operator++(int) noexcept {
      const_iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    bool operator==(const const_iterator& other) const noexcept {
      return owner_ == other.owner_ && position_ == other.position_;
    }

    bool operator!=(const const_iterator& other) const noexcept {
      return !(*this == other);
    }

   private:
    friend class DenseOrderedSet;

    const_iterator(const DenseOrderedSet* owner, size_type position) noexcept
        : owner_(owner), position_(position) {}

    const DenseOrderedSet* owner_ = nullptr;
    size_type position_ = 0;
  };

  // --------------------------------------------------------------------------
  // Construction
  // --------------------------------------------------------------------------

  DenseOrderedSet() = default;

  explicit DenseOrderedSet(const KeyOfValue& key_of_value)
      : Base(key_of_value) {}

  // --------------------------------------------------------------------------
  // Capacity
  // --------------------------------------------------------------------------

  using Base::empty;
  using Base::size;

  void reserve(size_type n) {
    Base::reserve(n);
    ordered_keys_.reserve(n);
    ordered_indices_.reserve(n);
  }

  // --------------------------------------------------------------------------
  // Lookup
  // --------------------------------------------------------------------------

  using Base::at;
  using Base::contains;

  // --------------------------------------------------------------------------
  // Iteration
  // --------------------------------------------------------------------------

  const_iterator begin() const noexcept { return const_iterator{this, 0}; }

  const_iterator end() const noexcept {
    return const_iterator{this, ordered_indices_.size()};
  }

  // --------------------------------------------------------------------------
  // Ordered access
  // --------------------------------------------------------------------------

  const T& front() const {
    assert(!empty());

    return Base::values_[ordered_indices_.front()];
  }

  const T& back() const {
    assert(!empty());

    return Base::values_[ordered_indices_.back()];
  }

  // --------------------------------------------------------------------------
  // Insertion
  // --------------------------------------------------------------------------

  bool insert(const T& value) { return insertImpl(value); }

  bool insert(T&& value) { return insertImpl(std::move(value)); }

 private:
  template <typename U>
  bool insertImpl(U&& value) {
    // Extract the key before moving value.
    const Key key = Base::key_of_value_(value);

    size_type storage_index;

    // ------------------------------------------------------------------------
    // Common insertion:
    //
    //   1. uniqueness check
    //   2. append to contiguous storage
    //   3. add key -> storage index
    //
    // ------------------------------------------------------------------------

    if (!Base::emplaceValue(std::forward<U>(value), storage_index)) {
      return false;
    }

    // ------------------------------------------------------------------------
    // Fast path: first element or monotonically increasing key.
    //
    // This is just two push_backs and is the critical common case.
    // ------------------------------------------------------------------------

    if (ordered_keys_.empty() || ordered_keys_.back() < key) {
      ordered_keys_.push_back(key);
      ordered_indices_.push_back(storage_index);

      return true;
    }

    // ------------------------------------------------------------------------
    // General case.
    //
    // Search ONLY the contiguous key array.
    //
    // This avoids:
    //
    //   ordered_indices_[i]
    //       -> values_[...]
    //           -> key_of_value_(...)
    //
    // for every binary-search comparison.
    // ------------------------------------------------------------------------

    const auto position =
        std::lower_bound(ordered_keys_.begin(), ordered_keys_.end(), key);

    const size_type ordered_position =
        static_cast<size_type>(position - ordered_keys_.begin());

    // The two vectors have identical logical ordering, so the same position
    // can be used for both insertions.
    ordered_keys_.insert(position, key);

    ordered_indices_.insert(ordered_indices_.begin() + ordered_position,
                            storage_index);

    return true;
  }

  // Sorted keys. Used exclusively for binary search.
  std::vector<Key> ordered_keys_;

  // Maps sorted position -> contiguous storage index.
  std::vector<size_type> ordered_indices_;
};
// template <typename T, typename Key, typename KeyOfValue>
// class DenseOrderedSetBinarySearch {
//  public:
//   using value_type = T;
//   using key_type = Key;
//   using size_type = std::size_t;

//  private:
//   std::vector<T> values_;
//   std::vector<size_type> ordered_indices_;

//   KeyOfValue key_of_;

//  public:
//   class const_iterator {
//     public:
//       using iterator_category = std::forward_iterator_tag;
//       using value_type = const T;
//       using difference_type = std::ptrdiff_t;
//       using pointer = const T*;
//       using reference = const T&;

//       const_iterator() = default;

//       reference operator*() const {
//         return owner_->values_[owner_->ordered_indices_[position_]];
//       }

//       pointer operator->() const {
//         return &owner_->values_[owner_->ordered_indices_[position_]];
//       }

//       const_iterator& operator++() {
//         ++position_;
//         return *this;
//       }

//       const_iterator operator++(int) {
//         const_iterator copy = *this;
//         ++(*this);
//         return copy;
//       }

//       friend bool operator==(
//           const const_iterator& lhs,
//           const const_iterator& rhs) {
//         return lhs.owner_ == rhs.owner_ &&
//               lhs.position_ == rhs.position_;
//       }

//       friend bool operator!=(
//           const const_iterator& lhs,
//           const const_iterator& rhs) {
//         return !(lhs == rhs);
//       }

//     private:
//       friend class DenseOrderedSetBinarySearch;

//       const DenseOrderedSetBinarySearch* owner_ = nullptr;
//       size_type position_ = 0;

//       const_iterator(
//           const DenseOrderedSetBinarySearch* owner,
//           size_type position)
//           : owner_(owner),
//             position_(position) {}
//     };

//   const_iterator begin() const {
//     return const_iterator(this, 0);
//   }

//   const_iterator end() const {
//     return const_iterator(this, ordered_indices_.size());
//   }

//   void reserve(size_type n) {
//     values_.reserve(n);
//     ordered_indices_.reserve(n);
//   }

//   bool insert(const T& value) {
//     const Key key = key_of_(value);

//     // Empty container.
//     if (ordered_indices_.empty()) {
//       values_.push_back(value);
//       ordered_indices_.push_back(0);
//       return true;
//     }

//     // Very common case: monotonically increasing keys.
//     const Key& last_key =
//         key_of_(values_[ordered_indices_.back()]);

//     if (last_key < key) {
//       const size_type storage_index = values_.size();

//       values_.push_back(value);
//       ordered_indices_.push_back(storage_index);

//       return true;
//     }

//     // General case.
//     const auto it = std::lower_bound(
//         ordered_indices_.begin(),
//         ordered_indices_.end(),
//         key,
//         [&](size_type index, const Key& k) {
//           return key_of_(values_[index]) < k;
//         });

//     // Check uniqueness.
//     if (it != ordered_indices_.end() &&
//         key_of_(values_[*it]) == key) {
//       return false;
//     }

//     const size_type storage_index = values_.size();

//     values_.push_back(value);

//     // Important: values_ does not move after this point because
//     // ordered_indices_ contains indices, not pointers.
//     ordered_indices_.insert(it, storage_index);

//     return true;
//   }

//   const T& at(const Key& key) const {
//     return find(key);
//   }

//   const T& find(const Key& key) const {
//     const auto it = std::lower_bound(
//         ordered_indices_.begin(),
//         ordered_indices_.end(),
//         key,
//         [&](size_type index, const Key& k) {
//           return key_of_(values_[index]) < k;
//         });

//     if (it == ordered_indices_.end() ||
//         key_of_(values_[*it]) != key) {
//       throw std::out_of_range("Key not found");
//     }

//     return values_[*it];
//   }

//   bool contains(const Key& key) const {
//     const auto it = std::lower_bound(
//         ordered_indices_.begin(),
//         ordered_indices_.end(),
//         key,
//         [&](size_type index, const Key& k) {
//           return key_of_(values_[index]) < k;
//         });

//     return it != ordered_indices_.end() &&
//            key_of_(values_[*it]) == key;
//   }

//   const T& front() const {
//     assert(!ordered_indices_.empty());
//     return values_[ordered_indices_.front()];
//   }

//   const T& back() const {
//     assert(!ordered_indices_.empty());
//     return values_[ordered_indices_.back()];
//   }

//   size_type size() const {
//     return values_.size();
//   }

//   bool empty() const {
//     return values_.empty();
//   }
// };

template <typename T, typename Key, typename KeyOfValue,
          typename Compare = std::less<Key>>
class DenseOrderedSetBinary {
 public:
  using value_type = T;
  using key_type = Key;
  using size_type = std::size_t;

 private:
  struct OrderedKeyCompare {
    const Compare& compare;

    bool operator()(const Key& lhs, const Key& rhs) const {
      return compare(lhs, rhs);
    }

    bool operator()(const size_type index, const Key& key) const {
      return compare(key_of(index), key);
    }

    bool operator()(const Key& key, const size_type index) const {
      return compare(key, key_of(index));
    }

    const Key& key_of(const size_type index) const { return keys[index]; }

    const std::vector<Key>& keys;
  };

 public:
  class const_iterator {
   public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = const T;
    using difference_type = std::ptrdiff_t;
    using pointer = const T*;
    using reference = const T&;

    const_iterator() = default;

    reference operator*() const {
      return owner_->values_[owner_->ordered_indices_[position_]];
    }

    pointer operator->() const { return &(**this); }

    const_iterator& operator++() {
      ++position_;
      return *this;
    }

    const_iterator operator++(int) {
      auto tmp = *this;
      ++(*this);
      return tmp;
    }

    const_iterator& operator--() {
      --position_;
      return *this;
    }

    const_iterator operator--(int) {
      auto tmp = *this;
      --(*this);
      return tmp;
    }

    const_iterator& operator+=(difference_type n) {
      position_ += static_cast<size_type>(n);
      return *this;
    }

    const_iterator& operator-=(difference_type n) {
      position_ -= static_cast<size_type>(n);
      return *this;
    }

    const_iterator operator+(difference_type n) const {
      auto result = *this;
      result += n;
      return result;
    }

    const_iterator operator-(difference_type n) const {
      auto result = *this;
      result -= n;
      return result;
    }

    difference_type operator-(const const_iterator& other) const {
      return static_cast<difference_type>(position_) -
             static_cast<difference_type>(other.position_);
    }

    reference operator[](difference_type n) const { return *(*this + n); }

    bool operator==(const const_iterator& other) const {
      return owner_ == other.owner_ && position_ == other.position_;
    }

    bool operator!=(const const_iterator& other) const {
      return !(*this == other);
    }

    bool operator<(const const_iterator& other) const {
      return position_ < other.position_;
    }

    bool operator>(const const_iterator& other) const { return other < *this; }

    bool operator<=(const const_iterator& other) const {
      return !(other < *this);
    }

    bool operator>=(const const_iterator& other) const {
      return !(*this < other);
    }

   private:
    friend class DenseOrderedSetBinary;

    const_iterator(const DenseOrderedSetBinary* owner, size_type position)
        : owner_(owner), position_(position) {}

    const DenseOrderedSetBinary* owner_ = nullptr;
    size_type position_ = 0;
  };

  DenseOrderedSetBinary() = default;

  explicit DenseOrderedSetBinary(const KeyOfValue& key_of) : key_of_(key_of) {}

  void reserve(size_type n) {
    values_.reserve(n);
    ordered_keys_.reserve(n);
    ordered_indices_.reserve(n);
  }

  bool empty() const noexcept { return values_.empty(); }

  size_type size() const noexcept { return values_.size(); }

  bool insert(const T& value) { return emplaceImpl(value); }

  bool insert(T&& value) { return emplaceImpl(std::move(value)); }

  bool contains(const Key& key) const {
    const auto it = findPosition(key);
    return it != ordered_keys_.end() && equivalent(*it, key);
  }

  const T& at(const Key& key) const {
    const auto it = findPosition(key);

    assert(it != ordered_keys_.end() && equivalent(*it, key));

    const size_type position =
        static_cast<size_type>(it - ordered_keys_.begin());

    return values_[ordered_indices_[position]];
  }

  const T& front() const {
    assert(!empty());
    return values_[ordered_indices_.front()];
  }

  const T& back() const {
    assert(!empty());
    return values_[ordered_indices_.back()];
  }

  const_iterator find(const Key& key) const {
    const auto it = findPosition(key);

    if (it == ordered_keys_.end() || !equivalent(*it, key)) {
      return end();
    }

    return const_iterator{this,
                          static_cast<size_type>(it - ordered_keys_.begin())};
  }

  const_iterator begin() const noexcept { return const_iterator{this, 0}; }

  const_iterator end() const noexcept {
    return const_iterator{this, ordered_indices_.size()};
  }

 private:
  template <typename U>
  bool emplaceImpl(U&& value) {
    const Key key = key_of_(value);

    // Find the sorted position before modifying anything.
    const auto position = std::lower_bound(ordered_keys_.begin(),
                                           ordered_keys_.end(), key, compare_);

    const size_type ordered_position =
        static_cast<size_type>(position - ordered_keys_.begin());

    // Uniqueness check.
    if (position != ordered_keys_.end() && equivalent(*position, key)) {
      return false;
    }

    // Values remain in insertion order / storage order.
    const size_type value_index = values_.size();
    values_.emplace_back(std::forward<U>(value));

    // Maintain the sorted key/index mapping.
    ordered_keys_.insert(
        ordered_keys_.begin() + static_cast<std::ptrdiff_t>(ordered_position),
        key);

    ordered_indices_.insert(ordered_indices_.begin() +
                                static_cast<std::ptrdiff_t>(ordered_position),
                            value_index);

    return true;
  }

  // const_iterator findPosition(const Key& key) const {
  //   const auto it = std::lower_bound(
  //       ordered_keys_.begin(),
  //       ordered_keys_.end(),
  //       key,
  //       compare_);

  //   return const_iterator{
  //       this,
  //       static_cast<size_type>(it - ordered_keys_.begin())};
  // }

  using KeyIterator = typename std::vector<Key>::const_iterator;
  // This overload is only used internally for findPosition's
  // existence check, so return an actual vector iterator instead.
  KeyIterator findPosition(const Key& key) const {
    return std::lower_bound(ordered_keys_.begin(), ordered_keys_.end(), key,
                            compare_);
  }

  bool equivalent(const Key& lhs, const Key& rhs) const {
    return !compare_(lhs, rhs) && !compare_(rhs, lhs);
  }

  std::vector<T> values_;
  std::vector<Key> ordered_keys_;
  std::vector<size_type> ordered_indices_;

  KeyOfValue key_of_;
  Compare compare_;
};

// ============================================================================
// Dummy specialisation
// ============================================================================

struct DummyKey {
  std::uint64_t operator()(const Dummy& value) const noexcept {
    return value.id;
  }
};

struct DummyCompare {
  // enables heterogeneous lookup
  using is_transparent = void;

  bool operator()(const Dummy& a, const Dummy& b) const { return a.id < b.id; }

  bool operator()(const Dummy& a, std::uint64_t id) const { return a.id < id; }
  bool operator()(std::uint64_t id, const Dummy& a) const { return id < a.id; }
};

// ============================================================================
// Test container types
// ============================================================================

// Adjust these aliases/includes to your actual types.

using TestDenseSet = DenseSet<Dummy, std::uint64_t, DummyKey>;

using TestDenseOrderedSet =
    DenseOrderedSetBinary<Dummy, std::uint64_t, DummyKey>;

using TestDynoSet = dyno::FastSet<Dummy, DummyCompare>;

// ============================================================================
// TestContains
// ============================================================================

template <typename Container>
struct TestContains;

// DenseSet:
// O(1) average lookup through the unordered_map.
template <>
struct TestContains<TestDenseSet> {
  static bool contains(const TestDenseSet& container, std::uint64_t key) {
    return container.contains(key);
  }

  static Dummy at(const TestDenseSet& container, std::uint64_t key) {
    return container.at(key);
  }

  static auto find(const TestDenseSet& container, std::uint64_t key) {
    return container.find(key);
  }
};

// DenseOrderedSet:
// O(log N) lookup through ordered_keys_.
template <>
struct TestContains<TestDenseOrderedSet> {
  static bool contains(const TestDenseOrderedSet& container,
                       std::uint64_t key) {
    return container.contains(key);
  }

  static Dummy at(const TestDenseOrderedSet& container, std::uint64_t key) {
    return container.at(key);
  }

  static auto find(const TestDenseOrderedSet& container, std::uint64_t key) {
    return container.find(key);
  }
};

// std::set:
// Use find() because std::set has no contains()/at().
template <>
struct TestContains<TestDynoSet> {
  static bool contains(const TestDynoSet& container, std::uint64_t key) {
    return container.find(Dummy{key, 0}) != container.end();
  }

  static Dummy at(const TestDynoSet& container, std::uint64_t key) {
    return *container.find(Dummy{key, 0});
  }

  static auto find(const TestDynoSet& container, std::uint64_t key) {
    return container.find(key);
  }
};

// ============================================================================
// Helpers
// ============================================================================

Dummy makeDummy(std::uint64_t id) {
  Dummy value;
  value.id = id;

  // Make the object large enough that moving objects is non-trivial,
  // while still keeping this benchmark simple.
  value.payload = id * 0x9e3779b97f4a7c15ULL + 1234567;

  return value;
}

enum class InsertionOrder { Increasing, Decreasing, Random, NearlySorted };

std::vector<std::uint64_t> makeKeys(std::size_t n, InsertionOrder order,
                                    std::uint64_t seed = 12345) {
  std::vector<std::uint64_t> keys(n);

  std::iota(keys.begin(), keys.end(), std::uint64_t{0});

  switch (order) {
    case InsertionOrder::Increasing:
      break;

    case InsertionOrder::Decreasing:
      std::reverse(keys.begin(), keys.end());
      break;

    case InsertionOrder::Random: {
      std::mt19937_64 rng(seed);

      std::shuffle(keys.begin(), keys.end(), rng);

      break;
    }

    case InsertionOrder::NearlySorted: {
      if (n < 2) {
        break;
      }

      std::mt19937_64 rng(seed);

      const std::size_t swaps = std::max<std::size_t>(1, n / 100);

      std::uniform_int_distribution<std::size_t> dist(0, n - 1);

      for (std::size_t i = 0; i < swaps; ++i) {
        std::swap(keys[dist(rng)], keys[dist(rng)]);
      }

      break;
    }
  }

  return keys;
}

// ============================================================================
// Correctness: basic behaviour
// ============================================================================

TEST(DenseSetCorrectness, StartsEmpty) {
  TestDenseSet set;

  EXPECT_TRUE(set.empty());
  EXPECT_EQ(set.size(), 0);
}

TEST(DenseOrderedSetCorrectness, StartsEmpty) {
  TestDenseOrderedSet set;

  EXPECT_TRUE(set.empty());
  EXPECT_EQ(set.size(), 0);
}

TEST(DenseSetCorrectness, InsertSingleElement) {
  TestDenseSet set;

  EXPECT_TRUE(set.insert(makeDummy(42)));

  EXPECT_FALSE(set.empty());
  EXPECT_EQ(set.size(), 1);

  EXPECT_TRUE(set.contains(42));
  EXPECT_EQ(set.at(42).id, 42);

  EXPECT_EQ(set.front().id, 42);
  EXPECT_EQ(set.back().id, 42);
}

TEST(DenseOrderedSetCorrectness, InsertSingleElement) {
  TestDenseOrderedSet set;

  EXPECT_TRUE(set.insert(makeDummy(42)));

  EXPECT_FALSE(set.empty());
  EXPECT_EQ(set.size(), 1);

  EXPECT_TRUE(set.contains(42));
  EXPECT_EQ(set.at(42).id, 42);

  EXPECT_EQ(set.front().id, 42);
  EXPECT_EQ(set.back().id, 42);
}

TEST(DenseSetCorrectness, DuplicateKeysAreRejected) {
  TestDenseSet set;

  Dummy first = makeDummy(42);
  Dummy duplicate = makeDummy(42);

  first.payload = 100;
  duplicate.payload = 999;

  EXPECT_TRUE(set.insert(first));
  EXPECT_FALSE(set.insert(duplicate));

  EXPECT_EQ(set.size(), 1);
  EXPECT_EQ(set.at(42).payload, 100);
}

TEST(DenseOrderedSetCorrectness, DuplicateKeysAreRejected) {
  TestDenseOrderedSet set;

  Dummy first = makeDummy(42);
  Dummy duplicate = makeDummy(42);

  first.payload = 100;
  duplicate.payload = 999;

  EXPECT_TRUE(set.insert(first));
  EXPECT_FALSE(set.insert(duplicate));

  EXPECT_EQ(set.size(), 1);
  EXPECT_EQ(set.at(42).payload, 100);
}

TEST(DenseSetCorrectness, MissingKey) {
  TestDenseSet set;

  set.insert(makeDummy(10));
  set.insert(makeDummy(20));

  EXPECT_FALSE(set.contains(30));
}

TEST(DenseOrderedSetCorrectness, MissingKey) {
  TestDenseOrderedSet set;

  set.insert(makeDummy(10));
  set.insert(makeDummy(20));

  EXPECT_FALSE(set.contains(30));
}

TEST(DenseSetCorrectness, LookupReturnsCorrectValue) {
  TestDenseSet set;

  for (std::uint64_t i = 0; i < 1000; ++i) {
    EXPECT_TRUE(set.insert(makeDummy(i)));
  }

  for (std::uint64_t i = 0; i < 1000; ++i) {
    ASSERT_TRUE(set.contains(i));
    EXPECT_EQ(set.at(i).id, i);
  }
}

TEST(DenseOrderedSetCorrectness, LookupReturnsCorrectValue) {
  TestDenseOrderedSet set;

  for (std::uint64_t i = 0; i < 1000; ++i) {
    EXPECT_TRUE(set.insert(makeDummy(i)));
  }

  for (std::uint64_t i = 0; i < 1000; ++i) {
    ASSERT_TRUE(set.contains(i));
    EXPECT_EQ(set.at(i).id, i);
  }
}

// ============================================================================
// Correctness: DenseSet does NOT guarantee ordering
// ============================================================================

TEST(DenseSetCorrectness, IterationFollowsStorageInsertionOrder) {
  TestDenseSet set;

  const std::vector<std::uint64_t> keys = {7, 2, 9, 1, 5, 3};

  for (const auto key : keys) {
    ASSERT_TRUE(set.insert(makeDummy(key)));
  }

  ASSERT_EQ(set.size(), keys.size());

  std::size_t index = 0;

  for (const auto& value : set) {
    ASSERT_LT(index, keys.size());
    EXPECT_EQ(value.id, keys[index]);
    ++index;
  }

  EXPECT_EQ(index, keys.size());
}

TEST(DenseSetCorrectness, FrontAndBackFollowStorageOrder) {
  TestDenseSet set;

  ASSERT_TRUE(set.insert(makeDummy(50)));
  ASSERT_TRUE(set.insert(makeDummy(10)));
  ASSERT_TRUE(set.insert(makeDummy(90)));

  EXPECT_EQ(set.front().id, 50);
  EXPECT_EQ(set.back().id, 90);
}

// ============================================================================
// Correctness: DenseOrderedSet ordering
// ============================================================================

TEST(DenseOrderedSetCorrectness, IncreasingInsertionIsOrdered) {
  TestDenseOrderedSet set;

  for (std::uint64_t i = 0; i < 100; ++i) {
    ASSERT_TRUE(set.insert(makeDummy(i)));
  }

  ASSERT_EQ(set.front().id, 0);
  ASSERT_EQ(set.back().id, 99);

  std::uint64_t expected = 0;

  for (const auto& value : set) {
    EXPECT_EQ(value.id, expected);
    ++expected;
  }

  EXPECT_EQ(expected, 100);
}

TEST(DenseOrderedSetCorrectness, DecreasingInsertionIsOrdered) {
  TestDenseOrderedSet set;

  for (std::uint64_t i = 100; i > 0; --i) {
    ASSERT_TRUE(set.insert(makeDummy(i - 1)));
  }

  ASSERT_EQ(set.front().id, 0);
  ASSERT_EQ(set.back().id, 99);

  std::uint64_t expected = 0;

  for (const auto& value : set) {
    EXPECT_EQ(value.id, expected);
    ++expected;
  }

  EXPECT_EQ(expected, 100);
}

TEST(DenseOrderedSetCorrectness, RandomInsertionIsOrdered) {
  constexpr std::size_t N = 10000;

  const auto keys = makeKeys(N, InsertionOrder::Random, 12345);

  TestDenseOrderedSet set;

  for (const auto key : keys) {
    ASSERT_TRUE(set.insert(makeDummy(key)));
  }

  ASSERT_EQ(set.size(), N);
  ASSERT_EQ(set.front().id, 0);
  ASSERT_EQ(set.back().id, N - 1);

  std::uint64_t expected = 0;

  for (const auto& value : set) {
    ASSERT_EQ(value.id, expected);
    ++expected;
  }

  EXPECT_EQ(expected, N);
}

TEST(DenseOrderedSetCorrectness, NearlySortedInsertionIsOrdered) {
  constexpr std::size_t N = 10000;

  const auto keys = makeKeys(N, InsertionOrder::NearlySorted, 54321);

  TestDenseOrderedSet set;

  for (const auto key : keys) {
    ASSERT_TRUE(set.insert(makeDummy(key)));
  }

  ASSERT_EQ(set.size(), N);
  ASSERT_EQ(set.front().id, 0);
  ASSERT_EQ(set.back().id, N - 1);

  std::uint64_t previous = 0;
  bool first = true;

  for (const auto& value : set) {
    if (!first) {
      EXPECT_GT(value.id, previous);
    }

    previous = value.id;
    first = false;
  }
}

// ============================================================================
// Correctness: arbitrary insertion sequences
// ============================================================================

TEST(DenseOrderedSetCorrectness, AllInsertionOrdersProduceSameOrderedView) {
  constexpr std::size_t N = 5000;

  std::vector<std::uint64_t> expected(N);

  std::iota(expected.begin(), expected.end(), std::uint64_t{0});

  for (const auto order :
       {InsertionOrder::Increasing, InsertionOrder::Decreasing,
        InsertionOrder::Random, InsertionOrder::NearlySorted}) {
    TestDenseOrderedSet set;

    const auto keys = makeKeys(N, order, 1234);

    for (const auto key : keys) {
      ASSERT_TRUE(set.insert(makeDummy(key)));
    }

    ASSERT_EQ(set.size(), N);

    std::size_t index = 0;

    for (const auto& value : set) {
      ASSERT_LT(index, expected.size());

      EXPECT_EQ(value.id, expected[index]);

      ++index;
    }

    EXPECT_EQ(index, N);
    EXPECT_EQ(set.front().id, 0);
    EXPECT_EQ(set.back().id, N - 1);
  }
}

// ============================================================================
// Correctness: compare ordered dense set against std::set
// ============================================================================

TEST(DenseOrderedSetCorrectness, MatchesStdSetOrdering) {
  constexpr std::size_t N = 10000;

  const auto keys = makeKeys(N, InsertionOrder::Random, 98765);

  TestDenseOrderedSet dense;
  TestDynoSet tree;

  for (const auto key : keys) {
    const Dummy value = makeDummy(key);

    ASSERT_TRUE(dense.insert(value));

    ASSERT_TRUE(tree.insert(value).second);
  }

  ASSERT_EQ(dense.size(), tree.size());

  auto dense_it = dense.begin();
  auto tree_it = tree.begin();

  while (dense_it != dense.end() && tree_it != tree.end()) {
    EXPECT_EQ(dense_it->id, tree_it->id);

    EXPECT_EQ(dense_it->payload, tree_it->payload);

    ++dense_it;
    ++tree_it;
  }

  EXPECT_EQ(dense_it, dense.end());

  EXPECT_EQ(tree_it, tree.end());

  EXPECT_EQ(dense.front().id, tree.begin()->id);

  EXPECT_EQ(dense.back().id, tree.rbegin()->id);
}

// ============================================================================
// Correctness: duplicate stress
// ============================================================================

TEST(DenseSetCorrectness, DuplicateStress) {
  constexpr std::size_t N = 10000;

  TestDenseSet set;

  for (std::size_t i = 0; i < N; ++i) {
    ASSERT_TRUE(set.insert(makeDummy(i)));
  }

  ASSERT_EQ(set.size(), N);

  std::vector<std::uint64_t> keys;
  keys.reserve(N * 5);

  for (std::size_t repeat = 0; repeat < 5; ++repeat) {
    for (std::size_t i = 0; i < N; ++i) {
      keys.push_back(i);
    }
  }

  std::mt19937_64 rng(12345);

  std::shuffle(keys.begin(), keys.end(), rng);

  for (const auto key : keys) {
    EXPECT_FALSE(set.insert(makeDummy(key)));
  }

  EXPECT_EQ(set.size(), N);
}

TEST(DenseOrderedSetCorrectness, DuplicateStress) {
  constexpr std::size_t N = 10000;

  TestDenseOrderedSet set;

  for (std::size_t i = 0; i < N; ++i) {
    ASSERT_TRUE(set.insert(makeDummy(i)));
  }

  ASSERT_EQ(set.size(), N);

  std::vector<std::uint64_t> keys;
  keys.reserve(N * 5);

  for (std::size_t repeat = 0; repeat < 5; ++repeat) {
    for (std::size_t i = 0; i < N; ++i) {
      keys.push_back(i);
    }
  }

  std::mt19937_64 rng(12345);

  std::shuffle(keys.begin(), keys.end(), rng);

  for (const auto key : keys) {
    EXPECT_FALSE(set.insert(makeDummy(key)));
  }

  EXPECT_EQ(set.size(), N);
}

// ============================================================================
// Performance infrastructure
// ============================================================================

struct BenchmarkResult {
  double insertion_ns = 0.0;
  double lookup_ns = 0.0;
  double iteration_ns = 0.0;

  std::uint64_t checksum = 0;
};

template <typename Container>
BenchmarkResult benchmarkContainer(
    const std::vector<std::uint64_t>& insertion_keys,
    const std::vector<std::uint64_t>& lookup_keys, std::size_t repetitions) {
  using Clock = std::chrono::steady_clock;

  double insertion_total = 0.0;
  double lookup_total = 0.0;
  double iteration_total = 0.0;

  std::uint64_t checksum = 0;

  using TestContainsT = TestContains<Container>;

  for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
    Container container;

    // ------------------------------------------------------------------------
    // Insertion
    // ------------------------------------------------------------------------

    auto start = Clock::now();

    for (const auto key : insertion_keys) {
      container.insert(makeDummy(key));
    }

    auto end = Clock::now();

    insertion_total +=
        std::chrono::duration<double, std::nano>(end - start).count();

    // ------------------------------------------------------------------------
    // Lookup
    //
    // Important:
    // This performs one logical lookup per key.
    //
    // We use the adapter's `at()` directly rather than:
    //
    //   contains() + at()
    //
    // because that would perform TWO lookups for every successful key.
    //
    // The benchmark contains both existing and missing keys.
    // ------------------------------------------------------------------------

    std::uint64_t lookup_checksum = 0;

    start = Clock::now();

    for (const auto key : lookup_keys) {
      const auto it = TestContains<Container>::find(container, key);
      if (it != container.end()) {
        checksum += it->payload;
      }
      // if (TestContainsT::contains(
      //         container,
      //         key)) {

      //   lookup_checksum +=
      //       TestContainsT::at(
      //           container,
      //           key)
      //           .payload;
      // }
    }

    end = Clock::now();

    lookup_total +=
        std::chrono::duration<double, std::nano>(end - start).count();

    // ------------------------------------------------------------------------
    // Iteration
    // ------------------------------------------------------------------------

    std::uint64_t iteration_checksum = 0;

    start = Clock::now();

    for (const auto& value : container) {
      iteration_checksum += value.id;

      iteration_checksum += value.payload;
    }

    end = Clock::now();

    iteration_total +=
        std::chrono::duration<double, std::nano>(end - start).count();

    checksum ^= lookup_checksum + iteration_checksum;
  }

  return {insertion_total / static_cast<double>(repetitions),

          lookup_total / static_cast<double>(repetitions),

          iteration_total / static_cast<double>(repetitions),

          checksum};
}

// ============================================================================
// Performance output
// ============================================================================

const char* insertionOrderName(InsertionOrder order) {
  switch (order) {
    case InsertionOrder::Increasing:
      return "increasing";

    case InsertionOrder::Decreasing:
      return "decreasing";

    case InsertionOrder::Random:
      return "random";

    case InsertionOrder::NearlySorted:
      return "nearly-sorted";
  }

  return "unknown";
}

// ============================================================================
// Run one benchmark
// ============================================================================

void runBenchmark(InsertionOrder order, std::size_t n,
                  std::size_t repetitions) {
  const auto insertion_keys = makeKeys(n, order, 123456 + n);

  // --------------------------------------------------------------------------
  // Lookup workload
  //
  // 50% existing keys
  // 50% missing keys
  // --------------------------------------------------------------------------

  std::vector<std::uint64_t> lookup_keys;

  lookup_keys.reserve(2 * n);

  for (std::size_t i = 0; i < n; ++i) {
    lookup_keys.push_back(insertion_keys[i]);

    lookup_keys.push_back(static_cast<std::uint64_t>(n + i));
  }

  std::mt19937_64 rng(999999 + n);

  std::shuffle(lookup_keys.begin(), lookup_keys.end(), rng);

  // --------------------------------------------------------------------------
  // Benchmark all three containers
  // --------------------------------------------------------------------------

  const auto dense = benchmarkContainer<TestDenseSet>(insertion_keys,
                                                      lookup_keys, repetitions);

  const auto dense_ordered = benchmarkContainer<TestDenseOrderedSet>(
      insertion_keys, lookup_keys, repetitions);

  const auto tree =
      benchmarkContainer<TestDynoSet>(insertion_keys, lookup_keys, repetitions);

  // --------------------------------------------------------------------------
  // Sanity check
  //
  // All three containers should perform the same logical lookup/iteration
  // work.
  // --------------------------------------------------------------------------

  ASSERT_EQ(dense.checksum, dense_ordered.checksum);

  ASSERT_EQ(dense.checksum, tree.checksum);

  // --------------------------------------------------------------------------
  // Ratios relative to std::set
  //
  // > 1.0 = dense is faster
  // < 1.0 = dense is slower
  // --------------------------------------------------------------------------

  const double dense_insert_x = tree.insertion_ns / dense.insertion_ns;

  const double ordered_insert_x =
      tree.insertion_ns / dense_ordered.insertion_ns;

  const double dense_lookup_x = tree.lookup_ns / dense.lookup_ns;

  const double ordered_lookup_x = tree.lookup_ns / dense_ordered.lookup_ns;

  const double dense_iterate_x = tree.iteration_ns / dense.iteration_ns;

  const double ordered_iterate_x =
      tree.iteration_ns / dense_ordered.iteration_ns;

  std::cout << std::left << std::setw(16) << insertionOrderName(order)

            << std::right << std::setw(8) << n

            << std::setw(15) << dense.insertion_ns

            << std::setw(15) << dense_ordered.insertion_ns

            << std::setw(15) << tree.insertion_ns

            << std::setw(10) << dense_insert_x

            << std::setw(10) << ordered_insert_x

            << std::setw(15) << dense.lookup_ns

            << std::setw(15) << dense_ordered.lookup_ns

            << std::setw(15) << tree.lookup_ns

            << std::setw(10) << dense_lookup_x

            << std::setw(10) << ordered_lookup_x

            << std::setw(15) << dense.iteration_ns

            << std::setw(15) << dense_ordered.iteration_ns

            << std::setw(15) << tree.iteration_ns

            << std::setw(10) << dense_iterate_x

            << std::setw(10) << ordered_iterate_x

            << '\n';
}

// ============================================================================
// Performance test
// ============================================================================

TEST(DenseSetPerformance, CompareAllContainers) {
  std::cout << '\n';

  std::cout << std::left << std::setw(16) << "order"

            << std::setw(8) << "N"

            << std::setw(15) << "dense insert"

            << std::setw(15) << "ordered insert"

            << std::setw(15) << "set insert"

            << std::setw(10) << "dense x"

            << std::setw(10) << "ordered x"

            << std::setw(15) << "dense lookup"

            << std::setw(15) << "ordered lookup"

            << std::setw(15) << "set lookup"

            << std::setw(10) << "dense x"

            << std::setw(10) << "ordered x"

            << std::setw(15) << "dense iterate"

            << std::setw(15) << "ordered iterate"

            << std::setw(15) << "set iterate"

            << std::setw(10) << "dense x"

            << std::setw(10) << "ordered x"

            << '\n';

  std::cout << std::string(200, '-') << '\n';

  const std::vector<std::size_t> sizes = {16,   32,   64,   128,  256,  512,
                                          1024, 2048, 4096, 8192, 16384};

  const std::vector<InsertionOrder> orders = {
      InsertionOrder::Increasing, InsertionOrder::Decreasing,
      InsertionOrder::Random, InsertionOrder::NearlySorted};

  for (const auto order : orders) {
    for (const auto n : sizes) {
      const std::size_t repetitions = n <= 128    ? 1000
                                      : n <= 1024 ? 300
                                      : n <= 4096 ? 100
                                                  : 30;

      runBenchmark(order, n, repetitions);
    }
  }
}
