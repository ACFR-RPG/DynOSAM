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
#include "dynosam_common/Map.hpp"
#include "dynosam_common/Types.hpp"
#include "internal/helpers.hpp"

using namespace dyno;

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

  EXPECT_EQ(map->getStaticTrackletsByFrame(0), expected_tracklets);

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

  EXPECT_EQ(map->getStaticTrackletsByFrame(0), expected_tracklets_f0);
  EXPECT_EQ(map->getStaticTrackletsByFrame(1), expected_tracklets_f1);

  // check for frames in some landmarks
  // should be seen in frames 0 and 1
  LandmarkNode<Keypoint>::Ptr lmk1 = map->getLandmark(0);
  std::vector<FrameId> lmk_1_seen_frames =
      lmk1->getSeenFrames().collectIds<FrameId>();
  std::vector<FrameId> lmk_1_seen_frames_expected = {0, 1};
  EXPECT_EQ(lmk_1_seen_frames, lmk_1_seen_frames_expected);

  // should be seen in frames 0
  LandmarkNode<Keypoint>::Ptr lmk6 = map->getLandmark(6);
  std::vector<FrameId> lmk_6_seen_frames =
      lmk6->getSeenFrames().collectIds<FrameId>();
  std::vector<FrameId> lmk_6_seen_frames_expected = {0};
  EXPECT_EQ(lmk_6_seen_frames, lmk_6_seen_frames_expected);

  // check that the frames here are the ones in the map
  EXPECT_EQ(map->getFrame(lmk_1_seen_frames.at(0)),
            map->getFrame(lmk_6_seen_frames.at(0)));

  // finally check that there are no objects
  EXPECT_EQ(map->getFrame(lmk_1_seen_frames.at(0))->objects_seen.size(), 0);
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

  LandmarkNode<Keypoint>::Ptr lmk = map->getLandmark(1);
  EXPECT_TRUE(lmk != nullptr);

  FrameIds expected_frame_ids = {0, 1, 2, 3};
  EXPECT_EQ(lmk->getSeenFrames().collectIds<FrameId>(), expected_frame_ids);

  EXPECT_EQ(lmk->getSeenFrames().getFirstIndex(), 0u);
  EXPECT_EQ(lmk->getSeenFrames().getLastIndex(), 3u);
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

  ObjectNode2d::Ptr object1 = map->getObject(1);
  // object 1 should have 1 point seen at frames 0 and 1
  EXPECT_EQ(object1->dynamic_landmarks.collectIds<TrackletId>(),
            TrackletIds{0});
  EXPECT_EQ(object1->dynamic_landmarks.size(), 1u);

  // now check that the frames also have these measurements
  FrameNode2d::Ptr frame_0 = map->getFrame(0);
  FrameNode2d::Ptr frame_1 = map->getFrame(1);
  EXPECT_TRUE(frame_0 != nullptr);
  EXPECT_TRUE(frame_1 != nullptr);

  EXPECT_EQ(frame_0->dynamic_landmarks.collectIds<TrackletId>(),
            TrackletIds{0});
  EXPECT_EQ(frame_1->dynamic_landmarks.collectIds<TrackletId>(),
            TrackletIds{0});

  // sanity check that there are no static points
  EXPECT_EQ(frame_0->static_landmarks.size(), 0u);
  EXPECT_EQ(frame_1->static_landmarks.size(), 0u);

  // check object id and seen frames
  LandmarkNode2d::Ptr lmk_0 = map->getLandmark(0);
  EXPECT_EQ(lmk_0->object_id, 1);  // object Id 1;
  EXPECT_EQ(lmk_0->getSeenFrames().collectIds<FrameId>(),
            FrameIds({0, 1}));  // seen frames

  // finally check that the landmark referred to by the frames are the same one
  // as getLandmark(0) this also implicitly tests FastMapNodeSet::find(index)
  auto frame_0_dynamic_lmks = frame_0->dynamic_landmarks;
  auto frame_1_dynamic_lmks = frame_1->dynamic_landmarks;

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
  LandmarkNode2d::Ptr landmark_node =
      std::make_shared<LandmarkNode2d>(map->getptr());
  landmark_node->tracklet_id = 0;
  landmark_node->object_id = 0;

  EXPECT_EQ(landmark_node->numObservations(), 0);

  FrameNode2d::Ptr frame_node = std::make_shared<FrameNode2d>(map->getptr());
  frame_node->frame_id = 0;

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

  ObjectNode2d::Ptr object1 = map->getObject(1);
  ObjectNode2d::Ptr object2 = map->getObject(2);
  ObjectNode2d::Ptr object3 = map->getObject(3);

  FrameNodePtrSet<Keypoint> expected_frame_set_object1;
  expected_frame_set_object1.insert(CHECK_NOTNULL(map->getFrame(0)));
  expected_frame_set_object1.insert(CHECK_NOTNULL(map->getFrame(1)));

  FrameNodePtrSet<Keypoint> expected_frame_set_object2;
  expected_frame_set_object2.insert(CHECK_NOTNULL(map->getFrame(1)));
  expected_frame_set_object2.insert(CHECK_NOTNULL(map->getFrame(2)));

  FrameNodePtrSet<Keypoint> expected_frame_set_object3;
  expected_frame_set_object3.insert(CHECK_NOTNULL(map->getFrame(0)));
  expected_frame_set_object3.insert(CHECK_NOTNULL(map->getFrame(1)));
  expected_frame_set_object3.insert(CHECK_NOTNULL(map->getFrame(2)));

  EXPECT_EQ(object1->getSeenFrames(), expected_frame_set_object1);
  EXPECT_EQ(object2->getSeenFrames(), expected_frame_set_object2);
  EXPECT_EQ(object3->getSeenFrames(), expected_frame_set_object3);

  // frame 0 has seen object 1 and 3 -the reverse of the above testt
  EXPECT_EQ(map->getFrame(0)->objects_seen,
            ObjectNodePtrSet<Keypoint>({object1, object3}));
  // frame 1 has seen object 1, 2 and 3
  EXPECT_EQ(map->getFrame(1)->objects_seen,
            ObjectNodePtrSet<Keypoint>({object1, object3, object2}));
  // frame 2 has seen object 2 and 3
  EXPECT_EQ(map->getFrame(2)->objects_seen,
            ObjectNodePtrSet<Keypoint>({object2, object3}));

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
  EXPECT_FALSE(frame0->objectObservedInPrevious(1));
  EXPECT_FALSE(frame0->objectObservedInPrevious(3));

  // both object1 and 3 appear in frame 0 but not object 2
  EXPECT_TRUE(frame1->objectObservedInPrevious(1));
  EXPECT_TRUE(frame1->objectObservedInPrevious(3));
  EXPECT_FALSE(frame1->objectObservedInPrevious(2));

  // all objects are observed at frame 1
  EXPECT_TRUE(frame2->objectObservedInPrevious(1));
  EXPECT_TRUE(frame2->objectObservedInPrevious(3));
  EXPECT_TRUE(frame2->objectObservedInPrevious(2));

  // check objectMotionExpected (i.e objects are observed at both frames)
  EXPECT_FALSE(frame0->objectMotionExpected(1));
  EXPECT_FALSE(frame0->objectMotionExpected(3));

  // object 1 and 3 seen at frames 0 and 1, but not object 2
  EXPECT_TRUE(frame1->objectMotionExpected(1));
  EXPECT_TRUE(frame1->objectMotionExpected(3));
  EXPECT_FALSE(frame1->objectMotionExpected(2));
  // object 2 and 3 seen at frames 1 and 2, but not object 1
  EXPECT_FALSE(frame2->objectMotionExpected(1));
  EXPECT_TRUE(frame2->objectMotionExpected(3));
  EXPECT_TRUE(frame2->objectMotionExpected(2));
}

TEST(Map, getLandmarksSeenAtFrame) {
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

  ObjectNode2d::Ptr object1 = map->getObject(1);
  ObjectNode2d::Ptr object2 = map->getObject(2);
  ObjectNode2d::Ptr object3 = map->getObject(3);

  LandmarkNodePtrSet<Keypoint> expected_lmk_set_object1_frame0;
  expected_lmk_set_object1_frame0.insert(CHECK_NOTNULL(map->getLandmark(0)));

  LandmarkNodePtrSet<Keypoint> expected_lmk_set_object1_frame1;
  expected_lmk_set_object1_frame1.insert(CHECK_NOTNULL(map->getLandmark(0)));

  LandmarkNodePtrSet<Keypoint> expected_lmk_set_object2_frame1;
  expected_lmk_set_object2_frame1.insert(CHECK_NOTNULL(map->getLandmark(1)));
  expected_lmk_set_object2_frame1.insert(CHECK_NOTNULL(map->getLandmark(2)));

  LandmarkNodePtrSet<Keypoint> expected_lmk_set_object3_frame0;
  expected_lmk_set_object3_frame0.insert(CHECK_NOTNULL(map->getLandmark(3)));

  LandmarkNodePtrSet<Keypoint> expected_lmk_set_object3_frame1;
  expected_lmk_set_object3_frame1.insert(CHECK_NOTNULL(map->getLandmark(3)));
  expected_lmk_set_object3_frame1.insert(CHECK_NOTNULL(map->getLandmark(4)));

  EXPECT_EQ(object1->getLandmarksSeenAtFrame(0),
            expected_lmk_set_object1_frame0);
  EXPECT_EQ(object1->getLandmarksSeenAtFrame(1),
            expected_lmk_set_object1_frame1);
  EXPECT_EQ(object1->getLandmarksSeenAtFrame(2),
            LandmarkNodePtrSet<Keypoint>{});
  EXPECT_EQ(object2->getLandmarksSeenAtFrame(1),
            expected_lmk_set_object2_frame1);
  EXPECT_EQ(object3->getLandmarksSeenAtFrame(0),
            expected_lmk_set_object3_frame0);
  EXPECT_EQ(object3->getLandmarksSeenAtFrame(1),
            expected_lmk_set_object3_frame1);
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

// TEST(Map, testEstimateWithStaticAndDynamicViaLmk) {
//     Map2d::Ptr map = Map2d::create();
//     StatusKeypointMeasurements measurements;

//     //tracklet 0, static, frame 0
//     measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(0, 0,
//     0));
//     //tracklet 0, static, frame 1
//     measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(0, 0,
//     1));
//     //static points should return the same estimate

//     //tracklet 1, dynamic, frame 0
//     measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(1, 1,
//     0));
//     //tracklet 1, dynamic, frame 1
//     measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(1, 1,
//     1));
//     //dynamic points should return different estimates

//     gtsam::Point3 static_lmk(0, 0, 0);
//     gtsam::Point3 dynamic_lmk_1(0, 0, 1);
//     gtsam::Point3 dynamic_lmk_2(0, 0, 2);

//     gtsam::Values estimate;
//     estimate.insert(StaticLandmarkSymbol(0), static_lmk);
//     estimate.insert(DynamicLandmarkSymbol(0, 1), dynamic_lmk_1);
//     estimate.insert(DynamicLandmarkSymbol(1, 1), dynamic_lmk_2);

//     map->updateObservations(measurements);
//     map->updateEstimates(estimate,gtsam::NonlinearFactorGraph{}, 0 );

//     LandmarkNode2d::Ptr static_lmk_node = map->getLandmark(0);
//     EXPECT_TRUE(static_lmk_node->isStatic());
//     auto estimate_query = static_lmk_node->getStaticLandmarkEstimate();
//     EXPECT_TRUE(estimate_query);
//     EXPECT_TRUE(estimate_query.isValid());
//     EXPECT_TRUE(gtsam::assert_equal(estimate_query.get(), static_lmk));

//     LandmarkNode2d::Ptr dynamic_lmk_1_node = map->getLandmark(1);
//     EXPECT_FALSE(dynamic_lmk_1_node->isStatic());
//     estimate_query = dynamic_lmk_1_node->getDynamicLandmarkEstimate(0);
//     EXPECT_TRUE(estimate_query);
//     EXPECT_TRUE(estimate_query.isValid());
//     EXPECT_TRUE(gtsam::assert_equal(estimate_query.get(), dynamic_lmk_1));

//     LandmarkNode2d::Ptr dynamic_lmk_2_node = map->getLandmark(1);
//     EXPECT_FALSE(dynamic_lmk_2_node->isStatic());
//     estimate_query = dynamic_lmk_2_node->getDynamicLandmarkEstimate(1);
//     EXPECT_TRUE(estimate_query);
//     EXPECT_TRUE(estimate_query.isValid());
//     EXPECT_TRUE(gtsam::assert_equal(estimate_query.get(), dynamic_lmk_2));

// }

// TEST(Map, testEstimateWithStaticAndDynamicViaFrame) {

// }

// TEST(Map, testObjectNodeWithOnlyMotion) {
//     Map2d::Ptr map = Map2d::create();

//     //create dynamic observations for object 1 at frames 0 and 1
//     StatusKeypointMeasurements measurements;
//      //tracklet 1, dynamic, frame 0
//     measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(1, 1,
//     0));
//     //tracklet 1, dynamic, frame 1
//     measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(1, 1,
//     1));

//     gtsam::Values estimate;
//     //add motion only at frame 1
//     const gtsam::Pose3 pose_estimate =
//     utils::createRandomAroundIdentity<gtsam::Pose3>(0.3);
//     estimate.insert(ObjectMotionSymbol(1, 1), pose_estimate);

//     map->updateObservations(measurements);
//     map->updateEstimates(estimate,gtsam::NonlinearFactorGraph{}, 1 );

//     ObjectNode2d::Ptr object1 = map->getObject(1);
//     EXPECT_TRUE(object1->getMotionEstimate(1));

//     gtsam::Pose3 recovered_pose_estimate;
//     EXPECT_TRUE(object1->hasMotionEstimate(1, &recovered_pose_estimate));
//     //check nullptr version
//     EXPECT_TRUE(object1->hasMotionEstimate(1));
//     EXPECT_TRUE(gtsam::assert_equal(recovered_pose_estimate, pose_estimate));

//     //should not throw exception as we have seen this obejct at frames 0 and
//     1 (based on the measurements)
//     //we just dont have an estimate of the pose
//     EXPECT_FALSE(object1->getPoseEstimate(0));
//     EXPECT_FALSE(object1->getPoseEstimate(1));
// }

// TEST(Map, testObjectNodeWithOnlyPose) {
//     Map2d::Ptr map = Map2d::create();

//     //create dynamic observations for object 1 at frames 0 and 1
//     StatusKeypointMeasurements measurements;
//      //tracklet 1, dynamic, frame 0
//     measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(1, 1,
//     0));
//     //tracklet 1, dynamic, frame 1
//     measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(1, 1,
//     1));

//     gtsam::Values estimate;
//     //add motion only at frame 1
//     const gtsam::Pose3 pose_estimate =
//     utils::createRandomAroundIdentity<gtsam::Pose3>(0.3);
//     estimate.insert(ObjectPoseSymbol(1, 1), pose_estimate);

//     map->updateObservations(measurements);
//     map->updateEstimates(estimate,gtsam::NonlinearFactorGraph{}, 1 );

//     ObjectNode2d::Ptr object1 = map->getObject(1);
//     EXPECT_TRUE(object1->getPoseEstimate(1));

//     gtsam::Pose3 recovered_pose_estimate;
//     EXPECT_TRUE(object1->hasPoseEstimate(1, &recovered_pose_estimate));
//     //check nullptr version
//     EXPECT_TRUE(object1->hasPoseEstimate(1));
//     EXPECT_TRUE(gtsam::assert_equal(recovered_pose_estimate, pose_estimate));

//     //should not throw exception as we have seen this obejct at frames 0 and
//     1 (based on the measurements)
//     //we just dont have an estimate of the motion
//     EXPECT_FALSE(object1->getMotionEstimate(1));
// }
