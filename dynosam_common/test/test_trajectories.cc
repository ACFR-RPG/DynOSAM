
#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "dynosam_common/Trajectories.hpp"

using namespace dyno;

using TestTrajectoryInt = Trajectory<int>;
using TrajectoryEntryAlreadyExists =
    TestTrajectoryInt::TrajectoryEntryAlreadyExists;

TEST(TrajectoryBase, EmptyTrajectory) {
  TestTrajectoryInt traj;

  EXPECT_TRUE(traj.empty());
  EXPECT_EQ(traj.size(), 0u);
}

TEST(TrajectoryBase, InsertAndAccess) {
  TestTrajectoryInt traj;

  traj.insert(0, 0.0, 42);
  traj.insert(1, 0.1, 43);

  EXPECT_FALSE(traj.empty());
  EXPECT_EQ(traj.size(), 2u);

  EXPECT_EQ(traj.at(0), 42);
  EXPECT_EQ(traj.at(1), 43);
  EXPECT_DOUBLE_EQ(traj.timeAt(1), 0.1);
}

TEST(TrajectoryBase, DuplicateInsertThrows) {
  TestTrajectoryInt traj;

  traj.insert(0, 0.0, 42);

  EXPECT_THROW(traj.insert(0, 0.1, 43), TrajectoryEntryAlreadyExists);
}

TEST(TrajectoryBase, Exists) {
  TestTrajectoryInt traj;

  traj.insert(5, 0.5, 10);

  EXPECT_TRUE(traj.exists(5));
  EXPECT_FALSE(traj.exists(6));
}

TEST(TrajectoryBase, FirstAndLast) {
  TestTrajectoryInt traj;

  traj.insert(10, 1.0, 100);
  traj.insert(12, 1.2, 120);
  traj.insert(11, 1.1, 110);

  EXPECT_EQ(traj.first().frame_id, 10);
  EXPECT_EQ(traj.last().frame_id, 12);

  EXPECT_EQ(traj.first().data, 100);
  EXPECT_EQ(traj.last().data, 120);
}

TEST(TrajectoryBase, Update) {
  TestTrajectoryInt traj;
  traj.insert(0, 0.0, 1);

  EXPECT_TRUE(traj.update(0, 99));
  EXPECT_EQ(traj.at(0), 99);

  EXPECT_FALSE(traj.update(1, 10));
}

TEST(TrajectoryBase, GetPrevious) {
  TestTrajectoryInt traj;

  traj.insert(1, 0.1, 10);
  traj.insert(3, 0.3, 30);
  traj.insert(5, 0.5, 50);

  TestTrajectoryInt::Entry prev;

  EXPECT_TRUE(traj.getPrevious(prev, traj.get(3)));
  EXPECT_EQ(prev.frame_id, 1);

  EXPECT_FALSE(traj.getPrevious(prev, traj.first()));
}

TEST(TrajectoryBase, Segments) {
  TestTrajectoryInt traj;

  traj.insert(0, 0.0, 0);
  traj.insert(1, 0.1, 1);
  traj.insert(3, 0.3, 3);
  traj.insert(4, 0.4, 4);
  traj.insert(6, 0.6, 6);

  auto segments = traj.segments();

  ASSERT_EQ(segments.size(), 3u);

  EXPECT_EQ(segments[0].start_frame, 0);
  EXPECT_EQ(segments[0].end_frame, 1);
  EXPECT_TRUE(segments[0].check());

  EXPECT_EQ(segments[1].start_frame, 3);
  EXPECT_EQ(segments[1].end_frame, 4);
  EXPECT_TRUE(segments[1].check());

  EXPECT_EQ(segments[2].start_frame, 6);
  EXPECT_EQ(segments[2].end_frame, 6);
  EXPECT_TRUE(segments[2].check());
}

TEST(TrajectoryBase, ToVector) {
  TestTrajectoryInt traj;

  traj.insert(2, 0.2, 20);
  traj.insert(1, 0.1, 10);

  auto vec = traj.toVector();

  ASSERT_EQ(vec.size(), 2u);
  EXPECT_EQ(vec[0].frame_id, 1);
  EXPECT_EQ(vec[1].frame_id, 2);
}

TEST(TrajectoryBase, InsertFromOtherTrajectory) {
  TestTrajectoryInt src;
  src.insert(0, 0.0, 10).insert(1, 1.0, 20).insert(2, 2.0, 30);

  TestTrajectoryInt dst;
  dst.insert(5, 5.0, 100);

  dst.insert(src);

  EXPECT_EQ(dst.size(), 4);

  EXPECT_TRUE(dst.exists(0));
  EXPECT_TRUE(dst.exists(1));
  EXPECT_TRUE(dst.exists(2));
  EXPECT_TRUE(dst.exists(5));

  EXPECT_EQ(dst.at(0), 10);
  EXPECT_EQ(dst.at(1), 20);
  EXPECT_EQ(dst.at(2), 30);
  EXPECT_EQ(dst.at(5), 100);
}

TEST(TrajectoryBase, InsertThrowsIfFrameExists) {
  TestTrajectoryInt src;
  src.insert(0, 0.0, 10);

  TestTrajectoryInt dst;
  dst.insert(0, 0.0, 999);

  EXPECT_THROW(dst.insert(src), TrajectoryEntryAlreadyExists);
}

TEST(TrajectoryBase, InsertOrUpdateFromOtherTrajectory) {
  TestTrajectoryInt src;
  src.insert(0, 0.0, 10).insert(1, 1.0, 20);

  TestTrajectoryInt dst;
  dst.insert(1, 1.0, 999)  // will be updated
      .insert(2, 2.0, 30);

  dst.insertOrUpdate(src);

  EXPECT_EQ(dst.size(), 3);

  // inserted
  EXPECT_EQ(dst.at(0), 10);

  // updated
  EXPECT_EQ(dst.at(1), 20);

  // unchanged
  EXPECT_EQ(dst.at(2), 30);
}

TEST(TrajectoryBase, SelfInsertDoesNotCorrupt) {
  TestTrajectoryInt traj;
  traj.insert(0, 0.0, 10).insert(1, 1.0, 20);

  EXPECT_THROW(traj.insert(traj), TrajectoryEntryAlreadyExists);
}

TEST(TrajectoryBase, ReturnsPreviousEntryInContinuousTrajectory) {
  Trajectory<int> traj;
  traj.insert(0, 0.0, 10).insert(1, 1.0, 20).insert(2, 2.0, 30);

  Trajectory<int>::Entry query = traj.get(2);
  Trajectory<int>::Entry previous;

  bool found = traj.getPrevious(previous, query);

  EXPECT_TRUE(found);
  EXPECT_EQ(previous.frame_id, 1);
  EXPECT_EQ(previous.data, 20);
}

TEST(TrajectoryBase, ReturnsFalseForFirstElement) {
  Trajectory<int> traj;
  traj.insert(0, 0.0, 10).insert(1, 1.0, 20);

  Trajectory<int>::Entry query = traj.get(0);
  Trajectory<int>::Entry previous;

  bool found = traj.getPrevious(previous, query);

  EXPECT_FALSE(found);
}

TEST(TrajectoryBase, ReturnsFalseIfQueryDoesNotExist) {
  Trajectory<int> traj;
  traj.insert(0, 0.0, 10).insert(1, 1.0, 20);

  Trajectory<int>::Entry fake_query{5, 5.0, 999};
  Trajectory<int>::Entry previous;

  bool found = traj.getPrevious(previous, fake_query);

  EXPECT_FALSE(found);
}

TEST(TrajectoryBase, SingleElementTrajectoryReturnsFalse) {
  Trajectory<int> traj;
  traj.insert(0, 0.0, 10);

  Trajectory<int>::Entry query = traj.get(0);
  Trajectory<int>::Entry previous;

  bool found = traj.getPrevious(previous, query);

  EXPECT_FALSE(found);
}

TEST(TrajectoryBase, WorksAcrossDiscontinuousSegments) {
  Trajectory<int> traj;
  traj.insert(0, 0.0, 10)
      .insert(1, 1.0, 20)
      .insert(5, 5.0, 50);  // discontinuity

  Trajectory<int>::Entry query = traj.get(5);
  Trajectory<int>::Entry previous;

  bool found = traj.getPrevious(previous, query);

  EXPECT_TRUE(found);

  // Important: previous frame is 1, NOT 4
  EXPECT_EQ(previous.frame_id, 1);
  EXPECT_EQ(previous.data, 20);
}

TEST(TrajectoryBase, DoesNotModifyPreviousIfNotFound) {
  Trajectory<int> traj;
  traj.insert(0, 0.0, 10);

  Trajectory<int>::Entry query = traj.get(0);
  Trajectory<int>::Entry previous{999, 999.0, 999};

  bool found = traj.getPrevious(previous, query);

  EXPECT_FALSE(found);

  // Ensure previous not overwritten
  EXPECT_EQ(previous.frame_id, 999);
}

TEST(TrajectoryBase, GetRangeBasicAndClamping) {
  TestTrajectoryInt traj;

  // Insert frames 0..4
  for (FrameId i = 0; i <= 4; ++i) {
    traj.insert(i, i * 0.1, static_cast<int>(i));
  }

  // --- 1. Full range ---
  {
    auto sub = traj.range(std::nullopt, std::nullopt);
    EXPECT_EQ(sub.size(), 5);
    EXPECT_EQ(sub.first().frame_id, 0);
    EXPECT_EQ(sub.last().frame_id, 4);
  }

  // --- 2. Normal subrange ---
  {
    auto sub = traj.range(1, 3);
    EXPECT_EQ(sub.size(), 3);
    EXPECT_EQ(sub.first().frame_id, 1);
    EXPECT_EQ(sub.last().frame_id, 3);
  }

  // --- 3. end > max → clamp ---
  {
    auto sub = traj.range(2, 100);
    EXPECT_EQ(sub.size(), 3);  // 2,3,4
    EXPECT_EQ(sub.first().frame_id, 2);
    EXPECT_EQ(sub.last().frame_id, 4);
  }

  // --- 4. start > max → throw ---
  { EXPECT_THROW(traj.range(100, std::nullopt), DynosamException); }

  // --- 5. start <= max but start > end → throw ---
  { EXPECT_THROW(traj.range(3, 1), DynosamException); }
}

TEST(TrajectoryBase, Find) {
  TestTrajectoryInt traj;

  traj.insert(0, 0.0, 10);
  traj.insert(2, 2.0, 20);
  traj.insert(5, 5.0, 50);

  // Existing frame: find() should return a valid iterator.
  {
    auto it = traj.find(2);

    EXPECT_NE(it, traj.end());
    EXPECT_EQ(it->frame_id, 2);
    EXPECT_EQ(it->timestamp, 2.0);
    EXPECT_EQ(it->data, 20);
  }

  // Another existing frame.
  {
    auto it = traj.find(5);

    EXPECT_NE(it, traj.end());
    EXPECT_EQ(it->frame_id, 5);
    EXPECT_EQ(it->data, 50);
  }

  // Missing frame: find() should return end().
  {
    auto it = traj.find(3);

    EXPECT_EQ(it, traj.end());
  }

  // Missing frame before the trajectory.
  {
    auto it = traj.find(-1);

    EXPECT_EQ(it, traj.end());
  }

  // Missing frame after the trajectory.
  {
    auto it = traj.find(10);

    EXPECT_EQ(it, traj.end());
  }
}

TEST(TrajectoryBase, FindBegin) {
  TestTrajectoryInt traj;

  traj.insert(10, 10.0, 100);
  traj.insert(20, 20.0, 200);

  auto it = traj.find(10);

  ASSERT_NE(it, traj.end());
  EXPECT_EQ(it, traj.begin());
  EXPECT_EQ(it->data, 100);
}

TEST(TrajectoryBase, FindConst) {
  TestTrajectoryInt traj;

  traj.insert(10, 10.0, 100);
  traj.insert(20, 20.0, 200);

  const TestTrajectoryInt& const_traj = traj;

  auto it = const_traj.find(20);

  EXPECT_NE(it, const_traj.end());
  EXPECT_EQ(it->frame_id, 20);
  EXPECT_EQ(it->data, 200);

  EXPECT_EQ(const_traj.find(30), const_traj.end());
}
