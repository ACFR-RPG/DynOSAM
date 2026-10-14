#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "dynosam_core/GroundTruthPacket.hpp"

using namespace dyno;

TEST(GroundTruthInputPacket, findAssociatedObjectWithIdx) {
  ObjectPoseGT obj01;
  obj01.frame_id_ = 0;
  obj01.object_id_ = 1;

  ObjectPoseGT obj02;
  obj02.frame_id_ = 0;
  obj02.object_id_ = 2;

  ObjectPoseGT obj03;
  obj03.frame_id_ = 0;
  obj03.object_id_ = 3;

  ObjectPoseGT obj11;
  obj11.frame_id_ = 1;
  obj11.object_id_ = 1;

  ObjectPoseGT obj12;
  obj12.frame_id_ = 1;
  obj12.object_id_ = 2;

  GroundTruthInputPacket packet_0;
  packet_0.frame_id_ = 0;
  packet_0.object_poses_.push_back(obj01);
  packet_0.object_poses_.push_back(obj02);
  packet_0.object_poses_.push_back(obj03);

  GroundTruthInputPacket packet_1;
  packet_1.frame_id_ = 1;
  // put in out of order compared to packet_1
  packet_1.object_poses_.push_back(obj12);
  packet_1.object_poses_.push_back(obj11);

  size_t obj_idx, obj_other_idx;
  EXPECT_TRUE(
      packet_0.findAssociatedObject(1, packet_1, obj_idx, obj_other_idx));

  EXPECT_EQ(obj_idx, 0);
  EXPECT_EQ(obj_other_idx, 1);

  EXPECT_TRUE(
      packet_0.findAssociatedObject(2, packet_1, obj_idx, obj_other_idx));

  EXPECT_EQ(obj_idx, 1);
  EXPECT_EQ(obj_other_idx, 0);

  // object 3 is not in packet_1
  EXPECT_FALSE(
      packet_0.findAssociatedObject(3, packet_1, obj_idx, obj_other_idx));
}

TEST(GroundTruthInputPacket, findAssociatedObjectWithPtr) {
  ObjectPoseGT obj01;
  obj01.frame_id_ = 0;
  obj01.object_id_ = 1;

  ObjectPoseGT obj02;
  obj02.frame_id_ = 0;
  obj02.object_id_ = 2;

  ObjectPoseGT obj03;
  obj03.frame_id_ = 0;
  obj03.object_id_ = 3;

  ObjectPoseGT obj11;
  obj11.frame_id_ = 1;
  obj11.object_id_ = 1;

  ObjectPoseGT obj12;
  obj12.frame_id_ = 1;
  obj12.object_id_ = 2;

  GroundTruthInputPacket packet_0;
  packet_0.frame_id_ = 0;
  packet_0.object_poses_.push_back(obj01);
  packet_0.object_poses_.push_back(obj02);
  packet_0.object_poses_.push_back(obj03);

  GroundTruthInputPacket packet_1;
  packet_1.frame_id_ = 1;
  // put in out of order compared to packet_1
  packet_1.object_poses_.push_back(obj12);
  packet_1.object_poses_.push_back(obj11);

  ObjectPoseGT* obj;
  const ObjectPoseGT* obj_other;
  EXPECT_TRUE(packet_0.findAssociatedObject(2, packet_1, &obj, &obj_other));

  EXPECT_TRUE(obj != nullptr);
  EXPECT_TRUE(obj_other != nullptr);

  EXPECT_EQ(obj->object_id_, 2);
  EXPECT_EQ(obj_other->object_id_, 2);

  EXPECT_EQ(obj->frame_id_, 0);
  EXPECT_EQ(obj_other->frame_id_, 1);
}

TEST(SharedGroundTruth, DefaultConstructedIsInvalid) {
  SharedGroundTruth gt;

  EXPECT_FALSE(gt.valid());
  EXPECT_FALSE(gt.access().has_value());
}

TEST(SharedGroundTruth, PublisherProvidesValidHandle) {
  GroundTruthPublisher publisher;

  auto handle = publisher.handle();

  EXPECT_TRUE(handle.valid());
  EXPECT_FALSE(handle.access().has_value());
}

// ============================================================
// Insert + Access
// ============================================================

TEST(SharedGroundTruth, InsertSingleEntry) {
  GroundTruthPublisher publisher;
  auto handle = publisher.handle();

  GroundTruthInputPacket packet;
  packet.frame_id_ = 42;

  publisher.insert(1, packet);

  auto snapshot = handle.access();

  ASSERT_TRUE(snapshot.has_value());
  ASSERT_EQ(snapshot->size(), 1u);

  EXPECT_EQ(snapshot->at(1).frame_id_, 42);
}

TEST(SharedGroundTruth, MultipleInsertionsAccumulate) {
  GroundTruthPublisher publisher;
  auto handle = publisher.handle();

  for (FrameId i = 0; i < 5; ++i) {
    GroundTruthInputPacket p;
    p.frame_id_ = static_cast<int>(i);
    publisher.insert(i, p);
  }

  auto snapshot = handle.access();

  ASSERT_TRUE(snapshot.has_value());
  EXPECT_EQ(snapshot->size(), 5u);

  for (FrameId i = 0; i < 5; ++i) {
    EXPECT_EQ(snapshot->at(i).frame_id_, static_cast<int>(i));
  }
}

// ============================================================
// Copy Semantics
// ============================================================

TEST(SharedGroundTruth, CopiesShareState) {
  GroundTruthPublisher publisher;

  auto handleA = publisher.handle();
  auto handleB = handleA;  // copy

  GroundTruthInputPacket packet;
  packet.frame_id_ = 99;

  publisher.insert(10, packet);

  auto snapshotA = handleA.access();
  auto snapshotB = handleB.access();

  ASSERT_TRUE(snapshotA.has_value());
  ASSERT_TRUE(snapshotB.has_value());

  EXPECT_EQ(snapshotA->at(10).frame_id_, 99);
  EXPECT_EQ(snapshotB->at(10).frame_id_, 99);
}

// ============================================================
// Snapshot Isolation
// ============================================================

TEST(SharedGroundTruth, SnapshotIsImmutable) {
  GroundTruthPublisher publisher;
  auto handle = publisher.handle();

  GroundTruthInputPacket p1;
  p1.frame_id_ = 1;

  GroundTruthInputPacket p2;
  p2.frame_id_ = 2;

  publisher.insert(1, p1);

  auto snapshot_before = handle.access();
  ASSERT_TRUE(snapshot_before.has_value());
  ASSERT_EQ(snapshot_before->size(), 1u);

  publisher.insert(2, p2);

  // Old snapshot must remain unchanged
  EXPECT_EQ(snapshot_before->size(), 1u);
  EXPECT_TRUE(snapshot_before->count(1));
  EXPECT_FALSE(snapshot_before->count(2));

  // New snapshot must contain both
  auto snapshot_after = handle.access();
  ASSERT_TRUE(snapshot_after.has_value());
  EXPECT_EQ(snapshot_after->size(), 2u);
}

// ============================================================
// Clear
// ============================================================

TEST(SharedGroundTruth, ClearRemovesData) {
  GroundTruthPublisher publisher;
  auto handle = publisher.handle();

  GroundTruthInputPacket p;
  p.frame_id_ = 7;

  publisher.insert(1, p);

  ASSERT_TRUE(handle.access().has_value());

  publisher.clear();

  EXPECT_FALSE(handle.access().has_value());
}

// ============================================================
// Concurrent Access
// ============================================================

TEST(SharedGroundTruth, ConcurrentReaderWriter) {
  GroundTruthPublisher publisher;
  auto handle = publisher.handle();

  constexpr int num_inserts = 1000;

  std::atomic<bool> writer_done{false};

  std::thread writer([&]() {
    for (int i = 0; i < num_inserts; ++i) {
      GroundTruthInputPacket p;
      p.frame_id_ = i;
      publisher.insert(i, p);
    }
    writer_done = true;
  });

  std::thread reader([&]() {
    while (!writer_done) {
      auto snapshot = handle.access();
      if (snapshot) {
        // Basic sanity check:
        for (const auto& kv : *snapshot) {
          EXPECT_EQ(kv.first, static_cast<FrameId>(kv.second.frame_id_));
        }
      }
    }
  });

  writer.join();
  reader.join();

  auto final_snapshot = handle.access();

  ASSERT_TRUE(final_snapshot.has_value());
  EXPECT_EQ(final_snapshot->size(), static_cast<size_t>(num_inserts));
}

TEST(SharedGroundTruth, DelayedConstructionThenAssignment) {
  GroundTruthPublisher publisher;

  // Step 1: default constructed handle (invalid)
  SharedGroundTruth delayed;

  EXPECT_FALSE(delayed.valid());
  EXPECT_FALSE(delayed.access().has_value());

  // Step 2: later assignment from publisher handle
  delayed = publisher.handle();

  EXPECT_TRUE(delayed.valid());
  EXPECT_FALSE(delayed.access().has_value());  // still empty

  // Step 3: publish data
  GroundTruthInputPacket packet;
  packet.frame_id_ = 123;

  publisher.insert(42, packet);

  // Step 4: delayed handle should now see the data
  auto snapshot = delayed.access();

  ASSERT_TRUE(snapshot.has_value());
  ASSERT_EQ(snapshot->size(), 1u);
  EXPECT_EQ(snapshot->at(42).frame_id_, 123);
}
