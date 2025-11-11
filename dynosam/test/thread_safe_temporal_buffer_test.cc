/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
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

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include "dynosam/pipeline/ThreadSafeTemporalBuffer.hpp"

namespace dyno {
struct TestData {
  explicit TestData(Timestamp time) : timestamp(time) {}
  TestData() = default;

  Timestamp timestamp;
};

TEST(ThreadsafeTemporalBuffer, testEqualsOverloading) {
  ThreadsafeTemporalBuffer<TestData> buffer_1(-1);
  ThreadsafeTemporalBuffer<TestData> buffer_2(100);

  EXPECT_EQ(buffer_1.bufferLength(), -1);
  EXPECT_EQ(buffer_2.bufferLength(), 100);

  buffer_1.addValue(0, TestData(0));
  buffer_1.addValue(1, TestData(1));
  buffer_1.addValue(2, TestData(2));

  EXPECT_EQ(buffer_1.size(), 3u);
  EXPECT_EQ(buffer_2.size(), 0u);

  buffer_2 = buffer_1;
  EXPECT_EQ(buffer_1.size(), 3u);
  EXPECT_EQ(buffer_2.size(), 3u);

  EXPECT_EQ(buffer_1.bufferLength(), -1);
  EXPECT_EQ(buffer_2.bufferLength(), -1);

  // change buffer1 and buffer 2 should not change
  buffer_1.addValue(3, TestData(3));
  EXPECT_EQ(buffer_1.size(), 4u);
  EXPECT_EQ(buffer_2.size(), 3u);
}

TEST(ThreadsafeTemporalBuffer, testLagSizeInfinite) {
  ThreadsafeTemporalBuffer<TestData> buffer_(-1);
  buffer_.addValue(0, TestData(0));
  EXPECT_EQ(buffer_.size(), 1u);
  buffer_.addValue(1271839713, TestData(1271839713));
  EXPECT_EQ(buffer_.size(), 2u);
  buffer_.addValue(10000, TestData(10000));
  EXPECT_EQ(buffer_.size(), 3u);
}

TEST(ThreadsafeTemporalBuffer, testLagSize10) {
  ThreadsafeTemporalBuffer<TestData> buffer_(10);
  buffer_.addValue(0, TestData(0));

  const double kMaxDelta = 0.01;
  TestData retrieved_item;
  // check that we can retrieve this value now so wer can be sure we cannot
  // retrieve it later
  EXPECT_TRUE(buffer_.getNearestValueToTime(0, kMaxDelta, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 0);

  EXPECT_EQ(buffer_.size(), 1u);
  buffer_.addValue(4.3, TestData(4.3));
  EXPECT_EQ(buffer_.size(), 2u);
  buffer_.addValue(9.9, TestData(9.9));
  EXPECT_EQ(buffer_.size(), 3u);

  // add time past the buffer -> this data should stay in the buffer but push
  // out the 0th value
  buffer_.addValue(10.3, TestData(10.3));
  EXPECT_EQ(buffer_.size(), 3u);
  EXPECT_FALSE(buffer_.getNearestValueToTime(0, kMaxDelta, &retrieved_item));
  EXPECT_TRUE(
      buffer_.getNearestValueToTime(10.2999, kMaxDelta, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 10.3);
}

class ThreadsafeTemporalBufferFixture : public ::testing::Test {
 public:
  ThreadsafeTemporalBufferFixture() : buffer_(kBufferLengthS) {}

 protected:
  virtual void SetUp() {}
  virtual void TearDown() {}  //
  void addValue(const TestData& data) {
    buffer_.addValue(data.timestamp, data);
  }

  static constexpr Timestamp kBufferLengthS = 100;
  ThreadsafeTemporalBuffer<TestData> buffer_;
};

TEST_F(ThreadsafeTemporalBufferFixture, SizeEmptyClearWork) {
  EXPECT_TRUE(buffer_.empty());
  EXPECT_EQ(buffer_.size(), 0u);

  addValue(TestData(10));
  addValue(TestData(20));
  EXPECT_TRUE(!buffer_.empty());
  EXPECT_EQ(buffer_.size(), 2u);

  buffer_.clear();
  EXPECT_TRUE(buffer_.empty());
  EXPECT_EQ(buffer_.size(), 0u);
}

TEST_F(ThreadsafeTemporalBufferFixture, GetValueAtTimeWorks) {
  addValue(TestData(3.1));
  addValue(TestData(10));
  addValue(TestData(0.004));
  addValue(TestData(40.234));

  TestData retrieved_item;
  EXPECT_TRUE(buffer_.getValueAtTime(3.1, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 3.1);

  EXPECT_TRUE(buffer_.getValueAtTime(0.004, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 0.004);

  EXPECT_TRUE(!buffer_.getValueAtTime(40, &retrieved_item));

  EXPECT_TRUE(buffer_.getValueAtTime(3.1, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 3.1);

  EXPECT_TRUE(buffer_.getValueAtTime(40.234, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 40.234);
}

TEST_F(ThreadsafeTemporalBufferFixture, GetNearestValueToTimeWorks) {
  addValue(TestData(3.00));
  addValue(TestData(1.004));
  addValue(TestData(6.32));
  addValue(TestData(34));

  TestData retrieved_item;
  EXPECT_TRUE(buffer_.getNearestValueToTime(3, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 3);

  EXPECT_TRUE(buffer_.getNearestValueToTime(0, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 1.004);

  EXPECT_TRUE(buffer_.getNearestValueToTime(16, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 6.32);

  EXPECT_TRUE(buffer_.getNearestValueToTime(34.432421, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 34);

  EXPECT_TRUE(buffer_.getNearestValueToTime(32, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 34);

  EXPECT_TRUE(buffer_.getNearestValueToTime(1232, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 34);
}

TEST_F(ThreadsafeTemporalBufferFixture, GetNearestValueToTimeMaxDeltaWorks) {
  addValue(TestData(30));
  addValue(TestData(10));
  addValue(TestData(20));

  const double kMaxDelta = 5;

  TestData retrieved_item;
  EXPECT_TRUE(buffer_.getNearestValueToTime(10, kMaxDelta, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 10);

  EXPECT_TRUE(!buffer_.getNearestValueToTime(0, kMaxDelta, &retrieved_item));

  EXPECT_TRUE(buffer_.getNearestValueToTime(9, kMaxDelta, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 10);

  EXPECT_TRUE(buffer_.getNearestValueToTime(16, kMaxDelta, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 20);

  EXPECT_TRUE(buffer_.getNearestValueToTime(26, kMaxDelta, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 30);

  EXPECT_TRUE(buffer_.getNearestValueToTime(32, kMaxDelta, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 30);

  EXPECT_TRUE(!buffer_.getNearestValueToTime(36, kMaxDelta, &retrieved_item));

  buffer_.clear();
  addValue(TestData(10));
  addValue(TestData(20));

  EXPECT_TRUE(buffer_.getNearestValueToTime(9, kMaxDelta, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 10);

  EXPECT_TRUE(buffer_.getNearestValueToTime(12, kMaxDelta, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 10);

  EXPECT_TRUE(buffer_.getNearestValueToTime(16, kMaxDelta, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 20);

  EXPECT_TRUE(buffer_.getNearestValueToTime(22, kMaxDelta, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 20);

  buffer_.clear();
  addValue(TestData(10));

  EXPECT_TRUE(buffer_.getNearestValueToTime(6, kMaxDelta, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 10);

  EXPECT_TRUE(buffer_.getNearestValueToTime(14, kMaxDelta, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 10);

  EXPECT_TRUE(!buffer_.getNearestValueToTime(16, kMaxDelta, &retrieved_item));
}

TEST_F(ThreadsafeTemporalBufferFixture,
       GetNearestValueToTimeMaxDeltaSmallWorks) {
  addValue(TestData(0.05));
  addValue(TestData(0.06));
  addValue(TestData(0.057));

  const Timestamp kMaxDelta = 0.005;
  TestData retrieved_item;

  EXPECT_TRUE(buffer_.getNearestValueToTime(0.05, kMaxDelta, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 0.05);

  EXPECT_FALSE(
      buffer_.getNearestValueToTime(0.04, kMaxDelta, &retrieved_item));  //!

  EXPECT_TRUE(buffer_.getNearestValueToTime(0.058, kMaxDelta, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 0.057);

  EXPECT_FALSE(
      buffer_.getNearestValueToTime(0.066, kMaxDelta, &retrieved_item));  //!

  EXPECT_TRUE(buffer_.getNearestValueToTime(0.063, kMaxDelta, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 0.06);

  Timestamp retrieved_timestamp;
  EXPECT_TRUE(buffer_.getNearestValueToTime(0.063, kMaxDelta, &retrieved_item,
                                            &retrieved_timestamp));
  EXPECT_EQ(retrieved_timestamp, 0.06);

  buffer_.clear();
}

TEST_F(ThreadsafeTemporalBufferFixture, GetValueAtOrBeforeTimeWorks) {
  addValue(TestData(30));
  addValue(TestData(10));
  addValue(TestData(20));
  addValue(TestData(40));

  TestData retrieved_item;
  Timestamp timestamp;

  EXPECT_TRUE(buffer_.getValueAtOrBeforeTime(40, &timestamp, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 40);
  EXPECT_EQ(timestamp, 40);

  EXPECT_TRUE(buffer_.getValueAtOrBeforeTime(50, &timestamp, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 40);
  EXPECT_EQ(timestamp, 40);

  EXPECT_TRUE(buffer_.getValueAtOrBeforeTime(15, &timestamp, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 10);
  EXPECT_EQ(timestamp, 10);

  EXPECT_TRUE(buffer_.getValueAtOrBeforeTime(10, &timestamp, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 10);
  EXPECT_EQ(timestamp, 10);

  EXPECT_TRUE(!buffer_.getValueAtOrBeforeTime(5, &timestamp, &retrieved_item));
}

TEST_F(ThreadsafeTemporalBufferFixture, GetValueAtOrAfterTimeWorks) {
  addValue(TestData(30));
  addValue(TestData(10));
  addValue(TestData(20));
  addValue(TestData(40));

  TestData retrieved_item;
  Timestamp timestamp;

  EXPECT_TRUE(buffer_.getValueAtOrAfterTime(10, &timestamp, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 10);
  EXPECT_EQ(timestamp, 10);

  EXPECT_TRUE(buffer_.getValueAtOrAfterTime(5, &timestamp, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 10);
  EXPECT_EQ(timestamp, 10);

  EXPECT_TRUE(buffer_.getValueAtOrAfterTime(35, &timestamp, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 40);
  EXPECT_EQ(timestamp, 40);

  EXPECT_TRUE(buffer_.getValueAtOrAfterTime(40, &timestamp, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 40);
  EXPECT_EQ(timestamp, 40);

  EXPECT_TRUE(!buffer_.getValueAtOrAfterTime(45, &timestamp, &retrieved_item));
}

TEST_F(ThreadsafeTemporalBufferFixture, GetOldestNewestValueWork) {
  TestData retrieved_item;
  EXPECT_TRUE(!buffer_.getOldestValue(&retrieved_item));
  EXPECT_TRUE(!buffer_.getNewestValue(&retrieved_item));

  addValue(TestData(30.4));
  addValue(TestData(10.122));
  addValue(TestData(20.6));
  addValue(TestData(40.8));

  Timestamp timestamp = 0;
  EXPECT_TRUE(buffer_.getOldestValue(&retrieved_item, &timestamp));
  EXPECT_DOUBLE_EQ(retrieved_item.timestamp, 10.122);
  EXPECT_DOUBLE_EQ(timestamp, 10.122);

  EXPECT_TRUE(buffer_.getNewestValue(&retrieved_item, &timestamp));
  EXPECT_DOUBLE_EQ(timestamp, 40.8);
  EXPECT_DOUBLE_EQ(retrieved_item.timestamp, 40.8);
}

TEST_F(ThreadsafeTemporalBufferFixture, GetValuesBetweenTimesWorks) {
  addValue(TestData(10));
  addValue(TestData(20));
  addValue(TestData(30));
  addValue(TestData(40));
  addValue(TestData(50));

  // Test aligned borders.
  /// When the user does not ask for the lower bound.
  /// Implicitly also checks that it is default behaviour.
  std::vector<TestData> values;
  EXPECT_TRUE(buffer_.getValuesBetweenTimes(10, 50, &values));
  EXPECT_EQ(values.size(), 3u);
  EXPECT_EQ(values[0].timestamp, 20);
  EXPECT_EQ(values[1].timestamp, 30);
  EXPECT_EQ(values[2].timestamp, 40);

  // Test aligned borders.
  /// When the user does ask for the lower bound.
  EXPECT_TRUE(buffer_.getValuesBetweenTimes(10, 50, &values, true));
  EXPECT_EQ(values.size(), 4u);
  EXPECT_EQ(values[0].timestamp, 10);
  EXPECT_EQ(values[1].timestamp, 20);
  EXPECT_EQ(values[2].timestamp, 30);
  EXPECT_EQ(values[3].timestamp, 40);

  // Test unaligned borders.
  /// When the user does not ask for the lower bound.
  /// Implicitly also checks that it is default behaviour.
  EXPECT_TRUE(buffer_.getValuesBetweenTimes(15, 45, &values));
  EXPECT_EQ(values.size(), 3u);
  EXPECT_EQ(values[0].timestamp, 20);
  EXPECT_EQ(values[1].timestamp, 30);
  EXPECT_EQ(values[2].timestamp, 40);

  // Test unaligned borders.
  /// When the user does ask for the lower bound.
  EXPECT_TRUE(buffer_.getValuesBetweenTimes(15, 45, &values));
  EXPECT_EQ(values.size(), 3u);
  EXPECT_EQ(values[0].timestamp, 20);
  EXPECT_EQ(values[1].timestamp, 30);
  EXPECT_EQ(values[2].timestamp, 40);

  // Test unsuccessful queries.
  // Lower border oob.
  EXPECT_TRUE(!buffer_.getValuesBetweenTimes(5, 45, &values));
  EXPECT_TRUE(!buffer_.getValuesBetweenTimes(5, 45, &values, true));
  // Higher border oob.
  EXPECT_TRUE(!buffer_.getValuesBetweenTimes(30, 55, &values));
  EXPECT_TRUE(!buffer_.getValuesBetweenTimes(30, 55, &values, true));
  EXPECT_TRUE(values.empty());

  // The method should check-fail when the buffer is empty.
  buffer_.clear();
  EXPECT_TRUE(!buffer_.getValuesBetweenTimes(10, 50, &values));
  EXPECT_TRUE(!buffer_.getValuesBetweenTimes(10, 50, &values, true));
  // EXPECT_DEATH(buffer_.getValuesBetweenTimes(40, 30, &values), "^");
}

TEST_F(ThreadsafeTemporalBufferFixture, MaintaingBufferLengthWorks) {
  addValue(TestData(0));
  addValue(TestData(50));
  addValue(TestData(100));
  EXPECT_EQ(buffer_.size(), 3u);

  addValue(TestData(150));
  EXPECT_EQ(buffer_.size(), 3u);

  TestData retrieved_item;
  EXPECT_TRUE(buffer_.getOldestValue(&retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 50);

  EXPECT_TRUE(buffer_.getNewestValue(&retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 150);
}

TEST_F(ThreadsafeTemporalBufferFixture, DeltetingValuesAtTimestampsWork) {
  addValue(TestData(12.4));
  addValue(TestData(0.001));
  addValue(TestData(56));
  addValue(TestData(21));

  TestData retrieved_item;
  Timestamp timestamp;

  EXPECT_TRUE(buffer_.getValueAtOrAfterTime(12.4, &timestamp, &retrieved_item));
  EXPECT_EQ(buffer_.size(), 4u);
  EXPECT_TRUE(buffer_.deleteValueAtTime(timestamp));
  EXPECT_TRUE(
      buffer_.getValueAtOrBeforeTime(12.4, &timestamp, &retrieved_item));
  EXPECT_EQ(retrieved_item.timestamp, 0.001);
  EXPECT_FALSE(buffer_.getNearestValueToTime(12.2, 1, &retrieved_item));
  EXPECT_EQ(buffer_.size(), 3u);

  // delete value using non-exact timestamp
  Timestamp stored_timestamp;
  EXPECT_TRUE(
      buffer_.getNearestValueToTime(59, 4, &retrieved_item, &stored_timestamp));
  EXPECT_EQ(retrieved_item.timestamp, 56);
  EXPECT_EQ(stored_timestamp, 56);
  EXPECT_TRUE(buffer_.deleteValueAtTime(stored_timestamp));
  EXPECT_EQ(buffer_.size(), 2u);
}

}  // namespace dyno
