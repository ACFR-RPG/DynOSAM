
#include "dynosam2/frontend/FeatureBlockContainer.hpp"

using namespace dyno2;

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

FeatureBlockContainer::FeatureData createFeatureData(size_t n,
                                                     int id_offset = 0) {
  FeatureBlockContainer::FeatureData data;

  data.resize(n);

  for (size_t i = 0; i < n; ++i) {
    data.points[i] =
        cv::Point2f(static_cast<float>(i), static_cast<float>(i + 10));

    data.previous_points[i] =
        cv::Point2f(static_cast<float>(i + 100), static_cast<float>(i + 110));

    data.ids[i] = id_offset + static_cast<int>(i);

    data.inlier[i] = static_cast<uchar>(i % 2);

    data.errors[i] = static_cast<float>(i) * 0.5f;
  }

  return data;
}

FeatureBlockContainer createFeatureSet(
    const std::vector<int>& object_ids,
    const std::vector<size_t>& features_per_object) {
  EXPECT_EQ(object_ids.size(), features_per_object.size());

  std::vector<FeatureBlockContainer::BlockDim> specs;
  specs.reserve(object_ids.size());

  for (size_t i = 0; i < object_ids.size(); ++i) {
    specs.push_back({object_ids[i], features_per_object[i]});
  }

  return FeatureBlockContainer(specs);
}

void copyObjectData(FeatureBlockContainer& features, int object_id,
                    const FeatureBlockContainer::FeatureData& data) {
  features.objectView(object_id).copyFrom(data);
}

// Used to prevent the compiler from completely eliminating benchmark loops.

template <typename Function>
double benchmarkNsPerFeature(Function&& function, size_t n, size_t iterations) {
  for (size_t i = 0; i < 100; ++i) {
    function();
  }

  const auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < iterations; ++i) {
    function();
  }

  const auto end = std::chrono::high_resolution_clock::now();

  const double elapsed_ns =
      std::chrono::duration<double, std::nano>(end - start).count();

  return elapsed_ns / static_cast<double>(n * iterations);
}

// =============================================================================
// Construction
// =============================================================================

TEST(FeatureBlockContainerTest, ConstructsKnownObjects) {
  FeatureBlockContainer features({
      {10, 3},
      {20, 5},
      {30, 2},
  });

  EXPECT_EQ(features.objectCount(), 3);
  EXPECT_EQ(features.size(), 10);

  EXPECT_TRUE(features.containsObject(10));
  EXPECT_TRUE(features.containsObject(20));
  EXPECT_TRUE(features.containsObject(30));

  EXPECT_FALSE(features.containsObject(40));

  features.checkInvariants();
}

TEST(FeatureBlockContainerTest, ConstructsEmptyFeatureSet) {
  FeatureBlockContainer features({});

  EXPECT_EQ(features.objectCount(), 0);
  EXPECT_EQ(features.size(), 0);

  features.checkInvariants();
}

TEST(FeatureBlockContainerTest, AllowsObjectsWithZeroFeatures) {
  FeatureBlockContainer features({
      {10, 0},
      {20, 5},
      {30, 0},
  });

  EXPECT_EQ(features.objectCount(), 3);
  EXPECT_EQ(features.size(), 5);

  EXPECT_EQ(features.objectView(10).size(), 0);
  EXPECT_EQ(features.objectView(20).size(), 5);
  EXPECT_EQ(features.objectView(30).size(), 0);

  features.checkInvariants();
}

TEST(FeatureBlockContainerTest, RejectsDuplicateObjectIds) {
  EXPECT_THROW(FeatureBlockContainer({
                   {10, 3},
                   {20, 5},
                   {10, 2},
               }),
               std::invalid_argument);
}

TEST(FeatureBlockContainerTest, RejectsUnknownObject) {
  FeatureBlockContainer features({
      {10, 3},
      {20, 5},
  });

  EXPECT_THROW(features.objectView(99), std::out_of_range);
}

// =============================================================================
// ObjectView
// =============================================================================

TEST(FeatureBlockContainerTest, ObjectViewHasCorrectSize) {
  FeatureBlockContainer features({
      {10, 3},
      {20, 5},
      {30, 2},
  });

  EXPECT_EQ(features.objectView(10).size(), 3);
  EXPECT_EQ(features.objectView(20).size(), 5);
  EXPECT_EQ(features.objectView(30).size(), 2);
}

TEST(FeatureBlockContainerTest, ObjectViewHasCorrectObjectId) {
  FeatureBlockContainer features({
      {10, 3},
      {20, 5},
      {30, 2},
  });

  EXPECT_EQ(features.objectView(10).objectId(), 10);
  EXPECT_EQ(features.objectView(20).objectId(), 20);
  EXPECT_EQ(features.objectView(30).objectId(), 30);
}

TEST(FeatureBlockContainerTest, ObjectViewIsZeroCopy) {
  FeatureBlockContainer features({
      {10, 3},
      {20, 5},
      {30, 2},
  });

  auto object = features.objectView(20);

  // Object 10 occupies [0, 3)
  // Object 20 occupies [3, 8)
  // Object 30 occupies [8, 10)

  EXPECT_EQ(object.points(), features.points.data() + 3);

  EXPECT_EQ(object.previousPoints(), features.previous_points.data() + 3);

  EXPECT_EQ(object.ids(), features.ids.data() + 3);

  EXPECT_EQ(object.objectIds(), features.object_ids.data() + 3);

  EXPECT_EQ(object.inlier(), features.inlier.data() + 3);

  EXPECT_EQ(object.errors(), features.errors.data() + 3);
}

TEST(FeatureBlockContainerTest, ConstObjectViewIsZeroCopy) {
  const FeatureBlockContainer features({
      {10, 3},
      {20, 5},
      {30, 2},
  });

  auto object = features.objectView(20);

  EXPECT_EQ(object.points(), features.points.data() + 3);

  EXPECT_EQ(object.previousPoints(), features.previous_points.data() + 3);

  EXPECT_EQ(object.ids(), features.ids.data() + 3);

  EXPECT_EQ(object.objectIds(), features.object_ids.data() + 3);

  EXPECT_EQ(object.inlier(), features.inlier.data() + 3);

  EXPECT_EQ(object.errors(), features.errors.data() + 3);
}

TEST(FeatureBlockContainerTest, ObjectViewProvidesCorrectData) {
  FeatureBlockContainer features({
      {10, 3},
      {20, 5},
  });

  const FeatureBlockContainer::FeatureData data = createFeatureData(5, 100);

  features.objectView(20).copyFrom(data);

  const auto object = features.objectView(20);

  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(object.points()[i], data.points[i]);

    EXPECT_EQ(object.previousPoints()[i], data.previous_points[i]);

    EXPECT_EQ(object.ids()[i], data.ids[i]);

    EXPECT_EQ(object.inlier()[i], data.inlier[i]);

    EXPECT_EQ(object.errors()[i], data.errors[i]);

    EXPECT_EQ(object.objectIds()[i], 20);
  }
}

TEST(FeatureBlockContainerTest, ObjectViewModificationUpdatesFeatureSet) {
  FeatureBlockContainer features({
      {10, 3},
  });

  auto object = features.objectView(10);

  object.points()[0] = cv::Point2f(123.0f, 456.0f);
  object.ids()[1] = 42;
  object.inlier()[2] = 0;

  EXPECT_EQ(features.points[0], cv::Point2f(123.0f, 456.0f));

  EXPECT_EQ(features.ids[1], 42);

  EXPECT_EQ(features.inlier[2], 0);
}

// =============================================================================
// Object block layout
// =============================================================================

TEST(FeatureBlockContainerTest, ObjectBlocksAreContiguous) {
  FeatureBlockContainer features({
      {10, 3},
      {20, 5},
      {30, 2},
  });

  auto object10 = features.objectView(10);
  auto object20 = features.objectView(20);
  auto object30 = features.objectView(30);

  EXPECT_EQ(object10.points() + object10.size(), object20.points());

  EXPECT_EQ(object20.points() + object20.size(), object30.points());
}

TEST(FeatureBlockContainerTest, ObjectBlocksContainCorrectObjectIds) {
  FeatureBlockContainer features({
      {10, 3},
      {20, 5},
      {30, 2},
  });

  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(features.object_ids[i], 10);
  }

  for (size_t i = 3; i < 8; ++i) {
    EXPECT_EQ(features.object_ids[i], 20);
  }

  for (size_t i = 8; i < 10; ++i) {
    EXPECT_EQ(features.object_ids[i], 30);
  }
}

// =============================================================================
// FeatureData
// =============================================================================

TEST(FeatureBlockContainerTest, FeatureDataReportsCorrectSize) {
  const FeatureBlockContainer::FeatureData data = createFeatureData(10);

  EXPECT_EQ(data.size(), 10);
  EXPECT_FALSE(data.empty());
}

TEST(FeatureBlockContainerTest, FeatureDataReportsEmpty) {
  FeatureBlockContainer::FeatureData data;

  EXPECT_EQ(data.size(), 0);
  EXPECT_TRUE(data.empty());

  EXPECT_NO_THROW(data.checkSizes());
}

TEST(FeatureBlockContainerTest, FeatureDataRejectsMismatchedArrays) {
  FeatureBlockContainer::FeatureData data = createFeatureData(5);

  data.ids.resize(4);

  EXPECT_THROW(data.checkSizes(), std::invalid_argument);
}

TEST(FeatureBlockContainerTest, CopyFeatureData) {
  const FeatureBlockContainer::FeatureData data = createFeatureData(5, 100);

  FeatureBlockContainer features({
      {42, 5},
  });

  features.objectView(42).copyFrom(data);

  const auto object = features.objectView(42);

  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(object.points()[i], data.points[i]);
    EXPECT_EQ(object.previousPoints()[i], data.previous_points[i]);
    EXPECT_EQ(object.ids()[i], data.ids[i]);
    EXPECT_EQ(object.inlier()[i], data.inlier[i]);
    EXPECT_EQ(object.errors()[i], data.errors[i]);

    // object_ids are generated by FeatureBlockContainer.
    EXPECT_EQ(object.objectIds()[i], 42);
  }

  features.checkInvariants();
}

TEST(FeatureBlockContainerTest, CopyFeatureDataRejectsIncorrectSize) {
  FeatureBlockContainer features({
      {42, 5},
  });

  const FeatureBlockContainer::FeatureData data = createFeatureData(4);

  EXPECT_THROW(features.objectView(42).copyFrom(data), std::invalid_argument);
}

TEST(FeatureBlockContainerTest, CopyFeatureDataDoesNotModifySource) {
  const FeatureBlockContainer::FeatureData data = createFeatureData(5, 100);

  const FeatureBlockContainer::FeatureData original = data;

  FeatureBlockContainer features({
      {42, 5},
  });

  features.objectView(42).copyFrom(data);

  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(data.points[i], original.points[i]);
    EXPECT_EQ(data.previous_points[i], original.previous_points[i]);
    EXPECT_EQ(data.ids[i], original.ids[i]);
    EXPECT_EQ(data.inlier[i], original.inlier[i]);
    EXPECT_EQ(data.errors[i], original.errors[i]);
  }
}

TEST(FeatureBlockContainerTest, ObjectIdsAreGeneratedFromObjectSpecification) {
  const FeatureBlockContainer::FeatureData data = createFeatureData(5);

  FeatureBlockContainer features({
      {42, 5},
  });

  features.objectView(42).copyFrom(data);

  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(features.object_ids[i], 42);
  }
}

TEST(FeatureBlockContainer, MergeRejectsDuplicateFeatureIdsAcrossObjects) {
  FeatureBlockContainer first = createFeatureSet({1}, {2});

  FeatureBlockContainer second = createFeatureSet({2}, {2});

  first.objectView(1).copyFrom(createFeatureData(2, 10));

  second.objectView(2).copyFrom(createFeatureData(2, 11));

  // first:  object 1 -> IDs [10, 11]
  // second: object 2 -> IDs [11, 12]
  //
  // ID 11 is duplicated even though the object IDs differ.

  EXPECT_THROW(first.merge(second), std::invalid_argument);
}

// =============================================================================
// Multiple objects
// =============================================================================

TEST(FeatureBlockContainerTest, CopiesMultipleObjects) {
  const FeatureBlockContainer::FeatureData object1 = createFeatureData(3, 100);

  const FeatureBlockContainer::FeatureData object2 = createFeatureData(5, 200);

  const FeatureBlockContainer::FeatureData object3 = createFeatureData(2, 300);

  FeatureBlockContainer features({
      {10, object1.size()},
      {20, object2.size()},
      {30, object3.size()},
  });

  features.objectView(10).copyFrom(object1);
  features.objectView(20).copyFrom(object2);
  features.objectView(30).copyFrom(object3);

  EXPECT_EQ(features.size(), 10);

  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(features.points[i], object1.points[i]);

    EXPECT_EQ(features.previous_points[i], object1.previous_points[i]);

    EXPECT_EQ(features.ids[i], object1.ids[i]);

    EXPECT_EQ(features.inlier[i], object1.inlier[i]);

    EXPECT_EQ(features.errors[i], object1.errors[i]);

    EXPECT_EQ(features.object_ids[i], 10);
  }

  for (size_t i = 0; i < 5; ++i) {
    const size_t index = 3 + i;

    EXPECT_EQ(features.points[index], object2.points[i]);

    EXPECT_EQ(features.previous_points[index], object2.previous_points[i]);

    EXPECT_EQ(features.ids[index], object2.ids[i]);

    EXPECT_EQ(features.inlier[index], object2.inlier[i]);

    EXPECT_EQ(features.errors[index], object2.errors[i]);

    EXPECT_EQ(features.object_ids[index], 20);
  }

  for (size_t i = 0; i < 2; ++i) {
    const size_t index = 8 + i;

    EXPECT_EQ(features.points[index], object3.points[i]);

    EXPECT_EQ(features.previous_points[index], object3.previous_points[i]);

    EXPECT_EQ(features.ids[index], object3.ids[i]);

    EXPECT_EQ(features.inlier[index], object3.inlier[i]);

    EXPECT_EQ(features.errors[index], object3.errors[i]);

    EXPECT_EQ(features.object_ids[index], 30);
  }

  features.checkInvariants();
}

// =============================================================================
// Merge
// =============================================================================

TEST(FeatureBlockContainerTest, MergePreservesObjectOrder) {
  FeatureBlockContainer features({
      {1, 2},
      {2, 2},
      {3, 2},
      {4, 2},
  });

  FeatureBlockContainer new_features({
      {1, 1},
      {3, 1},
      {5, 1},
  });

  FeatureBlockContainer merged = features.merge(new_features);

  EXPECT_TRUE(merged.containsObject(1));
  EXPECT_TRUE(merged.containsObject(2));
  EXPECT_TRUE(merged.containsObject(3));
  EXPECT_TRUE(merged.containsObject(4));
  EXPECT_TRUE(merged.containsObject(5));

  // The public API intentionally does not expose physical metadata.
  // Verify ordering through the contiguous object blocks.
  EXPECT_EQ(merged.objectView(1).objectId(), 1);
  EXPECT_EQ(merged.objectView(2).objectId(), 2);
  EXPECT_EQ(merged.objectView(3).objectId(), 3);
  EXPECT_EQ(merged.objectView(4).objectId(), 4);
  EXPECT_EQ(merged.objectView(5).objectId(), 5);
}

TEST(FeatureBlockContainerTest, MergeProducesCorrectTotalSize) {
  FeatureBlockContainer features({
      {1, 3},
      {2, 5},
      {3, 2},
  });

  FeatureBlockContainer new_features({
      {1, 4},
      {2, 1},
      {3, 6},
  });

  FeatureBlockContainer merged = features.merge(new_features);

  EXPECT_EQ(merged.size(), 3 + 4 + 5 + 1 + 2 + 6);
}

TEST(FeatureBlockContainerTest, MergePreservesExistingFeatures) {
  FeatureBlockContainer features({
      {1, 3},
      {2, 2},
  });

  const FeatureBlockContainer::FeatureData object1 = createFeatureData(3, 100);

  const FeatureBlockContainer::FeatureData object2 = createFeatureData(2, 200);

  features.objectView(1).copyFrom(object1);
  features.objectView(2).copyFrom(object2);

  FeatureBlockContainer new_features({
      {1, 2},
      {2, 3},
  });

  FeatureBlockContainer merged = features.merge(new_features);

  const auto merged1 = merged.objectView(1);

  const auto merged2 = merged.objectView(2);

  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(merged1.points()[i], object1.points[i]);

    EXPECT_EQ(merged1.previousPoints()[i], object1.previous_points[i]);

    EXPECT_EQ(merged1.ids()[i], object1.ids[i]);
  }

  for (size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(merged2.points()[i], object2.points[i]);

    EXPECT_EQ(merged2.previousPoints()[i], object2.previous_points[i]);

    EXPECT_EQ(merged2.ids()[i], object2.ids[i]);
  }
}

TEST(FeatureBlockContainerTest, MergeAppendsNewFeaturesToEachObject) {
  FeatureBlockContainer features({
      {1, 2},
      {2, 3},
  });

  const FeatureBlockContainer::FeatureData old1 = createFeatureData(2, 100);

  const FeatureBlockContainer::FeatureData old2 = createFeatureData(3, 200);

  features.objectView(1).copyFrom(old1);
  features.objectView(2).copyFrom(old2);

  FeatureBlockContainer new_features({
      {1, 3},
      {2, 2},
  });

  const FeatureBlockContainer::FeatureData new1 = createFeatureData(3, 300);

  const FeatureBlockContainer::FeatureData new2 = createFeatureData(2, 400);

  new_features.objectView(1).copyFrom(new1);
  new_features.objectView(2).copyFrom(new2);

  FeatureBlockContainer merged = features.merge(new_features);

  const auto merged1 = merged.objectView(1);

  const auto merged2 = merged.objectView(2);

  EXPECT_EQ(merged1.size(), 5);
  EXPECT_EQ(merged2.size(), 5);

  for (size_t i = 0; i < old1.size(); ++i) {
    EXPECT_EQ(merged1.points()[i], old1.points[i]);

    EXPECT_EQ(merged1.ids()[i], old1.ids[i]);
  }

  for (size_t i = 0; i < new1.size(); ++i) {
    EXPECT_EQ(merged1.points()[old1.size() + i], new1.points[i]);

    EXPECT_EQ(merged1.ids()[old1.size() + i], new1.ids[i]);
  }

  for (size_t i = 0; i < old2.size(); ++i) {
    EXPECT_EQ(merged2.points()[i], old2.points[i]);

    EXPECT_EQ(merged2.ids()[i], old2.ids[i]);
  }

  for (size_t i = 0; i < new2.size(); ++i) {
    EXPECT_EQ(merged2.points()[old2.size() + i], new2.points[i]);

    EXPECT_EQ(merged2.ids()[old2.size() + i], new2.ids[i]);
  }

  merged.checkInvariants();
}

TEST(FeatureBlockContainerTest, MergeHandlesObjectsWithNoNewFeatures) {
  FeatureBlockContainer features({
      {1, 3},
      {2, 4},
      {3, 2},
  });

  FeatureBlockContainer new_features({
      {1, 2},
      {3, 5},
  });

  FeatureBlockContainer merged = features.merge(new_features);

  EXPECT_EQ(merged.objectView(1).size(), 5);

  EXPECT_EQ(merged.objectView(2).size(), 4);

  EXPECT_EQ(merged.objectView(3).size(), 7);

  merged.checkInvariants();
}

TEST(FeatureBlockContainerTest, MergeHandlesObjectsWithNoExistingFeatures) {
  FeatureBlockContainer features({
      {1, 3},
      {2, 4},
  });

  FeatureBlockContainer new_features({
      {1, 2},
      {3, 5},
  });

  FeatureBlockContainer merged = features.merge(new_features);

  EXPECT_EQ(merged.objectView(1).size(), 5);

  EXPECT_EQ(merged.objectView(2).size(), 4);

  EXPECT_EQ(merged.objectView(3).size(), 5);

  EXPECT_TRUE(merged.containsObject(3));

  merged.checkInvariants();
}

TEST(FeatureBlockContainerTest, MergeAllowsMissingObjects) {
  FeatureBlockContainer features({
      {1, 3},
      {2, 4},
      {3, 5},
      {4, 2},
  });

  // Objects 2 and 4 do not require new features.
  FeatureBlockContainer new_features({
      {1, 2},
      {3, 3},
  });

  FeatureBlockContainer merged = features.merge(new_features);

  EXPECT_EQ(merged.objectView(1).size(), 5);

  EXPECT_EQ(merged.objectView(2).size(), 4);

  EXPECT_EQ(merged.objectView(3).size(), 8);

  EXPECT_EQ(merged.objectView(4).size(), 2);

  EXPECT_EQ(merged.objectCount(), 4);

  merged.checkInvariants();
}

TEST(FeatureBlockContainerTest, MergeAllowsAdditionalObjects) {
  FeatureBlockContainer features({
      {1, 3},
      {2, 4},
  });
  features.objectView(1).copyFrom(createFeatureData(3, 0));
  features.objectView(2).copyFrom(createFeatureData(4, 3));

  FeatureBlockContainer new_features({
      {1, 2},
      {3, 5},
      {4, 2},
  });
  new_features.objectView(1).copyFrom(createFeatureData(2, 7));
  new_features.objectView(3).copyFrom(createFeatureData(5, 9));
  new_features.objectView(4).copyFrom(createFeatureData(2, 14));

  FeatureBlockContainer merged = features.merge(new_features);

  EXPECT_EQ(merged.objectCount(), 4);

  EXPECT_EQ(merged.objectView(1).size(), 5);

  EXPECT_EQ(merged.objectView(2).size(), 4);

  EXPECT_EQ(merged.objectView(3).size(), 5);

  EXPECT_EQ(merged.objectView(4).size(), 2);

  merged.checkInvariants();
}

TEST(FeatureBlockContainerTest, MergeAllowsDifferentObjectSets) {
  FeatureBlockContainer features({
      {1, 3},
      {2, 4},
      {3, 2},
  });

  FeatureBlockContainer new_features({
      {3, 5},
      {4, 6},
      {5, 1},
  });

  FeatureBlockContainer merged = features.merge(new_features);

  EXPECT_EQ(merged.objectCount(), 5);

  EXPECT_EQ(merged.objectView(1).size(), 3);

  EXPECT_EQ(merged.objectView(2).size(), 4);

  EXPECT_EQ(merged.objectView(3).size(), 7);

  EXPECT_EQ(merged.objectView(4).size(), 6);

  EXPECT_EQ(merged.objectView(5).size(), 1);

  merged.checkInvariants();
}

// TEST(FeatureBlockContainerTest, MergeHandlesEmptyFeatureSets)
// {
//     FeatureBlockContainer features({
//         {1, 3},
//         {2, 4},
//     });

//     FeatureBlockContainer empty;

//     FeatureBlockContainer merged =
//         features.merge(empty);

//     EXPECT_EQ(merged.objectCount(), 2);
//     EXPECT_EQ(merged.size(), 7);

//     EXPECT_EQ(
//         merged.objectView(1).size(),
//         3);

//     EXPECT_EQ(
//         merged.objectView(2).size(),
//         4);

//     merged.checkInvariants();
// }

// TEST(FeatureBlockContainerTest, MergeEmptyIntoPopulated)
// {
//     FeatureBlockContainer empty;

//     FeatureBlockContainer new_features({
//         {1, 3},
//         {2, 5},
//     });

//     const FeatureBlockContainer::FeatureData object1 =
//         createFeatureData(3, 100);

//     const FeatureBlockContainer::FeatureData object2 =
//         createFeatureData(5, 200);

//     new_features.objectView(1).copyFrom(object1);
//     new_features.objectView(2).copyFrom(object2);

//     FeatureBlockContainer merged =
//         empty.merge(new_features);

//     EXPECT_EQ(merged.objectCount(), 2);
//     EXPECT_EQ(merged.size(), 8);

//     EXPECT_EQ(
//         merged.objectView(1).size(),
//         3);

//     EXPECT_EQ(
//         merged.objectView(2).size(),
//         5);

//     for (size_t i = 0; i < 3; ++i)
//     {
//         EXPECT_EQ(
//             merged.objectView(1).points()[i],
//             object1.points[i]);
//     }

//     for (size_t i = 0; i < 5; ++i)
//     {
//         EXPECT_EQ(
//             merged.objectView(2).points()[i],
//             object2.points[i]);
//     }

//     merged.checkInvariants();
// }

TEST(FeatureBlockContainerTest, ObjectViewPointsMatIsZeroCopy) {
  FeatureBlockContainer features({{1, 3}, {2, 2}});

  auto object = features.objectView(1);

  object.points()[0] = cv::Point2f(1.0f, 2.0f);
  object.points()[1] = cv::Point2f(3.0f, 4.0f);
  object.points()[2] = cv::Point2f(5.0f, 6.0f);

  cv::Mat points = object.pointsMat();

  ASSERT_EQ(points.rows, 3);
  ASSERT_EQ(points.cols, 1);
  ASSERT_EQ(points.type(), CV_32FC2);

  // Check the Mat sees the original FeatureBlockContainer data.
  EXPECT_FLOAT_EQ(points.at<cv::Point2f>(0, 0).x, 1.0f);
  EXPECT_FLOAT_EQ(points.at<cv::Point2f>(0, 0).y, 2.0f);
  EXPECT_FLOAT_EQ(points.at<cv::Point2f>(1, 0).x, 3.0f);
  EXPECT_FLOAT_EQ(points.at<cv::Point2f>(1, 0).y, 4.0f);
  EXPECT_FLOAT_EQ(points.at<cv::Point2f>(2, 0).x, 5.0f);
  EXPECT_FLOAT_EQ(points.at<cv::Point2f>(2, 0).y, 6.0f);

  // Modify through cv::Mat.
  points.at<cv::Point2f>(1, 0) = cv::Point2f(10.0f, 20.0f);

  // The underlying FeatureBlockContainer must see the modification.
  EXPECT_FLOAT_EQ(object.points()[1].x, 10.0f);
  EXPECT_FLOAT_EQ(object.points()[1].y, 20.0f);
}

TEST(FeatureBlockContainerTest, ObjectViewPointsMatContainsOnlyObjectFeatures) {
  FeatureBlockContainer features({{10, 3}, {20, 2}, {30, 4}});

  for (size_t i = 0; i < features.size(); ++i) {
    features.points[i] =
        cv::Point2f(static_cast<float>(i), static_cast<float>(i + 100));
  }

  auto object = features.objectView(20);
  cv::Mat points = object.pointsMat();

  ASSERT_EQ(points.rows, 2);
  ASSERT_EQ(points.cols, 1);
  ASSERT_EQ(points.type(), CV_32FC2);

  EXPECT_EQ(points.at<cv::Point2f>(0, 0), cv::Point2f(3.0f, 103.0f));

  EXPECT_EQ(points.at<cv::Point2f>(1, 0), cv::Point2f(4.0f, 104.0f));
}

TEST(FeatureBlockContainerTest, MergeDoesNotModifyInputs) {
  FeatureBlockContainer features({
      {1, 3},
      {2, 2},
  });

  FeatureBlockContainer new_features({
      {1, 2},
      {3, 4},
  });

  const FeatureBlockContainer::FeatureData old1 = createFeatureData(3, 100);

  const FeatureBlockContainer::FeatureData old2 = createFeatureData(2, 200);

  const FeatureBlockContainer::FeatureData new1 = createFeatureData(2, 300);

  const FeatureBlockContainer::FeatureData new3 = createFeatureData(4, 400);

  features.objectView(1).copyFrom(old1);
  features.objectView(2).copyFrom(old2);

  new_features.objectView(1).copyFrom(new1);
  new_features.objectView(3).copyFrom(new3);

  const auto original_points = features.points;
  const auto original_previous_points = features.previous_points;
  const auto original_ids = features.ids;
  const auto original_object_ids = features.object_ids;
  const auto original_status = features.inlier;
  const auto original_errors = features.errors;

  const auto original_new_points = new_features.points;
  const auto original_new_previous_points = new_features.previous_points;
  const auto original_new_ids = new_features.ids;
  const auto original_new_object_ids = new_features.object_ids;
  const auto original_new_status = new_features.inlier;
  const auto original_new_errors = new_features.errors;

  FeatureBlockContainer merged = features.merge(new_features);

  EXPECT_EQ(features.points, original_points);
  EXPECT_EQ(features.previous_points, original_previous_points);
  EXPECT_EQ(features.ids, original_ids);
  EXPECT_EQ(features.object_ids, original_object_ids);
  EXPECT_EQ(features.inlier, original_status);
  EXPECT_EQ(features.errors, original_errors);

  EXPECT_EQ(new_features.points, original_new_points);
  EXPECT_EQ(new_features.previous_points, original_new_previous_points);
  EXPECT_EQ(new_features.ids, original_new_ids);
  EXPECT_EQ(new_features.object_ids, original_new_object_ids);
  EXPECT_EQ(new_features.inlier, original_new_status);
  EXPECT_EQ(new_features.errors, original_new_errors);

  merged.checkInvariants();
}

TEST(FeatureBlockContainerTest, ConstructsFromFeatureData) {
  FeatureBlockContainer::FeatureData object1 = createFeatureData(3, 10);
  FeatureBlockContainer::FeatureData object2 = createFeatureData(2, 20);

  std::vector<std::pair<ObjectId, FeatureBlockContainer::FeatureData>> terms;
  terms.emplace_back(1, object1);
  terms.emplace_back(2, object2);

  FeatureBlockContainer features(terms);

  ASSERT_EQ(features.objectCount(), 2u);
  ASSERT_EQ(features.size(), 5u);

  const auto first = features.objectView(1);
  const auto second = features.objectView(2);

  EXPECT_EQ(first.size(), 3u);
  EXPECT_EQ(second.size(), 2u);

  EXPECT_EQ(first.objectId(), 1);
  EXPECT_EQ(second.objectId(), 2);
}

TEST(FeatureBlockContainerTest, ConstructsFromFeatureDataCopiesEveryField) {
  FeatureBlockContainer::FeatureData object1 = createFeatureData(3, 100);
  FeatureBlockContainer::FeatureData object2 = createFeatureData(2, 200);

  std::vector<std::pair<ObjectId, FeatureBlockContainer::FeatureData>> terms;
  terms.emplace_back(10, object1);
  terms.emplace_back(20, object2);

  FeatureBlockContainer features(terms);

  const auto first = features.objectView(10);
  const auto second = features.objectView(20);

  for (size_t i = 0; i < first.size(); ++i) {
    EXPECT_EQ(first.points()[i], object1.points[i]);
    EXPECT_EQ(first.previousPoints()[i], object1.previous_points[i]);
    EXPECT_EQ(first.ids()[i], object1.ids[i]);
    EXPECT_EQ(first.inlier()[i], object1.inlier[i]);
    EXPECT_EQ(first.errors()[i], object1.errors[i]);
    EXPECT_EQ(first.objectIds()[i], 10);
  }

  for (size_t i = 0; i < second.size(); ++i) {
    EXPECT_EQ(second.points()[i], object2.points[i]);
    EXPECT_EQ(second.previousPoints()[i], object2.previous_points[i]);
    EXPECT_EQ(second.ids()[i], object2.ids[i]);
    EXPECT_EQ(second.inlier()[i], object2.inlier[i]);
    EXPECT_EQ(second.errors()[i], object2.errors[i]);
    EXPECT_EQ(second.objectIds()[i], 20);
  }
}

TEST(FeatureBlockContainerTest, ConstructsFromEmptyFeatureData) {
  FeatureBlockContainer::FeatureData empty;

  std::vector<std::pair<ObjectId, FeatureBlockContainer::FeatureData>> terms;
  terms.emplace_back(42, empty);

  FeatureBlockContainer features(terms);

  EXPECT_EQ(features.objectCount(), 1u);
  EXPECT_EQ(features.size(), 0u);
  EXPECT_TRUE(features.containsObject(42));

  const auto object = features.objectView(42);

  EXPECT_EQ(object.objectId(), 42);
  EXPECT_EQ(object.size(), 0u);
}

TEST(FeatureBlockContainerTest,
     ConstructsFromFeatureDataRejectsMismatchedArrays) {
  FeatureBlockContainer::FeatureData data;
  data.points.resize(3);
  data.previous_points.resize(3);
  data.ids.resize(2);  // Incorrect size.
  data.inlier.resize(3);
  data.errors.resize(3);

  std::vector<std::pair<ObjectId, FeatureBlockContainer::FeatureData>> terms;
  terms.emplace_back(1, data);

  EXPECT_THROW(FeatureBlockContainer features(terms), std::invalid_argument);
}

TEST(FeatureBlockContainerTest, ConstructsFromFeatureDataDoesNotAliasSource) {
  FeatureBlockContainer::FeatureData data = createFeatureData(3, 100);

  const cv::Point2f original_point = data.points[0];
  const int original_id = data.ids[0];

  std::vector<std::pair<ObjectId, FeatureBlockContainer::FeatureData>> terms;
  terms.emplace_back(1, data);

  FeatureBlockContainer features(terms);

  data.points[0] = cv::Point2f(999.0f, 888.0f);
  data.ids[0] = 9999;

  const auto object = features.objectView(1);

  EXPECT_EQ(object.points()[0], original_point);
  EXPECT_EQ(object.ids()[0], original_id);
}

TEST(FeatureBlockContainerTest, ConstructsFromMap) {
  std::map<ObjectId, FeatureBlockContainer::FeatureData> terms;

  terms.emplace(10, createFeatureData(3, 100));
  terms.emplace(20, createFeatureData(2, 200));

  FeatureBlockContainer features(terms);

  EXPECT_EQ(features.objectCount(), 2u);
  EXPECT_EQ(features.size(), 5u);

  EXPECT_EQ(features.objectView(10).size(), 3u);
  EXPECT_EQ(features.objectView(20).size(), 2u);
}

TEST(FeatureBlockContainerTest, MergeCopiesEverySoAFieldCorrectly) {
  FeatureBlockContainer features({
      {1, 2},
      {2, 3},
  });

  FeatureBlockContainer new_features({
      {1, 2},
      {2, 1},
  });

  const FeatureBlockContainer::FeatureData old1 = createFeatureData(2, 10);

  const FeatureBlockContainer::FeatureData old2 = createFeatureData(3, 20);

  const FeatureBlockContainer::FeatureData new1 = createFeatureData(2, 30);

  const FeatureBlockContainer::FeatureData new2 = createFeatureData(1, 40);

  features.objectView(1).copyFrom(old1);
  features.objectView(2).copyFrom(old2);

  new_features.objectView(1).copyFrom(new1);
  new_features.objectView(2).copyFrom(new2);

  FeatureBlockContainer merged = features.merge(new_features);

  const auto object1 = merged.objectView(1);

  const auto object2 = merged.objectView(2);

  // Object 1: old + new.
  for (size_t i = 0; i < old1.size(); ++i) {
    EXPECT_EQ(object1.points()[i], old1.points[i]);

    EXPECT_EQ(object1.previousPoints()[i], old1.previous_points[i]);

    EXPECT_EQ(object1.ids()[i], old1.ids[i]);

    EXPECT_EQ(object1.objectIds()[i], 1);

    EXPECT_EQ(object1.inlier()[i], old1.inlier[i]);

    EXPECT_EQ(object1.errors()[i], old1.errors[i]);
  }

  for (size_t i = 0; i < new1.size(); ++i) {
    const size_t index = old1.size() + i;

    EXPECT_EQ(object1.points()[index], new1.points[i]);

    EXPECT_EQ(object1.previousPoints()[index], new1.previous_points[i]);

    EXPECT_EQ(object1.ids()[index], new1.ids[i]);

    EXPECT_EQ(object1.objectIds()[index], 1);

    EXPECT_EQ(object1.inlier()[index], new1.inlier[i]);

    EXPECT_EQ(object1.errors()[index], new1.errors[i]);
  }

  // Object 2: old + new.
  for (size_t i = 0; i < old2.size(); ++i) {
    EXPECT_EQ(object2.points()[i], old2.points[i]);

    EXPECT_EQ(object2.previousPoints()[i], old2.previous_points[i]);

    EXPECT_EQ(object2.ids()[i], old2.ids[i]);

    EXPECT_EQ(object2.objectIds()[i], 2);

    EXPECT_EQ(object2.inlier()[i], old2.inlier[i]);

    EXPECT_EQ(object2.errors()[i], old2.errors[i]);
  }

  for (size_t i = 0; i < new2.size(); ++i) {
    const size_t index = old2.size() + i;

    EXPECT_EQ(object2.points()[index], new2.points[i]);

    EXPECT_EQ(object2.previousPoints()[index], new2.previous_points[i]);

    EXPECT_EQ(object2.ids()[index], new2.ids[i]);

    EXPECT_EQ(object2.objectIds()[index], 2);

    EXPECT_EQ(object2.inlier()[index], new2.inlier[i]);

    EXPECT_EQ(object2.errors()[index], new2.errors[i]);
  }

  merged.checkInvariants();
}

TEST(FeatureBlockContainerTest, reduceToInliers) {
  FeatureBlockContainer features({
      {10, 4},
      {20, 3},
      {30, 2},
  });

  // Give every feature unique identifiable data.
  for (size_t i = 0; i < features.size(); ++i) {
    features.points[i] =
        cv::Point2f(static_cast<float>(i), static_cast<float>(i + 100));

    features.previous_points[i] =
        cv::Point2f(static_cast<float>(i + 10), static_cast<float>(i + 110));

    features.ids[i] = static_cast<TrackletId>(1000 + i);
    features.age[i] = 10 + i;
    features.inlier[i] = 0;
    features.errors[i] = static_cast<float>(i) + 0.5f;
  }

  // Original layout:
  //
  // object 10: [0 1 2 3]
  // object 20: [4 5 6]
  // object 30: [7 8]
  //
  // Keep:
  // object 10: [0 2]
  // object 20: [5]
  // object 30: [7 8]

  features.inlier[0] = 1;
  features.inlier[2] = 1;
  features.inlier[5] = 1;
  features.inlier[7] = 1;
  features.inlier[8] = 1;

  IndexMapping mapping;

  const FeatureBlockContainer result = features.reduceToInliers(&mapping);

  // -------------------------------------------------------------------------
  // Size / blocks
  // -------------------------------------------------------------------------

  ASSERT_EQ(result.size(), 5);
  ASSERT_EQ(result.objectCount(), 3);

  auto blocks = result.objectViews();

  ASSERT_EQ(blocks.size(), 3);

  EXPECT_EQ(blocks[0].objectId(), 10);
  EXPECT_EQ(blocks[0].size(), 2);

  EXPECT_EQ(blocks[1].objectId(), 20);
  EXPECT_EQ(blocks[1].size(), 1);

  EXPECT_EQ(blocks[2].objectId(), 30);
  EXPECT_EQ(blocks[2].size(), 2);

  // -------------------------------------------------------------------------
  // Mapping: old index -> new index
  // -------------------------------------------------------------------------

  ASSERT_EQ(mapping.size(), 5);

  EXPECT_EQ(mapping.at(0), 0);
  EXPECT_EQ(mapping.at(2), 1);
  EXPECT_EQ(mapping.at(5), 2);
  EXPECT_EQ(mapping.at(7), 3);
  EXPECT_EQ(mapping.at(8), 4);

  // -------------------------------------------------------------------------
  // Data
  // -------------------------------------------------------------------------

  const std::vector<size_t> expected_old_indices = {0, 2, 5, 7, 8};

  for (size_t new_index = 0; new_index < expected_old_indices.size();
       ++new_index) {
    const size_t old_index = expected_old_indices[new_index];

    EXPECT_EQ(result.points[new_index], features.points[old_index]);

    EXPECT_EQ(result.previous_points[new_index],
              features.previous_points[old_index]);

    EXPECT_EQ(result.ids[new_index], features.ids[old_index]);

    EXPECT_EQ(result.age[new_index], features.age[old_index]);

    EXPECT_EQ(result.object_ids[new_index], features.object_ids[old_index]);

    EXPECT_EQ(result.errors[new_index], features.errors[old_index]);

    EXPECT_EQ(result.inlier[new_index], 1);

    EXPECT_EQ(mapping.at(old_index), new_index);
  }
}

TEST(FeatureBlockContainerTest, reduceToInliersRemovesEmptyBlocks) {
  FeatureBlockContainer features({
      {10, 3},
      {20, 3},
      {30, 3},
  });

  features.inlier[0] = 1;
  features.inlier[1] = 0;
  features.inlier[2] = 0;

  // Object 20 has no inliers.
  features.inlier[3] = 0;
  features.inlier[4] = 0;
  features.inlier[5] = 0;

  features.inlier[6] = 1;
  features.inlier[7] = 1;
  features.inlier[8] = 0;

  IndexMapping mapping;

  const auto result = features.reduceToInliers(&mapping);

  ASSERT_EQ(result.size(), 3);
  ASSERT_EQ(result.objectCount(), 2);

  const auto blocks = result.objectViews();

  ASSERT_EQ(blocks.size(), 2);

  EXPECT_EQ(blocks[0].objectId(), 10);
  EXPECT_EQ(blocks[0].size(), 1);

  EXPECT_EQ(blocks[1].objectId(), 30);
  EXPECT_EQ(blocks[1].size(), 2);

  // Mapping: old index -> new index.
  ASSERT_EQ(mapping.size(), 3);

  EXPECT_EQ(mapping.at(0), 0);
  EXPECT_EQ(mapping.at(6), 1);
  EXPECT_EQ(mapping.at(7), 2);
}

TEST(FeatureBlockContainerTest, reduceToInliersNoInliers) {
  FeatureBlockContainer features({
      {10, 3},
      {20, 2},
  });

  std::fill(features.inlier.begin(), features.inlier.end(),
            static_cast<uchar>(0));

  IndexMapping mapping;

  const auto result = features.reduceToInliers(&mapping);

  EXPECT_EQ(result.size(), 0);
  EXPECT_EQ(result.objectCount(), 0);
  EXPECT_TRUE(mapping.empty());
  EXPECT_TRUE(result.objectViews().empty());
}

TEST(FeatureBlockContainerTest, reduceToInliersAllFeatures) {
  FeatureBlockContainer features({
      {10, 3},
      {20, 2},
  });

  for (auto& value : features.inlier) {
    value = 1;
  }

  IndexMapping mapping;

  const auto result = features.reduceToInliers(&mapping);

  ASSERT_EQ(result.size(), features.size());
  ASSERT_EQ(result.objectCount(), features.objectCount());

  // ASSERT_EQ(mapping.size(), features.size());

  for (size_t i = 0; i < features.size(); ++i) {
    // EXPECT_EQ(mapping.at(i), i);

    EXPECT_EQ(result.points[i], features.points[i]);
    EXPECT_EQ(result.ids[i], features.ids[i]);
    EXPECT_EQ(result.age[i], features.age[i]);
    EXPECT_EQ(result.errors[i], features.errors[i]);
    EXPECT_EQ(result.inlier[i], 1);
  }
}

TEST(FeatureBlockContainerTest, FilterInliersInPlace) {
  FeatureBlockContainer features({
      {10, 3},
      {20, 4},
      {30, 2},
  });

  // Fill identifiable feature data.
  for (size_t i = 0; i < features.size(); ++i) {
    features.points[i] =
        cv::Point2f(static_cast<float>(i), static_cast<float>(i + 100));

    features.previous_points[i] =
        cv::Point2f(static_cast<float>(i + 10), static_cast<float>(i + 110));

    features.ids[i] = static_cast<TrackletId>(1000 + i);
    features.age[i] = i + 1;
    features.errors[i] = static_cast<float>(i) * 0.5f;
  }

  // Flat layout:
  //
  // Object 10: indices 0, 1, 2
  // Object 20: indices 3, 4, 5, 6
  // Object 30: indices 7, 8
  //
  // Keep:
  //   object 10 -> 0, 2
  //   object 20 -> 4, 6
  //   object 30 -> 8
  //
  // Old -> new:
  //   0 -> 0
  //   2 -> 1
  //   4 -> 2
  //   6 -> 3
  //   8 -> 4

  features.inlier = {
      1, 0, 1, 0, 1, 0, 1, 0, 1,
  };

  IndexMapping mapping;

  features.reduceToInliersInplace(&mapping);

  // --------------------------------------------------------------------------
  // Container structure
  // --------------------------------------------------------------------------

  EXPECT_EQ(features.size(), 5u);
  EXPECT_EQ(features.objectCount(), 3u);

  const auto views = features.objectViews();

  ASSERT_EQ(views.size(), 3u);

  EXPECT_EQ(views[0].objectId(), 10);
  EXPECT_EQ(views[0].size(), 2u);

  EXPECT_EQ(views[1].objectId(), 20);
  EXPECT_EQ(views[1].size(), 2u);

  EXPECT_EQ(views[2].objectId(), 30);
  EXPECT_EQ(views[2].size(), 1u);

  // --------------------------------------------------------------------------
  // Index mapping: old index -> new index
  // --------------------------------------------------------------------------

  ASSERT_EQ(mapping.size(), 5u);

  EXPECT_EQ(mapping.at(0u), 0u);
  EXPECT_EQ(mapping.at(2u), 1u);
  EXPECT_EQ(mapping.at(4u), 2u);
  EXPECT_EQ(mapping.at(6u), 3u);
  EXPECT_EQ(mapping.at(8u), 4u);

  // --------------------------------------------------------------------------
  // Feature data
  // --------------------------------------------------------------------------

  const std::vector<size_t> expected_old_indices = {
      0, 2, 4, 6, 8,
  };

  ASSERT_EQ(features.points.size(), expected_old_indices.size());
  ASSERT_EQ(features.previous_points.size(), expected_old_indices.size());
  ASSERT_EQ(features.ids.size(), expected_old_indices.size());
  ASSERT_EQ(features.age.size(), expected_old_indices.size());
  ASSERT_EQ(features.object_ids.size(), expected_old_indices.size());
  ASSERT_EQ(features.inlier.size(), expected_old_indices.size());
  ASSERT_EQ(features.errors.size(), expected_old_indices.size());

  for (size_t new_index = 0; new_index < expected_old_indices.size();
       ++new_index) {
    const size_t old_index = expected_old_indices[new_index];

    EXPECT_EQ(features.points[new_index],
              cv::Point2f(static_cast<float>(old_index),
                          static_cast<float>(old_index + 100)));

    EXPECT_EQ(features.previous_points[new_index],
              cv::Point2f(static_cast<float>(old_index + 10),
                          static_cast<float>(old_index + 110)));

    EXPECT_EQ(features.ids[new_index],
              static_cast<TrackletId>(1000 + old_index));

    EXPECT_EQ(features.age[new_index], old_index + 1);

    EXPECT_EQ(features.errors[new_index], static_cast<float>(old_index) * 0.5f);

    EXPECT_EQ(features.inlier[new_index], 1);

    EXPECT_EQ(mapping.at(old_index), new_index);
  }

  // Object IDs should correspond to the rebuilt blocks.
  EXPECT_EQ(features.object_ids[0], 10);
  EXPECT_EQ(features.object_ids[1], 10);
  EXPECT_EQ(features.object_ids[2], 20);
  EXPECT_EQ(features.object_ids[3], 20);
  EXPECT_EQ(features.object_ids[4], 30);

  features.checkInvariants();
}

TEST(FeatureBlockContainerTest, FilterInliersInPlaceWithNoInliers) {
  FeatureBlockContainer features({
      {10, 3},
      {20, 2},
  });

  features.inlier = {
      0, 0, 0, 0, 0,
  };

  IndexMapping mapping;

  features.reduceToInliersInplace(&mapping);

  EXPECT_EQ(features.size(), 0u);
  EXPECT_EQ(features.objectCount(), 0u);
  EXPECT_TRUE(features.objectViews().empty());
  EXPECT_TRUE(mapping.empty());

  EXPECT_TRUE(features.points.empty());
  EXPECT_TRUE(features.previous_points.empty());
  EXPECT_TRUE(features.ids.empty());
  EXPECT_TRUE(features.age.empty());
  EXPECT_TRUE(features.object_ids.empty());
  EXPECT_TRUE(features.inlier.empty());
  EXPECT_TRUE(features.errors.empty());

  features.checkInvariants();
}

// =============================================================================
// Performance
// =============================================================================

volatile float benchmark_sink = 0.0f;

TEST(FeatureBlockContainerTest, ObjectViewMatchesSoAPerformance) {
  constexpr size_t N = 10000;
  constexpr size_t ITERATIONS = 1000;

  FeatureBlockContainer features({
      {1, N},
  });

  for (size_t i = 0; i < N; ++i) {
    features.points[i] =
        cv::Point2f(static_cast<float>(i), static_cast<float>(i + 1));
  }

  const auto object = features.objectView(1);

  const double soa_ns = benchmarkNsPerFeature(
      [&]() {
        const cv::Point2f* points = features.points.data();

        float sum = 0.0f;

        for (size_t i = 0; i < N; ++i) {
          sum += points[i].x;
          sum += points[i].y;
        }

        benchmark_sink = sum;
      },
      N, ITERATIONS);

  const double view_ns = benchmarkNsPerFeature(
      [&]() {
        const cv::Point2f* points = object.points();

        float sum = 0.0f;

        for (size_t i = 0; i < object.size(); ++i) {
          sum += points[i].x;
          sum += points[i].y;
        }

        benchmark_sink = sum;
      },
      N, ITERATIONS);

  const double overhead = ((view_ns - soa_ns) / soa_ns) * 100.0;

  std::cout << "N=" << N << " | SoA=" << soa_ns << " ns/feature"
            << " | ObjectView=" << view_ns << " ns/feature"
            << " | Overhead=" << overhead << "%\n";

  EXPECT_LT(view_ns, soa_ns * 1.10);
}

TEST(FeatureBlockContainerTest, PerObjectIterationPerformance) {
  constexpr size_t FEATURES_PER_OBJECT = 2500;
  constexpr size_t OBJECT_COUNT = 4;
  constexpr size_t N = FEATURES_PER_OBJECT * OBJECT_COUNT;
  constexpr size_t ITERATIONS = 1000;

  FeatureBlockContainer features({
      {1, FEATURES_PER_OBJECT},
      {2, FEATURES_PER_OBJECT},
      {3, FEATURES_PER_OBJECT},
      {4, FEATURES_PER_OBJECT},
  });

  for (size_t i = 0; i < N; ++i) {
    features.points[i] =
        cv::Point2f(static_cast<float>(i), static_cast<float>(i + 1));
  }

  const double direct_ns = benchmarkNsPerFeature(
      [&]() {
        float sum = 0.0f;

        size_t begin = 0;

        for (size_t object = 0; object < OBJECT_COUNT; ++object) {
          const size_t end = begin + FEATURES_PER_OBJECT;

          const cv::Point2f* points = features.points.data();

          for (size_t i = begin; i < end; ++i) {
            sum += points[i].x;
            sum += points[i].y;
          }

          begin = end;
        }

        benchmark_sink = sum;
      },
      N, ITERATIONS);

  const double view_ns = benchmarkNsPerFeature(
      [&]() {
        float sum = 0.0f;

        for (int object_id = 1; object_id <= 4; ++object_id) {
          const auto object = features.objectView(object_id);

          const cv::Point2f* points = object.points();

          for (size_t i = 0; i < object.size(); ++i) {
            sum += points[i].x;
            sum += points[i].y;
          }
        }

        benchmark_sink = sum;
      },
      N, ITERATIONS);

  const double overhead = ((view_ns - direct_ns) / direct_ns) * 100.0;

  std::cout << "N=" << N << " | Direct ranges=" << direct_ns << " ns/feature"
            << " | ObjectView=" << view_ns << " ns/feature"
            << " | Overhead=" << overhead << "%\n";

  EXPECT_LT(view_ns, direct_ns * 1.10);
}
