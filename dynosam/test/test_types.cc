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

#include <filesystem>
#include <nlohmann/json.hpp>  //for gt packet seralize tests

#include "dynosam/frontend/VIFrontendInput.hpp"
#include "dynosam/frontend/vision/FeatureTrackerBase.hpp"
#include "dynosam_common/Exceptions.hpp"
#include "dynosam_common/GroundTruthPacket.hpp"
#include "dynosam_common/logger/Logger.hpp"
#include "dynosam_common/utils/JsonUtils.hpp"
#include "dynosam_common/utils/Statistics.hpp"
#include "dynosam_common/utils/TimingStats.hpp"
#include "dynosam_common/utils/Variant.hpp"
#include "dynosam_sensors/Feature.hpp"
#include "internal/helpers.hpp"
#include "internal/simulator.hpp"

using namespace dyno;

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

// //custom type with dyno::to_string defined. Must be inside dyno namespace
// namespace dyno {
//     struct CustomToString {};
// }

// template<>
// std::string dyno::to_string(const CustomToString&) {
//     return "custom_to_string";
// }

// TEST(IOTraits, testToString) {

//     EXPECT_EQ(traits<decltype(4)>::ToString(4), "4");
//     EXPECT_EQ(traits<CustomToString>::ToString(CustomToString{}),
//     "custom_to_string");
// }

// TEST(Exceptions, testExceptionStream) {
//     EXPECT_THROW({ExceptionStream::Create<DynosamException>();},
//     std::runtime_error); EXPECT_NO_THROW({ExceptionStream::Create();});
// }

// TEST(Exceptions, testExceptionStreamMessage) {
//     //would be preferable to use gmock like
//     //Throws<std::runtime_error>(Property(&std::runtime_error::what,
//     //      HasSubstr("message"))));
//     //but currently issues getting the gmock library to be found...
//     // try {
//     //     ExceptionStream::Create<std::runtime_error>() << "A message";
//     // }
//     // catch(const std::runtime_error& expected) {
//     //     EXPECT_EQ(std::string(expected.what()), "A message");
//     // }
//     // catch(...) {
//     //     FAIL() << "An excpetion was thrown but it was not
//     std::runtime_error";
//     // }
//     // FAIL() << "Exception should be thrown but was not";
//     ExceptionStream::Create<std::runtime_error>() << "A message";
// }

// // TEST(Exceptions, testBasicThrow) {
// //     checkAndThrow(false);
// //     // EXPECT_THROW({checkAndThrow(false);}, DynosamException);
// //     // EXPECT_NO_THROW({checkAndThrow(true);});
// // }

TEST(GtsamUtils, isGtsamValueType) {
  EXPECT_TRUE(is_gtsam_value_v<gtsam::Pose3>);
  EXPECT_TRUE(is_gtsam_value_v<gtsam::Point3>);
  // EXPECT_FALSE(is_gtsam_value_v<double>);
  EXPECT_FALSE(is_gtsam_value_v<ImageType::RGBMono>);
}

TEST(VariantTypes, isVariant) {
  using Var = std::variant<int, std::string>;
  EXPECT_TRUE(is_variant_v<Var>);
  EXPECT_FALSE(is_variant_v<int>);
}

TEST(VariantTypes, variantContains) {
  using Var = std::variant<int, std::string>;
  // for some reason EXPECT_TRUE doenst work?
  //  EXPECT_TRUE(isvariantmember_v<int, Var>);
  bool r = is_variant_member_v<int, Var>;
  EXPECT_EQ(r, true);

  r = is_variant_member_v<std::string, Var>;
  EXPECT_EQ(r, true);

  r = is_variant_member_v<double, Var>;
  EXPECT_EQ(r, false);
}

TEST(ImageType, testRGBMonoValidation) {
  {
    // invalid type
    cv::Mat input(cv::Size(50, 50), CV_64F);
    EXPECT_THROW({ ImageType::RGBMono::validate(input); },
                 InvalidImageTypeException);
  }

  {
    // okay type
    cv::Mat input(cv::Size(50, 50), CV_8UC1);
    EXPECT_NO_THROW({ ImageType::RGBMono::validate(input); });
  }

  {
    // okay type
    cv::Mat input(cv::Size(50, 50), CV_8UC3);
    EXPECT_NO_THROW({ ImageType::RGBMono::validate(input); });
  }
}

TEST(ImageType, testDepthValidation) {
  {
    // okay type
    cv::Mat input(cv::Size(50, 50), CV_64F);
    EXPECT_NO_THROW({ ImageType::Depth::validate(input); });
  }

  {
    // invalud type
    cv::Mat input(cv::Size(50, 50), CV_8UC3);
    EXPECT_THROW({ ImageType::Depth::validate(input); },
                 InvalidImageTypeException);
  }
}

TEST(ImageType, testOpticalFlowValidation) {
  {
    // okay type
    cv::Mat input(cv::Size(50, 50), CV_32FC2);
    EXPECT_NO_THROW({ ImageType::OpticalFlow::validate(input); });
  }

  {
    // invalud type
    cv::Mat input(cv::Size(50, 50), CV_8UC3);
    EXPECT_THROW({ ImageType::OpticalFlow::validate(input); },
                 InvalidImageTypeException);
  }
}

TEST(ImageType, testSemanticMaskValidation) {
  // TODO:
}

TEST(ImageType, testMotionMaskValidation) {
  // TODO:
}

TEST(ImageContainerV2, testBasicAdd) {
  ImageContainer container;
  EXPECT_EQ(container.size(), 0u);
  EXPECT_TRUE(container.exists("rgb") == false);

  cv::Mat input(cv::Size(50, 50), CV_8UC3);
  container.add<ImageType::RGBMono>("rgb", input);
  EXPECT_TRUE(container.exists("rgb"));

  EXPECT_EQ(container.size(), 1u);
  ImageWrapper<ImageType::RGBMono> wrapped =
      container.at<ImageType::RGBMono>("rgb");
  EXPECT_TRUE(wrapped.exists());

  cv::Mat retrieved_image = wrapped;
  EXPECT_EQ(retrieved_image.data, input.data);
}

TEST(ImageContainerV2, testBasicAddWrongType) {
  ImageContainer container;
  EXPECT_EQ(container.size(), 0u);

  cv::Mat input(cv::Size(50, 50), CV_8UC3);
  container.add<ImageType::RGBMono>("rgb", input);
  EXPECT_THROW({ container.at<ImageType::OpticalFlow>("rgb"); },
               MismatchedImageWrapperTypes);
}

TEST(ImageContainerV2, testInvalidImageInput) {
  ImageContainer container;
  EXPECT_EQ(container.size(), 0u);

  cv::Mat optical_flow(cv::Size(25, 25), CV_32FC2);
  // request RGBMono but give optical flow type!!
  EXPECT_THROW({ container.add<ImageType::RGBMono>("rgb", optical_flow); },
               InvalidImageTypeException);
}

TEST(ImageContainerV2, testMultiAdd) {
  ImageContainer container;
  EXPECT_EQ(container.size(), 0u);

  cv::Mat input(cv::Size(50, 50), CV_8UC3);
  cv::Mat optical_flow(cv::Size(25, 25), CV_32FC2);
  container.add<ImageType::RGBMono>("rgb", input);
  container.add<ImageType::OpticalFlow>("flow", optical_flow);
  EXPECT_TRUE(container.exists("rgb"));
  EXPECT_TRUE(container.exists("flow"));

  EXPECT_EQ(container.size(), 2u);
  {
    ImageWrapper<ImageType::RGBMono> wrapped =
        container.at<ImageType::RGBMono>("rgb");
    EXPECT_TRUE(wrapped.exists());
  }

  EXPECT_EQ(container.size(), 2u);
  {
    ImageWrapper<ImageType::OpticalFlow> wrapped =
        container.at<ImageType::OpticalFlow>("flow");
    EXPECT_TRUE(wrapped.exists());
  }
}

TEST(ImageContainerV2, CopySharesCvMatData) {
  ImageContainer container1;
  cv::Mat img = cv::Mat::ones(10, 10, CV_8UC1);
  container1.rgb(img);

  // Copy container
  ImageContainer container2 = container1;

  // Check they share the same underlying data pointer
  auto& mat1 = container1.rgb();
  auto& mat2 = container2.rgb();

  // cv::Mat::data returns the underlying pixel data pointer
  EXPECT_EQ(mat1.image().data, mat2.image().data);
  EXPECT_EQ(container1.frameId(), container2.frameId());
  EXPECT_EQ(container1.timestamp(), container2.timestamp());

  // Modifying one should affect the other (since shared)
  mat1.image().at<uint8_t>(0, 0) = 42;
  EXPECT_EQ(mat2.image().at<uint8_t>(0, 0), 42);
}

TEST(ImageContainerV2, ExplicitDeepCopyCreatesNewData) {
  ImageContainer container1;
  cv::Mat img = cv::Mat::ones(10, 10, CV_8UC1);
  container1.rgb(img);

  // Make a deep copy of the cv::Mat inside container2
  ImageContainer container2 = container1.clone();

  EXPECT_EQ(container1.frameId(), container2.frameId());
  EXPECT_EQ(container1.timestamp(), container2.timestamp());

  auto& mat1 = container1.rgb();
  auto& mat2 = container2.rgb();

  EXPECT_NE(mat1.image().data, mat2.image().data);

  // Changing one does NOT affect the other
  mat1.image().at<uint8_t>(0, 0) = 42;
  EXPECT_NE(mat2.image().at<uint8_t>(0, 0), 42);
}

TEST(ImageContainerV2, MoveConstructorTransfersOwnership) {
  ImageContainer original;
  cv::Mat img = cv::Mat::ones(5, 5, CV_8UC1);
  original.rgb(img);
  EXPECT_TRUE(original.hasRgb());
  EXPECT_EQ(original.size(), 1);

  ImageContainer moved_to = std::move(original);

  EXPECT_TRUE(original.hasRgb());
  EXPECT_EQ(original.size(), 1);

  // Original is in valid, empty state
  EXPECT_EQ(original.size(), 0);

  cv::Mat& mat1 = moved_to.rgb();

  EXPECT_THROW({ original.rgb(); }, ImageKeyDoesNotExist);

  // Changing the moved image works as expected
  mat1.at<uint8_t>(0, 0) = 99;
  EXPECT_EQ(mat1.at<uint8_t>(0, 0), 99);

  // The original container should now be empty
  EXPECT_EQ(original.size(), 0);
}

enum class TestOptions : std::uint8_t {
  None = 0,
  A = 1 << 0,
  B = 1 << 1,
  C = 1 << 2
};

template <>
struct dyno::internal::EnableBitMaskOperators<TestOptions> : std::true_type {};

TEST(BitwiseFlags, testUnderlyingTypeSpecalization) {
  using TestFlags = Flags<TestOptions>;

  using U = std::underlying_type_t<TestFlags>;
  static_assert(std::is_same_v<U, uint8_t>);
}

TEST(BitwiseFlags, testCombinesFlagsCorrectly) {
  using TestFlags = Flags<TestOptions>;
  TestOptions ab = TestOptions::A | TestOptions::B;
  EXPECT_EQ(static_cast<uint8_t>(ab), (1 << 0) | (1 << 1));

  TestOptions masked = ab & TestOptions::A;
  EXPECT_EQ(static_cast<uint8_t>(masked), 1 << 0);

  TestOptions inverted = ~TestOptions::None;
  EXPECT_EQ(static_cast<uint8_t>(inverted), 0xFF);  // for uint8_t
}

TEST(BitwiseFlags, testCanSetAndCheckFlags) {
  using TestFlags = Flags<TestOptions>;
  TestFlags flags;
  flags.set(TestOptions::A).set(TestOptions::C);

  EXPECT_TRUE(flags.has(TestOptions::A));
  EXPECT_TRUE(flags.has(TestOptions::C));
  EXPECT_FALSE(flags.has(TestOptions::B));

  EXPECT_TRUE(flags.hasAny(TestOptions::A | TestOptions::B));
  EXPECT_FALSE(flags.hasAll(TestOptions::A | TestOptions::B));
  EXPECT_TRUE(flags.hasAll(TestOptions::A | TestOptions::C));
  EXPECT_FALSE(flags.hasAll(TestOptions::A | TestOptions::B | TestOptions::C));
}

TEST(BitwiseFlags, testEqualsOperators) {
  using TestFlags = Flags<TestOptions>;
  TestFlags flags;
  flags.set(TestOptions::A).set(TestOptions::C);

  TestFlags flags1(TestOptions::A | TestOptions::C);
  EXPECT_TRUE(flags == flags1);

  EXPECT_TRUE(flags == TestOptions::A);
  EXPECT_TRUE(flags != TestOptions::B);
  EXPECT_TRUE(flags == TestOptions::C);
}

TEST(BitwiseFlags, testAssignmentOperator) {
  using TestFlags = Flags<TestOptions>;
  TestFlags flags;
  EXPECT_TRUE(flags != (TestOptions::A | TestOptions::B));
  flags = TestOptions::A | TestOptions::B;
  EXPECT_TRUE(flags == (TestOptions::A | TestOptions::B));
}

TEST(FeatureContainer, basicAdd) {
  FeatureContainer fc;
  EXPECT_EQ(fc.size(), 0u);
  EXPECT_FALSE(fc.exists(1));

  Feature f;
  f.trackletId(1);
  f.objectId(0);

  fc.add(f);
  EXPECT_EQ(fc.size(), 1u);
  EXPECT_TRUE(fc.exists(1));

  auto tracklets = fc.getByObject(0);
  EXPECT_EQ(tracklets.size(), 1);
  EXPECT_EQ(tracklets.at(0), 1);

  // this implicitly tests map access
  auto fr = fc.getByTrackletId(1);
  EXPECT_TRUE(fr != nullptr);
  EXPECT_EQ(*fr, f);
}

TEST(FeatureContainer, basicAddMultipleObjects) {
  FeatureContainer fc;

  {
    Feature f;
    f.trackletId(1);
    f.objectId(1);
    fc.add(f);
  }

  {
    Feature f;
    f.trackletId(2);
    f.objectId(1);
    fc.add(f);
  }

  {
    Feature f;
    f.trackletId(3);
    f.objectId(1);
    fc.add(f);
  }

  {
    Feature f;
    f.trackletId(4);
    f.objectId(2);
    fc.add(f);
  }

  EXPECT_EQ(fc.size(), 4u);
  EXPECT_TRUE(fc.exists(1));

  {
    auto tracklets = fc.getByObject(1);
    EXPECT_THAT(tracklets,
                ::testing::UnorderedElementsAreArray(TrackletIds{1, 2, 3}));
  }

  {
    auto tracklets = fc.getByObject(2);
    EXPECT_THAT(tracklets,
                ::testing::UnorderedElementsAreArray(TrackletIds{4}));
  }
}

TEST(FeatureContainer, basicRemove) {
  FeatureContainer fc;
  EXPECT_EQ(fc.size(), 0u);

  Feature f;
  f.trackletId(1);
  f.objectId(1);

  fc.add(f);
  EXPECT_EQ(fc.size(), 1u);
  EXPECT_EQ(fc.size(1), 1);
  EXPECT_EQ(fc.size(2), 0);
  EXPECT_EQ(fc.getByObject(1).size(), 1);

  fc.remove(1);
  EXPECT_FALSE(fc.exists(1));
  EXPECT_EQ(fc.size(1), 0);
  EXPECT_EQ(fc.size(2), 0);
  EXPECT_EQ(fc.getByObject(1).size(), 0);

  auto fr = fc.getByTrackletId(1);
  EXPECT_TRUE(fr == nullptr);
}

TEST(FeatureContainer, testVectorLikeIteration) {
  FeatureContainer fc;

  for (size_t i = 0; i < 10u; i++) {
    Feature f;
    f.trackletId(i);
    fc.add(f);
  }

  int count = 0;
  for (const auto& i : fc) {
    EXPECT_TRUE(i != nullptr);
    count++;
  }

  EXPECT_EQ(count, 10);
  count = 0;

  fc.remove(0);
  fc.remove(1);

  for (const auto& i : fc) {
    EXPECT_TRUE(i != nullptr);
    EXPECT_TRUE(i->trackletId() != 0 || i->trackletId() != 1);
    count++;
  }

  EXPECT_EQ(count, 8);
}

TEST(FeatureContainer, testusableIterator) {
  FeatureContainer fc;

  for (size_t i = 0; i < 10u; i++) {
    Feature f;
    f.trackletId(i);
    fc.add(f);
    EXPECT_TRUE(f.usable());
  }

  {
    auto usable_iterator = fc.usableIterator();
    EXPECT_EQ(std::distance(usable_iterator.begin(), usable_iterator.end()),
              10);
  }

  fc.markOutliers({3});
  fc.markOutliers({4});

  fc.getByTrackletId(1)->markOutlier();

  {
    auto usable_iterator = fc.usableIterator();
    EXPECT_EQ(std::distance(usable_iterator.begin(), usable_iterator.end()), 7);

    for (const auto& f : fc) {
      if (f->trackletId() == 3 || f->trackletId() == 4 ||
          f->trackletId() == 1) {
        EXPECT_FALSE(f->usable());
        EXPECT_FALSE(f->inlier());
      } else {
        EXPECT_TRUE(f->inlier());
        EXPECT_TRUE(f->usable());
      }
    }

    for (const auto& f : usable_iterator) {
      EXPECT_TRUE(f->trackletId() == 0 || f->trackletId() == 2 ||
                  f->trackletId() == 5 || f->trackletId() == 6 ||
                  f->trackletId() == 7 || f->trackletId() == 8 ||
                  f->trackletId() == 9);
    }
  }
}

Feature::Ptr makeFeature(TrackletId tid, ObjectId oid, bool usable = true) {
  auto f = std::make_shared<Feature>();
  f->trackletId(tid);
  f->objectId(oid);
  if (usable)
    f->markInlier();
  else
    f->markOutlier();
  return f;
}

TEST(FeatureContainer, AddFeatures) {
  FeatureContainer container;

  auto f1 = makeFeature(1, 100);
  auto f2 = makeFeature(2, 100, false);
  auto f3 = makeFeature(3, 200);

  container.add(f1);
  container.add(f2);
  container.add(f3);

  // Tracklet map
  EXPECT_TRUE(container.exists(f1->trackletId()));
  EXPECT_TRUE(container.exists(f2->trackletId()));
  EXPECT_TRUE(container.exists(f3->trackletId()));

  // Container size
  EXPECT_EQ(container.size(), 3u);
  EXPECT_EQ(container.size(100), 2u);
  EXPECT_EQ(container.size(200), 1u);

  // Object map
  EXPECT_TRUE(container.hasObject(f1->objectId()));
  EXPECT_TRUE(container.hasObject(f3->objectId()));
}

// -----------------------------------------------------------
// Test ObjectFeatureViewT iteration
TEST(FeatureContainer, ObjectFeatureViewIteration) {
  FeatureContainer container;

  auto f1 = makeFeature(1, 100);
  auto f2 = makeFeature(2, 100);
  container.add(f1);
  container.add(f2);

  // Object map validation before removal
  auto featuresForObjectBefore = container.getByObject(100);
  EXPECT_TRUE(std::find(featuresForObjectBefore.begin(),
                        featuresForObjectBefore.end(),
                        f1->trackletId()) != featuresForObjectBefore.end());
  EXPECT_TRUE(std::find(featuresForObjectBefore.begin(),
                        featuresForObjectBefore.end(),
                        f2->trackletId()) != featuresForObjectBefore.end());

  // Remove the first feature by tracklet
  container.remove(f1->trackletId());

  // Tracklet map validation
  EXPECT_FALSE(container.exists(f1->trackletId()));
  EXPECT_TRUE(container.exists(f2->trackletId()));

  // Object map validation after removal
  auto featuresForObjectAfter = container.getByObject(100);
  EXPECT_TRUE(std::find(featuresForObjectAfter.begin(),
                        featuresForObjectAfter.end(), f1->trackletId()) ==
              featuresForObjectAfter.end());  // f1 removed
  EXPECT_TRUE(std::find(featuresForObjectAfter.begin(),
                        featuresForObjectAfter.end(), f2->trackletId()) !=
              featuresForObjectAfter.end());  // f2 still present
}

TEST(FeatureContainer, ObjectFeatureViewIterationNonExistantObject) {
  FeatureContainer container;

  auto f1 = makeFeature(1, 100);
  auto f2 = makeFeature(2, 100);
  container.add(f1);
  container.add(f2);

  auto usable_it = container.usableIterator(500);

  int size = std::distance(usable_it.begin(), usable_it.end());
  EXPECT_EQ(size, 0);
}

// -----------------------------------------------------------
// Test FastUsableObjectIterator filtering
TEST(FeatureContainer, FilterUsableFeaturesPerObject) {
  FeatureContainer container;

  auto f1 = makeFeature(1, 100, true);
  auto f2 = makeFeature(2, 100, false);
  container.add(f1);
  container.add(f2);

  auto usable_it = container.usableIterator(100);
  std::vector<Feature::Ptr> collected;

  for (auto& f : usable_it) {
    collected.push_back(f);
  }

  ASSERT_EQ(collected.size(), 1u);  // only f1 is usable
  EXPECT_EQ(collected[0]->trackletId(), f1->trackletId());

  collected.clear();
  f1->markOutlier();
  for (auto& f : usable_it) {
    collected.push_back(f);
  }
  ASSERT_EQ(collected.size(), 0u);
}

// -----------------------------------------------------------
// Test clearing the container
TEST(FeatureContainer, ClearContainer) {
  FeatureContainer container;

  auto f1 = makeFeature(1, 100);
  container.add(f1);

  EXPECT_FALSE(container.empty());
  container.clear();
  EXPECT_TRUE(container.empty());
  EXPECT_EQ(container.size(), 0u);
}

TEST(FeatureContainer, IterateNonExistantObject) {
  FeatureContainer container;

  auto f1 = makeFeature(1, 1);
  container.add(f1);
  EXPECT_TRUE(container.hasObject(1));

  auto object1_view = container.usableIterator(1);
  EXPECT_TRUE(object1_view.begin() != object1_view.end());

  for (const auto& f : object1_view) {
    (void)f;
  }

  // there is no object 2
  auto object2_view = container.usableIterator(2);
  EXPECT_TRUE(object2_view.begin() == object2_view.end());
  for (const auto& f : object2_view) {
    (void)f;
  }
}

TEST(FeatureContainer, testFailureCaseFromCopy) {
  FeatureContainer container;
  // make feature for object 1 - this creates an object view for j=1
  auto f1 = makeFeature(1, 1);
  container.add(f1);
  EXPECT_TRUE(container.hasObject(1));

  // assign container
  FeatureContainer container1 = container;
  // make new feature for object 1 which does not exist for the original
  // container
  auto f2 = makeFeature(2, 1);
  // in original implementation the object view for j=1 will will point to
  // container (not container1) which does not have f2
  container1.add(f2);

  auto it_obj1 = container1.usableIterator(1);
  for (const auto& f : it_obj1) {
  }
}

// -----------------------------------------------------------
// Test removing features by tracklet
TEST(FeatureContainer, RemoveFeatureByTracklet) {
  FeatureContainer container;

  auto f1 = makeFeature(1, 100);
  auto f2 = makeFeature(2, 100);
  container.add(f1);
  container.add(f2);

  // Object map validation before removal
  auto featuresForObjectBefore = container.getByObject(100);
  EXPECT_NE(std::find(featuresForObjectBefore.begin(),
                      featuresForObjectBefore.end(), f1->trackletId()),
            featuresForObjectBefore.end());
  EXPECT_NE(std::find(featuresForObjectBefore.begin(),
                      featuresForObjectBefore.end(), f2->trackletId()),
            featuresForObjectBefore.end());

  // Remove the first feature by tracklet
  container.remove(f1->trackletId());

  // Tracklet map validation
  EXPECT_FALSE(container.exists(f1->trackletId()));
  EXPECT_TRUE(container.exists(f2->trackletId()));

  // Object map validation after removal
  auto featuresForObjectAfter = container.getByObject(100);
  EXPECT_EQ(std::find(featuresForObjectAfter.begin(),
                      featuresForObjectAfter.end(), f1->trackletId()),
            featuresForObjectAfter.end());  // f1 removed
  EXPECT_NE(std::find(featuresForObjectAfter.begin(),
                      featuresForObjectAfter.end(), f2->trackletId()),
            featuresForObjectAfter.end());  // f2 still present
}

// -----------------------------------------------------------
// Test removing features by object
TEST(FeatureContainer, RemoveFeaturesByObject) {
  FeatureContainer container;

  auto f1 = makeFeature(1, 100);
  auto f2 = makeFeature(2, 200);
  container.add(f1);
  container.add(f2);

  container.removeByObjectId(100);
  EXPECT_FALSE(container.hasObject(100));
  EXPECT_TRUE(container.hasObject(200));
}

// -----------------------------------------------------------
// Stress test: many features, multiple objects
TEST(FeatureContainer, StressTestMultipleObjectsAndFilter) {
  FeatureContainer container;
  constexpr int numObjects = 10;
  constexpr int featuresPerObject = 100;

  std::vector<Feature::Ptr> allFeatures;

  for (int obj = 1; obj <= numObjects; ++obj) {
    for (int tid = 1; tid <= featuresPerObject; ++tid) {
      // Alternate usable flag
      bool usable = (tid % 2 == 0);
      auto f = makeFeature(tid + obj * 1000, obj, usable);
      container.add(f);
      allFeatures.push_back(f);
    }
  }

  // Check container size
  EXPECT_EQ(container.size(), numObjects * featuresPerObject);

  // Iterate per object using FastUsableObjectIterator
  for (int obj = 1; obj <= numObjects; ++obj) {
    auto usable_it = container.usableIterator(obj);
    int count = 0;
    for (auto& f : usable_it) {
      EXPECT_TRUE(f->usable());
      EXPECT_EQ(f->objectId(), obj);
      ++count;
    }
    EXPECT_EQ(count, featuresPerObject / 2);  // half are usable
  }
}

TEST(Feature, checkInvalidState) {
  Feature f;
  EXPECT_TRUE(f.inlier());
  EXPECT_FALSE(f.usable());  // inlier initally but invalid tracking label

  f.trackletId(10);
  EXPECT_TRUE(f.usable());

  f.markInvalid();
  EXPECT_FALSE(f.usable());

  f.trackletId(10u);
  EXPECT_TRUE(f.usable());

  f.markOutlier();
  EXPECT_FALSE(f.usable());
}

TEST(Feature, checkDepth) {
  Feature f;
  EXPECT_FALSE(f.hasDepth());

  f.depth(12.0);
  EXPECT_TRUE(f.hasDepth());
}

TEST(SensorTypes, MeasurementWithCovarianceConstructionEmpty) {
  MeasurementWithCovariance<Landmark> measurement;
  EXPECT_FALSE(measurement.hasModel());
}

TEST(SensorTypes, MeasurementWithCovarianceConstructionMeasurement) {
  Landmark lmk(10, 12.4, 0.001);
  MeasurementWithCovariance<Landmark> measurement(lmk);
  EXPECT_FALSE(measurement.hasModel());
  EXPECT_EQ(measurement.measurement(), lmk);
}

TEST(SensorTypes, MeasurementWithCovarianceConstructionMeasurementAndSigmas) {
  Landmark lmk(10, 12.4, 0.001);
  gtsam::Vector3 sigmas;
  sigmas << 0.1, 0.2, 0.3;
  MeasurementWithCovariance<Landmark> measurement =
      MeasurementWithCovariance<Landmark>::FromSigmas(lmk, sigmas);
  EXPECT_TRUE(measurement.hasModel());
  EXPECT_EQ(measurement.measurement(), lmk);

  MeasurementWithCovariance<Landmark>::Covariance cov =
      measurement.covariance();

  MeasurementWithCovariance<Landmark>::Covariance expected_cov =
      sigmas.array().pow(2).matrix().asDiagonal();
  EXPECT_TRUE(gtsam::assert_equal(expected_cov, cov));
}

TEST(SensorTypes, MeasurementWithCovarianceConstructionMeasurementAndCov) {
  Landmark lmk(10, 12.4, 0.001);
  MeasurementWithCovariance<Landmark>::Covariance expected_cov;
  expected_cov << 0.1, 0, 0, 0, 0.2, 0, 0, 0, 0.4;
  MeasurementWithCovariance<Landmark> measurement(lmk, expected_cov);
  EXPECT_TRUE(measurement.hasModel());
  EXPECT_EQ(measurement.measurement(), lmk);

  MeasurementWithCovariance<Landmark>::Covariance cov =
      measurement.covariance();
  EXPECT_TRUE(gtsam::assert_equal(expected_cov, cov));
}

TEST(JsonIO, ReferenceFrameValue) {
  ReferenceFrameValue<gtsam::Pose3> ref_frame(gtsam::Pose3::Identity(),
                                              ReferenceFrame::GLOBAL);

  using json = nlohmann::json;
  json j = ref_frame;

  auto ref_frame_load = j.template get<ReferenceFrameValue<gtsam::Pose3>>();
  // TODO: needs equals operator
  //  EXPECT_EQ(kp_load, kp);
}

TEST(JsonIO, ObjectPoseGTIO) {
  ObjectPoseGT object_pose_gt;

  object_pose_gt.frame_id_ = 0;
  object_pose_gt.object_id_ = 1;
  object_pose_gt.L_camera_ = gtsam::Pose3::Identity();
  object_pose_gt.L_world_ = gtsam::Pose3::Identity();
  object_pose_gt.prev_H_current_L_ = gtsam::Pose3::Identity();

  using json = nlohmann::json;
  json j = object_pose_gt;

  auto object_pose_gt_2 = j.template get<ObjectPoseGT>();
  EXPECT_EQ(object_pose_gt, object_pose_gt_2);
}

TEST(JsonIO, MeasurementWithCovSigmas) {
  using json = nlohmann::json;
  Landmark lmk(10, 12.4, 0.001);
  MeasurementWithCovariance<Landmark>::Covariance expected_cov;
  expected_cov << 0.1, 0, 0, 0, 0.2, 0, 0, 0, 0.4;
  MeasurementWithCovariance<Landmark> measurement(lmk, expected_cov);
  json j = measurement;
  auto measurements_load =
      j.template get<MeasurementWithCovariance<Landmark>>();
  EXPECT_TRUE(gtsam::assert_equal(measurements_load, measurement));
}

TEST(JsonIO, MeasurementWithCov) {
  using json = nlohmann::json;
  Landmark lmk(10, 12.4, 0.001);
  gtsam::Vector3 sigmas;
  sigmas << 0.1, 0.2, 0.3;
  MeasurementWithCovariance<Landmark> measurement =
      MeasurementWithCovariance<Landmark>::FromSigmas(lmk, sigmas);

  json j = measurement;
  auto measurements_load =
      j.template get<MeasurementWithCovariance<Landmark>>();
  EXPECT_TRUE(gtsam::assert_equal(measurements_load, measurement));
}

TEST(JsonIO, MeasurementWithNoCov) {
  using json = nlohmann::json;
  Landmark lmk(10, 12.4, 0.001);
  MeasurementWithCovariance<Landmark> measurement(lmk);

  json j = measurement;
  auto measurements_load =
      j.template get<MeasurementWithCovariance<Landmark>>();
  EXPECT_TRUE(gtsam::assert_equal(measurements_load, measurement));
}

TEST(JsonIO, GenericValueTrackKp) {
  KeypointStatus kp =
      dyno_testing::makeStatusKeypointMeasurement(4, 3, 1, Keypoint(0, 1));

  using json = nlohmann::json;
  json j = (KeypointStatus)kp;

  auto kp_load = j.template get<KeypointStatus>();
  // TODO: needs equals operator
  EXPECT_EQ(kp_load, kp);
}

TEST(JsonIO, GenericValueTrackKps) {
  StatusKeypointVector measurements;
  for (size_t i = 0; i < 10; i++) {
    measurements.push_back(
        dyno_testing::makeStatusKeypointMeasurement(i, background_label, 0));
  }
  using json = nlohmann::json;
  json j = measurements;

  auto measurements_load = j.template get<StatusKeypointVector>();
  EXPECT_EQ(measurements, measurements);
}

TEST(JsonIO, RGBDInstanceOutputPacket) {
  auto scenario = dyno_testing::makeDefaultScenario();

  std::map<FrameId, VisionImuPacket> rgbd_output;

  for (size_t i = 0; i < 10; i++) {
    auto output = scenario.getOutput(i);
    rgbd_output.insert({i, *output.first});
  }

  using json = nlohmann::json;
  json j = rgbd_output;
  std::map<FrameId, VisionImuPacket> rgbd_output_loaded =
      j.template get<std::map<FrameId, VisionImuPacket>>();
  EXPECT_EQ(rgbd_output_loaded, rgbd_output);
}

TEST(JsonIO, GroundTruthInputPacketIO) {
  GroundTruthInputPacket gt_packet;

  using json = nlohmann::json;
  json j = gt_packet;

  auto gt_packet_2 = j.template get<GroundTruthInputPacket>();
}

TEST(JsonIO, GroundTruthPacketMapIO) {
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

  GroundTruthPacketMap gt_packet_map;
  gt_packet_map.insert2(0, packet_0);
  gt_packet_map.insert2(1, packet_1);

  using json = nlohmann::json;
  json j = gt_packet_map;

  auto gt_packet_map_2 = j.template get<GroundTruthPacketMap>();
  EXPECT_EQ(gt_packet_map, gt_packet_map_2);
}

TEST(JsonIO, eigenJsonIO) {
  Eigen::Matrix4d m;
  m << 1.0, 2.0, 3.0, 4.0, 11.0, 12.0, 13.0, 14.0, 21.0, 22.0, 23.0, 24.0, 31.0,
      32.0, 33.0, 34.0;
  nlohmann::json j = m;
  // std::cerr << j.dump() << std::endl;
  Eigen::Matrix4d m2 = j.get<Eigen::Matrix4d>();

  EXPECT_TRUE(gtsam::assert_equal(m, m2));
}

TEST(JsonIO, testTemporalObjectCentricMap) {
  using Map = TemporalObjectCentricMap<gtsam::Pose3>;

  Map map;
  // add two frames for object 1
  map.insert22(1, 1, gtsam::Pose3::Identity());
  map.insert22(1, 2, gtsam::Pose3::Identity());

  // one frame for object 2
  map.insert22(2, 1, gtsam::Pose3::Identity());
  nlohmann::json j = map;
  std::cout << j << std::endl;

  gtsam::FastMap<FrameId, int> gtsam_map;
  gtsam_map.insert2(1, 10);
  gtsam_map.insert2(2, 10);
  j = gtsam_map;
  std::cout << "gtsam map " << j << std::endl;

  std::map<std::string, int> std_map;
  std_map["1"] = 10;
  std_map["2"] = 10;
  j = std_map;
  std::cout << "std_map " << j << std::endl;
}

namespace fs = std::filesystem;
class JsonIOWithFiles : public ::testing::Test {
 public:
  JsonIOWithFiles() {}

 protected:
  virtual void SetUp() { fs::create_directory(sandbox); }
  virtual void TearDown() { fs::remove_all(sandbox); }

  const fs::path sandbox{"/tmp/sandbox_json"};
};

TEST_F(JsonIOWithFiles, testSimpleBison) {
  StatusKeypointVector measurements;
  for (size_t i = 0; i < 10; i++) {
    measurements.push_back(
        dyno_testing::makeStatusKeypointMeasurement(i, background_label, 0));
  }

  fs::path tmp_bison_path = sandbox / "simple_bison.bson";
  std::string tmp_bison_path_str = tmp_bison_path;

  JsonConverter::WriteOutJson(measurements, tmp_bison_path_str,
                              JsonConverter::Format::BSON);

  StatusKeypointVector measurements_read;
  EXPECT_TRUE(JsonConverter::ReadInJson(measurements_read, tmp_bison_path_str,
                                        JsonConverter::Format::BSON));
  EXPECT_EQ(measurements_read, measurements);
}

TEST(GenericValueTrack, testIsTimeInvariant) {
  GenericValueTrack<Keypoint> status_time_invariant(
      MeasurementWithCovariance<Keypoint>{Keypoint()},
      GenericValueTrack<Keypoint>::MeaninglessFrame, 0, 0.0, 0,
      ReferenceFrame::GLOBAL);

  EXPECT_TRUE(status_time_invariant.isTimeInvariant());

  GenericValueTrack<Keypoint> status_time_variant(
      MeasurementWithCovariance<Keypoint>{Keypoint()},
      0,  // use zero,
      0.0, 0, 0, ReferenceFrame::GLOBAL);
  EXPECT_FALSE(status_time_variant.isTimeInvariant());
}

TEST(Statistics, testGetModules) {
  utils::StatsCollector("global_stats").IncrementOne();
  utils::StatsCollector("ns.spin").IncrementOne();
  utils::StatsCollector("ns.spin1").IncrementOne();

  EXPECT_EQ(utils::Statistics::getTagByModule(),
            std::vector<std::string>({"global_stats"}));
  EXPECT_EQ(utils::Statistics::getTagByModule("ns"),
            std::vector<std::string>({"ns.spin", "ns.spin1"}));
}

// some nice hacky global variables for testing the mock timing generator ;)
size_t start_called{0};
size_t stop_called{0};

struct MockTimingGenerator {
  void onStart() { start_called++; }
  void onStop() { stop_called++; }

  double calcDelta() const { return 0.0; }
};

class MockTimingStats
    : public utils::BaseTimingStatsCollector<MockTimingGenerator> {
 public:
  using Base = utils::BaseTimingStatsCollector<MockTimingGenerator>;
  MockTimingStats(const std::string& tag, int glog_level = 0,
                  bool construct_stopped = false)
      : Base(MockTimingGenerator{}, tag, glog_level, construct_stopped) {}
};

MockTimingStats createMockTimingStats() {
  return MockTimingStats{"mock_timing_stats"};
}

TEST(TimingStats, functionReturnDoesNotTriggerLog) {
  MockTimingStats timing_stats = createMockTimingStats();

  EXPECT_EQ(start_called, 1);
  EXPECT_EQ(stop_called, 0);
  EXPECT_TRUE(timing_stats.isTiming());
}

class FeatureTrackerBaseTest : public FeatureTrackerBase {
 public:
  FeatureTrackerBaseTest(Camera::Ptr camera)
      : FeatureTrackerBase(TrackerParams{}, camera, nullptr) {}

  using FeatureTrackerBase::isWithinShrunkenImage;
};

TEST(FeatureTrackerBase, isWithinShrunkenImage1) {
  CameraParams::IntrinsicsCoeffs intrinsics(4);
  CameraParams::DistortionCoeffs distortion(4);
  intrinsics.at(0) = 554.256;  // fx
  intrinsics.at(1) = 554.256;  // fy
  intrinsics.at(2) = 640 / 2;  // u0
  intrinsics.at(3) = 480 / 2;  // v0

  // specicific test case to fail
  CameraParams cam_params(intrinsics, distortion, cv::Size(752, 480), "radtan");
  FeatureTrackerBaseTest test(std::make_shared<Camera>(cam_params));

  cv::Mat img = cv::Mat::zeros(480, 752, CV_8U);
  Keypoint kp(440.197, 479.823);
  cv::Point2f kp_f = utils::gtsamPointToCv(kp);
  cv::Point2i kp_i = utils::gtsamPointToCv<int>(kp);
  LOG(INFO) << kp_f;
  LOG(INFO) << kp_i;

  LOG(INFO) << kp << " " << dyno::to_string(img.size());

  LOG(INFO) << test.isWithinShrunkenImage(kp);
  LOG(INFO) << test.isWithinShrunkenImage(kp_f);
  img.at<unsigned char>(kp_i);
}

namespace dyno {

class FeatureIterationBenchmark : public ::testing::Test {
 protected:
  using Clock = std::chrono::steady_clock;

  struct SoAFeatures {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<size_t> age;
    std::vector<ObjectId> object_id;
    std::vector<TrackletId> tracklet_id;
    std::vector<bool> inlier;

    void reserve(size_t n) {
      x.reserve(n);
      y.reserve(n);
      age.reserve(n);
      object_id.reserve(n);
      tracklet_id.reserve(n);
      inlier.reserve(n);
    }

    void add(float x_, float y_, size_t age_, ObjectId object_id_,
             TrackletId tracklet_id_, bool inlier_) {
      x.push_back(x_);
      y.push_back(y_);
      age.push_back(age_);
      object_id.push_back(object_id_);
      tracklet_id.push_back(tracklet_id_);
      inlier.push_back(inlier_);
    }

    size_t size() const { return x.size(); }
  };

  struct TestData {
    FeatureContainer container;
    SoAFeatures soa;
  };

  static constexpr size_t kNumFeatures = 10000;
  static constexpr size_t kNumObjects = 10;

  static TestData makeData(size_t num_features = kNumFeatures) {
    TestData data;
    data.soa.reserve(num_features);

    for (size_t i = 0; i < num_features; ++i) {
      // Distribute features across objects.
      // object 0 is background, objects 1..N are dynamic.
      const ObjectId object_id = static_cast<ObjectId>(i % (kNumObjects + 1));

      const float x = static_cast<float>(i % 640);
      const float y = static_cast<float>((i / 640) % 480);
      const size_t age = i % 20;
      const TrackletId tracklet_id = static_cast<TrackletId>(i);
      const bool inlier = (i % 5) != 0;

      Feature feature;
      feature.keypoint(cv::Point2f(x, y))
          .age(age)
          .trackletId(tracklet_id)
          .objectId(object_id);

      if (inlier) {
        feature.markInlier();
      } else {
        feature.markOutlier();
      }

      data.container.add(feature);

      data.soa.add(x, y, age, object_id, tracklet_id, inlier);
    }

    return data;
  }

  template <typename Func>
  static double timeFunction(Func&& func, size_t iterations,
                             size_t work_items) {
    // Warm up.
    for (size_t i = 0; i < 5; ++i) {
      func();
    }

    const auto start = Clock::now();

    for (size_t i = 0; i < iterations; ++i) {
      func();
    }

    const auto end = Clock::now();

    const double total_ns =
        std::chrono::duration<double, std::nano>(end - start).count();

    return total_ns / static_cast<double>(iterations * work_items);
  }

  static void printResult(const std::string& name, double ns_per_feature) {
    std::cout << name << ": " << ns_per_feature << " ns/feature"
              << " (" << ns_per_feature / 1000.0 << " us/feature)" << std::endl;
  }
};

// -----------------------------------------------------------------------------
// 1. Simple global iteration
//
// Measures:
//   unordered_map traversal
//   shared_ptr dereference
//
// Does NOT call Feature getters, so this isolates the container structure.
// -----------------------------------------------------------------------------

TEST_F(FeatureIterationBenchmark, GlobalIterationOnly) {
  auto data = makeData();

  volatile uintptr_t sink = 0;

  constexpr size_t iterations = 1000;

  const double container_ns = timeFunction(
      [&]() {
        uintptr_t local = 0;

        for (const auto& feature : data.container) {
          local += reinterpret_cast<uintptr_t>(feature.get());
        }

        sink += local;
      },
      iterations, data.container.size());

  const double soa_ns = timeFunction(
      [&]() {
        uintptr_t local = 0;

        for (size_t i = 0; i < data.soa.size(); ++i) {
          local += reinterpret_cast<uintptr_t>(&data.soa.x[i]);
        }

        sink += local;
      },
      iterations, data.soa.size());

  printResult("FeatureContainer global iteration", container_ns);
  printResult("SoA iteration", soa_ns);

  std::cout << "Container / SoA ratio: " << container_ns / soa_ns << "x"
            << std::endl;

  EXPECT_NE(sink, 0u);
}

// -----------------------------------------------------------------------------
// 2. Global iteration + feature property access
//
// This is probably the more realistic test for your tracking code.
//
// It deliberately accesses several fields.
// -----------------------------------------------------------------------------

TEST_F(FeatureIterationBenchmark, GlobalIterationAndPropertyAccess) {
  auto data = makeData();

  volatile double sink = 0.0;

  constexpr size_t iterations = 500;

  const double container_ns = timeFunction(
      [&]() {
        double local = 0.0;

        for (const auto& feature : data.container) {
          const auto kp = feature->keypoint();

          local += kp(0);
          local += kp(1);
          local += static_cast<double>(feature->age());
          local += static_cast<double>(feature->objectId());
          local += static_cast<double>(feature->trackletId());
          local += feature->inlier() ? 1.0 : 0.0;
        }

        sink += local;
      },
      iterations, data.container.size());

  const double soa_ns = timeFunction(
      [&]() {
        double local = 0.0;

        for (size_t i = 0; i < data.soa.size(); ++i) {
          local += data.soa.x[i];
          local += data.soa.y[i];
          local += static_cast<double>(data.soa.age[i]);
          local += static_cast<double>(data.soa.object_id[i]);
          local += static_cast<double>(data.soa.tracklet_id[i]);
          local += data.soa.inlier[i] ? 1.0 : 0.0;
        }

        sink += local;
      },
      iterations, data.soa.size());

  printResult("FeatureContainer + property access", container_ns);
  printResult("SoA + property access", soa_ns);

  std::cout << "Container / SoA ratio: " << container_ns / soa_ns << "x"
            << std::endl;

  EXPECT_NE(sink, 0.0);
}

// -----------------------------------------------------------------------------
// 3. Object iteration
//
// This specifically measures:
//
//   ObjectFeatureView
//       -> unordered_set<TrackletId>
//       -> feature_map_.find()
//       -> feature_map_.at()
//       -> Feature::Ptr
//
// versus:
//
//   SoA indices corresponding to a particular object.
//
// -----------------------------------------------------------------------------

TEST_F(FeatureIterationBenchmark, ObjectIteration) {
  auto data = makeData();

  constexpr ObjectId object_id = 5;
  constexpr size_t iterations = 1000;

  volatile double sink = 0.0;

  const size_t object_count = data.container.size(object_id);

  ASSERT_GT(object_count, 0u);

  // Build equivalent SoA index list.
  std::vector<size_t> object_indices;
  object_indices.reserve(object_count);

  for (size_t i = 0; i < data.soa.size(); ++i) {
    if (data.soa.object_id[i] == object_id) {
      object_indices.push_back(i);
    }
  }

  ASSERT_EQ(object_indices.size(), object_count);

  const double container_ns = timeFunction(
      [&]() {
        double local = 0.0;

        for (const auto& feature : data.container.featuresByObject(object_id)) {
          local += feature->keypoint()(0);
          local += feature->keypoint()(1);
          local += static_cast<double>(feature->age());
          local += static_cast<double>(feature->trackletId());
        }

        sink += local;
      },
      iterations, object_count);

  const double soa_ns = timeFunction(
      [&]() {
        double local = 0.0;

        for (const size_t i : object_indices) {
          local += data.soa.x[i];
          local += data.soa.y[i];
          local += static_cast<double>(data.soa.age[i]);
          local += static_cast<double>(data.soa.tracklet_id[i]);
        }

        sink += local;
      },
      iterations, object_count);

  printResult("FeatureContainer object iteration", container_ns);
  printResult("SoA object iteration", soa_ns);

  std::cout << "Object iteration Container / SoA ratio: "
            << container_ns / soa_ns << "x" << std::endl;

  EXPECT_NE(sink, 0.0);
}

// -----------------------------------------------------------------------------
// 4. Object iteration WITHOUT Feature getters
//
// This isolates the actual object-view mechanism from Feature's accessor cost.
// -----------------------------------------------------------------------------

TEST_F(FeatureIterationBenchmark, ObjectIterationOnly) {
  auto data = makeData();

  constexpr ObjectId object_id = 5;
  constexpr size_t iterations = 1000;

  volatile uintptr_t sink = 0;

  const size_t object_count = data.container.size(object_id);

  ASSERT_GT(object_count, 0u);

  std::vector<size_t> object_indices;
  object_indices.reserve(object_count);

  for (size_t i = 0; i < data.soa.size(); ++i) {
    if (data.soa.object_id[i] == object_id) {
      object_indices.push_back(i);
    }
  }

  const double container_ns = timeFunction(
      [&]() {
        uintptr_t local = 0;

        for (const auto& feature : data.container.featuresByObject(object_id)) {
          local += reinterpret_cast<uintptr_t>(feature.get());
        }

        sink += local;
      },
      iterations, object_count);

  const double soa_ns = timeFunction(
      [&]() {
        uintptr_t local = 0;

        for (const size_t i : object_indices) {
          local += reinterpret_cast<uintptr_t>(&data.soa.x[i]);
        }

        sink += local;
      },
      iterations, object_count);

  printResult("FeatureContainer object traversal", container_ns);
  printResult("SoA object traversal", soa_ns);

  std::cout << "Object traversal Container / SoA ratio: "
            << container_ns / soa_ns << "x" << std::endl;

  EXPECT_NE(sink, 0u);
}

// -----------------------------------------------------------------------------
// 5. Scaling test
//
// Useful for seeing whether the relative cost changes with the number of
// features.
// -----------------------------------------------------------------------------

TEST_F(FeatureIterationBenchmark, GlobalIterationScaling) {
  constexpr size_t iterations = 500;

  volatile double sink = 0.0;

  for (const size_t n : {100u, 500u, 1000u, 5000u, 10000u, 20000u}) {
    auto data = makeData(n);

    const double container_ns = timeFunction(
        [&]() {
          double local = 0.0;

          for (const auto& feature : data.container) {
            local += feature->keypoint()(0);
            local += feature->keypoint()(1);
            local += feature->age();
            local += feature->objectId();
          }

          sink += local;
        },
        iterations, data.container.size());

    const double soa_ns = timeFunction(
        [&]() {
          double local = 0.0;

          for (size_t i = 0; i < data.soa.size(); ++i) {
            local += data.soa.x[i];
            local += data.soa.y[i];
            local += data.soa.age[i];
            local += data.soa.object_id[i];
          }

          sink += local;
        },
        iterations, data.soa.size());

    std::cout << "N=" << n << " | Container=" << container_ns << " ns/feature"
              << " | SoA=" << soa_ns << " ns/feature"
              << " | ratio=" << container_ns / soa_ns << "x" << std::endl;
  }

  EXPECT_NE(sink, 0.0);
}

class FeatureStorageBenchmark : public ::testing::Test {
 protected:
  using Clock = std::chrono::steady_clock;

  struct SoAFeatures {
    std::vector<double> x;
    std::vector<double> y;
    std::vector<size_t> age;
    std::vector<ObjectId> object_id;
    std::vector<TrackletId> tracklet_id;
    std::vector<bool> inlier;

    void reserve(size_t n) {
      x.reserve(n);
      y.reserve(n);
      age.reserve(n);
      object_id.reserve(n);
      tracklet_id.reserve(n);
      inlier.reserve(n);
    }

    void add(double x_, double y_, size_t age_, ObjectId object_id_,
             TrackletId tracklet_id_, bool inlier_) {
      x.push_back(x_);
      y.push_back(y_);
      age.push_back(age_);
      object_id.push_back(object_id_);
      tracklet_id.push_back(tracklet_id_);
      inlier.push_back(inlier_);
    }

    size_t size() const { return x.size(); }
  };

  struct TestData {
    FeatureContainer container;

    // Same features stored as a contiguous vector of pointers.
    std::vector<Feature::Ptr> feature_ptrs;

    // Same features stored directly and contiguously.
    std::vector<Feature> features;

    // Structure-of-arrays representation.
    SoAFeatures soa;
  };

  static constexpr size_t kNumObjects = 10;

  static TestData makeData(size_t num_features) {
    TestData data;

    data.feature_ptrs.reserve(num_features);
    data.features.reserve(num_features);
    data.soa.reserve(num_features);

    for (size_t i = 0; i < num_features; ++i) {
      const ObjectId object_id = static_cast<ObjectId>(i % (kNumObjects + 1));

      const double x = static_cast<double>(i % 640);

      const double y = static_cast<double>((i / 640) % 480);

      const size_t age = i % 20;

      const TrackletId tracklet_id = static_cast<TrackletId>(i);

      const bool inlier = (i % 5) != 0;

      Feature feature;

      feature.keypoint(cv::Point2f(x, y))
          .age(age)
          .trackletId(tracklet_id)
          .objectId(object_id);

      if (inlier) {
        feature.markInlier();
      } else {
        feature.markOutlier();
      }

      // Current FeatureContainer implementation.
      data.container.add(feature);

      // Contiguous vector of Feature objects.
      data.features.push_back(feature);

      // Contiguous vector of Feature pointers.
      data.feature_ptrs.push_back(std::make_shared<Feature>(feature));

      // Structure of arrays.
      data.soa.add(x, y, age, object_id, tracklet_id, inlier);
    }

    return data;
  }

  template <typename Func>
  static double timeFunction(Func&& func, size_t iterations,
                             size_t work_items) {
    // Warm-up.
    for (size_t i = 0; i < 5; ++i) {
      func();
    }

    const auto start = Clock::now();

    for (size_t i = 0; i < iterations; ++i) {
      func();
    }

    const auto end = Clock::now();

    const double total_ns =
        std::chrono::duration<double, std::nano>(end - start).count();

    return total_ns / static_cast<double>(iterations * work_items);
  }

  static void printResult(const std::string& name, double ns_per_feature) {
    std::cout << name << ": " << ns_per_feature << " ns/feature"
              << " (" << ns_per_feature / 1000.0 << " us/feature)" << std::endl;
  }
};

// =============================================================================
// 1. PURE TRAVERSAL
//
// No Feature getters are called.
//
// This isolates the cost of:
//
//   unordered_map<TrackletId, Feature::Ptr>
//   vector<Feature::Ptr>
//   vector<Feature>
//   SoA
// =============================================================================

TEST_F(FeatureStorageBenchmark, PureTraversal) {
  constexpr size_t n = 10000;
  constexpr size_t iterations = 1000;

  auto data = makeData(n);

  volatile uintptr_t sink = 0;

  const double container_ns = timeFunction(
      [&]() {
        uintptr_t local = 0;

        for (const auto& feature : data.container) {
          local += reinterpret_cast<uintptr_t>(feature.get());
        }

        sink += local;
      },
      iterations, n);

  const double ptr_vector_ns = timeFunction(
      [&]() {
        uintptr_t local = 0;

        for (const auto& feature : data.feature_ptrs) {
          local += reinterpret_cast<uintptr_t>(feature.get());
        }

        sink += local;
      },
      iterations, n);

  const double vector_feature_ns = timeFunction(
      [&]() {
        uintptr_t local = 0;

        for (const auto& feature : data.features) {
          local += reinterpret_cast<uintptr_t>(&feature);
        }

        sink += local;
      },
      iterations, n);

  const double soa_ns = timeFunction(
      [&]() {
        uintptr_t local = 0;

        for (size_t i = 0; i < data.soa.size(); ++i) {
          local += reinterpret_cast<uintptr_t>(&data.soa.x[i]);
        }

        sink += local;
      },
      iterations, n);

  printResult("unordered_map<id, Feature::Ptr>", container_ns);

  printResult("vector<Feature::Ptr>", ptr_vector_ns);

  printResult("vector<Feature>", vector_feature_ns);

  printResult("SoA", soa_ns);

  std::cout << std::endl;

  std::cout << "Map / vector<Ptr>: " << container_ns / ptr_vector_ns << "x"
            << std::endl;

  std::cout << "Map / vector<Feature>: " << container_ns / vector_feature_ns
            << "x" << std::endl;

  std::cout << "Map / SoA: " << container_ns / soa_ns << "x" << std::endl;

  EXPECT_NE(sink, 0u);
}

// =============================================================================
// 2. PROPERTY ACCESS
//
// Access several properties that resemble a typical VO frontend.
//
// This measures the complete traversal + Feature access cost.
// =============================================================================

TEST_F(FeatureStorageBenchmark, PropertyAccess) {
  constexpr size_t n = 10000;
  constexpr size_t iterations = 500;

  auto data = makeData(n);

  volatile double sink = 0.0;

  const double container_ns = timeFunction(
      [&]() {
        double local = 0.0;

        for (const auto& feature : data.container) {
          const auto kp = feature->keypoint();

          local += kp(0);
          local += kp(1);
          local += static_cast<double>(feature->age());
          local += static_cast<double>(feature->objectId());
          local += static_cast<double>(feature->trackletId());

          if (feature->inlier()) {
            local += 1.0;
          }
        }

        sink += local;
      },
      iterations, n);

  const double ptr_vector_ns = timeFunction(
      [&]() {
        double local = 0.0;

        for (const auto& feature : data.feature_ptrs) {
          const auto kp = feature->keypoint();

          local += kp(0);
          local += kp(1);
          local += static_cast<double>(feature->age());
          local += static_cast<double>(feature->objectId());
          local += static_cast<double>(feature->trackletId());

          if (feature->inlier()) {
            local += 1.0;
          }
        }

        sink += local;
      },
      iterations, n);

  const double vector_feature_ns = timeFunction(
      [&]() {
        double local = 0.0;

        for (const auto& feature : data.features) {
          const auto kp = feature.keypoint();

          local += kp(0);
          local += kp(1);
          local += static_cast<double>(feature.age());
          local += static_cast<double>(feature.objectId());
          local += static_cast<double>(feature.trackletId());

          if (feature.inlier()) {
            local += 1.0;
          }
        }

        sink += local;
      },
      iterations, n);

  const double soa_ns = timeFunction(
      [&]() {
        double local = 0.0;

        for (size_t i = 0; i < data.soa.size(); ++i) {
          local += data.soa.x[i];
          local += data.soa.y[i];
          local += static_cast<double>(data.soa.age[i]);
          local += static_cast<double>(data.soa.object_id[i]);
          local += static_cast<double>(data.soa.tracklet_id[i]);

          if (data.soa.inlier[i]) {
            local += 1.0;
          }
        }

        sink += local;
      },
      iterations, n);

  printResult("unordered_map<id, Feature::Ptr>", container_ns);

  printResult("vector<Feature::Ptr>", ptr_vector_ns);

  printResult("vector<Feature>", vector_feature_ns);

  printResult("SoA", soa_ns);

  std::cout << std::endl;

  std::cout << "Map / vector<Ptr>: " << container_ns / ptr_vector_ns << "x"
            << std::endl;

  std::cout << "Map / vector<Feature>: " << container_ns / vector_feature_ns
            << "x" << std::endl;

  std::cout << "Map / SoA: " << container_ns / soa_ns << "x" << std::endl;

  EXPECT_NE(sink, 0.0);
}

// =============================================================================
// 3. VO-STYLE ITERATION
//
// Similar to a typical frontend operation:
//
//   - check usability
//   - obtain keypoint
//   - process the point
//
// This is likely the most representative benchmark for your use case.
// =============================================================================

TEST_F(FeatureStorageBenchmark, VOStyleIteration) {
  constexpr size_t n = 10000;
  constexpr size_t iterations = 500;

  auto data = makeData(n);

  volatile double sink = 0.0;

  const double container_ns = timeFunction(
      [&]() {
        double local = 0.0;

        for (const auto& feature : data.container) {
          if (!feature->usable()) {
            continue;
          }

          const auto kp = feature->keypoint();

          local += kp(0);
          local += kp(1);
        }

        sink += local;
      },
      iterations, n);

  const double ptr_vector_ns = timeFunction(
      [&]() {
        double local = 0.0;

        for (const auto& feature : data.feature_ptrs) {
          if (!feature->usable()) {
            continue;
          }

          const auto kp = feature->keypoint();

          local += kp(0);
          local += kp(1);
        }

        sink += local;
      },
      iterations, n);

  const double vector_feature_ns = timeFunction(
      [&]() {
        double local = 0.0;

        for (const auto& feature : data.features) {
          if (!feature.usable()) {
            continue;
          }

          const auto kp = feature.keypoint();

          local += kp(0);
          local += kp(1);
        }

        sink += local;
      },
      iterations, n);

  const double soa_ns = timeFunction(
      [&]() {
        double local = 0.0;

        for (size_t i = 0; i < data.soa.size(); ++i) {
          if (!data.soa.inlier[i] ||
              data.soa.tracklet_id[i] == Feature::invalid_id) {
            continue;
          }

          local += data.soa.x[i];
          local += data.soa.y[i];
        }

        sink += local;
      },
      iterations, n);

  printResult("unordered_map<id, Feature::Ptr>", container_ns);

  printResult("vector<Feature::Ptr>", ptr_vector_ns);

  printResult("vector<Feature>", vector_feature_ns);

  printResult("SoA", soa_ns);

  std::cout << std::endl;

  std::cout << "Map / vector<Ptr>: " << container_ns / ptr_vector_ns << "x"
            << std::endl;

  std::cout << "Map / vector<Feature>: " << container_ns / vector_feature_ns
            << "x" << std::endl;

  std::cout << "Map / SoA: " << container_ns / soa_ns << "x" << std::endl;

  EXPECT_NE(sink, 0.0);
}

// =============================================================================
// 4. SCALING
//
// See how the different representations behave as the number of features
// increases.
// =============================================================================

TEST_F(FeatureStorageBenchmark, Scaling) {
  constexpr size_t iterations = 500;

  volatile double sink = 0.0;

  for (const size_t n : {100u, 500u, 1000u, 5000u, 10000u, 20000u}) {
    auto data = makeData(n);

    const double container_ns = timeFunction(
        [&]() {
          double local = 0.0;

          for (const auto& feature : data.container) {
            const auto kp = feature->keypoint();

            local += kp(0);
            local += kp(1);
            local += feature->age();
            local += feature->objectId();
          }

          sink += local;
        },
        iterations, n);

    const double ptr_vector_ns = timeFunction(
        [&]() {
          double local = 0.0;

          for (const auto& feature : data.feature_ptrs) {
            const auto kp = feature->keypoint();

            local += kp(0);
            local += kp(1);
            local += feature->age();
            local += feature->objectId();
          }

          sink += local;
        },
        iterations, n);

    const double vector_feature_ns = timeFunction(
        [&]() {
          double local = 0.0;

          for (const auto& feature : data.features) {
            const auto kp = feature.keypoint();

            local += kp(0);
            local += kp(1);
            local += feature.age();
            local += feature.objectId();
          }

          sink += local;
        },
        iterations, n);

    const double soa_ns = timeFunction(
        [&]() {
          double local = 0.0;

          for (size_t i = 0; i < data.soa.size(); ++i) {
            local += data.soa.x[i];
            local += data.soa.y[i];
            local += data.soa.age[i];
            local += data.soa.object_id[i];
          }

          sink += local;
        },
        iterations, n);

    std::cout << "N=" << n << " | Map=" << container_ns << " ns/feature"
              << " | vector<Ptr>=" << ptr_vector_ns << " ns/feature"
              << " | vector<Feature>=" << vector_feature_ns << " ns/feature"
              << " | SoA=" << soa_ns << " ns/feature" << std::endl;
  }

  EXPECT_NE(sink, 0.0);
}

class FeatureSet {
 public:
  // =========================================================================
  // External feature data
  //
  // This is the convenient representation used when constructing/filling a
  // FeatureSet. All vectors must have the same size.
  // =========================================================================

  struct FeatureData {
    std::vector<cv::Point2f> points;
    std::vector<cv::Point2f> previous_points;
    std::vector<int> ids;
    std::vector<uchar> status;
    std::vector<float> errors;

    size_t size() const { return points.size(); }

    bool empty() const { return points.empty(); }

    void checkSizes() const {
      const size_t n = points.size();

      if (previous_points.size() != n || ids.size() != n ||
          status.size() != n || errors.size() != n) {
        throw std::invalid_argument(
            "FeatureData: all feature arrays must have the same size");
      }
    }
  };

  // =========================================================================
  // Object specification
  //
  // The caller specifies the logical layout only.
  // Physical begin/end indices are completely internal to FeatureSet.
  // =========================================================================

  struct ObjectSpec {
    int object_id;
    size_t size;
  };

 private:
  // =========================================================================
  // Internal object metadata
  // =========================================================================

  struct ObjectMetadata {
    int object_id;
    size_t begin;
    size_t end;

    size_t size() const { return end - begin; }
  };

 public:
  // =========================================================================
  // Writable object view
  // =========================================================================

  class ObjectView {
   public:
    size_t size() const { return end_ - begin_; }

    int objectId() const { return object_id_; }

    cv::Point2f* points() { return features_->points.data() + begin_; }

    const cv::Point2f* points() const {
      return features_->points.data() + begin_;
    }

    cv::Point2f* previousPoints() {
      return features_->previous_points.data() + begin_;
    }

    const cv::Point2f* previousPoints() const {
      return features_->previous_points.data() + begin_;
    }

    int* ids() { return features_->ids.data() + begin_; }

    const int* ids() const { return features_->ids.data() + begin_; }

    int* objectIds() { return features_->object_ids.data() + begin_; }

    const int* objectIds() const {
      return features_->object_ids.data() + begin_;
    }

    uchar* status() { return features_->status.data() + begin_; }

    const uchar* status() const { return features_->status.data() + begin_; }

    float* errors() { return features_->errors.data() + begin_; }

    const float* errors() const { return features_->errors.data() + begin_; }

    // potentiall dangerous as we could modify the points mat!
    cv::Mat pointsMat() {
      return cv::Mat(static_cast<int>(size()), 1, CV_32FC2, points());
    }

    // ---------------------------------------------------------------------
    // Convenient bulk assignment
    // ---------------------------------------------------------------------

    void copyFrom(const FeatureData& data) {
      data.checkSizes();

      if (data.size() != size()) {
        throw std::invalid_argument(
            "FeatureSet::ObjectView::copyFrom: "
            "FeatureData size does not match object size");
      }

      copyBlock(points(), data.points.data(), size());
      copyBlock(previousPoints(), data.previous_points.data(), size());
      copyBlock(ids(), data.ids.data(), size());
      copyBlock(status(), data.status.data(), size());
      copyBlock(errors(), data.errors.data(), size());
    }

   private:
    friend class FeatureSet;

    ObjectView(FeatureSet* features, int object_id, size_t begin, size_t end)
        : features_(features),
          object_id_(object_id),
          begin_(begin),
          end_(end) {}

    template <typename T>
    static void copyBlock(T* destination, const T* source, size_t count) {
      static_assert(std::is_trivially_copyable<T>::value,
                    "FeatureSet fields must be trivially copyable");

      if (count > 0) {
        std::memcpy(destination, source, count * sizeof(T));
      }
    }

    // in reality might be pointer to const FeatureSet.
    // TODO: redesign with template as before
    FeatureSet* features_;
    int object_id_;
    size_t begin_;
    size_t end_;
  };

  // =========================================================================
  // Read-only object view
  // =========================================================================

  // =========================================================================
  // Construction
  // =========================================================================

  FeatureSet(std::initializer_list<ObjectSpec> specs) {
    initialize(specs.begin(), specs.end());
  }

  explicit FeatureSet(const std::vector<ObjectSpec>& specs) {
    initialize(specs.begin(), specs.end());
  }

  //@tparam TERMS A container whose value type is std::pair<ObjectId,
  // FeatureData>
  template <typename TERMS>
  explicit FeatureSet(const TERMS& terms) {
    std::vector<ObjectSpec> specs;
    specs.reserve(terms.size());
    for (typename TERMS::const_iterator it = terms.begin(); it != terms.end();
         ++it) {
      const auto& term = *it;

      const ObjectId object_id = term.first;
      const FeatureData& data = term.second;

      data.checkSizes();

      specs.push_back({object_id, data.size()});
    }

    initialize(specs.begin(), specs.end());

    for (typename TERMS::const_iterator it = terms.begin(); it != terms.end();
         ++it) {
      const auto& term = *it;

      const ObjectId object_id = term.first;
      const FeatureData& source = term.second;

      ObjectView destination = objectView(term.first);
      destination.copyFrom(source);
    }

    checkInvariants();
  }

  // =========================================================================
  // Basic information
  // =========================================================================

  size_t size() const { return points.size(); }

  size_t objectCount() const { return objects_.size(); }

  bool containsObject(int object_id) const {
    return object_lookup_.find(object_id) != object_lookup_.end();
  }

  FeatureSet merge(const FeatureSet& other) const {
    // check for tracklet ids dupliactes
    // TODO: comment out for now - this makes creating empty or initalised but
    // inassigned FeatureSets invalid becuase all objectids/trackletids will
    // have the same id!
    //  std::unordered_set<int> feature_ids;
    //  feature_ids.reserve(ids.size() + other.ids.size());

    // for (const int id : ids)
    // {
    //     feature_ids.insert(id);
    // }

    // for (const int id : other.ids)
    // {
    //     if (!feature_ids.insert(id).second)
    //     {
    //         throw std::invalid_argument(
    //             "FeatureSet::merge: duplicate feature ID " +
    //             std::to_string(id));
    //     }
    // }

    // -------------------------------------------------------------------------
    // Build the resulting object layout.
    //
    // Existing objects retain their order.
    // New objects from `other` are appended in `other`'s order.
    // -------------------------------------------------------------------------

    std::vector<ObjectSpec> specs;
    specs.reserve(objects_.size() + other.objects_.size());

    // Existing objects.
    for (const ObjectMetadata& object : objects_) {
      const auto other_it = other.object_lookup_.find(object.object_id);

      const size_t other_size = other_it != other.object_lookup_.end()
                                    ? other.objects_[other_it->second].size()
                                    : 0;

      specs.push_back({object.object_id, object.size() + other_size});
    }

    // Objects which only exist in `other`.
    for (const ObjectMetadata& object : other.objects_) {
      if (object_lookup_.find(object.object_id) == object_lookup_.end()) {
        specs.push_back({object.object_id, object.size()});
      }
    }

    // -------------------------------------------------------------------------
    // Allocate the final FeatureSet exactly once.
    // -------------------------------------------------------------------------

    FeatureSet result(specs);

    // -------------------------------------------------------------------------
    // Copy the existing features into their final locations.
    // -------------------------------------------------------------------------

    for (const ObjectMetadata& object : objects_) {
      const ObjectView source = objectView(object.object_id);

      ObjectView destination = result.objectView(object.object_id);

      // copyBlock(
      //     destination.points(),
      //     source.points(),
      //     source.size());

      // copyBlock(
      //     destination.previousPoints(),
      //     source.previousPoints(),
      //     source.size());

      // copyBlock(
      //     destination.ids(),
      //     source.ids(),
      //     source.size());

      // copyBlock(
      //     destination.status(),
      //     source.status(),
      //     source.size());

      // copyBlock(
      //     destination.errors(),
      //     source.errors(),
      //     source.size());
      copyFeatures(destination, source);
    }

    // -------------------------------------------------------------------------
    // Append features from `other`.
    //
    // Existing objects are appended after their existing features.
    // New-only objects are copied starting at offset zero.
    // -------------------------------------------------------------------------

    for (const ObjectMetadata& other_object : other.objects_) {
      const ObjectView source = other.objectView(other_object.object_id);

      ObjectView destination = result.objectView(other_object.object_id);

      const auto existing_it = object_lookup_.find(other_object.object_id);

      const size_t destination_offset =
          existing_it != object_lookup_.end()
              ? objects_[existing_it->second].size()
              : 0;

      if (source.size() == 0) continue;

      copyFeatures(destination, source, destination_offset);

      // copyBlock(
      //     destination.points() + destination_offset,
      //     source.points(),
      //     source.size());

      // copyBlock(
      //     destination.previousPoints() + destination_offset,
      //     source.previousPoints(),
      //     source.size());

      // copyBlock(
      //     destination.ids() + destination_offset,
      //     source.ids(),
      //     source.size());

      // copyBlock(
      //     destination.status() + destination_offset,
      //     source.status(),
      //     source.size());

      // copyBlock(
      //     destination.errors() + destination_offset,
      //     source.errors(),
      //     source.size());
    }

    // object_ids are established by the FeatureSet constructor and therefore
    // don't need to be copied during the merge.

    result.checkInvariants();

    return result;
  }

  // =========================================================================
  // Object access
  // =========================================================================

  ObjectView objectView(int object_id) {
    const ObjectMetadata& metadata = objectMetadata(object_id);

    return ObjectView(this, metadata.object_id, metadata.begin, metadata.end);
  }

  const ObjectView objectView(int object_id) const {
    const ObjectMetadata& metadata = objectMetadata(object_id);

    return ObjectView(const_cast<FeatureSet*>(this), metadata.object_id,
                      metadata.begin, metadata.end);
  }

  // =========================================================================
  // Public SoA storage
  // =========================================================================

  std::vector<cv::Point2f> points;
  std::vector<cv::Point2f> previous_points;
  std::vector<int> ids;
  std::vector<int> object_ids;
  std::vector<uchar> status;
  std::vector<float> errors;

  // =========================================================================
  // Debug invariant checking
  // =========================================================================

  void checkInvariants() const {
#ifndef NDEBUG
    const size_t n = points.size();

    assert(previous_points.size() == n);
    assert(ids.size() == n);
    assert(object_ids.size() == n);
    assert(status.size() == n);
    assert(errors.size() == n);

    size_t expected_begin = 0;

    for (const ObjectMetadata& object : objects_) {
      assert(object.begin == expected_begin);
      assert(object.begin <= object.end);
      assert(object.end <= n);

      for (size_t i = object.begin; i < object.end; ++i) {
        assert(object_ids[i] == object.object_id);
      }

      expected_begin = object.end;
    }

    assert(expected_begin == n);
#endif
  }

 private:
  template <typename T>
  static void copyBlock(T* destination, const T* source, size_t count) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "FeatureSet fields must be trivially copyable");

    if (count > 0) {
      std::memcpy(destination, source, count * sizeof(T));
    }
  }

  static void copyFeatures(ObjectView destination, const ObjectView& source,
                           size_t destination_offset = 0) {
    copyBlock(destination.points() + destination_offset, source.points(),
              source.size());

    copyBlock(destination.previousPoints() + destination_offset,
              source.previousPoints(), source.size());

    copyBlock(destination.ids() + destination_offset, source.ids(),
              source.size());

    copyBlock(destination.status() + destination_offset, source.status(),
              source.size());

    copyBlock(destination.errors() + destination_offset, source.errors(),
              source.size());
  }

  // =========================================================================
  // Layout construction
  // =========================================================================

  template <typename Iterator>
  void initialize(Iterator begin, Iterator end) {
    const size_t object_count = static_cast<size_t>(std::distance(begin, end));

    objects_.reserve(object_count);
    object_lookup_.reserve(object_count);

    size_t total_size = 0;

    for (Iterator it = begin; it != end; ++it) {
      const ObjectSpec& spec = *it;

      if (object_lookup_.find(spec.object_id) != object_lookup_.end()) {
        throw std::invalid_argument("FeatureSet: duplicate object ID " +
                                    std::to_string(spec.object_id));
      }

      const size_t object_begin = total_size;
      const size_t object_end = total_size + spec.size;

      object_lookup_.emplace(spec.object_id, objects_.size());

      objects_.push_back(
          ObjectMetadata{spec.object_id, object_begin, object_end});

      total_size = object_end;
    }

    points.resize(total_size);
    previous_points.resize(total_size);
    ids.resize(total_size);
    object_ids.resize(total_size);
    status.resize(total_size);
    errors.resize(total_size);

    // Initialise object IDs immediately so the FeatureSet is valid even
    // before the caller fills the feature data.
    for (const ObjectMetadata& object : objects_) {
      std::fill(object_ids.begin() + object.begin,
                object_ids.begin() + object.end, object.object_id);
    }

    checkInvariants();
  }

  const ObjectMetadata& objectMetadata(int object_id) const {
    auto it = object_lookup_.find(object_id);

    if (it == object_lookup_.end()) {
      throw std::out_of_range("FeatureSet: unknown object ID " +
                              std::to_string(object_id));
    }

    return objects_[it->second];
  }

  std::vector<ObjectMetadata> objects_;
  std::unordered_map<int, size_t> object_lookup_;
};

volatile float benchmark_sink = 0.0f;

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

// FeatureSet createFeatureSet(
//     const std::vector<int>& object_ids,
//     const std::vector<size_t>& features_per_object)
// {
//     EXPECT_EQ(object_ids.size(), features_per_object.size());

//     FeatureSet features(object_ids);

//     size_t total_features = 0;

//     for (size_t n : features_per_object)
//     {
//         total_features += n;
//     }

//     features.points.resize(total_features);
//     features.previous_points.resize(total_features);
//     features.ids.resize(total_features);
//     features.object_ids.resize(total_features);
//     features.status.resize(total_features);
//     features.errors.resize(total_features);

//     size_t offset = 0;
//     int feature_id = 0;

//     for (size_t object_idx = 0;
//          object_idx < object_ids.size();
//          ++object_idx)
//     {
//         const int object_id = object_ids[object_idx];
//         const size_t count = features_per_object[object_idx];

//         auto& metadata = features.object(object_id);

//         metadata.begin = offset;
//         metadata.end = offset + count;

//         for (size_t i = 0; i < count; ++i)
//         {
//             features.points[offset + i] = cv::Point2f(
//                 static_cast<float>(feature_id),
//                 static_cast<float>(feature_id) + 0.5f);

//             features.previous_points[offset + i] = cv::Point2f(
//                 static_cast<float>(feature_id) - 0.25f,
//                 static_cast<float>(feature_id) + 0.25f);

//             features.ids[offset + i] = feature_id;
//             features.object_ids[offset + i] = object_id;
//             features.status[offset + i] = 1;
//             features.errors[offset + i] =
//                 static_cast<float>(feature_id) * 0.1f;

//             ++feature_id;
//         }

//         offset += count;
//     }

//     features.checkInvariants();

//     return features;
// }

FeatureSet::FeatureData createFeatureData(size_t n, int id_offset = 0) {
  FeatureSet::FeatureData data;

  data.points.resize(n);
  data.previous_points.resize(n);
  data.ids.resize(n);
  data.status.resize(n);
  data.errors.resize(n);

  for (size_t i = 0; i < n; ++i) {
    data.points[i] =
        cv::Point2f(static_cast<float>(i), static_cast<float>(i + 10));

    data.previous_points[i] =
        cv::Point2f(static_cast<float>(i + 100), static_cast<float>(i + 110));

    data.ids[i] = id_offset + static_cast<int>(i);

    data.status[i] = static_cast<uchar>(i % 2);

    data.errors[i] = static_cast<float>(i) * 0.5f;
  }

  return data;
}

FeatureSet createFeatureSet(const std::vector<int>& object_ids,
                            const std::vector<size_t>& features_per_object) {
  EXPECT_EQ(object_ids.size(), features_per_object.size());

  std::vector<FeatureSet::ObjectSpec> specs;
  specs.reserve(object_ids.size());

  for (size_t i = 0; i < object_ids.size(); ++i) {
    specs.push_back({object_ids[i], features_per_object[i]});
  }

  return FeatureSet(specs);
}

void copyObjectData(FeatureSet& features, int object_id,
                    const FeatureSet::FeatureData& data) {
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

TEST(FeatureSetTest, ConstructsKnownObjects) {
  FeatureSet features({
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

TEST(FeatureSetTest, ConstructsEmptyFeatureSet) {
  FeatureSet features({});

  EXPECT_EQ(features.objectCount(), 0);
  EXPECT_EQ(features.size(), 0);

  features.checkInvariants();
}

TEST(FeatureSetTest, AllowsObjectsWithZeroFeatures) {
  FeatureSet features({
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

TEST(FeatureSetTest, RejectsDuplicateObjectIds) {
  EXPECT_THROW(FeatureSet({
                   {10, 3},
                   {20, 5},
                   {10, 2},
               }),
               std::invalid_argument);
}

TEST(FeatureSetTest, RejectsUnknownObject) {
  FeatureSet features({
      {10, 3},
      {20, 5},
  });

  EXPECT_THROW(features.objectView(99), std::out_of_range);
}

// =============================================================================
// ObjectView
// =============================================================================

TEST(FeatureSetTest, ObjectViewHasCorrectSize) {
  FeatureSet features({
      {10, 3},
      {20, 5},
      {30, 2},
  });

  EXPECT_EQ(features.objectView(10).size(), 3);
  EXPECT_EQ(features.objectView(20).size(), 5);
  EXPECT_EQ(features.objectView(30).size(), 2);
}

TEST(FeatureSetTest, ObjectViewHasCorrectObjectId) {
  FeatureSet features({
      {10, 3},
      {20, 5},
      {30, 2},
  });

  EXPECT_EQ(features.objectView(10).objectId(), 10);
  EXPECT_EQ(features.objectView(20).objectId(), 20);
  EXPECT_EQ(features.objectView(30).objectId(), 30);
}

TEST(FeatureSetTest, ObjectViewIsZeroCopy) {
  FeatureSet features({
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

  EXPECT_EQ(object.status(), features.status.data() + 3);

  EXPECT_EQ(object.errors(), features.errors.data() + 3);
}

TEST(FeatureSetTest, ConstObjectViewIsZeroCopy) {
  const FeatureSet features({
      {10, 3},
      {20, 5},
      {30, 2},
  });

  auto object = features.objectView(20);

  EXPECT_EQ(object.points(), features.points.data() + 3);

  EXPECT_EQ(object.previousPoints(), features.previous_points.data() + 3);

  EXPECT_EQ(object.ids(), features.ids.data() + 3);

  EXPECT_EQ(object.objectIds(), features.object_ids.data() + 3);

  EXPECT_EQ(object.status(), features.status.data() + 3);

  EXPECT_EQ(object.errors(), features.errors.data() + 3);
}

TEST(FeatureSetTest, ObjectViewProvidesCorrectData) {
  FeatureSet features({
      {10, 3},
      {20, 5},
  });

  const FeatureSet::FeatureData data = createFeatureData(5, 100);

  features.objectView(20).copyFrom(data);

  const auto object = features.objectView(20);

  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(object.points()[i], data.points[i]);

    EXPECT_EQ(object.previousPoints()[i], data.previous_points[i]);

    EXPECT_EQ(object.ids()[i], data.ids[i]);

    EXPECT_EQ(object.status()[i], data.status[i]);

    EXPECT_EQ(object.errors()[i], data.errors[i]);

    EXPECT_EQ(object.objectIds()[i], 20);
  }
}

TEST(FeatureSetTest, ObjectViewModificationUpdatesFeatureSet) {
  FeatureSet features({
      {10, 3},
  });

  auto object = features.objectView(10);

  object.points()[0] = cv::Point2f(123.0f, 456.0f);
  object.ids()[1] = 42;
  object.status()[2] = 0;

  EXPECT_EQ(features.points[0], cv::Point2f(123.0f, 456.0f));

  EXPECT_EQ(features.ids[1], 42);

  EXPECT_EQ(features.status[2], 0);
}

// =============================================================================
// Object block layout
// =============================================================================

TEST(FeatureSetTest, ObjectBlocksAreContiguous) {
  FeatureSet features({
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

TEST(FeatureSetTest, ObjectBlocksContainCorrectObjectIds) {
  FeatureSet features({
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

TEST(FeatureSetTest, FeatureDataReportsCorrectSize) {
  const FeatureSet::FeatureData data = createFeatureData(10);

  EXPECT_EQ(data.size(), 10);
  EXPECT_FALSE(data.empty());
}

TEST(FeatureSetTest, FeatureDataReportsEmpty) {
  FeatureSet::FeatureData data;

  EXPECT_EQ(data.size(), 0);
  EXPECT_TRUE(data.empty());

  EXPECT_NO_THROW(data.checkSizes());
}

TEST(FeatureSetTest, FeatureDataRejectsMismatchedArrays) {
  FeatureSet::FeatureData data = createFeatureData(5);

  data.ids.resize(4);

  EXPECT_THROW(data.checkSizes(), std::invalid_argument);
}

TEST(FeatureSetTest, CopyFeatureData) {
  const FeatureSet::FeatureData data = createFeatureData(5, 100);

  FeatureSet features({
      {42, 5},
  });

  features.objectView(42).copyFrom(data);

  const auto object = features.objectView(42);

  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(object.points()[i], data.points[i]);
    EXPECT_EQ(object.previousPoints()[i], data.previous_points[i]);
    EXPECT_EQ(object.ids()[i], data.ids[i]);
    EXPECT_EQ(object.status()[i], data.status[i]);
    EXPECT_EQ(object.errors()[i], data.errors[i]);

    // object_ids are generated by FeatureSet.
    EXPECT_EQ(object.objectIds()[i], 42);
  }

  features.checkInvariants();
}

TEST(FeatureSetTest, CopyFeatureDataRejectsIncorrectSize) {
  FeatureSet features({
      {42, 5},
  });

  const FeatureSet::FeatureData data = createFeatureData(4);

  EXPECT_THROW(features.objectView(42).copyFrom(data), std::invalid_argument);
}

TEST(FeatureSetTest, CopyFeatureDataDoesNotModifySource) {
  const FeatureSet::FeatureData data = createFeatureData(5, 100);

  const FeatureSet::FeatureData original = data;

  FeatureSet features({
      {42, 5},
  });

  features.objectView(42).copyFrom(data);

  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(data.points[i], original.points[i]);
    EXPECT_EQ(data.previous_points[i], original.previous_points[i]);
    EXPECT_EQ(data.ids[i], original.ids[i]);
    EXPECT_EQ(data.status[i], original.status[i]);
    EXPECT_EQ(data.errors[i], original.errors[i]);
  }
}

TEST(FeatureSetTest, ObjectIdsAreGeneratedFromObjectSpecification) {
  const FeatureSet::FeatureData data = createFeatureData(5);

  FeatureSet features({
      {42, 5},
  });

  features.objectView(42).copyFrom(data);

  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(features.object_ids[i], 42);
  }
}

TEST(FeatureSet, MergeRejectsDuplicateFeatureIdsAcrossObjects) {
  FeatureSet first = createFeatureSet({1}, {2});

  FeatureSet second = createFeatureSet({2}, {2});

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

TEST(FeatureSetTest, CopiesMultipleObjects) {
  const FeatureSet::FeatureData object1 = createFeatureData(3, 100);

  const FeatureSet::FeatureData object2 = createFeatureData(5, 200);

  const FeatureSet::FeatureData object3 = createFeatureData(2, 300);

  FeatureSet features({
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

    EXPECT_EQ(features.status[i], object1.status[i]);

    EXPECT_EQ(features.errors[i], object1.errors[i]);

    EXPECT_EQ(features.object_ids[i], 10);
  }

  for (size_t i = 0; i < 5; ++i) {
    const size_t index = 3 + i;

    EXPECT_EQ(features.points[index], object2.points[i]);

    EXPECT_EQ(features.previous_points[index], object2.previous_points[i]);

    EXPECT_EQ(features.ids[index], object2.ids[i]);

    EXPECT_EQ(features.status[index], object2.status[i]);

    EXPECT_EQ(features.errors[index], object2.errors[i]);

    EXPECT_EQ(features.object_ids[index], 20);
  }

  for (size_t i = 0; i < 2; ++i) {
    const size_t index = 8 + i;

    EXPECT_EQ(features.points[index], object3.points[i]);

    EXPECT_EQ(features.previous_points[index], object3.previous_points[i]);

    EXPECT_EQ(features.ids[index], object3.ids[i]);

    EXPECT_EQ(features.status[index], object3.status[i]);

    EXPECT_EQ(features.errors[index], object3.errors[i]);

    EXPECT_EQ(features.object_ids[index], 30);
  }

  features.checkInvariants();
}

// =============================================================================
// Merge
// =============================================================================

TEST(FeatureSetTest, MergePreservesObjectOrder) {
  FeatureSet features({
      {1, 2},
      {2, 2},
      {3, 2},
      {4, 2},
  });

  FeatureSet new_features({
      {1, 1},
      {3, 1},
      {5, 1},
  });

  FeatureSet merged = features.merge(new_features);

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

TEST(FeatureSetTest, MergeProducesCorrectTotalSize) {
  FeatureSet features({
      {1, 3},
      {2, 5},
      {3, 2},
  });

  FeatureSet new_features({
      {1, 4},
      {2, 1},
      {3, 6},
  });

  FeatureSet merged = features.merge(new_features);

  EXPECT_EQ(merged.size(), 3 + 4 + 5 + 1 + 2 + 6);
}

TEST(FeatureSetTest, MergePreservesExistingFeatures) {
  FeatureSet features({
      {1, 3},
      {2, 2},
  });

  const FeatureSet::FeatureData object1 = createFeatureData(3, 100);

  const FeatureSet::FeatureData object2 = createFeatureData(2, 200);

  features.objectView(1).copyFrom(object1);
  features.objectView(2).copyFrom(object2);

  FeatureSet new_features({
      {1, 2},
      {2, 3},
  });

  FeatureSet merged = features.merge(new_features);

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

TEST(FeatureSetTest, MergeAppendsNewFeaturesToEachObject) {
  FeatureSet features({
      {1, 2},
      {2, 3},
  });

  const FeatureSet::FeatureData old1 = createFeatureData(2, 100);

  const FeatureSet::FeatureData old2 = createFeatureData(3, 200);

  features.objectView(1).copyFrom(old1);
  features.objectView(2).copyFrom(old2);

  FeatureSet new_features({
      {1, 3},
      {2, 2},
  });

  const FeatureSet::FeatureData new1 = createFeatureData(3, 300);

  const FeatureSet::FeatureData new2 = createFeatureData(2, 400);

  new_features.objectView(1).copyFrom(new1);
  new_features.objectView(2).copyFrom(new2);

  FeatureSet merged = features.merge(new_features);

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

TEST(FeatureSetTest, MergeHandlesObjectsWithNoNewFeatures) {
  FeatureSet features({
      {1, 3},
      {2, 4},
      {3, 2},
  });

  FeatureSet new_features({
      {1, 2},
      {3, 5},
  });

  FeatureSet merged = features.merge(new_features);

  EXPECT_EQ(merged.objectView(1).size(), 5);

  EXPECT_EQ(merged.objectView(2).size(), 4);

  EXPECT_EQ(merged.objectView(3).size(), 7);

  merged.checkInvariants();
}

TEST(FeatureSetTest, MergeHandlesObjectsWithNoExistingFeatures) {
  FeatureSet features({
      {1, 3},
      {2, 4},
  });

  FeatureSet new_features({
      {1, 2},
      {3, 5},
  });

  FeatureSet merged = features.merge(new_features);

  EXPECT_EQ(merged.objectView(1).size(), 5);

  EXPECT_EQ(merged.objectView(2).size(), 4);

  EXPECT_EQ(merged.objectView(3).size(), 5);

  EXPECT_TRUE(merged.containsObject(3));

  merged.checkInvariants();
}

TEST(FeatureSetTest, MergeAllowsMissingObjects) {
  FeatureSet features({
      {1, 3},
      {2, 4},
      {3, 5},
      {4, 2},
  });

  // Objects 2 and 4 do not require new features.
  FeatureSet new_features({
      {1, 2},
      {3, 3},
  });

  FeatureSet merged = features.merge(new_features);

  EXPECT_EQ(merged.objectView(1).size(), 5);

  EXPECT_EQ(merged.objectView(2).size(), 4);

  EXPECT_EQ(merged.objectView(3).size(), 8);

  EXPECT_EQ(merged.objectView(4).size(), 2);

  EXPECT_EQ(merged.objectCount(), 4);

  merged.checkInvariants();
}

TEST(FeatureSetTest, MergeAllowsAdditionalObjects) {
  FeatureSet features({
      {1, 3},
      {2, 4},
  });
  features.objectView(1).copyFrom(createFeatureData(3, 0));
  features.objectView(2).copyFrom(createFeatureData(4, 3));

  FeatureSet new_features({
      {1, 2},
      {3, 5},
      {4, 2},
  });
  new_features.objectView(1).copyFrom(createFeatureData(2, 7));
  new_features.objectView(3).copyFrom(createFeatureData(5, 9));
  new_features.objectView(4).copyFrom(createFeatureData(2, 14));

  FeatureSet merged = features.merge(new_features);

  EXPECT_EQ(merged.objectCount(), 4);

  EXPECT_EQ(merged.objectView(1).size(), 5);

  EXPECT_EQ(merged.objectView(2).size(), 4);

  EXPECT_EQ(merged.objectView(3).size(), 5);

  EXPECT_EQ(merged.objectView(4).size(), 2);

  merged.checkInvariants();
}

TEST(FeatureSetTest, MergeAllowsDifferentObjectSets) {
  FeatureSet features({
      {1, 3},
      {2, 4},
      {3, 2},
  });

  FeatureSet new_features({
      {3, 5},
      {4, 6},
      {5, 1},
  });

  FeatureSet merged = features.merge(new_features);

  EXPECT_EQ(merged.objectCount(), 5);

  EXPECT_EQ(merged.objectView(1).size(), 3);

  EXPECT_EQ(merged.objectView(2).size(), 4);

  EXPECT_EQ(merged.objectView(3).size(), 7);

  EXPECT_EQ(merged.objectView(4).size(), 6);

  EXPECT_EQ(merged.objectView(5).size(), 1);

  merged.checkInvariants();
}

// TEST(FeatureSetTest, MergeHandlesEmptyFeatureSets)
// {
//     FeatureSet features({
//         {1, 3},
//         {2, 4},
//     });

//     FeatureSet empty;

//     FeatureSet merged =
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

// TEST(FeatureSetTest, MergeEmptyIntoPopulated)
// {
//     FeatureSet empty;

//     FeatureSet new_features({
//         {1, 3},
//         {2, 5},
//     });

//     const FeatureSet::FeatureData object1 =
//         createFeatureData(3, 100);

//     const FeatureSet::FeatureData object2 =
//         createFeatureData(5, 200);

//     new_features.objectView(1).copyFrom(object1);
//     new_features.objectView(2).copyFrom(object2);

//     FeatureSet merged =
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

TEST(FeatureSetTest, ObjectViewPointsMatIsZeroCopy) {
  FeatureSet features({{1, 3}, {2, 2}});

  auto object = features.objectView(1);

  object.points()[0] = cv::Point2f(1.0f, 2.0f);
  object.points()[1] = cv::Point2f(3.0f, 4.0f);
  object.points()[2] = cv::Point2f(5.0f, 6.0f);

  cv::Mat points = object.pointsMat();

  ASSERT_EQ(points.rows, 3);
  ASSERT_EQ(points.cols, 1);
  ASSERT_EQ(points.type(), CV_32FC2);

  // Check the Mat sees the original FeatureSet data.
  EXPECT_FLOAT_EQ(points.at<cv::Point2f>(0, 0).x, 1.0f);
  EXPECT_FLOAT_EQ(points.at<cv::Point2f>(0, 0).y, 2.0f);
  EXPECT_FLOAT_EQ(points.at<cv::Point2f>(1, 0).x, 3.0f);
  EXPECT_FLOAT_EQ(points.at<cv::Point2f>(1, 0).y, 4.0f);
  EXPECT_FLOAT_EQ(points.at<cv::Point2f>(2, 0).x, 5.0f);
  EXPECT_FLOAT_EQ(points.at<cv::Point2f>(2, 0).y, 6.0f);

  // Modify through cv::Mat.
  points.at<cv::Point2f>(1, 0) = cv::Point2f(10.0f, 20.0f);

  // The underlying FeatureSet must see the modification.
  EXPECT_FLOAT_EQ(object.points()[1].x, 10.0f);
  EXPECT_FLOAT_EQ(object.points()[1].y, 20.0f);
}

TEST(FeatureSetTest, ObjectViewPointsMatContainsOnlyObjectFeatures) {
  FeatureSet features({{10, 3}, {20, 2}, {30, 4}});

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

TEST(FeatureSetTest, MergeDoesNotModifyInputs) {
  FeatureSet features({
      {1, 3},
      {2, 2},
  });

  FeatureSet new_features({
      {1, 2},
      {3, 4},
  });

  const FeatureSet::FeatureData old1 = createFeatureData(3, 100);

  const FeatureSet::FeatureData old2 = createFeatureData(2, 200);

  const FeatureSet::FeatureData new1 = createFeatureData(2, 300);

  const FeatureSet::FeatureData new3 = createFeatureData(4, 400);

  features.objectView(1).copyFrom(old1);
  features.objectView(2).copyFrom(old2);

  new_features.objectView(1).copyFrom(new1);
  new_features.objectView(3).copyFrom(new3);

  const auto original_points = features.points;
  const auto original_previous_points = features.previous_points;
  const auto original_ids = features.ids;
  const auto original_object_ids = features.object_ids;
  const auto original_status = features.status;
  const auto original_errors = features.errors;

  const auto original_new_points = new_features.points;
  const auto original_new_previous_points = new_features.previous_points;
  const auto original_new_ids = new_features.ids;
  const auto original_new_object_ids = new_features.object_ids;
  const auto original_new_status = new_features.status;
  const auto original_new_errors = new_features.errors;

  FeatureSet merged = features.merge(new_features);

  EXPECT_EQ(features.points, original_points);
  EXPECT_EQ(features.previous_points, original_previous_points);
  EXPECT_EQ(features.ids, original_ids);
  EXPECT_EQ(features.object_ids, original_object_ids);
  EXPECT_EQ(features.status, original_status);
  EXPECT_EQ(features.errors, original_errors);

  EXPECT_EQ(new_features.points, original_new_points);
  EXPECT_EQ(new_features.previous_points, original_new_previous_points);
  EXPECT_EQ(new_features.ids, original_new_ids);
  EXPECT_EQ(new_features.object_ids, original_new_object_ids);
  EXPECT_EQ(new_features.status, original_new_status);
  EXPECT_EQ(new_features.errors, original_new_errors);

  merged.checkInvariants();
}

TEST(FeatureSet, ConstructsFromFeatureData) {
  FeatureSet::FeatureData object1 = createFeatureData(3, 10);
  FeatureSet::FeatureData object2 = createFeatureData(2, 20);

  std::vector<std::pair<ObjectId, FeatureSet::FeatureData>> terms;
  terms.emplace_back(1, object1);
  terms.emplace_back(2, object2);

  FeatureSet features(terms);

  ASSERT_EQ(features.objectCount(), 2u);
  ASSERT_EQ(features.size(), 5u);

  const auto first = features.objectView(1);
  const auto second = features.objectView(2);

  EXPECT_EQ(first.size(), 3u);
  EXPECT_EQ(second.size(), 2u);

  EXPECT_EQ(first.objectId(), 1);
  EXPECT_EQ(second.objectId(), 2);
}

TEST(FeatureSet, ConstructsFromFeatureDataCopiesEveryField) {
  FeatureSet::FeatureData object1 = createFeatureData(3, 100);
  FeatureSet::FeatureData object2 = createFeatureData(2, 200);

  std::vector<std::pair<ObjectId, FeatureSet::FeatureData>> terms;
  terms.emplace_back(10, object1);
  terms.emplace_back(20, object2);

  FeatureSet features(terms);

  const auto first = features.objectView(10);
  const auto second = features.objectView(20);

  for (size_t i = 0; i < first.size(); ++i) {
    EXPECT_EQ(first.points()[i], object1.points[i]);
    EXPECT_EQ(first.previousPoints()[i], object1.previous_points[i]);
    EXPECT_EQ(first.ids()[i], object1.ids[i]);
    EXPECT_EQ(first.status()[i], object1.status[i]);
    EXPECT_EQ(first.errors()[i], object1.errors[i]);
    EXPECT_EQ(first.objectIds()[i], 10);
  }

  for (size_t i = 0; i < second.size(); ++i) {
    EXPECT_EQ(second.points()[i], object2.points[i]);
    EXPECT_EQ(second.previousPoints()[i], object2.previous_points[i]);
    EXPECT_EQ(second.ids()[i], object2.ids[i]);
    EXPECT_EQ(second.status()[i], object2.status[i]);
    EXPECT_EQ(second.errors()[i], object2.errors[i]);
    EXPECT_EQ(second.objectIds()[i], 20);
  }
}

TEST(FeatureSet, ConstructsFromEmptyFeatureData) {
  FeatureSet::FeatureData empty;

  std::vector<std::pair<ObjectId, FeatureSet::FeatureData>> terms;
  terms.emplace_back(42, empty);

  FeatureSet features(terms);

  EXPECT_EQ(features.objectCount(), 1u);
  EXPECT_EQ(features.size(), 0u);
  EXPECT_TRUE(features.containsObject(42));

  const auto object = features.objectView(42);

  EXPECT_EQ(object.objectId(), 42);
  EXPECT_EQ(object.size(), 0u);
}

TEST(FeatureSet, ConstructsFromFeatureDataRejectsMismatchedArrays) {
  FeatureSet::FeatureData data;
  data.points.resize(3);
  data.previous_points.resize(3);
  data.ids.resize(2);  // Incorrect size.
  data.status.resize(3);
  data.errors.resize(3);

  std::vector<std::pair<ObjectId, FeatureSet::FeatureData>> terms;
  terms.emplace_back(1, data);

  EXPECT_THROW(FeatureSet features(terms), std::invalid_argument);
}

TEST(FeatureSet, ConstructsFromFeatureDataDoesNotAliasSource) {
  FeatureSet::FeatureData data = createFeatureData(3, 100);

  const cv::Point2f original_point = data.points[0];
  const int original_id = data.ids[0];

  std::vector<std::pair<ObjectId, FeatureSet::FeatureData>> terms;
  terms.emplace_back(1, data);

  FeatureSet features(terms);

  data.points[0] = cv::Point2f(999.0f, 888.0f);
  data.ids[0] = 9999;

  const auto object = features.objectView(1);

  EXPECT_EQ(object.points()[0], original_point);
  EXPECT_EQ(object.ids()[0], original_id);
}

TEST(FeatureSet, ConstructsFromMap) {
  std::map<ObjectId, FeatureSet::FeatureData> terms;

  terms.emplace(10, createFeatureData(3, 100));
  terms.emplace(20, createFeatureData(2, 200));

  FeatureSet features(terms);

  EXPECT_EQ(features.objectCount(), 2u);
  EXPECT_EQ(features.size(), 5u);

  EXPECT_EQ(features.objectView(10).size(), 3u);
  EXPECT_EQ(features.objectView(20).size(), 2u);
}

TEST(FeatureSetTest, MergeCopiesEverySoAFieldCorrectly) {
  FeatureSet features({
      {1, 2},
      {2, 3},
  });

  FeatureSet new_features({
      {1, 2},
      {2, 1},
  });

  const FeatureSet::FeatureData old1 = createFeatureData(2, 10);

  const FeatureSet::FeatureData old2 = createFeatureData(3, 20);

  const FeatureSet::FeatureData new1 = createFeatureData(2, 30);

  const FeatureSet::FeatureData new2 = createFeatureData(1, 40);

  features.objectView(1).copyFrom(old1);
  features.objectView(2).copyFrom(old2);

  new_features.objectView(1).copyFrom(new1);
  new_features.objectView(2).copyFrom(new2);

  FeatureSet merged = features.merge(new_features);

  const auto object1 = merged.objectView(1);

  const auto object2 = merged.objectView(2);

  // Object 1: old + new.
  for (size_t i = 0; i < old1.size(); ++i) {
    EXPECT_EQ(object1.points()[i], old1.points[i]);

    EXPECT_EQ(object1.previousPoints()[i], old1.previous_points[i]);

    EXPECT_EQ(object1.ids()[i], old1.ids[i]);

    EXPECT_EQ(object1.objectIds()[i], 1);

    EXPECT_EQ(object1.status()[i], old1.status[i]);

    EXPECT_EQ(object1.errors()[i], old1.errors[i]);
  }

  for (size_t i = 0; i < new1.size(); ++i) {
    const size_t index = old1.size() + i;

    EXPECT_EQ(object1.points()[index], new1.points[i]);

    EXPECT_EQ(object1.previousPoints()[index], new1.previous_points[i]);

    EXPECT_EQ(object1.ids()[index], new1.ids[i]);

    EXPECT_EQ(object1.objectIds()[index], 1);

    EXPECT_EQ(object1.status()[index], new1.status[i]);

    EXPECT_EQ(object1.errors()[index], new1.errors[i]);
  }

  // Object 2: old + new.
  for (size_t i = 0; i < old2.size(); ++i) {
    EXPECT_EQ(object2.points()[i], old2.points[i]);

    EXPECT_EQ(object2.previousPoints()[i], old2.previous_points[i]);

    EXPECT_EQ(object2.ids()[i], old2.ids[i]);

    EXPECT_EQ(object2.objectIds()[i], 2);

    EXPECT_EQ(object2.status()[i], old2.status[i]);

    EXPECT_EQ(object2.errors()[i], old2.errors[i]);
  }

  for (size_t i = 0; i < new2.size(); ++i) {
    const size_t index = old2.size() + i;

    EXPECT_EQ(object2.points()[index], new2.points[i]);

    EXPECT_EQ(object2.previousPoints()[index], new2.previous_points[i]);

    EXPECT_EQ(object2.ids()[index], new2.ids[i]);

    EXPECT_EQ(object2.objectIds()[index], 2);

    EXPECT_EQ(object2.status()[index], new2.status[i]);

    EXPECT_EQ(object2.errors()[index], new2.errors[i]);
  }

  merged.checkInvariants();
}

// =============================================================================
// Performance
// =============================================================================

TEST(FeatureSetPerformance, ObjectViewMatchesSoA) {
  constexpr size_t N = 10000;
  constexpr size_t ITERATIONS = 1000;

  FeatureSet features({
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

TEST(FeatureSetPerformance, PerObjectIteration) {
  constexpr size_t FEATURES_PER_OBJECT = 2500;
  constexpr size_t OBJECT_COUNT = 4;
  constexpr size_t N = FEATURES_PER_OBJECT * OBJECT_COUNT;
  constexpr size_t ITERATIONS = 1000;

  FeatureSet features({
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

}  // namespace dyno
