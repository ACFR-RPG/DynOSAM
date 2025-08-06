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

#include <dynosam/utils/JsonUtils.hpp>
#include <filesystem>
#include <nlohmann/json.hpp>  //for gt packet seralize tests

#include "dynosam/common/Exceptions.hpp"
#include "dynosam/common/GroundTruthPacket.hpp"
#include "dynosam/frontend/FrontendInputPacket.hpp"
#include "dynosam/frontend/vision/Feature.hpp"
#include "dynosam/logger/Logger.hpp"
#include "dynosam/utils/Statistics.hpp"
#include "dynosam/utils/Variant.hpp"
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
  EXPECT_EQ(mat1.image.data, mat2.image.data);
  EXPECT_EQ(container1.frameId(), container2.frameId());
  EXPECT_EQ(container1.timestamp(), container2.timestamp());

  // Modifying one should affect the other (since shared)
  mat1.image.at<uint8_t>(0, 0) = 42;
  EXPECT_EQ(mat2.image.at<uint8_t>(0, 0), 42);
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

  EXPECT_NE(mat1.image.data, mat2.image.data);

  // Changing one does NOT affect the other
  mat1.image.at<uint8_t>(0, 0) = 42;
  EXPECT_NE(mat2.image.at<uint8_t>(0, 0), 42);
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
struct internal::EnableBitMaskOperators<TestOptions> : std::true_type {};

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

TEST(ImageContainerSubset, testBasicSubsetContainer) {
  cv::Mat depth(cv::Size(25, 25), CV_64F);
  uchar* depth_ptr = depth.data;

  cv::Mat optical_flow(cv::Size(25, 25), CV_32FC2);
  uchar* optical_flow_ptr = optical_flow.data;
  ImageContainerSubset<ImageType::Depth, ImageType::OpticalFlow> ics{
      ImageWrapper<ImageType::Depth>(depth),
      ImageWrapper<ImageType::OpticalFlow>(optical_flow)};

  cv::Mat retrieved_depth = ics.get<ImageType::Depth>();
  cv::Mat retrieved_of = ics.get<ImageType::OpticalFlow>();

  EXPECT_EQ(retrieved_depth.data, depth_ptr);
  EXPECT_EQ(retrieved_of.data, optical_flow_ptr);

  EXPECT_EQ(depth.size(), retrieved_depth.size());
  EXPECT_EQ(optical_flow.size(), retrieved_of.size());
}

TEST(ImageContainerSubset, testSubsetContainerExists) {
  cv::Mat depth(cv::Size(25, 25), CV_64F);
  cv::Mat optical_flow(cv::Size(25, 25), CV_32FC2);
  ImageContainerSubset<ImageType::Depth, ImageType::OpticalFlow,
                       ImageType::RGBMono>
      ics{ImageWrapper<ImageType::Depth>(depth),
          ImageWrapper<ImageType::OpticalFlow>(optical_flow),
          ImageWrapper<ImageType::RGBMono>()};

  EXPECT_TRUE(ics.exists<ImageType::Depth>());
  EXPECT_TRUE(ics.exists<ImageType::OpticalFlow>());

  EXPECT_FALSE(ics.exists<ImageType::RGBMono>());
}

TEST(ImageContainerSubset, testBasicMakeSubset) {
  cv::Mat depth(cv::Size(50, 50), CV_64F);
  cv::Mat optical_flow(cv::Size(50, 50), CV_32FC2);
  uchar* optical_flow_ptr = optical_flow.data;
  ImageContainerSubset<ImageType::Depth, ImageType::OpticalFlow> ics{
      ImageWrapper<ImageType::Depth>(depth),
      ImageWrapper<ImageType::OpticalFlow>(optical_flow)};

  EXPECT_EQ(ics.index<ImageType::OpticalFlow>(), 1u);

  ImageContainerSubset<ImageType::OpticalFlow> subset =
      ics.makeSubset<ImageType::OpticalFlow>();
  EXPECT_EQ(subset.index<ImageType::OpticalFlow>(), 0u);

  cv::Mat retrieved_of = ics.get<ImageType::OpticalFlow>();
  cv::Mat subset_retrieved_of = subset.get<ImageType::OpticalFlow>();

  EXPECT_EQ(retrieved_of.data, optical_flow_ptr);
  EXPECT_EQ(subset_retrieved_of.data, optical_flow_ptr);
}

TEST(ImageContainerSubset, testSafeGet) {
  cv::Mat depth(cv::Size(25, 25), CV_64F);
  cv::Mat optical_flow(cv::Size(25, 25), CV_32FC2);
  ImageContainerSubset<ImageType::Depth, ImageType::OpticalFlow,
                       ImageType::RGBMono>
      ics{ImageWrapper<ImageType::Depth>(depth),
          ImageWrapper<ImageType::OpticalFlow>(optical_flow),
          ImageWrapper<ImageType::RGBMono>()};

  cv::Mat tmp;
  EXPECT_FALSE(ics.safeGet<ImageType::RGBMono>(tmp));

  // tmp shoudl now be the same as the depth ptr
  EXPECT_TRUE(ics.safeGet<ImageType::Depth>(tmp));
  EXPECT_EQ(depth.data, tmp.data);
  EXPECT_EQ(depth.size(), tmp.size());

  // tmp shoudl now be the same as the optical flow ptr
  EXPECT_TRUE(ics.safeGet<ImageType::OpticalFlow>(tmp));
  EXPECT_EQ(optical_flow.data, tmp.data);
  EXPECT_EQ(optical_flow.size(), tmp.size());
}

TEST(ImageContainerSubset, testSafeClone) {
  cv::Mat depth(cv::Size(50, 50), CV_64F);
  cv::Mat optical_flow(cv::Size(50, 50), CV_32FC2);
  ImageContainerSubset<ImageType::Depth, ImageType::OpticalFlow,
                       ImageType::RGBMono>
      ics{ImageWrapper<ImageType::Depth>(depth),
          ImageWrapper<ImageType::OpticalFlow>(optical_flow),
          ImageWrapper<ImageType::RGBMono>()};

  cv::Mat tmp;
  EXPECT_FALSE(ics.cloneImage<ImageType::RGBMono>(tmp));
  EXPECT_TRUE(tmp.empty());

  EXPECT_TRUE(ics.cloneImage<ImageType::Depth>(tmp));
  EXPECT_NE(depth.data, tmp.data);  // no longer the same data
  EXPECT_EQ(depth.size(), tmp.size());

  EXPECT_TRUE(ics.cloneImage<ImageType::OpticalFlow>(tmp));
  EXPECT_NE(optical_flow.data, tmp.data);  // no longer the same data
  EXPECT_EQ(optical_flow.size(), tmp.size());
}

TEST(ImageContainerDeprecate, testImageContainerIndexing) {
  EXPECT_EQ(ImageContainerDeprecate::Index<ImageType::RGBMono>(), 0u);
  EXPECT_EQ(ImageContainerDeprecate::Index<ImageType::Depth>(), 1u);
  EXPECT_EQ(ImageContainerDeprecate::Index<ImageType::OpticalFlow>(), 2u);
  EXPECT_EQ(ImageContainerDeprecate::Index<ImageType::SemanticMask>(), 3u);
  EXPECT_EQ(ImageContainerDeprecate::Index<ImageType::MotionMask>(), 4u);
}

TEST(ImageContainerDeprecate, CreateRGBDSemantic) {
  cv::Mat rgb(cv::Size(50, 50), CV_8UC1);
  cv::Mat depth(cv::Size(50, 50), CV_64F);
  cv::Mat optical_flow(cv::Size(50, 50), CV_32FC2);
  cv::Mat semantic_mask(cv::Size(50, 50), CV_32SC1);
  ImageContainerDeprecate::Ptr rgbd_semantic = ImageContainerDeprecate::Create(
      0u, 0u, ImageWrapper<ImageType::RGBMono>(rgb),
      ImageWrapper<ImageType::Depth>(depth),
      ImageWrapper<ImageType::OpticalFlow>(optical_flow),
      ImageWrapper<ImageType::SemanticMask>(semantic_mask));

  EXPECT_TRUE(rgbd_semantic->hasSemanticMask());
  EXPECT_TRUE(rgbd_semantic->hasDepth());
  EXPECT_FALSE(rgbd_semantic->hasMotionMask());
  EXPECT_FALSE(rgbd_semantic->isMonocular());
}

TEST(ImageContainerDeprecate, CreateRGBDSemanticWithInvalidSizes) {
  cv::Mat rgb(cv::Size(25, 25), CV_8UC1);
  cv::Mat depth(cv::Size(50, 50), CV_64F);
  cv::Mat optical_flow(cv::Size(50, 50), CV_32FC2);
  cv::Mat semantic_mask(cv::Size(50, 50), CV_32SC1);
  EXPECT_THROW(
      {
        ImageContainerDeprecate::Create(
            0u, 0u, ImageWrapper<ImageType::RGBMono>(rgb),
            ImageWrapper<ImageType::Depth>(depth),
            ImageWrapper<ImageType::OpticalFlow>(optical_flow),
            ImageWrapper<ImageType::SemanticMask>(semantic_mask));
      },
      ImageContainerConstructionException);
}

TEST(FeatureContainer, basicAdd) {
  FeatureContainer fc;
  EXPECT_EQ(fc.size(), 0u);
  EXPECT_FALSE(fc.exists(1));

  Feature f;
  f.trackletId(1);

  fc.add(f);
  EXPECT_EQ(fc.size(), 1u);
  EXPECT_TRUE(fc.exists(1));

  // this implicitly tests map access
  auto fr = fc.getByTrackletId(1);
  EXPECT_TRUE(fr != nullptr);
  EXPECT_EQ(*fr, f);
}

TEST(FeatureContainer, basicRemove) {
  FeatureContainer fc;
  EXPECT_EQ(fc.size(), 0u);

  Feature f;
  f.trackletId(1);

  fc.add(f);
  EXPECT_EQ(fc.size(), 1u);

  fc.remove(1);
  EXPECT_FALSE(fc.exists(1));

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
    auto usable_iterator = fc.beginUsable();
    EXPECT_EQ(std::distance(usable_iterator.begin(), usable_iterator.end()),
              10);
  }

  fc.markOutliers({3});
  fc.markOutliers({4});

  {
    auto usable_iterator = fc.beginUsable();
    EXPECT_EQ(std::distance(usable_iterator.begin(), usable_iterator.end()), 8);

    for (const auto& i : usable_iterator) {
      EXPECT_TRUE(i->trackletId() != 3 || i->trackletId() != 4);
    }
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

TEST(JsonIO, TrackedValueStatusKp) {
  KeypointStatus kp =
      dyno_testing::makeStatusKeypointMeasurement(4, 3, 1, Keypoint(0, 1));

  using json = nlohmann::json;
  json j = (KeypointStatus)kp;

  auto kp_load = j.template get<KeypointStatus>();
  // TODO: needs equals operator
  EXPECT_EQ(kp_load, kp);
}

TEST(JsonIO, TrackedValueStatusKps) {
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

TEST(JsonIO, testGenericObjectCentricMap) {
  using Map = GenericObjectCentricMap<gtsam::Pose3>;

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

TEST(TrackedValueStatus, testIsTimeInvariant) {
  TrackedValueStatus<Keypoint> status_time_invariant(
      MeasurementWithCovariance<Keypoint>{Keypoint()},
      TrackedValueStatus<Keypoint>::MeaninglessFrame, 0, 0,
      ReferenceFrame::GLOBAL);

  EXPECT_TRUE(status_time_invariant.isTimeInvariant());

  TrackedValueStatus<Keypoint> status_time_variant(
      MeasurementWithCovariance<Keypoint>{Keypoint()},
      0,  // use zero
      0, 0, ReferenceFrame::GLOBAL);
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
