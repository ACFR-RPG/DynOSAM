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

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam_opt/FactorGraphTools.hpp"

using namespace dyno;

#include <glog/logging.h>
#include <gtest/gtest.h>

TEST(DynamicObjectSymbol, testReconstructMotionInfo) {
  gtsam::Key motion_key = ObjectMotionSymbol(12, 10);

  ObjectId recovered_object_id;
  FrameId recovered_frame_id;
  EXPECT_TRUE(reconstructMotionInfo(motion_key, recovered_object_id,
                                    recovered_frame_id));

  EXPECT_EQ(recovered_object_id, 12);
  EXPECT_EQ(recovered_frame_id, 10);
}

TEST(DynamicObjectSymbol, testReconstrucPoseInfo) {
  gtsam::Key pose_key = ObjectPoseSymbol(12, 12);

  ObjectId recovered_object_id;
  FrameId recovered_frame_id;
  EXPECT_TRUE(
      reconstructPoseInfo(pose_key, recovered_object_id, recovered_frame_id));

  EXPECT_EQ(recovered_object_id, 12);
  EXPECT_EQ(recovered_frame_id, 12);
}

TEST(DynamicObjectSymbol, testReconstructMotionInfoFromOtherSymbol) {
  {
    gtsam::Key motion_key = CameraPoseSymbol(10);

    ObjectId recovered_object_id;
    FrameId recovered_frame_id;
    EXPECT_FALSE(reconstructMotionInfo(motion_key, recovered_object_id,
                                       recovered_frame_id));
  }

  {
    gtsam::Key pose_key = ObjectPoseSymbol(10, 12);
    ObjectId recovered_object_id;
    FrameId recovered_frame_id;
    EXPECT_FALSE(reconstructMotionInfo(pose_key, recovered_object_id,
                                       recovered_frame_id));
  }
}

TEST(BackendDefinitions, testDynoChrExtractor) {
  gtsam::Key motion_key = ObjectMotionSymbol(12, 10);
  EXPECT_EQ(DynoChrExtractor(motion_key), kObjectMotionSymbolChar);

  gtsam::Key cam_pose_key = CameraPoseSymbol(2);
  EXPECT_EQ(DynoChrExtractor(cam_pose_key), kPoseSymbolChar);

  gtsam::Key dynamic_point_key = DynamicLandmarkSymbol(2, 10);
  EXPECT_EQ(DynoChrExtractor(dynamic_point_key), kDynamicLandmarkSymbolChar);

  gtsam::Key static_point_key = StaticLandmarkSymbol(2);
  EXPECT_EQ(DynoChrExtractor(static_point_key), kStaticLandmarkSymbolChar);
}

// TEST(FactorGraphTools, testReconstructMotionInfoFromOtherSymbol) {

//     gtsam::Key motion_key = CameraPoseSymbol(10);

//     ObjectId recovered_object_id;
//     FrameId recovered_frame_id;
//     EXPECT_FALSE(reconstructMotionInfo(motion_key, recovered_object_id,
//     recovered_frame_id));
// }
