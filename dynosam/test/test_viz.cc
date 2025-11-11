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

#include "dynosam_common/viz/Colour.hpp"

using namespace dyno;

TEST(Color, testRGBDDownCasting) {
  RGBA<uint8_t> rgbd_int(255, 255, 0);
  EXPECT_EQ(rgbd_int.r, 255);
  EXPECT_EQ(rgbd_int.g, 255);
  EXPECT_EQ(rgbd_int.b, 0);
  EXPECT_EQ(rgbd_int.a, 255);

  RGBA<float> rgbd_float(rgbd_int);
  EXPECT_EQ(rgbd_float.r, 1.0);
  EXPECT_EQ(rgbd_float.g, 1.0);
  EXPECT_EQ(rgbd_float.b, 0);
  EXPECT_EQ(rgbd_float.a, 1.0);
}

TEST(Color, testRGBDDownCasting1) {
  RGBA<uint8_t> rgbd_int(128, 100, 10);
  EXPECT_EQ(rgbd_int.r, 128);
  EXPECT_EQ(rgbd_int.g, 100);
  EXPECT_EQ(rgbd_int.b, 10);
  EXPECT_EQ(rgbd_int.a, 255);

  RGBA<float> rgbd_float(rgbd_int);
  EXPECT_FLOAT_EQ(rgbd_float.r, 128.0 / 255.0);
  EXPECT_FLOAT_EQ(rgbd_float.g, 100.0 / 255.0);
  EXPECT_FLOAT_EQ(rgbd_float.b, 10.0 / 255.0);
  EXPECT_FLOAT_EQ(rgbd_float.a, 1.0);
}

TEST(Color, testRGBDUpCasting) {
  RGBA<float> rgbd_float(0.5, 0.2, 0.1);
  EXPECT_FLOAT_EQ(rgbd_float.r, 0.5);
  EXPECT_FLOAT_EQ(rgbd_float.g, 0.2);
  EXPECT_FLOAT_EQ(rgbd_float.b, 0.1);
  EXPECT_FLOAT_EQ(rgbd_float.a, 1.0);

  RGBA<uint8_t> rgbd_int(rgbd_float);
  EXPECT_EQ(rgbd_int.r, static_cast<uint8_t>(0.5 * 255.0));
  EXPECT_EQ(rgbd_int.g, static_cast<uint8_t>(0.2 * 255.0));
  EXPECT_EQ(rgbd_int.b, static_cast<uint8_t>(0.1 * 255.0));
  EXPECT_EQ(rgbd_int.a, static_cast<uint8_t>(255.0));
}
