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

#include "dynosam/factors/LandmarkQuadricFactor.hpp"

using namespace dyno;

TEST(LandmarkQuadricFactor, residualBasicWithPointOnEllipse) {
  // test when radii = (3, 3, 3) (so denom = 9)
  // and then we construct a point m = 1, 2, 2 = 1 + 4 + 4 = 9
  gtsam::Point3 m_local(1, 2, 2);
  gtsam::Vector3 P(3, 3, 3);
  gtsam::Pose3 I = gtsam::Pose3::Identity();

  const gtsam::Vector1 error = LandmarkQuadricFactor::residual(m_local, I, P);
  EXPECT_TRUE(gtsam::assert_equal(error, gtsam::Vector1{0.0}));
}

TEST(LandmarkQuadricFactor, residualOnSphere) {
  // plug into ellipoide equation
  // matches distance to center of sphere + 1
  gtsam::Point3 m_local(6, 6, 3);
  gtsam::Vector3 P(3, 3, 3);
  gtsam::Pose3 I = gtsam::Pose3::Identity();

  const gtsam::Vector1 error = LandmarkQuadricFactor::residual(m_local, I, P);
  EXPECT_TRUE(gtsam::assert_equal(error, gtsam::Vector1{8.0}));
}
