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

#include <cmath>
#include <exception>

#include "dynosam_common/Algorithms.hpp"

using namespace dyno;

// Taken from https://www.hungarianalgorithm.com/examplehungarianalgorithm.php
TEST(Algorithms, HungarianSanityCheckSquareMatrix) {
  gtsam::Matrix44 costs;
  costs << 82, 83, 69, 92, 77, 37, 49, 92, 11, 69, 5, 86, 8, 9, 98, 23;

  Eigen::VectorXi assignment;
  const double assigned_cost =
      internal::HungarianAlgorithm().solve(costs, assignment);

  EXPECT_EQ(assigned_cost, 140.0);

  // answer should be assignment of:
  //  W1 -> J3, row0 -> col2
  //  W2 -> J2, row1 -> col1
  //  W3 -> J1, row2 -> col0,
  //  W4 -> J4, row3 -> col3
  Eigen::VectorXd expected_assignment(4);
  expected_assignment << 2, 1, 0, 3;

  Eigen::VectorXd assignmentd = assignment.cast<double>();
  // need to cast to Eigen::VectorXd to make gtsam templating/traits happy as
  // all their matrix operations are templated on double's!
  EXPECT_TRUE(gtsam::assert_equal(expected_assignment, assignmentd));
}

TEST(Algorithms, HungarianSanityCheckMoreJobs) {
  gtsam::Matrix45 costs;
  costs << 82, 83, 69, 92, 23, 77, 37, 49, 92, 7, 11, 69, 5, 86, 9, 8, 9, 98,
      23, 85;

  Eigen::VectorXi assignment;
  internal::HungarianAlgorithm().solve(costs, assignment);

  Eigen::VectorXd expected_assignment(4);
  expected_assignment << 2, 1, 0, 3;

  Eigen::VectorXd assignmentd = assignment.cast<double>();
  EXPECT_EQ(assignment.rows(), 4u);
}

TEST(Algorithms, HungarianTestSimpleArgMax) {
  // a simple assignment where we want to find the MAX score
  // testing that we just use -log in the score
  gtsam::Matrix33 costs;
  costs << 3, 12, 20, 2, 51, 4, 33, 14, 5;

  // assignment should be
  // W1 -> J3
  // W2 -> J2
  // W3 -> J1
  // apply scalign to the costs to turn the cost function from an argmax to an
  // argmin problem which the hungrian problem sovles
  gtsam::Matrix33 loged_costs =
      costs.unaryExpr([](double x) { return 1.0 / x * 10; });

  LOG(INFO) << loged_costs;

  Eigen::VectorXi assignment;
  internal::HungarianAlgorithm().solve(loged_costs, assignment);

  Eigen::VectorXd expected_assignment(3);
  expected_assignment << 2, 1, 0;

  Eigen::VectorXd assignmentd = assignment.cast<double>();
  EXPECT_TRUE(gtsam::assert_equal(expected_assignment, assignmentd));
}

TEST(Algorithms, OptimalAssignment) {}
