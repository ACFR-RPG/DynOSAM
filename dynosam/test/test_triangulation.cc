/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Yiduo Wang
 (yiduo.wang@sydney.edu.au)
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
#include <gtest/gtest.h>
#include <gtsam/base/Vector.h>

#include "dynosam/backend/MonoBackendTools.hpp"
#include "dynosam_common/Types.hpp"

using namespace dyno;

TEST(MonoBackendTools, triangulatePoint3Vector) {
  gtsam::Pose3 X_world_camera_prev;
  // Use Quaternion and translation to construct the current pose
  // Around y axis, 15 degree - equivalent to 15 degree to the right in yaw
  gtsam::Pose3 X_world_camera_curr(gtsam::Rot3(0.9914449, 0.0, 0.1305262, 0.0),
                                   gtsam::Point3(0.1, 0.1, 2.0));

  // std::cout << X_world_camera_prev << "\n";
  // std::cout << X_world_camera_curr << "\n";

  gtsam::Matrix3 obj_rot = Eigen::Matrix3d::Identity();
  // Euler angle x = 5 degree, y = 10 degree, z = 15 degree, order XYZ
  obj_rot << 0.98106026, -0.08583165, 0.17364818, 0.12895841, 0.95833311,
      -0.254887, -0.14453543, 0.2724529, 0.95125124;

  // std::cout << obj_rot << "\n";

  gtsam::Matrix3 intrinsic = Eigen::Matrix3d::Identity();
  intrinsic(0, 0) = 320.0;
  intrinsic(1, 1) = 320.0;
  intrinsic(0, 2) = 320.0;
  intrinsic(1, 2) = 240.0;

  // std::cout << intrinsic << "\n";

  gtsam::Point3Vector points_prev, points_curr;

  // a 2*2*2 cude
  points_prev.push_back(Eigen::Vector3d(5.0, 1.0, 5.0));
  points_prev.push_back(Eigen::Vector3d(7.0, 1.0, 5.0));
  points_prev.push_back(Eigen::Vector3d(5.0, 1.0, 7.0));
  points_prev.push_back(Eigen::Vector3d(7.0, 1.0, 7.0));
  points_prev.push_back(Eigen::Vector3d(5.0, -1.0, 5.0));
  points_prev.push_back(Eigen::Vector3d(7.0, -1.0, 5.0));
  points_prev.push_back(Eigen::Vector3d(5.0, -1.0, 7.0));
  points_prev.push_back(Eigen::Vector3d(7.0, -1.0, 7.0));

  // same rotation as obj_rot
  gtsam::Pose3 H_prev_curr_world(
      gtsam::Rot3(0.9862359, 0.1336749, 0.0806561, 0.0544469),
      gtsam::Point3(2.0, 0.0, 0.0));
  // std::cout << "H_prev_curr_world\n" << H_prev_curr_world << std::endl;
  for (int i = 0; i < 8; i++) {
    points_curr.push_back(H_prev_curr_world.transformFrom(points_prev[i]));
  }

  gtsam::Point2Vector observation_prev, observation_curr;

  for (int i = 0; i < 8; i++) {
    gtsam::Point3 local_point_prev =
        intrinsic * X_world_camera_prev.transformTo(points_prev[i]);
    gtsam::Point3 local_point_curr =
        intrinsic * X_world_camera_curr.transformTo(points_curr[i]);

    observation_prev.push_back(
        Eigen::Vector2d(local_point_prev.x() / local_point_prev.z(),
                        local_point_prev.y() / local_point_prev.z()));
    observation_curr.push_back(
        Eigen::Vector2d(local_point_curr.x() / local_point_curr.z(),
                        local_point_curr.y() / local_point_curr.z()));
  }

  gtsam::Point3Vector points_world =
      dyno::mono_backend_tools::triangulatePoint3Vector(
          X_world_camera_prev, X_world_camera_curr, intrinsic, observation_prev,
          observation_curr, obj_rot);

  gtsam::Point3Vector expected_points_world;
  // [ 3.07637456  4.69454556  2.5236143   3.96822456  3.30434543  5.0268543   2.68667343
  // 4.19481659] [ 0.61527491  0.67064937  0.50472286  0.56688922 -0.66086909
  // -0.71812204 -0.53733469 -0.59925951]
  // [ 3.07637456  3.35324683  3.53306001  3.96822456  3.30434543  3.59061022  3.7613428
  // 4.19481659]
  expected_points_world.push_back(
      gtsam::Point3(3.07637456, 0.61527491, 3.07637456));
  expected_points_world.push_back(
      gtsam::Point3(4.69454556, 0.67064937, 3.35324683));
  expected_points_world.push_back(
      gtsam::Point3(2.5236143, 0.50472286, 3.53306001));
  expected_points_world.push_back(
      gtsam::Point3(3.96822456, 0.56688922, 3.96822456));
  expected_points_world.push_back(
      gtsam::Point3(3.30434543, -0.66086909, 3.30434543));
  expected_points_world.push_back(
      gtsam::Point3(5.0268543, -0.71812204, 3.59061022));
  expected_points_world.push_back(
      gtsam::Point3(2.68667343, -0.53733469, 3.7613428));
  expected_points_world.push_back(
      gtsam::Point3(4.19481659, -0.59925951, 4.19481659));

  EXPECT_EQ(expected_points_world.size(), points_world.size());
  for (size_t i = 0; i < expected_points_world.size(); i++) {
    EXPECT_TRUE(gtsam::assert_equal(expected_points_world.at(i),
                                    points_world.at(i), 1.0e-5));
  }
}

TEST(MonoBackendTools, triangulatePoint3VectorNonExpanded) {
  gtsam::Pose3 X_world_camera_prev;
  // Use Quaternion and translation to construct the current pose
  // Around y axis, 15 degree - equivalent to 15 degree to the right in yaw
  gtsam::Pose3 X_world_camera_curr(gtsam::Rot3(0.9914449, 0.0, 0.1305262, 0.0),
                                   gtsam::Point3(0.1, 0.1, 2.0));

  // std::cout << X_world_camera_prev << "\n";
  // std::cout << X_world_camera_curr << "\n";

  gtsam::Matrix3 obj_rot = Eigen::Matrix3d::Identity();
  // Euler angle x = 5 degree, y = 10 degree, z = 15 degree, order XYZ
  obj_rot << 0.98106026, -0.08583165, 0.17364818, 0.12895841, 0.95833311,
      -0.254887, -0.14453543, 0.2724529, 0.95125124;

  // std::cout << obj_rot << "\n";

  gtsam::Matrix3 intrinsic = Eigen::Matrix3d::Identity();
  intrinsic(0, 0) = 320.0;
  intrinsic(1, 1) = 320.0;
  intrinsic(0, 2) = 320.0;
  intrinsic(1, 2) = 240.0;

  // std::cout << intrinsic << "\n";

  gtsam::Point3Vector points_prev, points_curr;

  // a 2*2*2 cude
  points_prev.push_back(Eigen::Vector3d(5.0, 1.0, 5.0));
  points_prev.push_back(Eigen::Vector3d(7.0, 1.0, 5.0));
  points_prev.push_back(Eigen::Vector3d(5.0, 1.0, 7.0));
  points_prev.push_back(Eigen::Vector3d(7.0, 1.0, 7.0));
  points_prev.push_back(Eigen::Vector3d(5.0, -1.0, 5.0));
  points_prev.push_back(Eigen::Vector3d(7.0, -1.0, 5.0));
  points_prev.push_back(Eigen::Vector3d(5.0, -1.0, 7.0));
  points_prev.push_back(Eigen::Vector3d(7.0, -1.0, 7.0));

  // same rotation as obj_rot
  gtsam::Pose3 H_prev_curr_world(
      gtsam::Rot3(0.9862359, 0.1336749, 0.0806561, 0.0544469),
      gtsam::Point3(2.0, 0.0, 0.0));
  // std::cout << "H_prev_curr_world\n" << H_prev_curr_world << std::endl;
  for (int i = 0; i < 8; i++) {
    points_curr.push_back(H_prev_curr_world.transformFrom(points_prev[i]));
  }

  gtsam::Point2Vector observation_prev, observation_curr;

  for (int i = 0; i < 8; i++) {
    gtsam::Point3 local_point_prev =
        intrinsic * X_world_camera_prev.transformTo(points_prev[i]);
    gtsam::Point3 local_point_curr =
        intrinsic * X_world_camera_curr.transformTo(points_curr[i]);

    observation_prev.push_back(
        Eigen::Vector2d(local_point_prev.x() / local_point_prev.z(),
                        local_point_prev.y() / local_point_prev.z()));
    observation_curr.push_back(
        Eigen::Vector2d(local_point_curr.x() / local_point_curr.z(),
                        local_point_curr.y() / local_point_curr.z()));
  }

  gtsam::Point3Vector points_world =
      dyno::mono_backend_tools::triangulatePoint3VectorNonExpanded(
          X_world_camera_prev, X_world_camera_curr, intrinsic, observation_prev,
          observation_curr, obj_rot);

  gtsam::Point3Vector expected_points_world;
  // [ 3.81828248  5.7540693   3.24108742  4.98904018  4.00304244  6.02650317  3.38370898
  // 5.18169805] [ 0.7636565   0.8220099   0.64821748  0.71272003 -0.80060849
  // -0.86092902 -0.6767418  -0.74024258]
  // [ 3.81828248  4.1100495   4.53752238  4.98904018  4.00304244  4.30464512  4.73719257
  // 5.18169805]
  expected_points_world.push_back(
      gtsam::Point3(3.81828248, 0.7636565, 3.81828248));
  expected_points_world.push_back(
      gtsam::Point3(5.7540693, 0.8220099, 4.1100495));
  expected_points_world.push_back(
      gtsam::Point3(3.24108742, 0.64821748, 4.53752238));
  expected_points_world.push_back(
      gtsam::Point3(4.98904018, 0.71272003, 4.98904018));
  expected_points_world.push_back(
      gtsam::Point3(4.00304244, -0.80060849, 4.00304244));
  expected_points_world.push_back(
      gtsam::Point3(6.02650317, -0.86092902, 4.30464512));
  expected_points_world.push_back(
      gtsam::Point3(3.38370898, -0.6767418, 4.73719257));
  expected_points_world.push_back(
      gtsam::Point3(5.18169805, -0.74024258, 5.18169805));

  EXPECT_EQ(expected_points_world.size(), points_world.size());
  for (size_t i = 0; i < expected_points_world.size(); i++) {
    EXPECT_TRUE(gtsam::assert_equal(expected_points_world.at(i),
                                    points_world.at(i), 1.0e-5));
  }
}
